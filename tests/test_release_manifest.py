from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_release_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_release_manifest", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

DEFAULT_DELIVERABLE_PATTERNS = _MODULE.DEFAULT_DELIVERABLE_PATTERNS
QA_SCHEMA_VERSION = _MODULE.QA_SCHEMA_VERSION
build_release_manifest = _MODULE.build_release_manifest
sha256_file = _MODULE.sha256_file


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _init_fixture(tmp_path: Path) -> dict[str, str]:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".gitignore").write_text("artifacts/release/\n", encoding="utf-8")
    for path, content in {
        "README.md": "# Test release\n",
        "CITATION.cff": "cff-version: 1.2.0\ntitle: Test\n",
        "LICENSE": "Test only\n",
        "configs/stage2_evaluation.yaml": "evaluation: validation\n",
        "data/processed/model_matrix.parquet": "fixture bytes\n",
        "artifacts/stage2/output.json": '{"metric": 1}\n',
        "artifacts/stage2/summary.json": '{"pilot": true}\n',
    }.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    config_sha = sha256_file(tmp_path / "configs/stage2_evaluation.yaml")
    input_sha = sha256_file(tmp_path / "data/processed/model_matrix.parquet")
    output_sha = sha256_file(tmp_path / "artifacts/stage2/output.json")
    receipt = {
        "experiment_id": "stage2_public_core_pilot_test",
        "git": {"commit": "PENDING", "dirty": False},
        "input_hashes": {
            "evaluation_config": {
                "path": "configs/stage2_evaluation.yaml",
                "sha256": config_sha,
            },
            "model_matrix": {
                "path": "data/processed/model_matrix.parquet",
                "sha256": input_sha,
            },
        },
        "outputs": {
            "result": {
                "path": "artifacts/stage2/output.json",
                "sha256": output_sha,
            }
        },
        "pilot": True,
        "status": "completed",
        "summary": {
            "experiment_id": "stage2_public_core_pilot_test",
            "label": "public_core_pilot",
            "pilot": True,
            "superiority_claim_permitted": False,
            "test_set_opened": False,
            "variants": [
                {
                    "model_variant": "base",
                    "pilot": True,
                    "superiority_claim_permitted": False,
                }
            ],
        },
        "test_set_opened": False,
    }
    receipt_path = tmp_path / "artifacts/stage2/run_receipt.json"
    _write_json(receipt_path, receipt)

    registry_path = tmp_path / "experiments/registry.csv"
    registry_path.parent.mkdir(parents=True)
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment_id",
                "stage",
                "status",
                "started_at_utc",
                "completed_at_utc",
                "data_manifest_sha256",
                "git_commit",
                "config_path",
                "summary_path",
            ]
        )
        writer.writerow(
            [
                "stage2_public_core_pilot_test",
                "stage2",
                "completed",
                "",
                "",
                "",
                "PENDING",
                "configs/stage2_evaluation.yaml",
                "artifacts/stage2/summary.json",
            ]
        )

    qa_path = tmp_path / "artifacts/release/qa_receipt.json"
    _write_json(
        qa_path,
        {
            "checks": [{"id": "pytest", "required": True, "status": "passed"}],
            "git": {"commit": "PENDING", "dirty": False},
            "schema_version": QA_SCHEMA_VERSION,
            "status": "passed",
        },
    )

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "fixture"],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Bind receipts and registry to the committed snapshot, then commit the binding.
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["git"]["commit"] = commit
    _write_json(receipt_path, receipt)
    rows = list(csv.reader(registry_path.open(encoding="utf-8", newline="")))
    rows[1][6] = commit
    with registry_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    subprocess.run(
        ["git", "add", "artifacts/stage2/run_receipt.json", "experiments/registry.csv"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "bind fixture"],
        cwd=tmp_path,
        check=True,
    )
    return {
        "qa": "artifacts/release/qa_receipt.json",
        "receipt": "artifacts/stage2/run_receipt.json",
    }


def _build(tmp_path: Path, fixture: dict[str, str], **kwargs: object) -> dict[str, object]:
    return build_release_manifest(
        tmp_path,
        receipt_paths=[fixture["receipt"]],
        qa_receipt_paths=[fixture["qa"]],
        deliverable_patterns=[
            "README.md",
            "CITATION.cff",
            "LICENSE",
            "artifacts/release/*.json",
        ],
        hash_bindings={},
        **kwargs,
    )


def test_default_deliverables_exclude_archives_and_distribution_packages() -> None:
    forbidden_suffixes = (".tar", ".tar.gz", ".tgz", ".whl", ".zip")
    assert all(not pattern.startswith("dist/") for pattern in DEFAULT_DELIVERABLE_PATTERNS)
    assert all(
        not pattern.casefold().endswith(forbidden_suffixes)
        for pattern in DEFAULT_DELIVERABLE_PATTERNS
    )


def test_manifest_is_deterministic_relative_and_acyclic(tmp_path: Path) -> None:
    fixture = _init_fixture(tmp_path)
    first = _build(tmp_path, fixture)
    manifest_path = tmp_path / "artifacts/release/release_manifest.json"
    first_bytes = manifest_path.read_bytes()
    second = _build(tmp_path, fixture)
    assert manifest_path.read_bytes() == first_bytes
    assert first == second
    assert {
        "configs",
        "inputs",
        "outputs",
        "evaluation",
        "claims",
        "qa_receipts",
        "deliverables",
    }.issubset(first)
    assert first["claims"]["stage2"]["pilot"] is True
    assert first["claims"]["stage2"]["superiority_claim_permitted"] is False

    manifest_relative = "artifacts/release/release_manifest.json"
    serialized = json.dumps(first)
    assert manifest_relative not in serialized
    for section in ("configs", "inputs", "outputs", "qa_receipts", "deliverables"):
        for record in first[section]:
            assert not Path(record["path"]).is_absolute()
            assert ".." not in Path(record["path"]).parts


def test_manifest_refuses_dirty_worktree_without_override(tmp_path: Path) -> None:
    fixture = _init_fixture(tmp_path)
    (tmp_path / "README.md").write_text("# Dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty Git worktree"):
        _build(tmp_path, fixture)
    manifest = _build(tmp_path, fixture, allow_dirty=True)
    assert manifest["git"]["dirty"] is True


def test_manifest_fails_on_receipt_hash_mismatch(tmp_path: Path) -> None:
    fixture = _init_fixture(tmp_path)
    output = tmp_path / "artifacts/stage2/output.json"
    output.write_text('{"metric": 999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _build(tmp_path, fixture, allow_dirty=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pilot", False),
        ("superiority_claim_permitted", True),
    ],
)
def test_manifest_enforces_stage2_claim_boundary(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    fixture = _init_fixture(tmp_path)
    receipt_path = tmp_path / fixture["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["summary"][field] = value
    _write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="Stage 2"):
        _build(tmp_path, fixture, allow_dirty=True)
