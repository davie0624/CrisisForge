"""Build a deterministic, fail-closed manifest for a CrisisForge release.

The release manifest is deliberately acyclic:

* it contains no generation timestamp;
* every referenced path is relative to the repository root;
* every referenced file is hashed and validated before the manifest is written;
* the manifest never includes its own path or hash.

Run the release QA first, commit the deliverables, and invoke this script from a
clean worktree.  ``--allow-dirty`` exists for development diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "crisisforge.release-manifest.v1"
QA_SCHEMA_VERSION = "crisisforge.release-qa-receipt.v1"

DEFAULT_RECEIPTS = (
    "artifacts/phase0/run_receipt.json",
    "artifacts/stage0_baselines/run_receipt.json",
    "artifacts/stage1_switching_factor/run_receipt.json",
    "artifacts/stage2_public_core_pilot/run_receipt.json",
    "artifacts/stage3_comparison/run_receipt.json",
    "artifacts/stage5_decisions/run_receipt.json",
    "artifacts/stage6_counterfactual/run_receipt.json",
)

DEFAULT_QA_RECEIPTS = ("artifacts/release/qa_receipt.json",)

DEFAULT_DELIVERABLE_PATTERNS = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    "docs/*.md",
    "docs/research_log/*.md",
    "reports/*.md",
    "reports/*.pdf",
    "reports/figures/*.json",
    "reports/figures/*.png",
    "reports/figures/*.svg",
    "slides/*.md",
    "slides/*.pdf",
    "slides/*.pptx",
    "linkedin/*.md",
    "linkedin/*.png",
    "linkedin/*.svg",
)

DEFAULT_REQUIRED_DELIVERABLES = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
)

# Some receipts predate the uniform {"path": ..., "sha256": ...} reference
# structure. These bindings make every hash-only declaration verifiable.
DEFAULT_HASH_BINDINGS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "artifacts/phase0/run_receipt.json": (
        ("config", "configs/data_catalog.yaml", ("phase0_manifest_file",)),
        ("config", "configs/pipeline.yaml", ("phase0_manifest_file",)),
        ("input", "data/processed/model_matrix.parquet", ("phase0_manifest_file",)),
        (
            "output",
            "artifacts/phase0/quality_report.json",
            ("phase0_manifest_file",),
        ),
    ),
    "artifacts/stage0_baselines/run_receipt.json": (
        ("config", "configs/stage0_baselines.yaml", ("config_sha256",)),
        (
            "input",
            "data/processed/model_matrix.parquet",
            ("model_matrix_sha256",),
        ),
        (
            "input",
            "artifacts/phase0/manifest.json",
            ("phase0_manifest_sha256",),
        ),
    ),
    "artifacts/stage1_switching_factor/run_receipt.json": (
        (
            "input",
            "data/processed/model_matrix.parquet",
            ("model_matrix_sha256",),
        ),
    ),
    "artifacts/stage3_comparison/run_receipt.json": (
        ("config", "configs/stage3_comparison.yaml", ("config_sha256",)),
        (
            "input",
            "artifacts/stage0_baselines/rolling_results.csv",
            ("input_hashes", "stage0_detail"),
        ),
        (
            "input",
            "artifacts/stage0_baselines/run_receipt.json",
            ("input_hashes", "stage0_receipt"),
        ),
        (
            "input",
            "artifacts/stage1_switching_factor/rolling_results.csv",
            ("input_hashes", "stage1_detail"),
        ),
        (
            "input",
            "artifacts/stage1_switching_factor/run_receipt.json",
            ("input_hashes", "stage1_receipt"),
        ),
    ),
    "artifacts/stage5_decisions/run_receipt.json": (
        ("config", "configs/stage5_decision.yaml", ("input_hashes", "config")),
        (
            "config",
            "configs/pipeline.yaml",
            ("input_hashes", "pipeline_config"),
        ),
        (
            "config",
            "configs/portfolio.yaml",
            ("input_hashes", "portfolio_model_config"),
        ),
        (
            "input",
            "data/processed/model_matrix.parquet",
            ("input_hashes", "model_matrix"),
        ),
        (
            "input",
            "artifacts/phase0/manifest.json",
            ("input_hashes", "phase0_manifest"),
        ),
        (
            "input",
            "artifacts/stage1_switching_factor/run_receipt.json",
            ("input_hashes", "stage1_run_receipt"),
        ),
        (
            "input",
            "artifacts/stage1_switching_factor/cumulative_scenarios.npz",
            ("input_hashes", "stage1_scenario_archive"),
        ),
    ),
    "artifacts/stage6_counterfactual/run_receipt.json": (
        (
            "input",
            "src/crisisforge/counterfactual/scm.py",
            ("scm_source_sha256",),
        ),
    ),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _repo_relative(project_root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"path must be repository-relative: {path}")
    resolved = (project_root / candidate).resolve()
    try:
        relative = resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes the repository: {path}") from exc
    return relative.as_posix()


def _file_record(
    project_root: Path,
    relative_path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    relative = _repo_relative(project_root, relative_path)
    path = project_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"declared file does not exist: {relative}")
    actual = sha256_file(path)
    if expected_sha256 is not None:
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(f"invalid SHA-256 declaration for {relative}")
        if actual != expected_sha256:
            raise ValueError(
                f"hash mismatch for {relative}: expected {expected_sha256}, found {actual}"
            )
    return {"bytes": path.stat().st_size, "path": relative, "sha256": actual}


def _git_state(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("a Git repository is required for release provenance") from exc
    return {"commit": commit, "dirty": bool(status.strip())}


def _nested(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            joined = ".".join(keys)
            raise ValueError(f"receipt has no declared field {joined}")
        value = value[key]
    return value


def _phase0_manifest_file_hash(project_root: Path, relative_path: str) -> str:
    manifest_path = project_root / "artifacts/phase0/manifest.json"
    manifest = _read_json(manifest_path)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Phase 0 manifest has no valid files list")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("path") == relative_path
    ]
    if len(matches) != 1:
        raise ValueError(f"Phase 0 manifest must bind exactly one copy of {relative_path}")
    declared = matches[0].get("sha256")
    if not isinstance(declared, str):
        raise ValueError(f"Phase 0 manifest has no SHA-256 for {relative_path}")
    return declared


def _experiment_id(receipt_path: str, receipt: Mapping[str, Any]) -> str:
    candidates = (
        receipt.get("experiment_id"),
        receipt.get("summary", {}).get("experiment_id")
        if isinstance(receipt.get("summary"), Mapping)
        else None,
        receipt.get("summary", {}).get("model_id")
        if isinstance(receipt.get("summary"), Mapping)
        else None,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    if receipt_path == "artifacts/phase0/run_receipt.json":
        return "phase0_public_data_v1"
    raise ValueError(f"cannot determine experiment ID from {receipt_path}")


def _read_registry(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"experiment registry does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "experiment_id",
        "stage",
        "status",
        "data_manifest_sha256",
        "git_commit",
        "config_path",
        "summary_path",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("experiment registry has an invalid schema")
    registry: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row["experiment_id"]
        if not identifier or identifier in registry:
            raise ValueError(f"duplicate or empty experiment ID: {identifier!r}")
        registry[identifier] = row
    return registry


def _add_unique(
    collection: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> None:
    previous = collection.get(record["path"])
    if previous is not None and previous["sha256"] != record["sha256"]:
        raise ValueError(f"conflicting hashes declared for {record['path']}")
    collection[record["path"]] = record


def _collect_uniform_references(
    project_root: Path,
    receipt: Mapping[str, Any],
    configs: dict[str, dict[str, Any]],
    inputs: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
) -> None:
    declared_outputs = receipt.get("outputs", {})
    if declared_outputs is None:
        declared_outputs = {}
    if not isinstance(declared_outputs, Mapping):
        raise ValueError("receipt outputs must be an object")
    for value in declared_outputs.values():
        if not isinstance(value, Mapping):
            raise ValueError("each receipt output must be an object")
        path = value.get("path")
        digest = value.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("each receipt output must declare path and sha256")
        _add_unique(
            outputs,
            _file_record(project_root, path, expected_sha256=digest),
        )

    declared_inputs = receipt.get("input_hashes", {})
    if isinstance(declared_inputs, Mapping):
        for value in declared_inputs.values():
            if not isinstance(value, Mapping):
                continue
            path = value.get("path")
            digest = value.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                continue
            target = configs if path.startswith("configs/") else inputs
            _add_unique(
                target,
                _file_record(project_root, path, expected_sha256=digest),
            )

    # Direct sibling declarations such as config/config_sha256,
    # model_config/model_config_sha256, or manifest/manifest_sha256.
    for key, value in receipt.items():
        if not isinstance(value, str):
            continue
        digest = receipt.get(f"{key}_sha256")
        if not isinstance(digest, str):
            continue
        target = configs if value.startswith("configs/") else inputs
        _add_unique(
            target,
            _file_record(project_root, value, expected_sha256=digest),
        )

    checkpoint_hashes = receipt.get("checkpoint_hashes")
    if isinstance(checkpoint_hashes, Mapping):
        checkpoint_output_names = {
            "base": "checkpoint_base",
            "tail_weighted": "checkpoint_tail_weighted",
        }
        for checkpoint_name, output_name in checkpoint_output_names.items():
            if checkpoint_name not in checkpoint_hashes:
                continue
            output = declared_outputs.get(output_name)
            if not isinstance(output, Mapping):
                raise ValueError(f"checkpoint hash {checkpoint_name!r} has no declared output")
            if checkpoint_hashes[checkpoint_name] != output.get("sha256"):
                raise ValueError(f"checkpoint hash declaration conflicts for {checkpoint_name}")
    if "standardizer_sha256" in receipt:
        standardizers = declared_outputs.get("standardizers")
        if not isinstance(standardizers, Mapping):
            raise ValueError("standardizer hash has no declared output")
        if receipt["standardizer_sha256"] != standardizers.get("sha256"):
            raise ValueError("standardizer hash declaration conflicts with output")


def _collect_bound_references(
    project_root: Path,
    receipt_path: str,
    receipt: Mapping[str, Any],
    configs: dict[str, dict[str, Any]],
    inputs: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    bindings: Mapping[str, Sequence[tuple[str, str, tuple[str, ...]]]],
) -> None:
    for category, path, hash_field in bindings.get(receipt_path, ()):
        if hash_field == ("phase0_manifest_file",):
            digest = _phase0_manifest_file_hash(project_root, path)
        else:
            digest = _nested(receipt, hash_field)
            if not isinstance(digest, str):
                joined = ".".join(hash_field)
                raise ValueError(f"{joined} in {receipt_path} is not a SHA-256 string")
        if category == "config":
            target = configs
        elif category == "output":
            target = outputs
        else:
            target = inputs
        _add_unique(
            target,
            _file_record(project_root, path, expected_sha256=digest),
        )


def _timing_free(payload: Any) -> Any:
    """Remove volatile timing metadata from embedded evaluation summaries."""
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(payload):
            lowered = key.lower()
            if (
                lowered.endswith("_seconds")
                or lowered == "elapsed_seconds"
                or lowered == "created_at_utc"
            ):
                continue
            result[key] = _timing_free(payload[key])
        return result
    if isinstance(payload, list):
        return [_timing_free(value) for value in payload]
    return payload


def _stage2_claims(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    matches = [
        receipt for identifier, receipt in receipts.items() if identifier.startswith("stage2_")
    ]
    if len(matches) != 1:
        raise ValueError("release must contain exactly one Stage 2 receipt")
    receipt = matches[0]
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("Stage 2 receipt has no summary object")
    if receipt.get("pilot") is not True or summary.get("pilot") is not True:
        raise ValueError("Stage 2 release must be labelled pilot=true")
    if summary.get("superiority_claim_permitted") is not False:
        raise ValueError("Stage 2 release must enforce superiority_claim_permitted=false")
    variants = summary.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Stage 2 receipt must include evaluated pilot variants")
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("Stage 2 variant must be an object")
        if variant.get("pilot") is not True:
            raise ValueError("every Stage 2 variant must be labelled pilot=true")
        if variant.get("superiority_claim_permitted") is not False:
            raise ValueError("every Stage 2 variant must prohibit superiority claims")
    return {
        "full_bayesian_parameter_posterior": summary.get("full_bayesian_parameter_posterior"),
        "future_regime_factor_joint_consistency": summary.get(
            "future_regime_factor_joint_consistency"
        ),
        "label": summary.get("label"),
        "pilot": True,
        "stage1_estimator": summary.get("stage1_estimator"),
        "superiority_claim_permitted": False,
    }


def _test_seal_claim(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for identifier, receipt in sorted(receipts.items()):
        summary = receipt.get("summary")
        value = None
        if isinstance(summary, Mapping):
            value = summary.get("test_set_opened")
        if value is None:
            value = receipt.get("test_set_opened")
        if value is None and receipt.get("evaluation_split") == "validation":
            value = False
        if value is not None:
            if value is not False:
                raise ValueError(f"test-set seal is not closed in {identifier}")
            evidence.append({"experiment_id": identifier, "test_set_opened": False})
    if not evidence:
        raise ValueError("release has no test-seal evidence")
    return {"evidence": evidence, "test_set_opened": False}


def _counterfactual_claims(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    matches = [
        receipt for identifier, receipt in receipts.items() if identifier.startswith("stage6_")
    ]
    if not matches:
        return {"included": False}
    if len(matches) != 1:
        raise ValueError("release has multiple Stage 6 receipts")
    boundary = matches[0].get("claims_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Stage 6 receipt must declare a claims boundary")
    if boundary.get("real_market_causal_identification") is not False:
        raise ValueError("Stage 6 cannot claim real-market causal identification")
    return {"included": True, **_timing_free(boundary)}


def _collect_deliverables(
    project_root: Path,
    *,
    patterns: Sequence[str],
    required: Sequence[str],
    manifest_output: str,
) -> list[dict[str, Any]]:
    excluded = _repo_relative(project_root, manifest_output)
    paths: set[str] = set()
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.is_file():
                relative = path.resolve().relative_to(project_root.resolve()).as_posix()
                if relative != excluded:
                    paths.add(relative)
    for required_path in required:
        relative = _repo_relative(project_root, required_path)
        if relative == excluded:
            raise ValueError("release manifest cannot include itself as a deliverable")
        if not (project_root / relative).is_file():
            raise FileNotFoundError(f"required deliverable does not exist: {relative}")
        paths.add(relative)
    return [_file_record(project_root, path) for path in sorted(paths)]


def build_release_manifest(
    project_root: Path,
    *,
    output_path: str = "artifacts/release/release_manifest.json",
    allow_dirty: bool = False,
    receipt_paths: Sequence[str] = DEFAULT_RECEIPTS,
    registry_path: str = "experiments/registry.csv",
    qa_receipt_paths: Sequence[str] = DEFAULT_QA_RECEIPTS,
    deliverable_patterns: Sequence[str] = DEFAULT_DELIVERABLE_PATTERNS,
    required_deliverables: Sequence[str] = DEFAULT_REQUIRED_DELIVERABLES,
    hash_bindings: Mapping[str, Sequence[tuple[str, str, tuple[str, ...]]]] = DEFAULT_HASH_BINDINGS,
) -> dict[str, Any]:
    """Validate release evidence, write the manifest, and return its payload."""
    root = project_root.resolve()
    output_relative = _repo_relative(root, output_path)
    git = _git_state(root)
    if git["dirty"] and not allow_dirty:
        raise RuntimeError("refusing to build a release manifest from a dirty Git worktree")

    registry = _read_registry(root / _repo_relative(root, registry_path))
    phase0_manifest_path = root / "artifacts/phase0/manifest.json"
    phase0_manifest_sha256 = (
        sha256_file(phase0_manifest_path) if phase0_manifest_path.is_file() else None
    )
    configs: dict[str, dict[str, Any]] = {}
    inputs: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    receipts_by_id: dict[str, dict[str, Any]] = {}
    experiment_records: list[dict[str, Any]] = []
    evaluation: dict[str, dict[str, Any]] = {}

    for receipt_path_raw in receipt_paths:
        receipt_relative = _repo_relative(root, receipt_path_raw)
        if receipt_relative == output_relative:
            raise ValueError("release manifest cannot be one of its own receipts")
        receipt_file = root / receipt_relative
        receipt = _read_json(receipt_file)
        identifier = _experiment_id(receipt_relative, receipt)
        if identifier in receipts_by_id:
            raise ValueError(f"duplicate receipt experiment ID: {identifier}")
        receipts_by_id[identifier] = receipt

        row = registry.get(identifier)
        if row is None:
            raise ValueError(f"{identifier} is missing from the experiment registry")
        if row["status"] != str(receipt.get("status")):
            raise ValueError(f"registry status mismatch for {identifier}")
        if row["data_manifest_sha256"]:
            if phase0_manifest_sha256 is None:
                raise FileNotFoundError("registry cites a missing Phase 0 manifest")
            if row["data_manifest_sha256"] != phase0_manifest_sha256:
                raise ValueError(f"registry data-manifest mismatch for {identifier}")
        config_record = _file_record(root, row["config_path"])
        summary_record = _file_record(root, row["summary_path"])
        _add_unique(configs, config_record)
        receipt_record = _file_record(root, receipt_relative)

        declared_commit = None
        if isinstance(receipt.get("git"), Mapping):
            declared_commit = receipt["git"].get("commit")
            if receipt["git"].get("dirty") is not False:
                raise ValueError(f"{identifier} was not produced from a clean worktree")
        elif isinstance(receipt.get("runtime"), Mapping):
            declared_commit = receipt["runtime"].get("git_commit")
        if declared_commit != row["git_commit"]:
            raise ValueError(f"registry Git commit mismatch for {identifier}")

        _collect_uniform_references(root, receipt, configs, inputs, outputs)
        _collect_bound_references(
            root,
            receipt_relative,
            receipt,
            configs,
            inputs,
            outputs,
            hash_bindings,
        )

        experiment_records.append(
            {
                "config": config_record,
                "experiment_id": identifier,
                "receipt": receipt_record,
                "stage": row["stage"],
                "status": row["status"],
                "summary": summary_record,
            }
        )
        evaluation[identifier] = {
            "evaluation_split": receipt.get("evaluation_split")
            or (
                receipt.get("summary", {}).get("evaluation_split")
                if isinstance(receipt.get("summary"), Mapping)
                else None
            ),
            "receipt_summary": _timing_free(receipt.get("summary", {})),
            "summary": summary_record,
        }

    stage2 = _stage2_claims(receipts_by_id)
    test_seal = _test_seal_claim(receipts_by_id)
    counterfactual = _counterfactual_claims(receipts_by_id)

    qa_receipts: list[dict[str, Any]] = []
    if not qa_receipt_paths:
        raise ValueError("release must include at least one QA receipt")
    for qa_path_raw in qa_receipt_paths:
        qa_relative = _repo_relative(root, qa_path_raw)
        if qa_relative == output_relative:
            raise ValueError("release manifest cannot include itself as QA evidence")
        qa_payload = _read_json(root / qa_relative)
        if qa_payload.get("schema_version") != QA_SCHEMA_VERSION:
            raise ValueError(f"unsupported QA receipt schema in {qa_relative}")
        if qa_payload.get("status") != "passed":
            raise ValueError(f"QA receipt did not pass: {qa_relative}")
        if _timing_free(qa_payload) != qa_payload:
            raise ValueError(f"QA receipt contains volatile timing fields: {qa_relative}")
        qa_receipts.append(
            {
                **_file_record(root, qa_relative),
                "checks": qa_payload.get("checks", []),
                "status": "passed",
            }
        )

    deliverables = _collect_deliverables(
        root,
        patterns=deliverable_patterns,
        required=required_deliverables,
        manifest_output=output_relative,
    )

    manifest: dict[str, Any] = {
        "claims": {
            "counterfactual": counterfactual,
            "stage2": stage2,
            "test_set": test_seal,
        },
        "configs": [configs[path] for path in sorted(configs)],
        "deliverables": deliverables,
        "evaluation": {identifier: evaluation[identifier] for identifier in sorted(evaluation)},
        "experiments": sorted(
            experiment_records,
            key=lambda record: (record["stage"], record["experiment_id"]),
        ),
        "git": git,
        "inputs": [inputs[path] for path in sorted(inputs)],
        "outputs": [outputs[path] for path in sorted(outputs)],
        "project": {"name": "CrisisForge"},
        "qa_receipts": sorted(qa_receipts, key=lambda record: record["path"]),
        "schema_version": SCHEMA_VERSION,
    }
    all_paths = {
        record["path"]
        for section in ("configs", "deliverables", "inputs", "outputs", "qa_receipts")
        for record in manifest[section]
    }
    all_paths.update(
        experiment[key]["path"]
        for experiment in manifest["experiments"]
        for key in ("config", "receipt", "summary")
    )
    if output_relative in all_paths:
        raise AssertionError("acyclicity violation: manifest includes its own path")

    _atomic_write(root / output_relative, _json_bytes(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        default="artifacts/release/release_manifest.json",
        help="Repository-relative output path.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only: permit a dirty Git worktree.",
    )
    parser.add_argument(
        "--qa-receipt",
        action="append",
        dest="qa_receipts",
        help="Repository-relative QA receipt; repeat for multiple receipts.",
    )
    parser.add_argument(
        "--deliverable",
        action="append",
        dest="deliverables",
        help="Additional repository-relative deliverable glob.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    patterns = list(DEFAULT_DELIVERABLE_PATTERNS)
    if args.deliverables:
        patterns.extend(args.deliverables)
    manifest = build_release_manifest(
        args.project_root,
        output_path=args.output,
        allow_dirty=args.allow_dirty,
        qa_receipt_paths=args.qa_receipts or DEFAULT_QA_RECEIPTS,
        deliverable_patterns=patterns,
    )
    print(
        json.dumps(
            {
                "manifest": _repo_relative(args.project_root.resolve(), args.output),
                "schema_version": manifest["schema_version"],
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
