from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_release_bundle.py"
_SPEC = importlib.util.spec_from_file_location("build_release_bundle", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ARCHIVE_ROOT = _MODULE.ARCHIVE_ROOT
build_release_bundle = _MODULE.build_release_bundle


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_git_repository(root: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "bundle-test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Bundle Test"],
        cwd=root,
        check=True,
    )


def test_release_bundle_is_git_tracked_allowlisted_and_deterministic(
    tmp_path: Path,
) -> None:
    _init_git_repository(tmp_path)
    _write(tmp_path / "README.md", "# Public fixture\n")
    _write(tmp_path / "src/model.py", "value = 1\n")

    denied_paths = (
        "data/raw/licensed.csv",
        "data/interim/panel.parquet",
        "data/processed/model.parquet",
        "artifacts/result.json",
        "configs/checkpoints/model.pt",
        "docs/artifacts/result.json",
        "docs/old-release.zip",
        "dist/package.whl",
        "dist/package.tar.gz",
        "notebooks/scenarios.npz",
        "reports/checkpoint.pt",
        "reports/results.parquet",
        "scripts/.env.local",
        "src/private.key",
        ".env",
        ".Rhistory",
    )
    for relative in denied_paths:
        _write(tmp_path / relative)
    subprocess.run(["git", "add", "-f", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "fixture"],
        cwd=tmp_path,
        check=True,
    )

    _write(tmp_path / "docs/untracked-note.md", "must not ship\n")

    first = build_release_bundle(tmp_path)
    archive_path = tmp_path / first["archive"]
    first_digest = _digest(archive_path)
    second = build_release_bundle(tmp_path)

    assert first_digest == _digest(archive_path)
    assert first["sha256"] == second["sha256"]
    checksum_path = tmp_path / first["checksum_file"]
    assert checksum_path.read_text(encoding="utf-8") == (
        f"{first['sha256']}  {archive_path.name}\n"
    )
    assert first["checksum_sha256"] == _digest(checksum_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        expected_files = {
            f"{ARCHIVE_ROOT}/README.md",
            f"{ARCHIVE_ROOT}/src/model.py",
            f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json",
        }
        assert names == expected_files
        assert f"{ARCHIVE_ROOT}/src/model.py" in names
        assert f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json" in names
        assert f"{ARCHIVE_ROOT}/docs/untracked-note.md" not in names
        assert f"{ARCHIVE_ROOT}/SHA256SUMS" not in names
        for relative in denied_paths:
            assert f"{ARCHIVE_ROOT}/{relative}" not in names
        manifest = json.loads(archive.read(f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json"))
        assert manifest["file_count"] == 2
        assert [record["path"] for record in manifest["files"]] == [
            "README.md",
            "src/model.py",
        ]


def test_release_bundle_requires_git_repository(tmp_path: Path) -> None:
    _write(tmp_path / "README.md")
    with pytest.raises(RuntimeError, match="Git repository"):
        build_release_bundle(tmp_path)
