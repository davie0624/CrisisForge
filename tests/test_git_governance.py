from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_transaction_leftovers_are_ignored(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    project_root = Path(__file__).resolve().parents[1]
    shutil.copy2(project_root / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(
        [git, "init", "--quiet"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    candidates = [
        "data/.raw-staging-123/returns/licensed.csv",
        "data/.raw-backup-123/macro/source.csv",
        "data/.raw-failed-123/macro/source.csv",
        "data/.interim-backup-123/target_returns.parquet",
        "data/.processed-backup-123/model_matrix.parquet",
    ]
    for relative_path in candidates:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        result = subprocess.run(
            [git, "check-ignore", "--quiet", relative_path],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, relative_path
