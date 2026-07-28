"""Fail-closed helpers for sealed data access and experiment provenance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from crisisforge.data.validation import hash_file


def read_validation_matrix(
    matrix_path: Path,
    *,
    validation_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Load only rows through validation_end at the Parquet scan boundary."""
    cutoff = pd.Timestamp(validation_end)
    schema = pq.read_schema(matrix_path)
    pandas_metadata = json.loads(schema.metadata[b"pandas"].decode("utf-8"))
    index_columns = pandas_metadata.get("index_columns", [])
    if len(index_columns) != 1 or not isinstance(index_columns[0], str):
        raise ValueError("model_matrix must have one materialized timestamp index")
    index_field = index_columns[0]
    frame = pd.read_parquet(
        matrix_path,
        filters=[(index_field, "<=", cutoff)],
    ).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("model_matrix must have a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("model_matrix index must be unique and increasing")
    if len(frame) and frame.index.max() > cutoff:
        raise AssertionError("test-seal violation: post-validation rows were loaded")
    return frame


def assert_manifest_binds_file(
    *,
    manifest_path: Path,
    project_root: Path,
    file_path: Path,
) -> str:
    """Require the manifest to bind an input's relative path and current hash."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Phase 0 manifest has no valid files list")
    relative = str(file_path.resolve().relative_to(project_root.resolve()))
    matches = [record for record in records if record.get("path") == relative]
    if len(matches) != 1:
        raise ValueError(f"Phase 0 manifest must contain exactly one record for {relative}")
    actual = hash_file(file_path)
    expected = matches[0].get("sha256")
    if expected != actual:
        raise ValueError(
            f"Phase 0 manifest hash mismatch for {relative}: expected {expected}, found {actual}"
        )
    return actual


def assert_receipt_binds_output(
    *,
    receipt_path: Path,
    project_root: Path,
    output_key: str,
    output_path: Path,
    allowed_statuses: set[str],
) -> dict[str, Any]:
    """Require an upstream receipt to bind a specific persisted output."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") not in allowed_statuses:
        raise ValueError(f"upstream receipt status {receipt.get('status')!r} is not allowed")
    output = receipt.get("outputs", {}).get(output_key)
    if not isinstance(output, dict):
        raise ValueError(f"upstream receipt has no hashed output {output_key!r}")
    relative = str(output_path.resolve().relative_to(project_root.resolve()))
    if output.get("path") != relative:
        raise ValueError(
            f"upstream receipt path mismatch for {output_key}: "
            f"expected {relative}, found {output.get('path')}"
        )
    actual = hash_file(output_path)
    if output.get("sha256") != actual:
        raise ValueError(
            f"upstream receipt hash mismatch for {output_key}: "
            f"expected {output.get('sha256')}, found {actual}"
        )
    return receipt


def git_state(project_root: Path) -> dict[str, Any]:
    """Return the current commit and dirty flag without making Git mandatory."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def output_hashes(
    project_root: Path,
    outputs: dict[str, Path],
) -> dict[str, dict[str, str]]:
    """Hash persisted outputs after they have been written."""
    return {
        name: {
            "path": str(path.relative_to(project_root)),
            "sha256": hash_file(path),
        }
        for name, path in outputs.items()
    }
