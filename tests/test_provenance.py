from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crisisforge.data.validation import hash_file
from crisisforge.evaluation.provenance import (
    assert_manifest_binds_file,
    assert_receipt_binds_output,
    read_validation_matrix,
)


def _write_matrix(path: Path) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"asset__a": [0.01, 0.02, 999.0]},
        index=pd.DatetimeIndex(
            ["2019-12-27", "2019-12-30", "2020-01-02"],
            name="date",
        ),
    ).to_parquet(path)


def test_validation_reader_filters_sealed_rows_at_parquet_scan(tmp_path: Path) -> None:
    matrix = tmp_path / "data/processed/model_matrix.parquet"
    _write_matrix(matrix)
    loaded = read_validation_matrix(matrix, validation_end="2019-12-31")
    assert loaded.index.max() == pd.Timestamp("2019-12-30")
    assert 999.0 not in loaded.to_numpy()


def test_manifest_binding_fails_closed_after_input_mutation(tmp_path: Path) -> None:
    matrix = tmp_path / "data/processed/model_matrix.parquet"
    _write_matrix(matrix)
    manifest = tmp_path / "artifacts/phase0/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "data/processed/model_matrix.parquet",
                        "sha256": hash_file(matrix),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert_manifest_binds_file(
        manifest_path=manifest,
        project_root=tmp_path,
        file_path=matrix,
    )
    frame = pd.read_parquet(matrix)
    frame.iloc[0, 0] = -0.25
    frame.to_parquet(matrix)
    with pytest.raises(ValueError, match="hash mismatch"):
        assert_manifest_binds_file(
            manifest_path=manifest,
            project_root=tmp_path,
            file_path=matrix,
        )


def test_upstream_receipt_binds_output_hash_and_status(tmp_path: Path) -> None:
    output = tmp_path / "artifacts/stage1/scenarios.npz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"registered scenario bytes")
    receipt = tmp_path / "artifacts/stage1/run_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "completed",
                "outputs": {
                    "cumulative_scenarios": {
                        "path": "artifacts/stage1/scenarios.npz",
                        "sha256": hash_file(output),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert_receipt_binds_output(
        receipt_path=receipt,
        project_root=tmp_path,
        output_key="cumulative_scenarios",
        output_path=output,
        allowed_statuses={"completed"},
    )
    output.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        assert_receipt_binds_output(
            receipt_path=receipt,
            project_root=tmp_path,
            output_key="cumulative_scenarios",
            output_path=output,
            allowed_statuses={"completed"},
        )
