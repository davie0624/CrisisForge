from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from crisisforge.data.providers import DownloadedFrame, read_cached_snapshot, write_snapshot


def test_cached_snapshot_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    download = DownloadedFrame(
        frame=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "value": [1.0, 2.0],
            }
        ),
        source_urls=("https://example.invalid/source.csv",),
        provider="test",
        retrieved_at_utc="2024-01-03T00:00:00+00:00",
        payload_sha256="remote",
    )
    write_snapshot(download, path, {"source_id": "test"})
    frame = pd.read_csv(path)
    frame.loc[0, "value"] = 999.0
    frame.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_cached_snapshot(path)


def test_cached_snapshot_rejects_row_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    download = DownloadedFrame(
        frame=pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "value": [1.0]}),
        source_urls=("https://example.invalid/source.csv",),
        provider="test",
        retrieved_at_utc="2024-01-02T00:00:00+00:00",
        payload_sha256="remote",
    )
    write_snapshot(download, path, {"source_id": "test"})
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["row_count"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        read_cached_snapshot(path)


def test_cached_snapshot_binds_catalog_identity_and_range(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    download = DownloadedFrame(
        frame=pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "value": [1.0, 2.0, 3.0],
            }
        ),
        source_urls=("https://example.invalid/source.csv",),
        provider="test_provider",
        retrieved_at_utc="2020-01-04T00:00:00+00:00",
        payload_sha256="remote",
    )
    write_snapshot(
        download,
        path,
        {
            "source_id": "expected_source",
            "page_url": "https://example.invalid/page",
            "requested_start": "2020-01-01",
            "requested_end": "2020-01-03",
        },
    )
    filtered = read_cached_snapshot(
        path,
        expected_source_id="expected_source",
        expected_provider="test_provider",
        expected_columns=["date", "value"],
        expected_page_url="https://example.invalid/page",
        start_date="2020-01-02",
        end_date="2020-01-03",
    )
    assert filtered["date"].min() == pd.Timestamp("2020-01-02")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        read_cached_snapshot(path, expected_source_id="wrong_source")
    with pytest.raises(RuntimeError, match="does not cover requested start"):
        read_cached_snapshot(
            path,
            start_date="2019-12-31",
            end_date="2020-01-03",
        )
