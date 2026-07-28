from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import crisisforge.data.pipeline as phase0_pipeline
from crisisforge.data.pipeline import run_phase0
from crisisforge.data.providers import DownloadedFrame, write_snapshot
from crisisforge.data.validation import DataQualityError, hash_file


def _write_yaml(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _synthetic_project(root: Path) -> None:
    dates = pd.date_range("2009-01-02", "2013-01-04", freq="B")
    return_frame = pd.DataFrame({"date": dates, "Industry": 0.001})
    yield_frame = pd.DataFrame({"date": dates, "YIELD": 2.0})
    return_path = root / "data/raw/returns/industries.csv"
    yield_path = root / "data/raw/macro/treasury.csv"
    shared_metadata = {
        "requested_start": "2009-01-01",
        "requested_end": "2013-01-04",
    }
    write_snapshot(
        DownloadedFrame(
            frame=return_frame,
            source_urls=("https://example.invalid/industries.zip",),
            provider="french_research_zip",
            retrieved_at_utc="2013-01-05T00:00:00+00:00",
            payload_sha256="returns",
        ),
        return_path,
        {
            **shared_metadata,
            "source_id": "industries",
            "name": "Synthetic industry returns",
            "page_url": "https://example.invalid/industries",
            "native_frequency": "daily",
            "units": "decimal_simple_return",
        },
    )
    write_snapshot(
        DownloadedFrame(
            frame=yield_frame,
            source_urls=("https://example.invalid/treasury.xml",),
            provider="treasury_xml",
            retrieved_at_utc="2013-01-05T00:00:00+00:00",
            payload_sha256="yields",
        ),
        yield_path,
        {
            **shared_metadata,
            "source_id": "treasury",
            "name": "Synthetic Treasury yield",
            "page_url": "https://example.invalid/treasury",
            "native_frequency": "daily",
        },
    )
    pipeline = {
        "project": {
            "name": "Synthetic",
            "timezone": "America/New_York",
            "decision_timestamp": "after_close",
            "random_seed": 1,
        },
        "sample": {
            "start_date": "2010-06-01",
            "end_date": "2012-06-29",
        },
        "splits": {
            "train_end": "2010-12-31",
            "validation_end": "2011-12-30",
        },
        "quality": {
            "minimum_asset_coverage": 0.90,
            "minimum_relative_target_calendar_density": 0.90,
            "minimum_model_context_source_coverage": 0.90,
            "minimum_validation_source_coverage": 0.90,
            "minimum_model_index_relative_density": 0.90,
            "maximum_daily_return_abs": 0.40,
            "maximum_target_source_internal_gap_calendar_days": 10,
            "maximum_model_context_source_internal_gap_calendar_days": 10,
            "maximum_validation_source_internal_gap_calendar_days": 10,
            "maximum_model_index_internal_gap_calendar_days": 10,
            "maximum_common_interval_calendar_days": 10,
            "maximum_interval_source_observation_count_gap": 5,
            "maximum_target_source_start_delay_calendar_days": 10,
            "fail_on_duplicate_dates": True,
            "warning_after_source_staleness_calendar_days": 10,
            "fail_after_source_staleness_calendar_days": 30,
        },
        "paths": {
            "raw": "data/raw",
            "interim": "data/interim",
            "processed": "data/processed",
            "artifacts": "artifacts/phase0",
        },
    }
    catalog = {
        "panel": {"id": "synthetic", "description": "test"},
        "target_return_sources": [
            {
                "id": "industries",
                "provider": "french_research_zip",
                "name": "Synthetic industry returns",
                "page_url": "https://example.invalid/industries",
                "download_url": "https://example.invalid/industries.zip",
                "native_frequency": "daily",
                "units": "decimal_simple_return",
                "columns": [
                    {
                        "source_column": "Industry",
                        "asset": "industry_test",
                    }
                ],
            },
            {
                "id": "treasury_proxy",
                "provider": "derived_treasury_return",
                "source_id": "treasury",
                "name": "Synthetic proxy",
                "native_frequency": "daily",
                "units": "approximate_decimal_simple_return",
                "columns": [
                    {
                        "source_column": "YIELD",
                        "asset": "treasury_test",
                        "maturity_years": 2,
                    }
                ],
            },
        ],
        "macro_sources": [
            {
                "id": "treasury",
                "provider": "treasury_xml",
                "name": "Synthetic Treasury yield",
                "page_url": "https://example.invalid/treasury",
                "year_url_template": "https://example.invalid/{year}",
                "native_frequency": "daily",
                "availability_lag_model_sessions": 1,
                "max_staleness_calendar_days": 5,
                "columns": [
                    {
                        "source_column": "YIELD",
                        "feature": "yield_level",
                        "transform": "level",
                    }
                ],
            }
        ],
        "derived_macro_features": [],
        "derived_market_features": [],
    }
    _write_yaml(root / "configs/pipeline.yaml", pipeline)
    _write_yaml(root / "configs/data_catalog.yaml", catalog)


def test_offline_pipeline_applies_requested_sample_range(tmp_path: Path) -> None:
    _synthetic_project(tmp_path)
    receipt = run_phase0(tmp_path, refresh=False, allow_network=False)
    assert receipt["first_model_date"] >= "2010-06-01"
    assert receipt["last_model_date"] <= "2012-06-29"
    matrix = pd.read_parquet(tmp_path / "data/processed/model_matrix.parquet")
    assert matrix.index.min() >= pd.Timestamp("2010-06-01")
    assert matrix.index.max() <= pd.Timestamp("2012-06-29")


def test_offline_pipeline_rejects_uncatalogued_raw_file(tmp_path: Path) -> None:
    _synthetic_project(tmp_path)
    unexpected = tmp_path / "data/raw/macro/cboe_vix.csv"
    unexpected.write_text("date,value\n2012-01-01,1\n", encoding="utf-8")
    with pytest.raises(DataQualityError, match="allowlist"):
        run_phase0(tmp_path, refresh=False, allow_network=False)


def test_phase0_manifest_excludes_mutable_experiment_registry(tmp_path: Path) -> None:
    _synthetic_project(tmp_path)
    registry = tmp_path / "experiments/registry.csv"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "experiment_id,data_manifest_sha256\nexample,see_run_receipt\n",
        encoding="utf-8",
    )
    run_phase0(tmp_path, refresh=False, allow_network=False)
    manifest = json.loads(
        (tmp_path / "artifacts/phase0/manifest.json").read_text(encoding="utf-8")
    )
    paths = {record["path"] for record in manifest["files"]}
    assert "experiments/registry.csv" not in paths


def _tree_hashes(root: Path, relative_roots: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_root in relative_roots:
        base = root / relative_root
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            hashes[str(path.relative_to(root))] = hash_file(path)
    return hashes


def _assert_manifest_verifies(root: Path) -> None:
    manifest = json.loads(
        (root / "artifacts/phase0/manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["files"]:
        path = root / record["path"]
        assert path.exists(), record["path"]
        assert hash_file(path) == record["sha256"], record["path"]


@pytest.mark.parametrize("failure_point", ["parquet", "manifest"])
def test_refresh_failure_rolls_back_raw_and_all_published_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    _synthetic_project(tmp_path)
    run_phase0(tmp_path, refresh=False, allow_network=False)
    protected_roots = (
        "data/raw",
        "data/interim",
        "data/processed",
        "artifacts/phase0",
    )
    before = _tree_hashes(tmp_path, protected_roots)

    dates = pd.date_range("2009-01-02", "2013-01-04", freq="B")

    def changed_returns(
        source: dict[str, object],
        start_date: str,
        end_date: str,
    ) -> DownloadedFrame:
        del source, start_date, end_date
        return DownloadedFrame(
            frame=pd.DataFrame({"date": dates, "Industry": 0.002}),
            source_urls=("https://example.invalid/industries.zip",),
            provider="french_research_zip",
            retrieved_at_utc="2013-01-06T00:00:00+00:00",
            payload_sha256="changed-returns",
        )

    def changed_macro(
        source: dict[str, object],
        start_date: str,
        end_date: str,
    ) -> DownloadedFrame:
        del source, start_date, end_date
        return DownloadedFrame(
            frame=pd.DataFrame({"date": dates, "YIELD": 2.1}),
            source_urls=("https://example.invalid/treasury.xml",),
            provider="treasury_xml",
            retrieved_at_utc="2013-01-06T00:00:00+00:00",
            payload_sha256="changed-yields",
        )

    monkeypatch.setattr(phase0_pipeline, "download_return_source", changed_returns)
    monkeypatch.setattr(phase0_pipeline, "download_macro_source", changed_macro)
    monkeypatch.setattr(phase0_pipeline, "polite_pause", lambda: None)

    if failure_point == "parquet":
        monkeypatch.setattr(
            phase0_pipeline,
            "_write_parquet",
            lambda frame, path: (_ for _ in ()).throw(RuntimeError("parquet failure")),
        )
    else:
        monkeypatch.setattr(
            phase0_pipeline,
            "write_manifest",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("manifest failure")
            ),
        )

    with pytest.raises(RuntimeError, match=f"{failure_point} failure"):
        run_phase0(tmp_path, refresh=True, allow_network=True)

    after = _tree_hashes(tmp_path, protected_roots)
    assert after == before
    _assert_manifest_verifies(tmp_path)
