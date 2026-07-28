from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crisisforge.data.pipeline import (
    _aligned_context_calendar_quality,
    _raw_allowlist_report,
    _target_calendar_quality,
    aggregate_simple_returns_to_endpoints,
    asof_align_feature,
    build_common_interval_audit,
    build_derived_market_features,
    chronological_split,
    compute_log_returns,
    duration_convexity_return_proxy,
    transform_macro,
)


def test_compute_log_returns() -> None:
    prices = pd.DataFrame(
        {"SPY": [100.0, 101.0, 99.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    result = compute_log_returns(prices)
    assert np.isclose(result.iloc[1, 0], np.log(101.0 / 100.0))
    assert np.isclose(result.iloc[2, 0], np.log(99.0 / 101.0))


def test_macro_transform_bps() -> None:
    values = pd.Series([4.00, 4.25, 4.10])
    result = transform_macro(values, "bps_difference")
    assert np.isnan(result.iloc[0])
    assert np.isclose(result.iloc[1], 25.0)
    assert np.isclose(result.iloc[2], -15.0)


def test_asof_alignment_respects_availability_lag() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
            "value": [1.0, 2.0],
        }
    )
    dates = pd.date_range("2020-01-01", "2020-01-17", freq="B")
    result = asof_align_feature(
        dates,
        source,
        feature_name="weekly_signal",
        availability_lag_sessions=5,
        max_staleness_calendar_days=14,
    )
    assert np.isnan(result.loc["2020-01-07", "macro__weekly_signal"])
    assert result.loc["2020-01-08", "macro__weekly_signal"] == 1.0
    assert result.loc["2020-01-15", "macro__weekly_signal"] == 2.0


def test_chronological_split_is_non_overlapping() -> None:
    frame = pd.DataFrame(
        {"x": range(10)},
        index=pd.date_range("2017-12-27", periods=10, freq="D"),
    )
    train, validation, test = chronological_split(
        frame,
        train_end="2017-12-29",
        validation_end="2018-01-02",
    )
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()


def test_duration_proxy_has_carry_when_yield_is_constant() -> None:
    yields = pd.Series([5.0, 5.0, 5.0])
    result = duration_convexity_return_proxy(yields, maturity_years=5)
    assert np.isnan(result.iloc[0])
    assert np.isclose(result.iloc[1], 0.05 / 252.0)
    assert np.isclose(result.iloc[2], 0.05 / 252.0)


def test_duration_proxy_loses_value_when_yield_rises() -> None:
    yields = pd.Series([4.0, 5.0])
    result = duration_convexity_return_proxy(yields, maturity_years=10)
    assert result.iloc[1] < 0.0


def test_common_endpoint_aggregation_handles_market_holiday_mismatch() -> None:
    source = pd.DataFrame(
        {"equity": [0.01, 0.02, -0.01]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    endpoints = pd.to_datetime(["2024-01-02", "2024-01-04"])
    result = aggregate_simple_returns_to_endpoints(source, endpoints)
    expected = (1.0 + 0.02) * (1.0 - 0.01) - 1.0
    assert np.isclose(result.loc["2024-01-04", "equity"], expected)


def test_interval_audit_records_source_observation_counts() -> None:
    endpoints = pd.to_datetime(["2024-01-02", "2024-01-04"])
    audit = build_common_interval_audit(
        endpoints,
        {
            "equity": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "rates": endpoints,
        },
    )
    assert audit.loc["2024-01-04", "calendar_days"] == 2.0
    assert audit.loc["2024-01-04", "observations__equity"] == 2.0
    assert audit.loc["2024-01-04", "observations__rates"] == 1.0
    assert bool(audit.loc["2024-01-04", "all_source_endpoints_present"])


def test_derived_market_features_are_backward_looking() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "asset__industry_a": [0.01, -0.02, 0.03, -0.04],
            "asset__industry_b": [0.03, -0.04, 0.01, -0.02],
        },
        index=dates,
    )
    specifications = [
        {
            "feature": "industry_market_return",
            "operation": "cross_sectional_mean",
            "source_prefix": "asset__industry_",
            "lookback": 1,
        },
        {
            "feature": "realized_volatility_2d",
            "operation": "rolling_volatility",
            "source": "industry_market_return",
            "lookback": 2,
            "annualization": 252,
        },
    ]
    result = build_derived_market_features(returns, specifications)
    assert np.isclose(result.iloc[0]["macro__industry_market_return"], 0.02)
    assert np.isnan(result.iloc[0]["macro__realized_volatility_2d"])
    expected = np.std([0.02, -0.03], ddof=1) * np.sqrt(252)
    assert np.isclose(result.iloc[1]["macro__realized_volatility_2d"], expected)


def test_elapsed_time_volatility_respects_irregular_intervals() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-05"])
    returns = pd.DataFrame(
        {
            "asset__industry_a": [0.01, 0.01],
            "asset__industry_b": [0.01, 0.01],
        },
        index=dates,
    )
    specifications = [
        {
            "feature": "industry_market_return",
            "operation": "cross_sectional_mean",
            "source_prefix": "asset__industry_",
            "lookback": 1,
        },
        {
            "feature": "realized_volatility_2d",
            "operation": "rolling_volatility",
            "source": "industry_market_return",
            "lookback": 2,
        },
    ]
    elapsed_years = pd.Series([1.0, 3.0], index=dates) / 365.25
    result = build_derived_market_features(
        returns,
        specifications,
        interval_years=elapsed_years,
    )
    expected = np.sqrt(2.0 * np.log1p(0.01) ** 2 / (4.0 / 365.25))
    assert np.isclose(result.iloc[1]["macro__realized_volatility_2d"], expected)


def test_context_calendar_quality_exposes_internal_source_hole() -> None:
    dates = pd.date_range("2008-01-02", "2010-12-31", freq="B")
    values = pd.Series(1.0, index=dates)
    values.loc["2009-01-01":"2009-12-31"] = np.nan
    audit = pd.DataFrame({"macro__policy_rate": values}, index=dates)
    sources = [
        {
            "id": "policy_source",
            "columns": [{"feature": "policy_rate", "use": "model"}],
        }
    ]

    quality = _aligned_context_calendar_quality(audit, sources, dates)
    metric = quality["model"]["policy_source"]
    assert metric["coverage_on_target_calendar"] < 0.70
    assert metric["maximum_internal_gap_calendar_days"] > 360


def test_raw_allowlist_rejects_uncatalogued_snapshot(tmp_path: Path) -> None:
    (tmp_path / "returns").mkdir()
    (tmp_path / "macro").mkdir()
    for relative_path in (
        "returns/core.csv",
        "returns/core.meta.json",
        "macro/rates.csv",
        "macro/rates.meta.json",
    ):
        (tmp_path / relative_path).write_text("", encoding="utf-8")
    catalog = {
        "target_return_sources": [
            {"id": "core", "provider": "french_research_zip"}
        ],
        "macro_sources": [{"id": "rates"}],
    }
    assert _raw_allowlist_report(tmp_path, catalog)["passed"]
    (tmp_path / "macro" / "cboe_vix.csv").write_text("", encoding="utf-8")
    report = _raw_allowlist_report(tmp_path, catalog)
    assert not report["passed"]
    assert report["unexpected_files"] == ["macro/cboe_vix.csv"]


def test_target_calendar_quality_surfaces_long_internal_gap() -> None:
    catalog = {
        "target_return_sources": [
            {
                "id": "industries",
                "provider": "french_research_zip",
                "columns": [{"source_column": "industry"}],
            },
            {
                "id": "treasury_proxy",
                "provider": "derived_treasury_return",
                "source_id": "treasury",
                "columns": [{"source_column": "yield"}],
            },
        ]
    }
    downloaded = {
        "industries": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-01", "2020-01-02", "2021-01-04"]
                ),
                "industry": [0.0, 0.0, 0.0],
            }
        )
    }
    macro = {
        "treasury": pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", "2021-01-04", freq="B"),
                "yield": 1.0,
            }
        )
    }
    quality = _target_calendar_quality(
        catalog,
        downloaded,
        macro,
        requested_start=pd.Timestamp("2020-01-01"),
    )
    assert (
        quality["sources"]["industries"]["maximum_internal_gap_calendar_days"]
        > 300
    )
