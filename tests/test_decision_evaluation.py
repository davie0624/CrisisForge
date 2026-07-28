from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crisisforge.evaluation.decision import (
    assert_validation_only,
    historical_cumulative_scenarios,
    load_stage1_scenario_archive,
    maximum_drawdown,
    realized_decision_arithmetic,
    summarize_decisions,
)


def _configuration() -> dict:
    return {
        "experiment": {
            "evaluation_split": "validation",
            "radius_grid_role": "exploratory_validation_only",
        },
        "claims_boundary": {
            "test_set_opened": False,
            "select_final_radius": False,
        },
    }


def _pipeline() -> dict:
    return {
        "splits": {
            "train_end": "2013-12-31",
            "validation_end": "2019-12-31",
        }
    }


def test_historical_scenarios_ignore_every_future_value() -> None:
    dates = pd.date_range("2010-01-04", periods=12, freq="B")
    original = pd.DataFrame(
        {
            "asset__a": np.linspace(-0.01, 0.01, len(dates)),
            "asset__b": np.linspace(0.02, -0.02, len(dates)),
        },
        index=dates,
    )
    origin = dates[7]
    first = historical_cumulative_scenarios(
        original,
        origin_date=origin,
        horizon=3,
        lookback_observations=None,
    )
    adversarial = original.copy()
    adversarial.loc[adversarial.index > origin] = np.array([0.99, -0.99])
    second = historical_cumulative_scenarios(
        adversarial,
        origin_date=origin,
        horizon=3,
        lookback_observations=None,
    )
    assert np.array_equal(first, second)
    assert len(first) == 6


@pytest.mark.parametrize("mismatch", ["origin", "asset"])
def test_archive_rejects_any_origin_or_asset_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    expected_dates = pd.DatetimeIndex(["2014-01-02", "2014-01-30"])
    expected_assets = ["asset__a", "asset__b"]
    archive_dates = np.array(["2014-01-02", "2014-01-30"])
    archive_assets = np.array(expected_assets)
    if mismatch == "origin":
        archive_dates[-1] = "2014-01-31"
    else:
        archive_assets = archive_assets[::-1]
    archive_path = tmp_path / "scenarios.npz"
    np.savez_compressed(
        archive_path,
        scenarios=np.zeros((2, 8, 2)),
        origin_dates=archive_dates,
        asset_columns=archive_assets,
    )
    with pytest.raises(ValueError, match="exactly match"):
        load_stage1_scenario_archive(
            archive_path,
            expected_origin_dates=expected_dates,
            expected_asset_columns=expected_assets,
        )


def test_realized_cost_arithmetic_and_post_return_drift_are_explicit() -> None:
    actual = np.array([0.10, -0.10])
    target = np.array([0.60, 0.40])
    previous = np.array([0.50, 0.50])
    result = realized_decision_arithmetic(
        actual,
        target,
        previous,
        transaction_cost_rates=0.01,
    )
    expected_turnover = 0.20
    expected_cost = 0.01 * expected_turnover
    expected_gross = 0.60 * 0.10 + 0.40 * -0.10
    assert np.isclose(result.l1_turnover, expected_turnover)
    assert np.isclose(result.transaction_cost, expected_cost)
    assert np.isclose(result.gross_return, expected_gross)
    assert np.isclose(result.net_return, expected_gross - expected_cost)
    expected_drift = target * (1.0 + actual) / (1.0 + expected_gross)
    assert np.allclose(result.end_drifted_weights, expected_drift)
    assert np.isclose(result.end_drifted_weights.sum(), 1.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evaluation_split", "test", "test split sealed"),
        ("radius_grid_role", "test_selected", "exploratory_validation_only"),
    ],
)
def test_test_seal_rejects_nonvalidation_configuration(
    field: str,
    value: str,
    message: str,
) -> None:
    configuration = deepcopy(_configuration())
    configuration["experiment"][field] = value
    with pytest.raises(ValueError, match=message):
        assert_validation_only(configuration, _pipeline())


def test_test_seal_requires_explicit_no_open_and_no_selection() -> None:
    opened = deepcopy(_configuration())
    opened["claims_boundary"]["test_set_opened"] = True
    with pytest.raises(ValueError, match="test_set_opened"):
        assert_validation_only(opened, _pipeline())
    selected = deepcopy(_configuration())
    selected["claims_boundary"]["select_final_radius"] = True
    with pytest.raises(ValueError, match="select_final_radius"):
        assert_validation_only(selected, _pipeline())


def test_summary_reports_realized_tail_drawdown_cost_and_worst_block() -> None:
    detail = pd.DataFrame(
        {
            "strategy_id": ["model"] * 3,
            "wasserstein_radius": [0.001] * 3,
            "origin_date": ["2014-01-02", "2014-01-30", "2014-02-27"],
            "gross_realized_return": [0.10, -0.20, 0.05],
            "net_realized_return": [0.09, -0.21, 0.04],
            "realized_loss": [-0.09, 0.21, -0.04],
            "l1_turnover": [0.2, 0.3, 0.1],
            "transaction_cost": [0.01, 0.01, 0.01],
        }
    )
    summary = summarize_decisions(
        detail,
        confidence_level=0.80,
        expected_origins=3,
    ).iloc[0]
    assert bool(summary["complete_sequence"])
    assert summary["worst_block_origin_date"] == "2014-01-30"
    assert np.isclose(summary["worst_block_net_return"], -0.21)
    assert np.isclose(summary["total_l1_turnover"], 0.6)
    assert np.isclose(summary["total_transaction_cost"], 0.03)
    assert np.isclose(
        summary["maximum_drawdown"],
        maximum_drawdown(np.array([0.09, -0.21, 0.04])),
    )
