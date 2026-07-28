"""Validation-only portfolio decisions from frozen Stage 1 scenario archives.

The experiment intentionally consumes *cumulative* H-period asset-return
scenarios saved by Stage 1.  It never selects a Wasserstein radius on the test
split: every radius in the registered grid remains a separate exploratory
validation strategy.

Historical scenarios are reconstructed at each forecast origin from rows whose
timestamps are no later than that origin.  The actual next-H block is used only
after target weights have been chosen.  Portfolio turnover uses the explicit
L1 convention ``sum(abs(w_new - w_previous))`` and transaction costs are
an additive linear return charge: ``net_return = gross_return - cost``.  This is
an auditable proportional-cost approximation, not a multiplicative wealth
withdrawal, intrablock execution model, or market-impact model.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisisforge.config import (
    load_yaml,
    project_root_from_module,
    resolve_config_path,
)
from crisisforge.data.validation import hash_file
from crisisforge.evaluation.provenance import (
    assert_manifest_binds_file,
    assert_receipt_binds_output,
    git_state,
    output_hashes,
    read_validation_matrix,
)
from crisisforge.evaluation.rolling import _origin_positions, rolling_cumulative_returns
from crisisforge.portfolio import (
    CVaRPortfolioResult,
    solve_empirical_cvar,
    solve_wasserstein_robust_cvar,
)
from crisisforge.risk import (
    empirical_expected_shortfall,
    empirical_var,
    portfolio_losses,
)


@dataclass(frozen=True)
class Stage1ScenarioArchive:
    """A schema-checked, origin-aligned Stage 1 cumulative-scenario archive."""

    scenarios: np.ndarray
    origin_dates: pd.DatetimeIndex
    asset_columns: tuple[str, ...]


@dataclass(frozen=True)
class RealizedDecision:
    """Auditable arithmetic for one target portfolio and holding block."""

    gross_return: float
    transaction_cost: float
    net_return: float
    realized_loss: float
    l1_turnover: float
    end_drifted_weights: np.ndarray


def assert_validation_only(
    configuration: dict[str, Any],
    pipeline_configuration: dict[str, Any],
) -> None:
    """Reject any configuration that could turn radius exploration into test use."""
    experiment = configuration.get("experiment", {})
    claims = configuration.get("claims_boundary", {})
    if experiment.get("evaluation_split") != "validation":
        raise ValueError(
            "The decision experiment keeps the test split sealed; "
            "evaluation_split must be validation"
        )
    if experiment.get("radius_grid_role") != "exploratory_validation_only":
        raise ValueError(
            "radius_grid_role must be exploratory_validation_only; "
            "test-set radius selection is forbidden"
        )
    if claims.get("test_set_opened") is not False:
        raise ValueError("claims_boundary.test_set_opened must be explicitly false")
    if claims.get("select_final_radius") is not False:
        raise ValueError("claims_boundary.select_final_radius must be explicitly false")

    splits = pipeline_configuration.get("splits", {})
    train_end = pd.Timestamp(splits["train_end"])
    validation_end = pd.Timestamp(splits["validation_end"])
    if train_end >= validation_end:
        raise ValueError("train_end must be strictly before validation_end")


def historical_cumulative_scenarios(
    returns: pd.DataFrame,
    *,
    origin_date: str | pd.Timestamp,
    horizon: int,
    lookback_observations: int | None,
) -> np.ndarray:
    """Build overlapping H-period scenarios using information through origin only."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("returns index must be sorted and unique")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if lookback_observations is not None and lookback_observations < horizon:
        raise ValueError("lookback_observations must be at least horizon")

    origin = pd.Timestamp(origin_date)
    if origin not in returns.index:
        raise ValueError("origin_date must exactly match a return-panel timestamp")
    available = returns.loc[:origin]
    if available.index[-1] != origin:
        raise AssertionError("historical slice unexpectedly extends beyond origin")
    if lookback_observations is not None:
        available = available.tail(lookback_observations)
    values = available.to_numpy(dtype=float)
    if len(values) < horizon:
        raise ValueError("insufficient historical observations for one H-period scenario")
    return rolling_cumulative_returns(values, horizon=horizon)


def load_stage1_scenario_archive(
    archive_path: Path,
    *,
    expected_origin_dates: pd.DatetimeIndex,
    expected_asset_columns: list[str] | tuple[str, ...],
) -> Stage1ScenarioArchive:
    """Load an NPZ archive and require exact origin and asset ordering."""
    required_keys = {"scenarios", "origin_dates", "asset_columns"}
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != required_keys:
            raise ValueError(
                "Stage 1 archive keys must be exactly "
                f"{sorted(required_keys)}; found {sorted(archive.files)}"
            )
        scenarios = np.asarray(archive["scenarios"], dtype=float)
        raw_dates = np.asarray(archive["origin_dates"])
        raw_assets = np.asarray(archive["asset_columns"])

    if scenarios.ndim != 3:
        raise ValueError("archive scenarios must have shape (origins, scenarios, assets)")
    if raw_dates.ndim != 1 or raw_assets.ndim != 1:
        raise ValueError("archive origin_dates and asset_columns must be one-dimensional")
    if not np.isfinite(scenarios).all():
        raise ValueError("archive scenarios contain non-finite values")
    if (scenarios < -1.0).any():
        raise ValueError("cumulative simple-return scenarios cannot be below -100%")

    origin_dates = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="raise"))
    if origin_dates.tz is not None:
        raise ValueError("archive origin_dates must be timezone-naive")
    if not origin_dates.is_unique or not origin_dates.is_monotonic_increasing:
        raise ValueError("archive origin_dates must be sorted and unique")
    expected_dates = pd.DatetimeIndex(expected_origin_dates)
    if not origin_dates.equals(expected_dates):
        raise ValueError(
            "Stage 1 archive origin_dates do not exactly match registered "
            "validation origins"
        )

    asset_columns = tuple(str(value) for value in raw_assets.tolist())
    expected_assets = tuple(expected_asset_columns)
    if asset_columns != expected_assets:
        raise ValueError(
            "Stage 1 archive asset_columns do not exactly match the model matrix "
            "in name and order"
        )
    expected_shape = (len(expected_dates), len(expected_assets))
    if scenarios.shape[0] != expected_shape[0] or scenarios.shape[2] != expected_shape[1]:
        raise ValueError(
            "Stage 1 scenario dimensions disagree with the registered origins/assets"
        )
    if scenarios.shape[1] < 2:
        raise ValueError("Stage 1 archive must contain at least two scenarios per origin")
    return Stage1ScenarioArchive(
        scenarios=scenarios,
        origin_dates=origin_dates,
        asset_columns=asset_columns,
    )


def realized_decision_arithmetic(
    actual_cumulative_returns: np.ndarray,
    target_weights: np.ndarray,
    previous_weights: np.ndarray,
    transaction_cost_rates: float | np.ndarray,
) -> RealizedDecision:
    """Apply target weights, subtract explicit costs, and drift end weights."""
    actual = np.asarray(actual_cumulative_returns, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    previous = np.asarray(previous_weights, dtype=float)
    if actual.ndim != 1 or target.shape != actual.shape or previous.shape != actual.shape:
        raise ValueError("actual returns, target weights, and previous weights must align")
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(target).all()
        or not np.isfinite(previous).all()
    ):
        raise ValueError("realized decision inputs must be finite")
    if (actual <= -1.0).any():
        raise ValueError("actual cumulative simple returns must exceed -100%")
    if (target < -1e-12).any() or not np.isclose(target.sum(), 1.0, atol=1e-8):
        raise ValueError("target_weights must be long-only and fully invested")
    if (previous < -1e-12).any() or not np.isclose(previous.sum(), 1.0, atol=1e-8):
        raise ValueError("previous_weights must be long-only and fully invested")

    rates = np.asarray(transaction_cost_rates, dtype=float)
    if rates.ndim == 0:
        rates = np.repeat(float(rates), len(actual))
    if rates.shape != actual.shape or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("transaction_cost_rates must be non-negative and asset-aligned")

    trades = np.abs(target - previous)
    turnover = float(trades.sum())
    cost = float(np.dot(rates, trades))
    gross = float(np.dot(target, actual))
    net = gross - cost
    gross_wealth = 1.0 + gross
    if gross_wealth <= 0.0:
        raise ValueError("holding block produced non-positive gross portfolio wealth")
    if 1.0 + net <= 0.0:
        raise ValueError("transaction costs produced non-positive net portfolio wealth")
    drifted = target * (1.0 + actual) / gross_wealth
    drifted /= drifted.sum()
    return RealizedDecision(
        gross_return=gross,
        transaction_cost=cost,
        net_return=net,
        realized_loss=-net,
        l1_turnover=turnover,
        end_drifted_weights=drifted,
    )


def maximum_drawdown(block_returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of sequential non-overlapping blocks."""
    values = np.asarray(block_returns, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("block_returns must be a finite non-empty vector")
    if (values <= -1.0).any():
        raise ValueError("net block returns must exceed -100%")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    running_peak = np.maximum.accumulate(wealth)
    return float(np.max(1.0 - wealth / running_peak))


def summarize_decisions(
    detail: pd.DataFrame,
    *,
    confidence_level: float,
    expected_origins: int,
) -> pd.DataFrame:
    """Aggregate realized decision outcomes without selecting a winning radius."""
    required = {
        "strategy_id",
        "origin_date",
        "net_realized_return",
        "gross_realized_return",
        "realized_loss",
        "l1_turnover",
        "transaction_cost",
    }
    missing = required.difference(detail.columns)
    if missing:
        raise ValueError(f"decision detail is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for strategy_id, group in detail.groupby("strategy_id", sort=True):
        ordered = group.sort_values("origin_date")
        net = ordered["net_realized_return"].to_numpy(dtype=float)
        losses = ordered["realized_loss"].to_numpy(dtype=float)
        worst_index = int(np.argmin(net))
        rows.append(
            {
                "strategy_id": strategy_id,
                "wasserstein_radius": float(ordered["wasserstein_radius"].iloc[0]),
                "completed_origins": int(len(ordered)),
                "expected_origins": int(expected_origins),
                "complete_sequence": bool(len(ordered) == expected_origins),
                "mean_gross_return": float(ordered["gross_realized_return"].mean()),
                "mean_net_return": float(net.mean()),
                "cumulative_net_return": float(np.prod(1.0 + net) - 1.0),
                "realized_value_at_risk": empirical_var(losses, confidence_level),
                "realized_expected_shortfall": empirical_expected_shortfall(
                    losses,
                    confidence_level,
                ),
                "maximum_drawdown": maximum_drawdown(net),
                "mean_l1_turnover": float(ordered["l1_turnover"].mean()),
                "total_l1_turnover": float(ordered["l1_turnover"].sum()),
                "mean_transaction_cost": float(ordered["transaction_cost"].mean()),
                "total_transaction_cost": float(ordered["transaction_cost"].sum()),
                "worst_block_origin_date": str(
                    ordered.iloc[worst_index]["origin_date"]
                ),
                "worst_block_net_return": float(net[worst_index]),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy_id").reset_index(drop=True)


def _registered_strategies(radius_grid: list[float]) -> list[dict[str, Any]]:
    if not radius_grid:
        raise ValueError("wasserstein radius_grid cannot be empty")
    radii = np.asarray(radius_grid, dtype=float)
    if not np.isfinite(radii).all() or (radii <= 0.0).any():
        raise ValueError("registered Wasserstein radii must be finite and positive")
    if len(np.unique(radii)) != len(radii) or np.any(np.diff(radii) <= 0.0):
        raise ValueError("registered Wasserstein radii must be unique and increasing")
    strategies: list[dict[str, Any]] = [
        {
            "strategy_id": "equal_weight",
            "kind": "equal_weight",
            "scenario_source": "historical_through_origin",
            "radius": 0.0,
        },
        {
            "strategy_id": "historical_empirical_cvar",
            "kind": "empirical_cvar",
            "scenario_source": "historical_through_origin",
            "radius": 0.0,
        },
        {
            "strategy_id": "stage1_empirical_cvar",
            "kind": "empirical_cvar",
            "scenario_source": "stage1_switching_factor",
            "radius": 0.0,
        },
    ]
    strategies.extend(
        {
            "strategy_id": f"stage1_wdro_rho_{radius:.8g}",
            "kind": "wasserstein_cvar",
            "scenario_source": "stage1_switching_factor",
            "radius": float(radius),
        }
        for radius in radii
    )
    return strategies


def _solve_strategy(
    specification: dict[str, Any],
    *,
    historical_scenarios: np.ndarray,
    stage1_scenarios: np.ndarray,
    equal_weights: np.ndarray,
    previous_weights: np.ndarray,
    confidence_level: float,
    minimum_position: float,
    maximum_position: float,
    turnover_limit: float,
    transaction_cost_rates: np.ndarray,
) -> tuple[np.ndarray, CVaRPortfolioResult | None, np.ndarray]:
    scenarios = (
        historical_scenarios
        if specification["scenario_source"] == "historical_through_origin"
        else stage1_scenarios
    )
    if specification["kind"] == "equal_weight":
        turnover = float(np.abs(equal_weights - previous_weights).sum())
        if turnover > turnover_limit + 1e-10:
            raise ValueError("equal-weight rebalance exceeds registered L1 turnover limit")
        if np.max(equal_weights) > maximum_position + 1e-10:
            raise ValueError("equal weights violate registered maximum position")
        if np.min(equal_weights) < minimum_position - 1e-10:
            raise ValueError("equal weights violate registered minimum position")
        return equal_weights.copy(), None, scenarios
    solver_arguments = {
        "confidence_level": confidence_level,
        "lower_bounds": minimum_position,
        "upper_bounds": maximum_position,
        "previous_weights": previous_weights,
        "turnover_limit": turnover_limit,
        "transaction_cost_rates": transaction_cost_rates,
    }
    if specification["kind"] == "empirical_cvar":
        result = solve_empirical_cvar(scenarios, **solver_arguments)
    elif specification["kind"] == "wasserstein_cvar":
        result = solve_wasserstein_robust_cvar(
            scenarios,
            wasserstein_radius=float(specification["radius"]),
            **solver_arguments,
        )
    else:
        raise ValueError(f"unsupported strategy kind: {specification['kind']}")
    return result.weights, result, scenarios


def run_decision_evaluation(
    project_root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the registered portfolio comparison on validation origins only."""
    config_path = resolve_config_path(
        project_root,
        config_path,
        default_relative="configs/stage5_decision.yaml",
    )
    configuration = load_yaml(config_path)
    pipeline_path = project_root / "configs" / "pipeline.yaml"
    pipeline = load_yaml(pipeline_path)
    assert_validation_only(configuration, pipeline)

    paths = configuration["paths"]
    matrix_path = project_root / paths["model_matrix"]
    manifest_path = project_root / paths["phase0_manifest"]
    portfolio_model_path = project_root / paths["portfolio_model_config"]
    archive_path = project_root / paths["stage1_scenario_archive"]
    stage1_receipt_path = project_root / paths["stage1_run_receipt"]
    output_root = project_root / paths["output"]
    output_root.mkdir(parents=True, exist_ok=True)

    validation_end = pd.Timestamp(pipeline["splits"]["validation_end"])
    assert_manifest_binds_file(
        manifest_path=manifest_path,
        project_root=project_root,
        file_path=matrix_path,
    )
    # Row-level filtering prevents test-period asset values from entering memory.
    matrix = read_validation_matrix(
        matrix_path,
        validation_end=validation_end,
    )
    asset_columns = [column for column in matrix if column.startswith("asset__")]
    returns = matrix[asset_columns]
    if returns.empty or returns.isna().any().any():
        raise ValueError("Decision evaluation requires a complete asset-return panel")
    if returns.index.max() > validation_end:
        raise AssertionError("test-seal violation: post-validation rows were loaded")

    forecast = configuration["forecast"]
    horizon = int(forecast["horizon"])
    origins = _origin_positions(
        returns.index,
        evaluation_split="validation",
        train_end=pipeline["splits"]["train_end"],
        validation_end=pipeline["splits"]["validation_end"],
        horizon=horizon,
        stride=int(forecast["origin_stride"]),
    )
    expected_origin_dates = pd.DatetimeIndex([returns.index[position] for position in origins])
    horizon_end_dates = pd.DatetimeIndex(
        [returns.index[position + horizon] for position in origins]
    )
    if horizon_end_dates.max() > validation_end:
        raise AssertionError("test-seal violation: a holding block crosses validation_end")
    archive = load_stage1_scenario_archive(
        archive_path,
        expected_origin_dates=expected_origin_dates,
        expected_asset_columns=asset_columns,
    )
    stage1_receipt = assert_receipt_binds_output(
        receipt_path=stage1_receipt_path,
        project_root=project_root,
        output_key="cumulative_scenarios",
        output_path=archive_path,
        allowed_statuses={"completed"},
    )
    current_manifest_hash = hash_file(manifest_path)
    current_matrix_hash = hash_file(matrix_path)
    if stage1_receipt.get("phase0_manifest_sha256") != current_manifest_hash:
        raise ValueError("Stage 1 receipt does not reference the active Phase 0 manifest")
    if stage1_receipt.get("model_matrix_sha256") != current_matrix_hash:
        raise ValueError("Stage 1 receipt does not reference the active model matrix")

    decision_model = load_yaml(portfolio_model_path)
    portfolio = decision_model["portfolio"]
    trading = decision_model["trading"]
    robustness = decision_model["distributional_robustness"]
    solver = decision_model["solver"]
    if not bool(portfolio["fully_invested"]) or not bool(portfolio["long_only"]):
        raise ValueError("Stage 5 public experiment requires long-only full investment")
    if (
        robustness["ambiguity_set"] != "one_wasserstein_ball"
        or robustness["ground_transport_norm"] != "l1"
        or robustness["dual_portfolio_norm"] != "linfinity"
        or robustness["support_assumption"] != "unbounded_return_space"
    ):
        raise ValueError("registered Wasserstein formulation is unsupported")
    if trading["turnover_definition"] != "l1":
        raise ValueError("Stage 5 implements only explicit L1 turnover")
    if solver["backend"] != "scipy_linprog_highs":
        raise ValueError("Stage 5 implements only the registered scipy/HiGHS backend")
    confidence = float(portfolio["confidence_level"])
    minimum_position = float(portfolio["lower_position_limit"])
    maximum_position = float(portfolio["upper_position_limit"])
    turnover_limit = float(trading["maximum_l1_turnover"])
    cost_rate = float(trading["proportional_transaction_cost"])
    cost_rates = np.repeat(cost_rate, len(asset_columns))
    equal_weights = np.repeat(1.0 / len(asset_columns), len(asset_columns))
    if maximum_position * len(asset_columns) < 1.0 - 1e-12:
        raise ValueError("maximum_position is incompatible with full investment")

    registered_radii = [
        float(value)
        for value in robustness["candidate_radii"]
        if float(value) > 0.0
    ]
    strategies = _registered_strategies(registered_radii)
    previous_holdings = {
        specification["strategy_id"]: equal_weights.copy()
        for specification in strategies
    }
    decision_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    disabled_strategies: set[str] = set()
    started = time.perf_counter()
    for origin_number, origin_position in enumerate(origins):
        origin_date = returns.index[origin_position]
        end_date = returns.index[origin_position + horizon]
        historical = historical_cumulative_scenarios(
            returns,
            origin_date=origin_date,
            horizon=horizon,
            lookback_observations=int(
                forecast["historical_lookback_observations"]
            ),
        )
        minimum_historical = int(forecast["minimum_historical_scenarios"])
        if len(historical) < minimum_historical:
            raise ValueError(
                f"only {len(historical)} historical scenarios at {origin_date}; "
                f"minimum is {minimum_historical}"
            )
        actual_path = returns.iloc[
            origin_position + 1 : origin_position + 1 + horizon
        ].to_numpy(dtype=float)
        actual_cumulative = np.prod(1.0 + actual_path, axis=0) - 1.0

        for specification in strategies:
            strategy_id = specification["strategy_id"]
            if strategy_id in disabled_strategies:
                continue
            previous = previous_holdings[strategy_id].copy()
            try:
                weights, solver_result, decision_scenarios = _solve_strategy(
                    specification,
                    historical_scenarios=historical,
                    stage1_scenarios=archive.scenarios[origin_number],
                    equal_weights=equal_weights,
                    previous_weights=previous,
                    confidence_level=confidence,
                    minimum_position=minimum_position,
                    maximum_position=maximum_position,
                    turnover_limit=turnover_limit,
                    transaction_cost_rates=cost_rates,
                )
                realized = realized_decision_arithmetic(
                    actual_cumulative,
                    weights,
                    previous,
                    cost_rates,
                )
                if realized.l1_turnover > turnover_limit + 1e-8:
                    raise AssertionError("realized trade exceeds registered turnover limit")
                if solver_result is not None and not np.isclose(
                    realized.transaction_cost,
                    solver_result.transaction_cost,
                    atol=1e-10,
                ):
                    raise AssertionError(
                        "solver and realized transaction-cost arithmetic disagree"
                    )
                forecast_losses = portfolio_losses(decision_scenarios, weights)
                forecast_losses = forecast_losses + realized.transaction_cost
                decision_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "strategy_kind": specification["kind"],
                        "scenario_source": specification["scenario_source"],
                        "wasserstein_radius": float(specification["radius"]),
                        "origin_date": origin_date.date().isoformat(),
                        "horizon_end_date": end_date.date().isoformat(),
                        "historical_last_input_date": origin_date.date().isoformat(),
                        "scenario_count": int(len(decision_scenarios)),
                        "forecast_value_at_risk": empirical_var(
                            forecast_losses,
                            confidence,
                        ),
                        "forecast_expected_shortfall": (
                            empirical_expected_shortfall(
                                forecast_losses,
                                confidence,
                            )
                        ),
                        "gross_realized_return": realized.gross_return,
                        "transaction_cost": realized.transaction_cost,
                        "net_realized_return": realized.net_return,
                        "realized_loss": realized.realized_loss,
                        "l1_turnover": realized.l1_turnover,
                        "empirical_cvar_objective": (
                            float(solver_result.empirical_cvar)
                            if solver_result is not None
                            else float("nan")
                        ),
                        "robust_penalty": (
                            float(solver_result.robust_penalty)
                            if solver_result is not None
                            else 0.0
                        ),
                        "optimization_objective": (
                            float(solver_result.objective_value)
                            if solver_result is not None
                            else float("nan")
                        ),
                    }
                )
                for asset_number, asset in enumerate(asset_columns):
                    weight_rows.append(
                        {
                            "strategy_id": strategy_id,
                            "wasserstein_radius": float(specification["radius"]),
                            "origin_date": origin_date.date().isoformat(),
                            "asset": asset,
                            "previous_weight": float(previous[asset_number]),
                            "target_weight": float(weights[asset_number]),
                            "absolute_trade": float(
                                abs(weights[asset_number] - previous[asset_number])
                            ),
                            "end_drifted_weight": float(
                                realized.end_drifted_weights[asset_number]
                            ),
                        }
                    )
                previous_holdings[strategy_id] = realized.end_drifted_weights
            except Exception as exc:
                # Terminate this strategy's sequence. Continuing with stale
                # pre-failure holdings would corrupt all later turnover and
                # transaction-cost arithmetic.
                disabled_strategies.add(strategy_id)
                failures.append(
                    {
                        "strategy_id": strategy_id,
                        "wasserstein_radius": float(specification["radius"]),
                        "origin_date": origin_date.date().isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    detail = pd.DataFrame(decision_rows)
    weights = pd.DataFrame(weight_rows)
    if detail.empty:
        raise RuntimeError("Every registered decision failed")
    summary = summarize_decisions(
        detail,
        confidence_level=confidence,
        expected_origins=len(origins),
    )
    detail_path = output_root / "decision_results.csv"
    weights_path = output_root / "weights.csv"
    summary_path = output_root / "summary.csv"
    failures_path = output_root / "failures.json"
    detail.to_csv(detail_path, index=False)
    weights.to_csv(weights_path, index=False)
    summary.to_csv(summary_path, index=False)
    failures_path.write_text(
        json.dumps(failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    persisted_outputs = {
        "detail": detail_path,
        "weights": weights_path,
        "summary": summary_path,
        "failures": failures_path,
    }

    receipt = {
        "status": "completed" if not failures else "completed_with_failures",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "experiment_id": configuration["experiment"]["id"],
        "evaluation_split": "validation",
        "test_set_opened": False,
        "radius_grid_role": "exploratory_validation_only",
        "final_radius_selected": False,
        "origin_count": len(origins),
        "strategy_count": len(strategies),
        "successful_decisions": len(detail),
        "failed_decisions": len(failures),
        "input_hashes": {
            "config": hash_file(config_path),
            "pipeline_config": hash_file(pipeline_path),
            "portfolio_model_config": hash_file(portfolio_model_path),
            "phase0_manifest": hash_file(manifest_path),
            "model_matrix": hash_file(matrix_path),
            "stage1_scenario_archive": hash_file(archive_path),
            "stage1_run_receipt": hash_file(stage1_receipt_path),
        },
        "git": git_state(project_root),
        "registered_wasserstein_radius_grid": [
            float(value) for value in registered_radii
        ],
        "outputs": output_hashes(project_root, persisted_outputs),
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation-only CrisisForge portfolio decisions"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else project_root_from_module()
    )
    config_path = args.config.resolve() if args.config is not None else None
    receipt = run_decision_evaluation(project_root, config_path=config_path)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
