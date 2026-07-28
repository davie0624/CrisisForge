from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisisforge.baselines import (
    EWMFilteredHistoricalGenerator,
    GaussianScenarioGenerator,
    HistoricalScenarioGenerator,
    MovingBlockBootstrapGenerator,
    StudentTCopulaScenarioGenerator,
    StudentTScenarioGenerator,
    VARResidualBootstrapGenerator,
)
from crisisforge.config import load_yaml, project_root_from_module
from crisisforge.data.validation import hash_file
from crisisforge.risk import (
    aggregate_path_returns,
    brier_score,
    christoffersen_conditional_coverage_test,
    co_crash_probability,
    empirical_expected_shortfall,
    empirical_var,
    energy_score,
    fit_co_crash_thresholds,
    joint_var_es_score,
    portfolio_losses,
    realized_co_crash,
    variogram_score,
)


def rolling_cumulative_returns(
    returns: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Overlapping forward cumulative simple returns for threshold estimation."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or horizon < 1 or len(values) < horizon:
        raise ValueError("returns must be 2D with at least horizon rows")
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite simple returns above -100%")
    windows = np.lib.stride_tricks.sliding_window_view(
        values,
        window_shape=horizon,
        axis=0,
    )
    # sliding_window_view is (origins, assets, horizon) for axis=0.
    return np.prod(1.0 + windows, axis=2) - 1.0


def build_generator(specification: dict[str, Any]) -> Any:
    kind = specification["kind"]
    factories: dict[str, Callable[[], Any]] = {
        "historical": HistoricalScenarioGenerator,
        "moving_block": lambda: MovingBlockBootstrapGenerator(
            block_length=int(specification["block_length"])
        ),
        "filtered_historical": lambda: EWMFilteredHistoricalGenerator(
            decay=float(specification["decay"])
        ),
        "gaussian": lambda: GaussianScenarioGenerator(
            shrinkage=float(specification["shrinkage"])
        ),
        "student_t": lambda: StudentTScenarioGenerator(
            shrinkage=float(specification["shrinkage"])
        ),
        "student_t_copula": lambda: StudentTCopulaScenarioGenerator(
            degrees_of_freedom=float(specification["degrees_of_freedom"]),
            shrinkage=float(specification["shrinkage"]),
        ),
        "var_residual_bootstrap": lambda: VARResidualBootstrapGenerator(
            ridge=float(specification["ridge"])
        ),
    }
    if kind not in factories:
        raise ValueError(f"Unsupported baseline kind: {kind}")
    return factories[kind]()


def _origin_positions(
    dates: pd.DatetimeIndex,
    *,
    evaluation_split: str,
    train_end: str,
    validation_end: str,
    horizon: int,
    stride: int,
) -> list[int]:
    train_cut = pd.Timestamp(train_end)
    validation_cut = pd.Timestamp(validation_end)
    if evaluation_split == "validation":
        start = int(dates.searchsorted(train_cut, side="right") - 1)
        final_future_date = validation_cut
    elif evaluation_split == "test":
        start = int(dates.searchsorted(validation_cut, side="right") - 1)
        final_future_date = dates[-1]
    else:
        raise ValueError("evaluation_split must be 'validation' or 'test'")
    positions: list[int] = []
    for position in range(start, len(dates) - horizon, stride):
        future_dates = dates[position + 1 : position + 1 + horizon]
        if len(future_dates) != horizon or future_dates[-1] > final_future_date:
            break
        positions.append(position)
    if not positions:
        raise ValueError("No valid rolling origins for the configured split")
    return positions


def run_stage0_baselines(
    project_root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path or project_root / "configs" / "stage0_baselines.yaml"
    configuration = load_yaml(config_path)
    pipeline = load_yaml(project_root / "configs" / "pipeline.yaml")
    matrix_path = project_root / configuration["paths"]["model_matrix"]
    phase0_manifest_path = project_root / configuration["paths"]["phase0_manifest"]
    output_root = project_root / configuration["paths"]["output"]
    output_root.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(matrix_path).sort_index()
    asset_columns = [column for column in matrix if column.startswith("asset__")]
    returns = matrix[asset_columns]
    if returns.isna().any().any():
        raise ValueError("Stage 0 requires complete target returns")

    forecast = configuration["forecast"]
    risk = configuration["risk"]
    horizon = int(forecast["horizon"])
    origins = _origin_positions(
        returns.index,
        evaluation_split=configuration["experiment"]["evaluation_split"],
        train_end=pipeline["splits"]["train_end"],
        validation_end=pipeline["splits"]["validation_end"],
        horizon=horizon,
        stride=int(forecast["origin_stride"]),
    )
    initial_train = returns.loc[
        returns.index <= pd.Timestamp(pipeline["splits"]["train_end"])
    ].to_numpy()
    threshold_outcomes = rolling_cumulative_returns(initial_train, horizon=horizon)
    thresholds = fit_co_crash_thresholds(
        threshold_outcomes,
        marginal_quantile=float(risk["co_crash_marginal_quantile"]),
    )
    equal_weights = np.repeat(1.0 / len(asset_columns), len(asset_columns))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    experiment_seed = int(configuration["experiment"]["random_seed"])
    start_time = time.perf_counter()
    for model_number, model_specification in enumerate(configuration["models"]):
        model_id = model_specification["id"]
        for origin_number, origin_position in enumerate(origins):
            training_start = max(
                0,
                origin_position - int(forecast["training_lookback"]) + 1,
            )
            training = returns.iloc[
                training_start : origin_position + 1
            ].to_numpy()
            actual_path = returns.iloc[
                origin_position + 1 : origin_position + 1 + horizon
            ].to_numpy()
            seed = experiment_seed + model_number * 1_000_003 + origin_number
            rng = np.random.default_rng(seed)
            try:
                generator = build_generator(model_specification).fit(training)
                scenarios = generator.sample(
                    num_scenarios=int(forecast["num_scenarios"]),
                    horizon=horizon,
                    rng=rng,
                )
                simulated_cumulative = aggregate_path_returns(scenarios)
                actual_cumulative = aggregate_path_returns(actual_path[None, :, :])[0]
                simulated_losses = portfolio_losses(
                    simulated_cumulative,
                    equal_weights,
                )
                actual_loss = float(-(actual_cumulative @ equal_weights))
                confidence = float(risk["confidence_level"])
                value_at_risk = empirical_var(simulated_losses, confidence)
                expected_shortfall = empirical_expected_shortfall(
                    simulated_losses,
                    confidence,
                )
                minimum_fraction = float(risk["co_crash_minimum_fraction"])
                predicted_co_crash = co_crash_probability(
                    simulated_cumulative,
                    thresholds,
                    minimum_fraction=minimum_fraction,
                )
                actual_co_crash = realized_co_crash(
                    actual_cumulative,
                    thresholds,
                    minimum_fraction=minimum_fraction,
                )
                records.append(
                    {
                        "model_id": model_id,
                        "origin_date": returns.index[
                            origin_position
                        ].date().isoformat(),
                        "horizon_end_date": returns.index[
                            origin_position + horizon
                        ].date().isoformat(),
                        "seed": seed,
                        "training_rows": len(training),
                        "energy_score": energy_score(
                            simulated_cumulative,
                            actual_cumulative,
                            rng=np.random.default_rng(seed + 17),
                        ),
                        "variogram_score": variogram_score(
                            simulated_cumulative,
                            actual_cumulative,
                        ),
                        "value_at_risk": value_at_risk,
                        "expected_shortfall": expected_shortfall,
                        "actual_loss": actual_loss,
                        "var_violation": actual_loss > value_at_risk,
                        "predicted_co_crash_probability": predicted_co_crash,
                        "actual_co_crash": actual_co_crash,
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "model_id": model_id,
                        "origin_date": returns.index[
                            origin_position
                        ].date().isoformat(),
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError("Every Stage 0 baseline run failed")
    detail = detail.sort_values(["model_id", "origin_date"]).reset_index(drop=True)
    detail.to_csv(output_root / "rolling_results.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    confidence = float(risk["confidence_level"])
    for model_id, group in detail.groupby("model_id", sort=True):
        violations = group["var_violation"].to_numpy(dtype=bool)
        coverage = christoffersen_conditional_coverage_test(
            violations,
            confidence,
        )
        summary_rows.append(
            {
                "model_id": model_id,
                "forecast_origins": len(group),
                "mean_energy_score": float(group["energy_score"].mean()),
                "mean_variogram_score": float(group["variogram_score"].mean()),
                "joint_var_es_score": joint_var_es_score(
                    group["actual_loss"].to_numpy(),
                    group["value_at_risk"].to_numpy(),
                    group["expected_shortfall"].to_numpy(),
                    confidence,
                ),
                "var_violation_rate": float(group["var_violation"].mean()),
                "kupiec_p_value": float(coverage["p_value"]),
                "christoffersen_independence_p_value": float(
                    coverage["p_value_independence"]
                ),
                "christoffersen_cc_p_value": float(coverage["p_value_cc"]),
                "co_crash_brier_score": brier_score(
                    group["predicted_co_crash_probability"].to_numpy(),
                    group["actual_co_crash"].to_numpy(dtype=float),
                ),
                "mean_realized_loss": float(group["actual_loss"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_energy_score")
    summary.to_csv(output_root / "summary.csv", index=False)
    (output_root / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    receipt = {
        "status": "passed" if not failures else "passed_with_failures",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": configuration["experiment"]["id"],
        "evaluation_split": configuration["experiment"]["evaluation_split"],
        "origin_count": len(origins),
        "model_count": len(configuration["models"]),
        "successful_model_origins": len(detail),
        "failed_model_origins": len(failures),
        "elapsed_seconds": time.perf_counter() - start_time,
        "phase0_manifest_sha256": hash_file(phase0_manifest_path),
        "config_sha256": hash_file(config_path),
        "outputs": {
            "detail": str((output_root / "rolling_results.csv").relative_to(project_root)),
            "summary": str((output_root / "summary.csv").relative_to(project_root)),
            "failures": str((output_root / "failures.json").relative_to(project_root)),
        },
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CrisisForge Stage 0 rolling baseline evaluation"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root_from_module(),
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    receipt = run_stage0_baselines(
        args.project_root.resolve(),
        config_path=args.config.resolve() if args.config else None,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
