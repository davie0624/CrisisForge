from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisisforge.config import (
    display_path,
    load_yaml,
    project_root_from_module,
    resolve_config_path,
)
from crisisforge.data.validation import hash_file
from crisisforge.evaluation.provenance import (
    assert_manifest_binds_file,
    git_state,
    output_hashes,
    read_validation_matrix,
)
from crisisforge.evaluation.rolling import _origin_positions, rolling_cumulative_returns
from crisisforge.regimes import SwitchingDynamicFactorBaseline
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


def build_switching_factor_model(configuration: dict[str, Any]) -> SwitchingDynamicFactorBaseline:
    """Construct the Stage 1 baseline without silently changing YAML choices."""
    factor = configuration["factor_model"]
    regime = configuration["regime_model"]
    dynamics = configuration["factor_dynamics"]
    mapping = configuration["observation_mapping"]
    data = configuration["data"]
    return SwitchingDynamicFactorBaseline(
        n_states=int(regime["n_states"]),
        n_factors=int(factor["n_factors"]),
        return_transform=str(data["return_transform"]),
        factor_scale_floor=float(factor["scale_floor"]),
        hmm_n_init=int(regime["n_initializations"]),
        hmm_max_iter=int(regime["maximum_iterations"]),
        hmm_tolerance=float(regime["tolerance"]),
        transition_pseudocount=float(regime["transition_pseudocount"]),
        sticky_pseudocount=float(regime["sticky_pseudocount"]),
        minimum_covar=float(regime["minimum_covariance_eigenvalue"]),
        hmm_minimum_state_weight=float(regime["minimum_state_weight"]),
        var_ridge=float(dynamics["ridge"]),
        factor_covariance_floor=float(dynamics["covariance_floor"]),
        maximum_spectral_radius=float(dynamics["maximum_spectral_radius"]),
        observation_ridge=float(mapping["ridge"]),
        residual_correlation_shrinkage=float(
            mapping["residual_correlation_shrinkage"]
        ),
        observation_scale_floor=float(mapping["idiosyncratic_scale_floor"]),
        random_state=int(regime["random_seed"]),
    )


def sample_fixed_estimation_paths(
    model: SwitchingDynamicFactorBaseline,
    observed_returns: pd.DataFrame,
    *,
    num_scenarios: int,
    horizon: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from fixed train parameters using information through one origin.

    ``observed_returns`` must end at the forecast origin. It contains no future
    path or future context. The factor transform and all model parameters remain
    fixed; only the origin factor and filtered state distribution are updated.
    """
    if num_scenarios < 1 or horizon < 1:
        raise ValueError("num_scenarios and horizon must be positive")
    factors = model.factor_model.transform(observed_returns)
    filtered = model.regime_model.forward_backward(factors).filtered_probabilities[-1]
    rng = np.random.default_rng(random_seed)
    regimes = model.regime_model.sample_posterior_predictive_paths(
        n_paths=num_scenarios,
        horizon=horizon,
        rng=rng,
        initial_filtered_probabilities=filtered,
    )
    factor_paths = model.factor_dynamics_.sample_paths(
        factors[-1],
        regimes,
        rng=rng,
    )
    observation_paths = model.observation_mapping_.sample_paths(
        factor_paths,
        regimes,
        rng=rng,
    )
    flat = observation_paths.reshape(-1, observation_paths.shape[-1])
    simple_returns = model.factor_model.observation_to_simple_returns(flat).reshape(
        observation_paths.shape
    )
    return simple_returns, filtered


def _state_profiles(
    returns: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    market = returns.mean(axis=1).to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    for state in range(probabilities.shape[1]):
        weights = probabilities[:, state]
        effective = float(weights.sum())
        mean = float(np.dot(weights, market) / effective)
        variance = float(np.dot(weights, np.square(market - mean)) / effective)
        rows.append(
            {
                "state": state,
                "effective_observations": effective,
                "occupancy": effective / len(returns),
                "weighted_market_mean_per_interval": mean,
                "weighted_market_volatility_per_interval": np.sqrt(variance),
            }
        )
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def run_stage1_evaluation(
    project_root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Fit once on training data and evaluate sequentially on validation only."""
    config_path = resolve_config_path(
        project_root,
        config_path,
        default_relative="configs/stage1_evaluation.yaml",
    )
    evaluation = load_yaml(config_path)
    if evaluation["experiment"]["evaluation_split"] != "validation":
        raise ValueError(
            "Stage 1 research evaluation keeps the test split sealed; "
            "evaluation_split must be validation"
        )
    pipeline = load_yaml(project_root / "configs/pipeline.yaml")
    model_config_path = project_root / evaluation["paths"]["model_config"]
    model_configuration = load_yaml(model_config_path)
    matrix_path = project_root / evaluation["paths"]["model_matrix"]
    phase0_manifest_path = project_root / evaluation["paths"]["phase0_manifest"]
    output_root = project_root / evaluation["paths"]["output"]
    output_root.mkdir(parents=True, exist_ok=True)

    model_matrix_sha256 = assert_manifest_binds_file(
        manifest_path=phase0_manifest_path,
        project_root=project_root,
        file_path=matrix_path,
    )
    matrix = read_validation_matrix(
        matrix_path,
        validation_end=pipeline["splits"]["validation_end"],
    )
    asset_columns = [column for column in matrix if column.startswith("asset__")]
    returns = matrix[asset_columns]
    if returns.empty or returns.isna().any().any():
        raise ValueError("Stage 1 requires a complete asset-return panel")

    train_end = pd.Timestamp(pipeline["splits"]["train_end"])
    train = returns.loc[returns.index <= train_end]
    if train.empty:
        raise ValueError("The configured chronological training split is empty")

    model = build_switching_factor_model(model_configuration)
    start_time = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - start_time
    diagnostics = model.diagnostics()
    diagnostics["fit_seconds"] = fit_seconds
    diagnostics["claims_boundary"] = {
        "estimator": "MAP/empirical-Bayes multi-stage baseline",
        "full_parameter_posterior": False,
        "economic_state_names_inferred": False,
        "test_set_opened": False,
    }
    diagnostics_path = output_root / "diagnostics.json"
    filtered_path = output_root / "train_filtered_probabilities.parquet"
    smoothed_path = output_root / "train_smoothed_probabilities.parquet"
    state_profiles_path = output_root / "state_profiles.csv"
    detail_path = output_root / "rolling_results.csv"
    scenarios_path = output_root / "cumulative_scenarios.npz"
    failures_path = output_root / "failures.json"
    summary_path = output_root / "summary.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )

    filtered_frame = pd.DataFrame(
        model.filtered_probabilities_,
        index=train.index,
        columns=[f"state_{index}" for index in range(model.n_states)],
    )
    smoothed_frame = pd.DataFrame(
        model.smoothed_probabilities_,
        index=train.index,
        columns=[f"state_{index}" for index in range(model.n_states)],
    )
    filtered_frame.to_parquet(filtered_path)
    smoothed_frame.to_parquet(smoothed_path)
    state_profiles = _state_profiles(train, model.smoothed_probabilities_)
    state_profiles.to_csv(state_profiles_path, index=False)

    forecast = evaluation["forecast"]
    risk = evaluation["risk"]
    horizon = int(forecast["horizon"])
    origins = _origin_positions(
        returns.index,
        evaluation_split="validation",
        train_end=pipeline["splits"]["train_end"],
        validation_end=pipeline["splits"]["validation_end"],
        horizon=horizon,
        stride=int(forecast["origin_stride"]),
    )
    threshold_outcomes = rolling_cumulative_returns(
        train.to_numpy(dtype=float),
        horizon=horizon,
    )
    thresholds = fit_co_crash_thresholds(
        threshold_outcomes,
        marginal_quantile=float(risk["co_crash_marginal_quantile"]),
    )
    equal_weights = np.repeat(1.0 / len(asset_columns), len(asset_columns))
    confidence = float(risk["confidence_level"])
    minimum_fraction = float(risk["co_crash_minimum_fraction"])
    base_seed = int(evaluation["experiment"]["random_seed"])
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cumulative_scenario_blocks: list[np.ndarray] = []
    scenario_origin_dates: list[str] = []

    evaluation_start = time.perf_counter()
    for origin_number, origin_position in enumerate(origins):
        seed = base_seed + origin_number
        origin_date = returns.index[origin_position]
        actual_path = returns.iloc[
            origin_position + 1 : origin_position + 1 + horizon
        ].to_numpy(dtype=float)
        try:
            scenarios, filtered = sample_fixed_estimation_paths(
                model,
                returns.iloc[: origin_position + 1],
                num_scenarios=int(forecast["num_scenarios"]),
                horizon=horizon,
                random_seed=seed,
            )
            simulated_cumulative = aggregate_path_returns(scenarios)
            actual_cumulative = aggregate_path_returns(actual_path[None, :, :])[0]
            simulated_losses = portfolio_losses(
                simulated_cumulative,
                equal_weights,
            )
            actual_loss = float(-(actual_cumulative @ equal_weights))
            value_at_risk = empirical_var(simulated_losses, confidence)
            expected_shortfall = empirical_expected_shortfall(
                simulated_losses,
                confidence,
            )
            records.append(
                {
                    "model_id": evaluation["experiment"]["id"],
                    "origin_date": origin_date.date().isoformat(),
                    "horizon_end_date": returns.index[
                        origin_position + horizon
                    ].date().isoformat(),
                    "seed": seed,
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
                    "predicted_co_crash_probability": co_crash_probability(
                        simulated_cumulative,
                        thresholds,
                        minimum_fraction=minimum_fraction,
                    ),
                    "actual_co_crash": realized_co_crash(
                        actual_cumulative,
                        thresholds,
                        minimum_fraction=minimum_fraction,
                    ),
                    "filtered_state_entropy": float(
                        -np.sum(filtered * np.log(np.maximum(filtered, 1e-300)))
                    ),
                    **{
                        f"filtered_state_{state}": float(probability)
                        for state, probability in enumerate(filtered)
                    },
                }
            )
            cumulative_scenario_blocks.append(simulated_cumulative)
            scenario_origin_dates.append(origin_date.date().isoformat())
        except Exception as exc:
            failures.append(
                {
                    "origin_date": origin_date.date().isoformat(),
                    "seed": seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    evaluation_seconds = time.perf_counter() - evaluation_start
    detail = pd.DataFrame(records)
    if detail.empty:
        raise RuntimeError("Every Stage 1 validation origin failed")
    detail.to_csv(detail_path, index=False)
    np.savez_compressed(
        scenarios_path,
        scenarios=np.stack(cumulative_scenario_blocks),
        origin_dates=np.asarray(scenario_origin_dates),
        asset_columns=np.asarray(asset_columns),
    )
    failures_path.write_text(
        json.dumps(failures, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    violations = detail["var_violation"].to_numpy(dtype=bool)
    coverage = christoffersen_conditional_coverage_test(violations, confidence)
    realized_losses = detail["actual_loss"].to_numpy(dtype=float)
    var_forecasts = detail["value_at_risk"].to_numpy(dtype=float)
    es_forecasts = detail["expected_shortfall"].to_numpy(dtype=float)
    actual_crashes = detail["actual_co_crash"].to_numpy(dtype=float)
    predicted_crashes = detail["predicted_co_crash_probability"].to_numpy(
        dtype=float
    )
    summary = {
        "model_id": evaluation["experiment"]["id"],
        "evaluation_split": "validation",
        "test_set_opened": False,
        "estimation_policy": evaluation["experiment"]["estimation_policy"],
        "origins": int(len(detail)),
        "failures": int(len(failures)),
        "mean_energy_score": float(detail["energy_score"].mean()),
        "mean_variogram_score": float(detail["variogram_score"].mean()),
        "joint_var_es_score": joint_var_es_score(
            realized_losses,
            var_forecasts,
            es_forecasts,
            confidence,
        ),
        "mean_brier_score": brier_score(predicted_crashes, actual_crashes),
        "actual_co_crash_count": int(actual_crashes.sum()),
        "mean_filtered_state_entropy": float(
            detail["filtered_state_entropy"].mean()
        ),
        "coverage": coverage,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    persisted_outputs = {
        "diagnostics": diagnostics_path,
        "train_filtered_probabilities": filtered_path,
        "train_smoothed_probabilities": smoothed_path,
        "state_profiles": state_profiles_path,
        "detail": detail_path,
        "cumulative_scenarios": scenarios_path,
        "failures": failures_path,
        "summary": summary_path,
    }

    receipt = {
        "status": "completed" if not failures else "completed_with_failures",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": evaluation["experiment"]["id"],
        "evaluation_config": display_path(config_path, project_root=project_root),
        "evaluation_config_sha256": hash_file(config_path),
        "model_config": str(model_config_path.relative_to(project_root)),
        "model_config_sha256": hash_file(model_config_path),
        "phase0_manifest": str(phase0_manifest_path.relative_to(project_root)),
        "phase0_manifest_sha256": hash_file(phase0_manifest_path),
        "model_matrix_sha256": model_matrix_sha256,
        "git": git_state(project_root),
        "outputs": output_hashes(project_root, persisted_outputs),
        "summary": summary,
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sealed-test Stage 1 switching-factor validation"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else project_root_from_module()
    )
    receipt = run_stage1_evaluation(project_root, config_path=args.config)
    print(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
