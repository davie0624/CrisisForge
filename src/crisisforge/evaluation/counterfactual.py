from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
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
from crisisforge.counterfactual import (
    Intervention,
    RegimeSwitchingSCM,
    SCMParameters,
    estimate_causal_effect,
    evaluate_counterfactual_error,
)
from crisisforge.data.validation import hash_file
from crisisforge.evaluation.provenance import git_state, output_hashes


def _constant_schedule(value: float, steps: int) -> dict[int, float]:
    if steps < 1 or not np.isfinite(value):
        raise ValueError("steps must be positive and value finite")
    return {time: float(value) for time in range(steps)}


def build_registered_interventions(
    configuration: dict[str, Any],
) -> tuple[Intervention, Intervention, Intervention]:
    """Build total and properly paired controlled-direct-effect interventions."""
    policy = configuration["policy_intervention"]
    active_steps = int(policy["active_steps"])
    factual_policy = _constant_schedule(
        float(policy["factual_control_value"]),
        active_steps,
    )
    treatment_policy = _constant_schedule(
        float(policy["treatment_value"]),
        active_steps,
    )
    controlled = configuration["controlled_direct_effect"]
    mediator_schedule = _constant_schedule(
        float(controlled["mediator_value"]),
        int(controlled["active_steps"]),
    )
    assignments = {
        str(mediator): dict(mediator_schedule)
        for mediator in controlled["mediators"]
    }
    return (
        Intervention(policy=treatment_policy, label="total_policy_treatment"),
        Intervention(
            policy=factual_policy,
            controlled=assignments,
            label="controlled_reference",
        ),
        Intervention(
            policy=treatment_policy,
            controlled=assignments,
            label="controlled_policy_treatment",
        ),
    )


def _misspecified_models(
    truth: SCMParameters,
    configuration: dict[str, Any],
) -> dict[str, RegimeSwitchingSCM]:
    misspecification = configuration["misspecification"]
    reduced = replace(
        truth,
        yield_policy=truth.yield_policy
        * float(misspecification["reduced_yield_policy_multiplier"]),
    )
    models = {"reduced_yield_transmission": RegimeSwitchingSCM(reduced)}
    if bool(misspecification["remove_lagged_equity_feedback"]):
        models["no_lagged_equity_feedback"] = RegimeSwitchingSCM(
            replace(truth, policy_equity_feedback=0.0)
        )
    return models


def _effect_record(
    *,
    model_id: str,
    outcome: str,
    effect_type: str,
    effect: Any,
    error: Any | None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "outcome": outcome,
        "effect_type": effect_type,
        "ate_terminal": effect.ate_terminal,
        "cumulative_ate": effect.cumulative_ate,
        "tail_effect": effect.tail_effect,
        "ate_error": None if error is None else error.ate_error,
        "path_rmse": None if error is None else error.path_rmse,
        "tail_effect_error": None if error is None else error.tail_effect_error,
    }


def run_counterfactual_evaluation(
    project_root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Validate AAP recovery and quantify structural misspecification sensitivity."""
    config_path = resolve_config_path(
        project_root,
        config_path,
        default_relative="configs/stage6_counterfactual_evaluation.yaml",
    )
    configuration = load_yaml(config_path)
    experiment = configuration["experiment"]
    model_config_path = project_root / configuration["paths"]["model_config"]
    model_configuration = load_yaml(model_config_path)
    output_root = project_root / configuration["paths"]["output"]
    output_root.mkdir(parents=True, exist_ok=True)
    confidence = float(experiment["confidence_level"])

    truth_parameters = SCMParameters(**model_configuration["parameters"])
    truth = RegimeSwitchingSCM(truth_parameters)
    noise = truth.generate_exogenous(
        num_paths=int(experiment["num_paths"]),
        horizon=int(experiment["horizon"]),
        seed=int(experiment["random_seed"]),
    )
    total_treatment, controlled_reference, controlled_treatment = (
        build_registered_interventions(configuration)
    )
    factual = truth.simulate(noise)
    true_total = truth.simulate(noise, total_treatment)
    oracle_total = truth.abduction_action_prediction(factual, total_treatment)
    controlled_factual = truth.simulate(noise, controlled_reference)
    controlled_counterfactual = truth.simulate(noise, controlled_treatment)

    summary_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for outcome in ("equity", "volatility"):
        truth_effect = estimate_causal_effect(
            factual,
            true_total,
            outcome=outcome,
            effect_type="total",
            confidence_level=confidence,
        )
        oracle_effect = estimate_causal_effect(
            factual,
            oracle_total,
            outcome=outcome,
            effect_type="total",
            confidence_level=confidence,
        )
        oracle_error = evaluate_counterfactual_error(
            factual=factual,
            estimated_counterfactual=oracle_total,
            ground_truth_counterfactual=true_total,
            outcome=outcome,
            effect_type="total",
            confidence_level=confidence,
        )
        summary_rows.append(
            _effect_record(
                model_id="known_scm_ground_truth",
                outcome=outcome,
                effect_type="total",
                effect=truth_effect,
                error=None,
            )
        )
        summary_rows.append(
            _effect_record(
                model_id="oracle_aap",
                outcome=outcome,
                effect_type="total",
                effect=oracle_effect,
                error=oracle_error,
            )
        )
        for time, value in enumerate(truth_effect.mean_path):
            path_rows.append(
                {
                    "model_id": "known_scm_ground_truth",
                    "outcome": outcome,
                    "effect_type": "total",
                    "time": time,
                    "mean_effect": float(value),
                }
            )
        for model_id, model in _misspecified_models(
            truth.parameters,
            configuration,
        ).items():
            estimated = model.abduction_action_prediction(factual, total_treatment)
            effect = estimate_causal_effect(
                factual,
                estimated,
                outcome=outcome,
                effect_type="total",
                confidence_level=confidence,
            )
            error = evaluate_counterfactual_error(
                factual=factual,
                estimated_counterfactual=estimated,
                ground_truth_counterfactual=true_total,
                outcome=outcome,
                effect_type="total",
                confidence_level=confidence,
            )
            summary_rows.append(
                _effect_record(
                    model_id=model_id,
                    outcome=outcome,
                    effect_type="total",
                    effect=effect,
                    error=error,
                )
            )
            for time, value in enumerate(effect.mean_path):
                path_rows.append(
                    {
                        "model_id": model_id,
                        "outcome": outcome,
                        "effect_type": "total",
                        "time": time,
                        "mean_effect": float(value),
                    }
                )

        controlled_effect = estimate_causal_effect(
            controlled_factual,
            controlled_counterfactual,
            outcome=outcome,
            effect_type="controlled",
            confidence_level=confidence,
        )
        summary_rows.append(
            _effect_record(
                model_id="known_scm_controlled_direct_effect",
                outcome=outcome,
                effect_type="controlled",
                effect=controlled_effect,
                error=None,
            )
        )
        for time, value in enumerate(controlled_effect.mean_path):
            path_rows.append(
                {
                    "model_id": "known_scm_controlled_direct_effect",
                    "outcome": outcome,
                    "effect_type": "controlled",
                    "time": time,
                    "mean_effect": float(value),
                }
            )

    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(path_rows)
    effect_summary_path = output_root / "effect_summary.csv"
    mean_paths_path = output_root / "mean_effect_paths.csv"
    summary_path = output_root / "summary.json"
    summary.to_csv(effect_summary_path, index=False)
    paths.to_csv(mean_paths_path, index=False)
    oracle_errors = summary.loc[summary["model_id"] == "oracle_aap"]
    maximum_oracle_error = float(
        oracle_errors[["ate_error", "path_rmse", "tail_effect_error"]]
        .to_numpy(dtype=float)
        .max()
    )
    misspecified_errors = summary.loc[
        summary["model_id"].isin(
            ["reduced_yield_transmission", "no_lagged_equity_feedback"]
        )
    ]
    result_summary = {
        "experiment_id": experiment["id"],
        "num_paths": int(experiment["num_paths"]),
        "horizon": int(experiment["horizon"]),
        "maximum_oracle_error": maximum_oracle_error,
        "minimum_misspecified_path_rmse": float(
            misspecified_errors["path_rmse"].min()
        ),
        "known_ground_truth_scope": "semi-synthetic only",
        "real_market_causal_identification": False,
        "controlled_effect_definition": (
            "treated and reference worlds hold identical mediator schedules"
        ),
    }
    summary_path.write_text(
        json.dumps(result_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_path = Path(__file__).resolve().parents[1] / "counterfactual/scm.py"
    persisted_outputs = {
        "effect_summary": effect_summary_path,
        "mean_effect_paths": mean_paths_path,
        "summary": summary_path,
    }
    receipt = {
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": display_path(config_path, project_root=project_root),
        "config_sha256": hash_file(config_path),
        "model_config": display_path(model_config_path, project_root=project_root),
        "model_config_sha256": hash_file(model_config_path),
        "scm_source_sha256": hash_file(source_path),
        "scm_parameters": asdict(truth_parameters),
        "git": git_state(project_root),
        "outputs": output_hashes(project_root, persisted_outputs),
        "summary": result_summary,
        "claims_boundary": configuration["claims_boundary"],
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the semi-synthetic structural counterfactual evaluation"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else project_root_from_module()
    )
    receipt = run_counterfactual_evaluation(
        project_root,
        config_path=args.config,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
