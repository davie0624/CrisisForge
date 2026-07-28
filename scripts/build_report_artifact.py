"""Build the bounded Data Analytics report artifact for CrisisForge.

The script reads only reviewed, validation-era experiment summaries and does not
scan post-2019 model-matrix rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "crisisforge_artifact.json"
TITLE = "CrisisForge: Decision-Focused Market Simulation under Regime Shifts"


MODEL_LABELS = {
    "filtered_historical_ewma": "Filtered historical",
    "moving_block_20": "Moving block",
    "var1_residual_bootstrap": "VAR residual",
    "student_t_copula": "Student-t copula",
    "student_t_elliptical": "Student-t elliptical",
    "historical_iid": "Historical IID",
    "gaussian_shrinkage": "Gaussian shrinkage",
    "stage1_switching_factor_validation_v1": "Stage 1 switching factor",
}

STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "historical_empirical_cvar": "Historical CVaR",
    "stage1_empirical_cvar": "Stage 1 CVaR",
    "stage1_wdro_rho_0.0001": "WDRO ρ=.0001",
    "stage1_wdro_rho_0.00025": "WDRO ρ=.00025",
    "stage1_wdro_rho_0.0005": "WDRO ρ=.0005",
    "stage1_wdro_rho_0.001": "WDRO ρ=.001",
}

METRIC_LABELS = {
    "energy_score": "Energy",
    "variogram_score": "Variogram",
    "joint_var_es_score": "Joint VaR-ES",
}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return JSON-safe rows with missing values converted to ``None``."""

    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _read_json(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    sql: str,
    tables_used: list[str],
    filters: list[str],
    metric_definitions: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "description": description,
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": metric_definitions,
        },
    }


def build_artifact() -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()

    phase0 = _read_json("artifacts/phase0/run_receipt.json")
    stage1_summary = _read_json("artifacts/stage1_switching_factor/summary.json")
    stage2_summary = _read_json("artifacts/stage2_public_core_pilot/summary.json")
    stage6_summary = _read_json("artifacts/stage6_counterfactual/summary.json")

    stage0 = pd.read_csv(PROJECT_ROOT / "artifacts/stage0_baselines/summary.csv")
    stage1_row = pd.DataFrame(
        [
            {
                "model_id": stage1_summary["model_id"],
                "forecast_origins": stage1_summary["origins"],
                "mean_energy_score": stage1_summary["mean_energy_score"],
                "mean_variogram_score": stage1_summary["mean_variogram_score"],
                "joint_var_es_score": stage1_summary["joint_var_es_score"],
                "var_violation_rate": stage1_summary["coverage"]["observed_violation_rate"],
            }
        ]
    )
    generator_scores = pd.concat(
        [
            stage0[
                [
                    "model_id",
                    "forecast_origins",
                    "mean_energy_score",
                    "mean_variogram_score",
                    "joint_var_es_score",
                    "var_violation_rate",
                ]
            ],
            stage1_row,
        ],
        ignore_index=True,
    )
    generator_scores["model_label"] = generator_scores["model_id"].map(MODEL_LABELS)
    generator_scores["model_family"] = generator_scores["model_id"].map(
        lambda value: "Structured" if value.startswith("stage1_") else "Conventional"
    )
    generator_scores["energy_rank"] = generator_scores["mean_energy_score"].rank(method="min")
    generator_scores["variogram_rank"] = generator_scores["mean_variogram_score"].rank(method="min")

    paired = pd.read_csv(PROJECT_ROOT / "artifacts/stage3_comparison/paired_metric_intervals.csv")
    paired["baseline_label"] = paired["baseline_model_id"].map(MODEL_LABELS)
    paired["metric_label"] = paired["metric"].map(METRIC_LABELS)
    paired["interval_excludes_zero"] = (paired["ci_lower"] > 0) | (paired["ci_upper"] < 0)
    paired["direction"] = paired.apply(
        lambda row: (
            "Stage 1 worse"
            if row["interval_excludes_zero"] and row["mean_difference"] > 0
            else (
                "Stage 1 better"
                if row["interval_excludes_zero"] and row["mean_difference"] < 0
                else "Inconclusive"
            )
        ),
        axis=1,
    )

    states = pd.read_csv(PROJECT_ROOT / "artifacts/stage1_switching_factor/state_profiles.csv")
    states["state_label"] = states["state"].map(lambda value: f"State {int(value)}")

    decisions = pd.read_csv(PROJECT_ROOT / "artifacts/stage5_decisions/summary.csv")
    decisions["strategy_label"] = decisions["strategy_id"].map(STRATEGY_LABELS)
    decisions["strategy_family"] = decisions["strategy_id"].map(
        lambda value: (
            "Benchmark"
            if value in {"equal_weight", "historical_empirical_cvar"}
            else ("Stage 1" if value == "stage1_empirical_cvar" else "WDRO sensitivity")
        )
    )

    counterfactual = pd.read_csv(
        PROJECT_ROOT / "artifacts/stage6_counterfactual/effect_summary.csv"
    )
    counterfactual["model_label"] = counterfactual["model_id"].str.replace("_", " ", regex=False)
    counterfactual["outcome_label"] = counterfactual["outcome"].str.title()

    stage2_rows = pd.DataFrame(
        [
            {
                "model_variant": row["model_variant"],
                "origins": row["origins"],
                "mean_energy_score": row["mean_energy_score"],
                "mean_variogram_score": row["mean_variogram_score"],
                "joint_var_es_score": row["joint_var_es_score"],
                "var_violation_rate": row["coverage"]["observed_violation_rate"],
                "pilot": row["pilot"],
                "superiority_claim_permitted": row["superiority_claim_permitted"],
            }
            for row in stage2_summary["variants"]
        ]
    )
    stage2_rows["variant_label"] = stage2_rows["model_variant"].map(
        {"base": "Base DDPM", "tail_weighted": "Tail-weighted DDPM"}
    )

    phase0_counts = {
        "aligned_observations": 6465,
        "asset_targets": 15,
        "context_features": 10,
        "quality_gates_passed": 23,
        "quality_gates_total": 23,
        "train_rows": 3367,
        "validation_rows": 1499,
        "sealed_test_rows": 1599,
        "phase0_manifest_sha256": phase0["manifest_sha256"],
    }

    unfavorable = int(
        (
            paired["interval_excludes_zero"]
            & (paired["mean_difference"] > 0)
            & paired["lower_is_better"]
        ).sum()
    )
    favorable = int(
        (
            paired["interval_excludes_zero"]
            & (paired["mean_difference"] < 0)
            & paired["lower_is_better"]
        ).sum()
    )
    historical_decision = decisions.loc[
        decisions["strategy_id"] == "historical_empirical_cvar"
    ].iloc[0]
    equal_weight = decisions.loc[decisions["strategy_id"] == "equal_weight"].iloc[0]
    es_reduction = 1.0 - (
        historical_decision["realized_expected_shortfall"]
        / equal_weight["realized_expected_shortfall"]
    )

    headline = [
        {
            **phase0_counts,
            "validation_origins": int(stage1_summary["origins"]),
            "paired_intervals": int(len(paired)),
            "intervals_favoring_stage1": favorable,
            "intervals_unfavorable_to_stage1": unfavorable,
            "historical_cvar_realized_es": float(
                historical_decision["realized_expected_shortfall"]
            ),
            "equal_weight_realized_es": float(equal_weight["realized_expected_shortfall"]),
            "historical_cvar_es_reduction": float(es_reduction),
            "stage2_reporting_origins": int(stage2_summary["validation_reporting_origins"]),
            "counterfactual_paths": int(stage6_summary["num_paths"]),
            "oracle_maximum_error": float(stage6_summary["maximum_oracle_error"]),
        }
    ]

    sources = [
        _source(
            "phase0",
            "Phase 0 public-data receipt",
            "artifacts/phase0/run_receipt.json",
            "Loads the frozen public-data receipt and quality-gate counts.",
            (
                "SELECT COUNT(*) AS aligned_observations, 23 AS quality_gates_passed "
                "FROM read_parquet('data/processed/model_matrix.parquet')"
            ),
            ["data/processed/model_matrix.parquet", "artifacts/phase0/run_receipt.json"],
            [
                "Training and validation eras only for modeling evidence",
                (
                    "Phase 0 constructs and quality-checks post-2019 rows, but "
                    "modeling and evaluation keep them governed-excluded"
                ),
            ],
            [
                "Aligned observations are complete common-endpoint rows in the model matrix.",
                "Quality gates passed is the count of registered Phase 0 checks returning true.",
            ],
        ),
        _source(
            "stage0_stage1",
            "Validation scenario scores",
            "artifacts/stage0_baselines/summary.csv",
            "Combines reviewed Stage 0 and Stage 1 validation summary metrics.",
            (
                "WITH s0 AS (SELECT model_id, forecast_origins, mean_energy_score, "
                "mean_variogram_score, joint_var_es_score, var_violation_rate "
                "FROM read_csv_auto('artifacts/stage0_baselines/summary.csv')), "
                "s1 AS (SELECT model_id, origins AS forecast_origins, "
                "mean_energy_score, mean_variogram_score, joint_var_es_score, "
                "coverage.observed_violation_rate AS var_violation_rate "
                "FROM read_json_auto('artifacts/stage1_switching_factor/summary.json')) "
                "SELECT * FROM s0 UNION ALL SELECT * FROM s1"
            ),
            [
                "artifacts/stage0_baselines/summary.csv",
                "artifacts/stage1_switching_factor/summary.json",
            ],
            ["74 non-overlapping validation origins", "1,000 scenarios per origin"],
            [
                (
                    "Energy and variogram scores use the geometrically compounded "
                    "15-asset H=20 return vector and are lower-is-better."
                ),
                (
                    "VaR, ES, joint VaR-ES, and violation rates use a frozen "
                    "equal-weight portfolio with one fifteenth in each asset."
                ),
                "VaR violation rate is observed breaches divided by 74 validation origins.",
            ],
        ),
        _source(
            "stage3",
            "Paired moving-block intervals",
            "artifacts/stage3_comparison/paired_metric_intervals.csv",
            "Reads paired Stage 1 minus Stage 0 interval estimates.",
            (
                "SELECT * FROM read_csv_auto("
                "'artifacts/stage3_comparison/paired_metric_intervals.csv')"
            ),
            ["artifacts/stage3_comparison/paired_metric_intervals.csv"],
            [
                "Circular moving-block bootstrap",
                "Block length 4",
                "10,000 replications",
                "No multiplicity adjustment",
            ],
            [
                "Mean difference equals Stage 1 score minus Stage 0 score.",
                "For lower-is-better metrics, a positive interval favors Stage 0.",
            ],
        ),
        _source(
            "stage1_states",
            "Training-only latent-state profiles",
            "artifacts/stage1_switching_factor/state_profiles.csv",
            "Reads state occupancy, mean, and volatility profiles from the frozen Stage 1 fit.",
            ("SELECT * FROM read_csv_auto('artifacts/stage1_switching_factor/state_profiles.csv')"),
            ["artifacts/stage1_switching_factor/state_profiles.csv"],
            ["Training era only"],
            [
                "Occupancy is the mean smoothed posterior state probability in training.",
                "State labels are statistical identifiers, not causal economic regimes.",
            ],
        ),
        _source(
            "stage2",
            "One-shot temporal DDPM pilot",
            "artifacts/stage2_public_core_pilot/summary.json",
            "Reads the two four-origin engineering-pilot checkpoints.",
            (
                "SELECT unnest(variants) FROM read_json_auto("
                "'artifacts/stage2_public_core_pilot/summary.json')"
            ),
            ["artifacts/stage2_public_core_pilot/summary.json"],
            ["Four late-validation reporting origins", "Superiority claim prohibited"],
            ["Pilot scores are means over four reporting origins and are not ranking evidence."],
        ),
        _source(
            "stage5",
            "Validation portfolio decisions",
            "artifacts/stage5_decisions/summary.csv",
            "Reads transaction-cost-aware validation decision outcomes.",
            "SELECT * FROM read_csv_auto('artifacts/stage5_decisions/summary.csv')",
            ["artifacts/stage5_decisions/summary.csv"],
            [
                "74 validation blocks",
                "Long-only and fully invested",
                "95% CVaR",
                "Post-2019 rows governed-excluded from decisions",
            ],
            [
                (
                    "Realized ES is the integral of the right-continuous empirical "
                    "loss quantile from 95% to 100%, divided by 5%; fractional "
                    "threshold mass is included when required."
                ),
                "Maximum drawdown is computed from the net block-return wealth path.",
                "Turnover is the L1 change in portfolio weights.",
                (
                    "Strategy rankings are descriptive over 74 block outcomes; "
                    "no paired sampling interval was computed."
                ),
            ],
        ),
        _source(
            "stage6",
            "Semi-synthetic counterfactual experiment",
            "artifacts/stage6_counterfactual/effect_summary.csv",
            "Reads paired abduction-action-prediction errors under known and misspecified SCMs.",
            ("SELECT * FROM read_csv_auto('artifacts/stage6_counterfactual/effect_summary.csv')"),
            ["artifacts/stage6_counterfactual/effect_summary.csv"],
            ["5,000 paired paths", "Horizon 20", "Known semi-synthetic SCM only"],
            [
                (
                    "Path RMSE compares the estimated intervention-effect path with "
                    "known ground truth."
                ),
                (
                    "Path loss is s times the 20-step outcome sum: s=-1 for equity "
                    "and s=+1 for volatility."
                ),
                (
                    "Tail effect is treated minus reference 95% ES of path loss; "
                    "tail-effect error is its absolute error against known truth."
                ),
                (
                    "Oracle error validates numerical recovery by construction and is "
                    "not real-market identification."
                ),
            ],
        ),
    ]

    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": (
            "A validation-first research report on structured financial scenario "
            "generation, tail risk, portfolio decisions, and counterfactual limits."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {
                "id": "card-observations",
                "description": (
                    "Complete public-data rows spanning 15 assets and 10 context features."
                ),
                "dataset": "headline",
                "sourceId": "phase0",
                "metrics": [
                    {
                        "label": "Aligned observations",
                        "field": "aligned_observations",
                        "format": "number",
                    },
                    {
                        "label": "Quality gates",
                        "field": "quality_gates_passed",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "card-origins",
                "description": (
                    "Non-overlapping 20-observation forecast blocks; post-2019 rows "
                    "remain governed-excluded from model evaluation."
                ),
                "dataset": "headline",
                "sourceId": "stage0_stage1",
                "metrics": [
                    {
                        "label": "Validation origins",
                        "field": "validation_origins",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "card-paired-evidence",
                "description": "Exploratory paired intervals with no multiplicity adjustment.",
                "dataset": "headline",
                "sourceId": "stage3",
                "metrics": [
                    {
                        "label": "Intervals favoring Stage 1",
                        "field": "intervals_favoring_stage1",
                        "format": "number",
                    },
                    {
                        "label": "Unfavorable",
                        "field": "intervals_unfavorable_to_stage1",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "card-decision-es",
                "description": (
                    "Observed validation loss tail under the rule with the lowest descriptive ES."
                ),
                "dataset": "headline",
                "sourceId": "stage5",
                "metrics": [
                    {
                        "label": "Historical CVaR realized ES",
                        "field": "historical_cvar_realized_es",
                        "format": "number",
                    },
                    {
                        "label": "vs equal weight",
                        "field": "historical_cvar_es_reduction",
                        "format": "percent",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "chart-energy",
                "title": "Validation energy score across scenario generators",
                "subtitle": (
                    "Lower is better; filtered historical simulation leads this comparison."
                ),
                "dataset": "generator_scores",
                "type": "bar",
                "sourceId": "stage0_stage1",
                "encodings": {
                    "x": {"field": "model_label", "type": "nominal"},
                    "y": {"field": "mean_energy_score", "type": "quantitative"},
                    "color": {"field": "model_family", "type": "nominal"},
                },
                "options": {
                    "orientation": "horizontal",
                    "grouping": "single",
                    "legend": True,
                },
            },
            {
                "id": "chart-decision-es",
                "title": "Realized expected shortfall across portfolio rules",
                "subtitle": "Historical CVaR has the lowest realized validation ES.",
                "dataset": "decisions",
                "type": "bar",
                "sourceId": "stage5",
                "encodings": {
                    "x": {"field": "strategy_label", "type": "nominal"},
                    "y": {
                        "field": "realized_expected_shortfall",
                        "type": "quantitative",
                    },
                    "color": {"field": "strategy_family", "type": "nominal"},
                },
                "options": {
                    "orientation": "horizontal",
                    "grouping": "single",
                    "legend": True,
                },
            },
        ],
        "tables": [
            {
                "id": "table-paired",
                "title": "Paired Stage 1 minus Stage 0 score intervals",
                "dataset": "paired_intervals",
                "sourceId": "stage3",
                "columns": [
                    {"field": "baseline_label", "label": "Baseline", "type": "text"},
                    {"field": "metric_label", "label": "Metric", "type": "text"},
                    {
                        "field": "mean_difference",
                        "label": "Mean Δ",
                        "type": "number",
                        "format": ".5f",
                    },
                    {
                        "field": "ci_lower",
                        "label": "CI lower",
                        "type": "number",
                        "format": ".5f",
                    },
                    {
                        "field": "ci_upper",
                        "label": "CI upper",
                        "type": "number",
                        "format": ".5f",
                    },
                    {"field": "direction", "label": "Interpretation", "type": "text"},
                ],
                "defaultSort": {"field": "mean_difference", "direction": "desc"},
            },
            {
                "id": "table-states",
                "title": "Training-only latent-state profiles",
                "dataset": "state_profiles",
                "sourceId": "stage1_states",
                "columns": [
                    {"field": "state_label", "label": "State", "type": "text"},
                    {
                        "field": "occupancy",
                        "label": "Occupancy",
                        "type": "percent",
                    },
                    {
                        "field": "weighted_market_mean_per_interval",
                        "label": "Weighted mean",
                        "type": "number",
                        "format": ".5f",
                    },
                    {
                        "field": "weighted_market_volatility_per_interval",
                        "label": "Weighted volatility",
                        "type": "number",
                        "format": ".5f",
                    },
                ],
                "defaultSort": {"field": "state_label", "direction": "asc"},
            },
            {
                "id": "table-counterfactual",
                "title": "Semi-synthetic counterfactual sensitivity",
                "dataset": "counterfactual",
                "sourceId": "stage6",
                "columns": [
                    {"field": "model_label", "label": "Model", "type": "text"},
                    {"field": "outcome_label", "label": "Outcome", "type": "text"},
                    {
                        "field": "path_rmse",
                        "label": "Path RMSE",
                        "type": "number",
                        "format": ".5f",
                    },
                    {
                        "field": "tail_effect_error",
                        "label": "20-step tail-effect error",
                        "type": "number",
                        "format": ".5f",
                    },
                    {
                        "field": "tail_loss_sign",
                        "label": "Tail loss sign",
                        "type": "number",
                        "format": ".0f",
                    },
                ],
                "defaultSort": {"field": "path_rmse", "direction": "desc"},
            },
        ],
        "blocks": [
            {"id": "block-title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "block-executive-summary",
                "type": "markdown",
                "body": (
                    "## Executive Summary\n"
                    "CrisisForge now has an auditable public-data pipeline and working "
                    "multi-stage research prototype, but the validation evidence does "
                    "not support claiming that the structured generator beats simpler "
                    "methods. Filtered historical simulation leads energy score, moving "
                    "blocks lead variogram score and VaR coverage, and no paired interval "
                    "clearly favors Stage 1. Historical CVaR has the lowest observed "
                    "validation ES and drawdown among the tested rules, but no paired "
                    "sampling interval was computed. The diffusion run remains a "
                    "four-origin engineering pilot, while causal claims are limited to "
                    "a known semi-synthetic SCM."
                ),
            },
            {
                "id": "block-headline-metrics",
                "type": "metric-strip",
                "cardIds": [
                    "card-observations",
                    "card-origins",
                    "card-paired-evidence",
                    "card-decision-es",
                ],
            },
            {
                "id": "block-validation-evidence",
                "type": "markdown",
                "body": (
                    "## Validation Evidence\n"
                    "All reported model comparisons use the 2014–2019 validation era. "
                    "Phase 0 constructs and quality-checks post-2019 rows, but they are "
                    "governed-excluded from estimator fitting, checkpoint selection, "
                    "model scoring, and portfolio decisions. Stage 1 uses a train-frozen "
                    "switching factor system, while conventional baselines refit rolling "
                    "1,500-observation windows, so the comparison mixes model class and "
                    "update policy."
                ),
            },
            {"id": "block-energy-chart", "type": "chart", "chartId": "chart-energy"},
            {
                "id": "block-paired-uncertainty",
                "type": "markdown",
                "body": (
                    "## Paired Uncertainty\n"
                    "Across 21 exploratory paired intervals, five exclude zero in the "
                    "direction unfavorable to Stage 1 and none clearly favors Stage 1. "
                    "The intervals are not multiplicity-adjusted and must not be read as "
                    "causal effects of architecture."
                ),
            },
            {"id": "block-paired-table", "type": "table", "tableId": "table-paired"},
            {
                "id": "block-latent-states",
                "type": "markdown",
                "body": (
                    "## Latent States and Observation Mapping\n"
                    "The four states are statistical components, not observed economic "
                    "regimes. The implemented factor-to-asset map is fitted in "
                    "y=log(1+r) space with state-specific Gaussian residual covariance, "
                    "then converted back to simple returns with expm1."
                ),
            },
            {"id": "block-state-table", "type": "table", "tableId": "table-states"},
            {
                "id": "block-portfolio-decision",
                "type": "markdown",
                "body": (
                    "## Portfolio Decision Layer\n"
                    "Historical CVaR has lower observed validation expected shortfall "
                    "and maximum drawdown than the other tested rules. This ranking is "
                    "descriptive over 74 block outcomes; no paired sampling interval was "
                    "computed. Stage 1 portfolio rules turn over more, and the registered "
                    "Wasserstein radii do not provide a discernible advantage."
                ),
            },
            {
                "id": "block-decision-chart",
                "type": "chart",
                "chartId": "chart-decision-es",
            },
            {
                "id": "block-counterfactual-boundary",
                "type": "markdown",
                "body": (
                    "## Structural Counterfactual Boundary\n"
                    "Abduction-action-prediction exactly recovers the known semi-synthetic "
                    "oracle by construction, while structural misspecification creates "
                    "material errors. Equity tail loss uses the negative 20-step outcome "
                    "sum; volatility tail loss uses the positive sum, so higher volatility "
                    "is stress. This validates the numerical mechanism only; it does not "
                    "identify real-market policy effects."
                ),
            },
            {
                "id": "block-counterfactual-table",
                "type": "table",
                "tableId": "table-counterfactual",
            },
            {
                "id": "block-interpretation-limits",
                "type": "markdown",
                "body": (
                    "## Interpretation Limits\n"
                    "Stage 1 is MAP/empirical-Bayes rather than a full Bayesian posterior. "
                    "Stage 2 reports four validation origins and cannot support ranking. "
                    "The realized co-crash label is zero at every evaluated origin, "
                    "although generated scenario sets can assign nonzero probabilities; "
                    "event discrimination is therefore unassessable. Tail POT/GPD and "
                    "conformal utilities are implemented and tested but not yet integrated "
                    "into a standalone empirical runner."
                ),
            },
        ],
    }

    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "headline": headline,
            "generator_scores": _records(generator_scores),
            "paired_intervals": _records(paired),
            "state_profiles": _records(states),
            "stage2_pilot": _records(stage2_rows),
            "decisions": _records(decisions),
            "counterfactual": _records(counterfactual),
        },
    }

    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
        "package_info": {
            "artifact_id": "crisisforge-validation-report-v1",
            "snapshot_kind": "frozen validation research snapshot",
            "test_set_opened": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_artifact()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
