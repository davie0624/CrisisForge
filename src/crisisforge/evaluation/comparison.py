"""Exploratory paired validation comparisons with block-bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisisforge.config import load_yaml, project_root_from_module, resolve_config_path
from crisisforge.data.validation import hash_file
from crisisforge.evaluation.provenance import (
    assert_receipt_binds_output,
    git_state,
    output_hashes,
)


def paired_block_bootstrap_mean(
    differences: np.ndarray,
    *,
    replications: int,
    block_length: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, float]:
    """Circular moving-block bootstrap interval for a paired mean difference."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite vector with at least two rows")
    if replications < 100 or block_length < 1 or block_length > len(values):
        raise ValueError("invalid bootstrap replication count or block length")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one")
    rng = np.random.default_rng(random_seed)
    blocks_per_draw = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    means = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = rng.integers(0, len(values), size=blocks_per_draw)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        draw = values[indices.ravel()[: len(values)]]
        means[replication] = float(draw.mean())
    tail = (1.0 - confidence_level) / 2.0
    return {
        "mean_difference": float(values.mean()),
        "ci_lower": float(np.quantile(means, tail)),
        "ci_upper": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_standard_error": float(means.std(ddof=1)),
    }


def _load_bound_detail(
    *,
    project_root: Path,
    detail_path: Path,
    receipt_path: Path,
    output_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt = assert_receipt_binds_output(
        receipt_path=receipt_path,
        project_root=project_root,
        output_key=output_key,
        output_path=detail_path,
        allowed_statuses={"passed", "completed"},
    )
    detail = pd.read_csv(detail_path)
    if detail.empty:
        raise ValueError(f"empty experiment detail: {detail_path}")
    return detail, receipt


def _add_rowwise_joint_var_es_score(
    detail: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    """Add the same exponential FZ score used by the aggregate risk engine."""
    required = {"actual_loss", "value_at_risk", "expected_shortfall"}
    missing = required.difference(detail.columns)
    if missing:
        raise ValueError(f"detail lacks VaR/ES columns: {sorted(missing)}")
    output = detail.copy()
    loss = output["actual_loss"].to_numpy(dtype=float)
    var = output["value_at_risk"].to_numpy(dtype=float)
    es = output["expected_shortfall"].to_numpy(dtype=float)
    if (es < var).any():
        raise ValueError("Expected Shortfall forecasts must not be below VaR")
    tail_probability = 1.0 - confidence_level
    transformed_y = -loss
    transformed_q = -var
    transformed_e = -es
    indicator = transformed_y <= transformed_q
    output["joint_var_es_score"] = np.exp(transformed_e) * (
        transformed_e
        - transformed_q
        + indicator * (transformed_q - transformed_y) / tail_probability
        - 1.0
    )
    return output


def run_paired_comparison(
    project_root: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Compare Stage 1 against every Stage 0 model on identical origins."""
    config_path = resolve_config_path(
        project_root,
        config_path,
        default_relative="configs/stage3_comparison.yaml",
    )
    configuration = load_yaml(config_path)
    if configuration["experiment"]["evaluation_split"] != "validation":
        raise ValueError("paired comparison is validation-only")
    if configuration["claims_boundary"]["test_set_opened"] is not False:
        raise ValueError("paired comparison must keep the test split sealed")
    paths = configuration["paths"]
    stage0_path = project_root / paths["stage0_detail"]
    stage0_receipt_path = project_root / paths["stage0_receipt"]
    stage1_path = project_root / paths["stage1_detail"]
    stage1_receipt_path = project_root / paths["stage1_receipt"]
    output_root = project_root / paths["output"]
    output_root.mkdir(parents=True, exist_ok=True)

    stage0, stage0_receipt = _load_bound_detail(
        project_root=project_root,
        detail_path=stage0_path,
        receipt_path=stage0_receipt_path,
        output_key="detail",
    )
    stage1, stage1_receipt = _load_bound_detail(
        project_root=project_root,
        detail_path=stage1_path,
        receipt_path=stage1_receipt_path,
        output_key="detail",
    )
    if (
        stage0_receipt.get("phase0_manifest_sha256")
        != stage1_receipt.get("phase0_manifest_sha256")
    ):
        raise ValueError("Stage 0 and Stage 1 do not share a Phase 0 manifest")
    if (
        stage0_receipt.get("model_matrix_sha256")
        != stage1_receipt.get("model_matrix_sha256")
    ):
        raise ValueError("Stage 0 and Stage 1 do not share a model matrix")
    confidence = float(configuration["risk"]["confidence_level"])
    stage0 = _add_rowwise_joint_var_es_score(
        stage0,
        confidence_level=confidence,
    )
    stage1 = _add_rowwise_joint_var_es_score(
        stage1,
        confidence_level=confidence,
    )

    stage1 = stage1.sort_values("origin_date")
    if stage1["origin_date"].duplicated().any():
        raise ValueError("Stage 1 detail has duplicate origins")
    bootstrap = configuration["bootstrap"]
    rows: list[dict[str, Any]] = []
    for model_id, baseline in stage0.groupby("model_id", sort=True):
        baseline = baseline.sort_values("origin_date")
        if baseline["origin_date"].duplicated().any():
            raise ValueError(f"Stage 0 model {model_id} has duplicate origins")
        merged = stage1.merge(
            baseline,
            on="origin_date",
            how="inner",
            suffixes=("_stage1", "_stage0"),
            validate="one_to_one",
        )
        if len(merged) != len(stage1) or len(merged) != len(baseline):
            raise ValueError(f"origin mismatch for Stage 0 model {model_id}")
        for metric_number, metric in enumerate(configuration["metrics"]):
            difference = (
                merged[f"{metric}_stage1"].to_numpy(dtype=float)
                - merged[f"{metric}_stage0"].to_numpy(dtype=float)
            )
            interval = paired_block_bootstrap_mean(
                difference,
                replications=int(bootstrap["replications"]),
                block_length=int(bootstrap["origin_block_length"]),
                confidence_level=float(bootstrap["confidence_level"]),
                random_seed=(
                    int(configuration["experiment"]["random_seed"])
                    + 1009 * metric_number
                    + len(rows)
                ),
            )
            rows.append(
                {
                    "baseline_model_id": model_id,
                    "metric": metric,
                    "difference_definition": "stage1_minus_stage0",
                    "lower_is_better": True,
                    "paired_origins": int(len(difference)),
                    "origin_block_length": int(bootstrap["origin_block_length"]),
                    "bootstrap_replications": int(bootstrap["replications"]),
                    **interval,
                    "directional_interval_favors_stage1": bool(
                        interval["ci_upper"] < 0.0
                    ),
                    "multiplicity_adjusted": False,
                }
            )
    result = pd.DataFrame(rows)
    result_path = output_root / "paired_metric_intervals.csv"
    summary_path = output_root / "summary.json"
    result.to_csv(result_path, index=False)
    summary = {
        "experiment_id": configuration["experiment"]["id"],
        "comparison_rows": int(len(result)),
        "paired_origins": int(len(stage1)),
        "test_set_opened": False,
        "model_class_effect_identified": False,
        "multiplicity_adjusted": False,
        "interpretation": configuration["claims_boundary"]["note"],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    persisted = {"intervals": result_path, "summary": summary_path}
    receipt = {
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": hash_file(config_path),
        "input_hashes": {
            "stage0_detail": hash_file(stage0_path),
            "stage0_receipt": hash_file(stage0_receipt_path),
            "stage1_detail": hash_file(stage1_path),
            "stage1_receipt": hash_file(stage1_receipt_path),
        },
        "git": git_state(project_root),
        "outputs": output_hashes(project_root, persisted),
        "summary": summary,
    }
    (output_root / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired validation comparisons for CrisisForge"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else project_root_from_module()
    )
    receipt = run_paired_comparison(project_root, config_path=args.config)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
