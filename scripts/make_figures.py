"""Render the registered CrisisForge research figures from saved artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE = "#2F5D8A"
LIGHT_BLUE = "#83A6C5"
AMBER = "#C78B35"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D7D7D7"
DARK = "#242424"
RED = "#9B4A46"

MODEL_LABELS = {
    "filtered_historical_ewma": "Filtered historical",
    "moving_block_20": "Moving block",
    "var1_residual_bootstrap": "VAR residual bootstrap",
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
    "stage1_wdro_rho_0.0001": "WDRO 0.00010",
    "stage1_wdro_rho_0.00025": "WDRO 0.00025",
    "stage1_wdro_rho_0.0005": "WDRO 0.00050",
    "stage1_wdro_rho_0.001": "WDRO 0.00100",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": LIGHT_GRAY,
            "axes.linewidth": 0.8,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#E8E8E8",
            "grid.linewidth": 0.8,
        }
    )


def _subtitle(fig: plt.Figure, text: str, *, y: float = 0.965) -> None:
    fig.text(0.01, y, text, ha="left", va="top", fontsize=9, color=GRAY)


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, y=0.995, ha="left", fontsize=15, weight="bold")
    _subtitle(fig, subtitle, y=0.945)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping in {path}")
    return payload


def figure_architecture(root: Path, output_dir: Path) -> list[Path]:
    del root
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        detail: str,
        color: str,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.12,
        )
        ax.add_patch(patch)
        ax.text(
            x + 0.018,
            y + height - 0.030,
            title,
            va="top",
            ha="left",
            fontsize=10.5,
            weight="bold",
            color=DARK,
        )
        ax.text(
            x + 0.018,
            y + height - 0.072,
            detail,
            va="top",
            ha="left",
            fontsize=8.2,
            color=GRAY,
            linespacing=1.25,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = LIGHT_BLUE,
        style: str = "-|>",
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=13,
                linewidth=1.6,
                color=color,
                connectionstyle="arc3,rad=0.0",
            )
        )

    x_positions = [0.03, 0.215, 0.40, 0.585, 0.77]
    width = 0.16
    height = 0.18
    main_y = 0.51
    box(
        x_positions[0],
        main_y,
        width,
        height,
        "Phase 0 · governed panel",
        "15 asset returns · 10 context fields\nchronological split · hashed snapshots",
        BLUE,
    )
    box(
        x_positions[1],
        main_y,
        width,
        height,
        "Stage 1 · switching factors",
        "train-only PCA · sticky Gaussian HMM\nstate-weighted VAR(1) · plug-in/MAP",
        BLUE,
    )
    box(
        x_positions[2],
        main_y,
        width,
        height,
        "Stage 2 · one-shot DDPM",
        "H×q future factor tensor\nsmall tail-weighted engineering pilot",
        AMBER,
    )
    box(
        x_positions[3],
        main_y,
        width,
        height,
        "Factor → asset mapping",
        "y = α(z) + B(z)f + D(z)ε\nr = expm1(y) · correlated residuals",
        BLUE,
    )
    box(
        x_positions[4],
        main_y,
        width,
        height,
        "Risk → Stage 5 decisions",
        "VaR · ES · co-crash\nempirical CVaR · Wasserstein DRO",
        BLUE,
    )
    for left, right in zip(x_positions[:-1], x_positions[1:], strict=True):
        arrow((left + width, main_y + height / 2), (right, main_y + height / 2))

    box(
        0.03,
        0.19,
        0.30,
        0.15,
        "Stage 0 / 3 · conventional comparators",
        "seven rolling generators · 74 common origins\npaired moving-block bootstrap intervals",
        GRAY,
    )
    arrow((0.18, 0.34), (0.295, main_y), color=GRAY)

    box(
        0.365,
        0.19,
        0.25,
        0.15,
        "Stage 4 · embedded, not standalone",
        "tail importance weighting is in Stage 2;\nPOT/GPD and conformal utilities are test-only",
        AMBER,
    )
    arrow((0.49, 0.34), (0.49, main_y), color=AMBER)

    box(
        0.65,
        0.19,
        0.28,
        0.15,
        "Stage 6 · separate counterfactual track",
        "time-unrolled known SCM · paired AAP\nsemi-synthetic truth, no real-market causal claim",
        GRAY,
    )

    _header(
        fig,
        "CrisisForge implemented research architecture",
        (
            "Blue modules are complete public-core components; amber marks pilot or "
            "partial evidence; gray marks separate comparators."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, output_dir, "00_implemented_research_architecture")


def figure_data_split(root: Path, output_dir: Path) -> list[Path]:
    pipeline = _read_yaml(root / "configs/pipeline.yaml")
    receipt = json.loads((root / "artifacts/phase0/run_receipt.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(receipt["first_model_date"])
    train_end = pd.Timestamp(pipeline["splits"]["train_end"])
    validation_end = pd.Timestamp(pipeline["splits"]["validation_end"])
    end = pd.Timestamp(receipt["last_model_date"])
    test_start = validation_end + pd.Timedelta(days=1)
    spans = [
        ("Train", start, train_end, int(receipt["train_rows"]), BLUE),
        (
            "Validation",
            train_end + pd.Timedelta(days=1),
            validation_end,
            int(receipt["validation_rows"]),
            AMBER,
        ),
        ("Sealed test", test_start, end, int(receipt["test_rows"]), GRAY),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 2.5))
    for row, (label, left, right, count, color) in enumerate(spans):
        width = (right - left).days
        ax.barh(row, width, left=left, height=0.56, color=color, alpha=0.92)
        ax.text(
            left + (right - left) / 2,
            row,
            f"{label}\n{count:,} rows",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )
    ax.set_yticks([])
    ax.set_xlabel("Calendar date")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x")
    ax.set_title("Chronological research split", loc="left", pad=28, weight="bold")
    _subtitle(
        fig,
        "Registered boundaries and row counts only; no post-2019 test results are opened.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, output_dir, "01_chronological_research_split")


def figure_model_scores(root: Path, output_dir: Path) -> list[Path]:
    stage0 = pd.read_csv(root / "artifacts/stage0_baselines/summary.csv")
    stage1 = json.loads(
        (root / "artifacts/stage1_switching_factor/summary.json").read_text(encoding="utf-8")
    )
    rows = stage0[["model_id", "mean_energy_score", "mean_variogram_score"]].copy()
    rows.loc[len(rows)] = [
        stage1["model_id"],
        stage1["mean_energy_score"],
        stage1["mean_variogram_score"],
    ]
    rows["label"] = rows["model_id"].map(MODEL_LABELS)
    rows = rows.sort_values("mean_energy_score", ascending=True).reset_index(drop=True)
    colors = [BLUE if value.startswith("Stage 1") else GRAY for value in rows["label"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for axis, metric, title in [
        (axes[0], "mean_energy_score", "Mean energy score"),
        (axes[1], "mean_variogram_score", "Mean variogram score"),
    ]:
        axis.scatter(rows[metric], np.arange(len(rows)), s=62, c=colors, zorder=3)
        axis.set_yticks(np.arange(len(rows)), rows["label"])
        axis.invert_yaxis()
        axis.set_xlabel("Score (lower is better)")
        axis.set_title(title, loc="left")
        axis.grid(axis="x")
    _header(
        fig,
        "Validation distribution scores",
        (
            "74 non-overlapping 20-observation origins; Stage 1 is blue and "
            "conventional baselines are gray."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, output_dir, "02_validation_distribution_scores")


def figure_paired_intervals(root: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(root / "artifacts/stage3_comparison/paired_metric_intervals.csv")
    metrics = [
        ("energy_score", "Energy score difference"),
        ("variogram_score", "Variogram score difference"),
        ("joint_var_es_score", "Joint VaR–ES score difference"),
    ]
    baseline_order = (
        frame.loc[frame["metric"] == "energy_score"]
        .sort_values("mean_difference")["baseline_model_id"]
        .tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0), sharey=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        subset = (
            frame.loc[frame["metric"] == metric]
            .set_index("baseline_model_id")
            .loc[baseline_order]
            .reset_index()
        )
        y = np.arange(len(subset))
        means = subset["mean_difference"].to_numpy()
        lower = subset["ci_lower"].to_numpy()
        upper = subset["ci_upper"].to_numpy()
        axis.errorbar(
            means,
            y,
            xerr=np.vstack((means - lower, upper - means)),
            fmt="o",
            color=BLUE,
            ecolor=LIGHT_BLUE,
            elinewidth=2,
            capsize=3,
        )
        axis.axvline(0.0, color=DARK, linewidth=1, linestyle="--")
        axis.locator_params(axis="x", nbins=5)
        axis.set_yticks(
            y,
            [MODEL_LABELS.get(value, value) for value in subset["baseline_model_id"]],
        )
        axis.invert_yaxis()
        axis.set_title(title, loc="left")
        axis.set_xlabel("Stage 1 minus Stage 0\n(negative favors Stage 1)")
        axis.grid(axis="x")
    _header(
        fig,
        "Paired validation score differences",
        (
            "Circular moving-block bootstrap 95% intervals, 10,000 draws; "
            "exploratory and not multiplicity-adjusted."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, output_dir, "03_paired_validation_intervals")


def figure_regime_profiles(root: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(root / "artifacts/stage1_switching_factor/state_profiles.csv")
    labels = [f"State {int(value)}" for value in frame["state"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].barh(labels, frame["occupancy"], color=BLUE)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Soft-probability occupancy")
    axes[0].set_title("Training-state occupancy", loc="left")
    axes[0].grid(axis="x")
    sizes = 900 * frame["occupancy"].to_numpy()
    axes[1].scatter(
        frame["weighted_market_volatility_per_interval"],
        frame["weighted_market_mean_per_interval"],
        s=sizes,
        color=AMBER,
        alpha=0.85,
        edgecolor="white",
        linewidth=1,
    )
    for _, row in frame.iterrows():
        axes[1].annotate(
            f"State {int(row['state'])}",
            (
                row["weighted_market_volatility_per_interval"],
                row["weighted_market_mean_per_interval"],
            ),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1].axhline(0.0, color=LIGHT_GRAY, linewidth=1)
    axes[1].set_xlabel("Weighted market volatility per interval")
    axes[1].set_ylabel("Weighted market mean per interval")
    axes[1].set_title("Statistical state profiles", loc="left")
    axes[1].grid()
    _header(
        fig,
        "Latent regime profiles in the training sample",
        "States are unlabeled statistical components; point size is state occupancy.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, output_dir, "04_latent_regime_profiles")


def figure_decision_outcomes(root: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(root / "artifacts/stage5_decisions/summary.csv")
    frame["label"] = frame["strategy_id"].map(STRATEGY_LABELS)
    palette = [
        AMBER if value == "Historical CVaR" else BLUE if "Stage 1" in value else GRAY
        for value in frame["label"]
    ]
    sizes = 2500 * frame["maximum_drawdown"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.scatter(
        frame["realized_expected_shortfall"],
        frame["cumulative_net_return"],
        s=sizes,
        c=palette,
        alpha=0.82,
        edgecolor="white",
        linewidth=1,
    )
    labeled = frame.loc[
        frame["strategy_id"].isin(
            ["equal_weight", "historical_empirical_cvar", "stage1_empirical_cvar"]
        )
    ]
    for _, row in labeled.iterrows():
        label = (
            "Stage 1 CVaR / WDRO grid"
            if row["strategy_id"] == "stage1_empirical_cvar"
            else row["label"]
        )
        ax.annotate(
            label,
            (row["realized_expected_shortfall"], row["cumulative_net_return"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel("Realized 95% Expected Shortfall (lower is better)")
    ax.set_ylabel("Cumulative net return")
    ax.grid()
    _header(
        fig,
        "Validation portfolio risk and return outcomes",
        (
            "74 non-overlapping 20-observation blocks; bubble area is maximum "
            "drawdown; transaction costs included."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, output_dir, "05_portfolio_decision_outcomes")


def figure_counterfactual_paths(root: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(root / "artifacts/stage6_counterfactual/mean_effect_paths.csv")
    frame = frame.loc[frame["effect_type"] == "total"].copy()
    styles = {
        "known_scm_ground_truth": (DARK, "-"),
        "reduced_yield_transmission": (BLUE, "--"),
        "no_lagged_equity_feedback": (AMBER, ":"),
    }
    labels = {
        "known_scm_ground_truth": "Known SCM truth",
        "reduced_yield_transmission": "Reduced yield transmission",
        "no_lagged_equity_feedback": "No lagged equity feedback",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, outcome in zip(axes, ["equity", "volatility"], strict=True):
        subset = frame.loc[frame["outcome"] == outcome]
        for model_id, group in subset.groupby("model_id", sort=False):
            color, linestyle = styles[model_id]
            axis.plot(
                group["time"],
                group["mean_effect"],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=labels[model_id],
            )
        axis.axhline(0.0, color=LIGHT_GRAY, linewidth=1)
        axis.set_xlabel("Horizon step")
        axis.set_ylabel("Mean paired effect (standardized units)")
        axis.set_title(f"{outcome.title()} effect path", loc="left")
        axis.grid()
    axes[1].legend(frameon=False, loc="best")
    _header(
        fig,
        "Structural misspecification sensitivity",
        (
            "5,000 paired semi-synthetic paths under a five-step policy intervention; "
            "not a real-market causal estimate."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _save(fig, output_dir, "06_counterfactual_misspecification")


def figure_diffusion_pilot(root: Path, output_dir: Path) -> list[Path]:
    history = pd.read_csv(root / "artifacts/stage2_public_core_pilot/training_history.csv")
    summary = json.loads(
        (root / "artifacts/stage2_public_core_pilot/summary.json").read_text(encoding="utf-8")
    )
    variants = pd.DataFrame(summary["variants"]).set_index("model_variant")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for stage, group in history.groupby("stage", sort=False):
        color = BLUE if stage == "base" else AMBER
        axes[0, 0].plot(
            group["epoch"],
            group["training_loss"],
            marker="o",
            color=color,
            label=f"{stage} training",
        )
        axes[0, 0].plot(
            group["epoch"],
            group["validation_denoising_loss"],
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"{stage} tuning",
        )
    axes[0, 0].set_title("Denoising loss", loc="left")
    axes[0, 0].set_xlabel("Epoch within stage")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid()
    axes[0, 0].legend(frameon=False, fontsize=8)

    for axis, metric, title in [
        (axes[0, 1], "mean_energy_score", "Energy score"),
        (axes[1, 0], "mean_variogram_score", "Variogram score"),
        (axes[1, 1], "joint_var_es_score", "Joint VaR–ES score"),
    ]:
        values = variants.loc[["base", "tail_weighted"], metric]
        axis.bar(
            ["Base", "Tail-weighted"],
            values,
            color=[BLUE, AMBER],
            width=0.58,
        )
        axis.set_title(title, loc="left")
        axis.set_ylabel("Score (lower is better)")
        axis.grid(axis="y")
    _header(
        fig,
        "One-shot diffusion engineering pilot",
        (
            "658 train windows, 37 tuning windows, 4 late-validation reporting "
            "origins; no superiority claim."
        ),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, output_dir, "07_diffusion_engineering_pilot")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CrisisForge research figures")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else root / "reports/figures"
    )
    _style()
    outputs: list[Path] = []
    for builder in (
        figure_architecture,
        figure_data_split,
        figure_model_scores,
        figure_paired_intervals,
        figure_regime_profiles,
        figure_decision_outcomes,
        figure_counterfactual_paths,
        figure_diffusion_pilot,
    ):
        outputs.extend(builder(root, output_dir))
    manifest = {
        "figure_count": len(outputs) // 2,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
