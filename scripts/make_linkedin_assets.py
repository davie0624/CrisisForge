"""Create the saved CrisisForge LinkedIn visual in PNG and SVG formats."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

NAVY = "#17324D"
BLUE = "#2F5D8A"
PALE_BLUE = "#E8F0F7"
AMBER = "#C78B35"
PALE_AMBER = "#F6ECDD"
INK = "#202428"
GRAY = "#6F7478"
WHITE = "#FFFFFF"


def _badge(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    value: str,
    label: str,
    face: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        0.18,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.024, y + 0.125, value, fontsize=20, weight="bold", color=NAVY)
    ax.text(
        x + 0.024,
        y + 0.048,
        label,
        fontsize=9.5,
        color=GRAY,
        va="center",
        linespacing=1.2,
    )


def render(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6.27))
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            0.022,
            1.0,
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=AMBER,
        )
    )
    ax.text(
        0.065,
        0.90,
        "CRISISFORGE · GENERATIVE FINANCIAL RISK RESEARCH",
        color=BLUE,
        fontsize=11,
        weight="bold",
        va="center",
    )
    ax.text(
        0.065,
        0.77,
        "Can a complex generative\nrisk system beat baselines?",
        color=INK,
        fontsize=30,
        weight="bold",
        va="center",
        linespacing=1.05,
    )
    ax.text(
        0.065,
        0.575,
        "Not in the registered validation.",
        color=AMBER,
        fontsize=20,
        weight="bold",
        va="center",
    )
    ax.text(
        0.065,
        0.495,
        "That negative result became the most valuable finding.",
        color=GRAY,
        fontsize=12.5,
        va="center",
    )

    _badge(
        ax,
        x=0.065,
        y=0.215,
        width=0.25,
        value="74",
        label="non-overlapping\nvalidation origins",
        face=PALE_BLUE,
    )
    _badge(
        ax,
        x=0.335,
        y=0.215,
        width=0.25,
        value="5",
        label="paired intervals favoring\nconventional baselines",
        face=PALE_AMBER,
    )
    _badge(
        ax,
        x=0.605,
        y=0.215,
        width=0.30,
        value="HELD OUT",
        label="post-2019 rows excluded\nfrom model evaluation",
        face=PALE_BLUE,
    )

    ax.text(
        0.065,
        0.105,
        (
            "Regime factors  ·  one-shot diffusion  ·  VaR / ES  ·  "
            "CVaR / WDRO  ·  structural counterfactuals"
        ),
        color=NAVY,
        fontsize=10.5,
        weight="bold",
        va="center",
    )
    ax.text(
        0.935,
        0.055,
        "CrisisForge v0.3",
        color=GRAY,
        fontsize=9,
        ha="right",
    )

    png = output_dir / "crisisforge_linkedin_banner.png"
    svg = output_dir / "crisisforge_linkedin_banner.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(svg, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return [png, svg]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "linkedin",
    )
    args = parser.parse_args()
    for path in render(args.output_dir.resolve()):
        print(path)


if __name__ == "__main__":
    main()
