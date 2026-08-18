#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "results" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, title, subtitle):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5,
        fill=False,
    )
    ax.add_patch(patch)

    ax.text(
        x,
        y + 0.15,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        x,
        y - 0.15,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.5,
    )


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.4,
        )
    )

    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.12,
            label,
            ha="center",
            va="center",
            fontsize=8,
        )


def main():
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ---------------------------------------------------------
    # RAN
    # ---------------------------------------------------------

    ax.text(
        6,
        9.45,
        "5G RADIO ACCESS NETWORK",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    box(
        ax,
        6,
        7.9,
        3.2,
        1.0,
        "gNB-1",
        "5G-LENA + Sionna RT | Cell ID 1",
    )

    box(
        ax,
        3.0,
        6.0,
        2.5,
        1.0,
        "UE-1",
        "RNTI=1 | HEALTHY",
    )

    box(
        ax,
        9.0,
        6.0,
        2.5,
        1.0,
        "UE-2",
        "RNTI=2 | DEGRADED",
    )

    arrow(ax, 5.1, 7.35, 3.9, 6.55, "NR Uu")
    arrow(ax, 6.9, 7.35, 8.1, 6.55, "NR Uu")

    # ---------------------------------------------------------
    # Core
    # ---------------------------------------------------------

    ax.text(
        6,
        4.85,
        "5G CORE / EPC",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    box(
        ax,
        3.6,
        3.55,
        2.2,
        0.85,
        "SGW",
        "Serving Gateway",
    )

    box(
        ax,
        6.0,
        3.55,
        2.2,
        0.85,
        "PGW",
        "Packet Gateway",
    )

    box(
        ax,
        9.2,
        6.0,
        2.5,
        1.0,
        "MME",
        "Mobility Management",
    )

    arrow(ax, 6, 7.4, 3.6, 4.0, "S1-U")
    arrow(ax, 3.6, 3.55, 6.0, 3.55, "S5")
    arrow(ax, 7.1, 7.4, 8.2, 6.55, "S1-MME")
    arrow(ax, 9.0, 5.5, 4.7, 4.0, "S11")

    # ---------------------------------------------------------
    # External network
    # ---------------------------------------------------------

    box(
        ax,
        6.0,
        1.55,
        3.0,
        0.95,
        "Remote Host",
        "External data network",
    )

    arrow(ax, 6.0, 3.1, 6.0, 2.05, "SGi")

    # ---------------------------------------------------------
    # Research note
    # ---------------------------------------------------------

    ax.text(
        6,
        0.55,
        "Simulation architecture represented independently from "
        "the quantitative measurement plots.",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()

    output = OUTPUT_DIR / "01_research_topology.png"

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
