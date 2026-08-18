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
        linewidth=1.6,
        fill=False,
    )
    ax.add_patch(patch)

    ax.text(
        x,
        y + 0.14,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        x,
        y - 0.16,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.5,
    )


def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    label=None,
    label_dx=0.0,
    label_dy=0.0,
    bidirectional=False,
):
    style = "<->" if bidirectional else "->"

    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=1.5,
        )
    )

    if label:
        ax.text(
            (x1 + x2) / 2 + label_dx,
            (y1 + y2) / 2 + label_dy,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                pad=1.5,
            ),
        )


def main():
    fig, ax = plt.subplots(figsize=(13, 9))

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # =========================================================
    # SECTION HEADERS
    # =========================================================

    ax.text(
        6.5,
        10.45,
        "5G RADIO ACCESS NETWORK",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
    )

    ax.text(
        6.5,
        5.55,
        "5G CORE / EPC",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    # =========================================================
    # RADIO ACCESS NETWORK
    # =========================================================

    box(
        ax,
        6.5,
        8.7,
        3.4,
        1.0,
        "gNB-1",
        "5G-LENA + Sionna RT | Cell ID 1",
    )

    box(
        ax,
        3.2,
        6.9,
        2.7,
        1.0,
        "UE-1",
        "RNTI=1 | HEALTHY",
    )

    box(
        ax,
        9.8,
        6.9,
        2.7,
        1.0,
        "UE-2",
        "RNTI=2 | DEGRADED",
    )

    # Conceptual radio associations.
    arrow(
        ax,
        5.25,
        8.25,
        4.45,
        7.45,
        "NR Uu",
        label_dy=0.08,
    )

    arrow(
        ax,
        7.75,
        8.25,
        8.55,
        7.45,
        "NR Uu",
        label_dy=0.08,
    )

    # =========================================================
    # CORE NETWORK
    # =========================================================

    box(
        ax,
        3.7,
        4.15,
        2.4,
        0.9,
        "SGW",
        "Serving Gateway",
    )

    box(
        ax,
        6.5,
        4.15,
        2.4,
        0.9,
        "PGW",
        "Packet Gateway",
    )

    box(
        ax,
        9.3,
        4.15,
        2.4,
        0.9,
        "MME",
        "Mobility Management",
    )

    # =========================================================
    # ACTUAL RECORDED CORE LINKS
    # =========================================================

    # gNB -> SGW : S1-U
    arrow(
        ax,
        5.65,
        8.2,
        4.0,
        4.7,
        "S1-U",
        label_dx=-0.18,
        label_dy=0.10,
    )

    # SGW <-> PGW : S5
    arrow(
        ax,
        4.95,
        4.15,
        5.25,
        4.15,
        "S5",
        label_dy=0.18,
        bidirectional=True,
    )

    # SGW <-> MME : S11
    arrow(
        ax,
        4.9,
        4.35,
        8.1,
        4.35,
        "S11",
        label_dy=0.22,
        bidirectional=True,
    )

    # PGW -> Remote Host : SGi
    box(
        ax,
        6.5,
        1.65,
        3.2,
        0.95,
        "Remote Host",
        "External data network",
    )

    arrow(
        ax,
        6.5,
        3.65,
        6.5,
        2.15,
        "SGi",
        label_dx=0.28,
    )

    # =========================================================
    # EXPLANATORY NOTE
    # =========================================================

    ax.text(
        6.5,
        0.55,
        "NR Uu shows the UE-gNB radio association; "
        "S1-U, S5, S11 and SGi correspond to the recorded core-network links.",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()

    output = OUTPUT_DIR / "01_research_topology.png"

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
