#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
NS3_DIR = Path.home() / "ns-3-dev"
OUTPUT_DIR = PROJECT_DIR / "results" / "visualizations"

RADIO_CSV = NS3_DIR / "sdt_radio_trace.csv"
NETWORK_CSV = NS3_DIR / "sdt_network_trace.csv"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clamp01(value):
    return max(0.0, min(1.0, value))


def main():
    radio = read_csv(RADIO_CSV)
    network = read_csv(NETWORK_CSV)

    # Latest radio value associated with each
    # (RNTI, timestamp).
    radio_by_key = {}

    for row in radio:
        rnti = int(row["rnti"])
        time_s = float(row["time_s"])

        radio_by_key[(rnti, round(time_s, 3))] = {
            "sinr_db": float(row["sinr_db"]),
        }

    times = sorted(
        {
            round(float(row["time_s"]), 3)
            for row in network
        }
    )

    rntis = sorted(
        {
            int(row["rnti"])
            for row in network
        }
    )

    matrix = np.zeros(
        (len(rntis), len(times)),
        dtype=float,
    )

    for i, rnti in enumerate(rntis):
        for j, time_s in enumerate(times):
            candidates = [
                value
                for (rt, rr), value
                in radio_by_key.items()
                if rr == rnti
                and abs(rt - time_s) <= 0.011
            ]

            radio_value = (
                candidates[-1]
                if candidates
                else None
            )

            network_rows = [
                row
                for row in network
                if int(row["rnti"]) == rnti
                and abs(
                    float(row["time_s"]) - time_s
                ) < 1e-6
            ]

            if not network_rows:
                matrix[i, j] = np.nan
                continue

            row = network_rows[-1]

            loss_pct = float(
                row["interval_packet_loss_pct"]
            )

            delay_ms = float(
                row["interval_mean_delay_ms"]
            )

            sinr_db = (
                radio_value["sinr_db"]
                if radio_value is not None
                else 30.0
            )

            # -------------------------------------------------
            # Visualization-only degradation score.
            #
            # 0 = strong cross-layer condition
            # 1 = severe cross-layer degradation
            #
            # This is NOT a new SDT model or ground truth.
            # It is only a compact way to visualize the
            # observed measurements.
            # -------------------------------------------------

            radio_severity = clamp01(
                (15.0 - sinr_db) / 15.0
            )

            loss_severity = clamp01(
                loss_pct / 60.0
            )

            delay_severity = clamp01(
                delay_ms / 40.0
            )

            score = (
                0.40 * radio_severity
                + 0.40 * loss_severity
                + 0.20 * delay_severity
            )

            matrix[i, j] = score

    fig, ax = plt.subplots(
        figsize=(11, 4.5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        extent=[
            0,
            len(times),
            -0.5,
            len(rntis) - 0.5,
    ])

    ax.set_yticks(
        range(len(rntis))
    )
    ax.set_yticklabels(
        [f"RNTI {r}" for r in rntis]
    )

    ax.set_xticks(
        range(len(times))
    )
    ax.set_xticklabels(
        [f"{t:.2f}" for t in times],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Simulation time (s)"
    )

    ax.set_ylabel(
        "UE identity"
    )

    ax.set_title(
        "Cross-Layer Degradation Index"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Normalized observed degradation index"
    )

    fig.tight_layout()

    output = (
        OUTPUT_DIR
        / "08_sdt_cross_layer_heatmap.png"
    )

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
