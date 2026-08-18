#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
NS3_DIR = Path.home() / "ns-3-dev"

RADIO_CSV = NS3_DIR / "sdt_radio_trace.csv"
NETWORK_CSV = NS3_DIR / "sdt_network_trace.csv"

OUTPUT_DIR = PROJECT_DIR / "results" / "visualizations"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def unique_sorted(values):
    return sorted(set(values))


def setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    radio = load_csv(RADIO_CSV)
    network = load_csv(NETWORK_CSV)

    if not radio:
        raise RuntimeError("Radio CSV contains no rows.")

    if not network:
        raise RuntimeError("Network CSV contains no rows.")

    return radio, network


def plot_sinr(radio):
    by_rnti = defaultdict(list)

    for row in radio:
        by_rnti[as_int(row, "rnti")].append(
            (
                as_float(row, "time_s"),
                as_float(row, "sinr_db"),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    for rnti in sorted(by_rnti):
        points = sorted(by_rnti[rnti])
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        ax.plot(
            x,
            y,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"RNTI {rnti}",
        )

    ax.set_title("NR Radio Quality: SINR vs Time")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("SINR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "02_sinr_vs_time.png",
        dpi=200,
    )
    plt.close(fig)


def plot_packet_loss(network):
    by_rnti = defaultdict(list)

    for row in network:
        rnti = as_int(row, "rnti")

        by_rnti[rnti].append(
            (
                as_float(row, "time_s"),
                as_float(row, "interval_packet_loss_pct"),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    for rnti in sorted(by_rnti):
        points = sorted(by_rnti[rnti])
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        ax.plot(
            x,
            y,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"RNTI {rnti}",
        )

    ax.set_title("Network Reliability: Packet Loss vs Time")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Packet loss (%)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "03_packet_loss_vs_time.png",
        dpi=200,
    )
    plt.close(fig)


def plot_throughput(network):
    by_rnti = defaultdict(list)

    for row in network:
        rnti = as_int(row, "rnti")

        by_rnti[rnti].append(
            (
                as_float(row, "time_s"),
                as_float(row, "interval_throughput_mbps"),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    for rnti in sorted(by_rnti):
        points = sorted(by_rnti[rnti])
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        ax.plot(
            x,
            y,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"RNTI {rnti}",
        )

    ax.set_title("Network Performance: Throughput vs Time")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "04_throughput_vs_time.png",
        dpi=200,
    )
    plt.close(fig)


def plot_delay(network):
    by_rnti = defaultdict(list)

    for row in network:
        rnti = as_int(row, "rnti")

        by_rnti[rnti].append(
            (
                as_float(row, "time_s"),
                as_float(row, "interval_mean_delay_ms"),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    for rnti in sorted(by_rnti):
        points = sorted(by_rnti[rnti])
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        ax.plot(
            x,
            y,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"RNTI {rnti}",
        )

    ax.set_title("Network Latency: Mean Delay vs Time")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Mean delay (ms)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "05_delay_vs_time.png",
        dpi=200,
    )
    plt.close(fig)


def plot_jitter(network):
    by_rnti = defaultdict(list)

    for row in network:
        rnti = as_int(row, "rnti")

        by_rnti[rnti].append(
            (
                as_float(row, "time_s"),
                as_float(row, "interval_mean_jitter_ms"),
            )
        )

    fig, ax = plt.subplots(figsize=(10, 5))

    for rnti in sorted(by_rnti):
        points = sorted(by_rnti[rnti])
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        ax.plot(
            x,
            y,
            marker=".",
            markersize=3,
            linewidth=1.5,
            label=f"RNTI {rnti}",
        )

    ax.set_title("Network Stability: Jitter vs Time")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Mean jitter (ms)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "06_jitter_vs_time.png",
        dpi=200,
    )
    plt.close(fig)


def latest_radio_state(radio):
    latest = {}

    for row in radio:
        rnti = as_int(row, "rnti")
        t = as_float(row, "time_s")

        if (
            rnti not in latest
            or t > latest[rnti]["time_s"]
        ):
            latest[rnti] = {
                "time_s": t,
                "sinr_db": as_float(row, "sinr_db"),
                "ue_x": as_float(row, "ue_x"),
                "ue_y": as_float(row, "ue_y"),
                "ue_z": as_float(row, "ue_z"),
            }

    return latest


def latest_network_state(network):
    latest = {}

    for row in network:
        rnti = as_int(row, "rnti")
        t = as_float(row, "time_s")

        if (
            rnti not in latest
            or t > latest[rnti]["time_s"]
        ):
            latest[rnti] = {
                "time_s": t,
                "ue_ip": row["ue_ip"],
                "loss_pct": as_float(
                    row,
                    "interval_packet_loss_pct",
                ),
                "throughput_mbps": as_float(
                    row,
                    "interval_throughput_mbps",
                ),
                "delay_ms": as_float(
                    row,
                    "interval_mean_delay_ms",
                ),
                "jitter_ms": as_float(
                    row,
                    "interval_mean_jitter_ms",
                ),
            }

    return latest


def plot_cross_layer_state(radio, network):
    radio_latest = latest_radio_state(radio)
    network_latest = latest_network_state(network)

    rntis = sorted(
        set(radio_latest)
        & set(network_latest)
    )

    if not rntis:
        raise RuntimeError(
            "No common RNTIs found between radio and network data."
        )

    fig, ax = plt.subplots(
        figsize=(11, max(4.5, len(rntis) * 1.8))
    )

    ax.axis("off")

    columns = [
        "RNTI",
        "UE IP",
        "SINR (dB)",
        "Loss (%)",
        "Throughput (Mbps)",
        "Delay (ms)",
        "Jitter (ms)",
    ]

    rows = []

    for rnti in rntis:
        r = radio_latest[rnti]
        n = network_latest[rnti]

        rows.append(
            [
                str(rnti),
                n["ue_ip"],
                f'{r["sinr_db"]:.2f}',
                f'{n["loss_pct"]:.2f}',
                f'{n["throughput_mbps"]:.2f}',
                f'{n["delay_ms"]:.2f}',
                f'{n["jitter_ms"]:.3f}',
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax.set_title(
        "Latest Cross-Layer UE State",
        pad=20,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "07_cross_layer_state.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temporal_state(network, radio):
    radio_latest = defaultdict(dict)

    for row in radio:
        rnti = as_int(row, "rnti")
        t = as_float(row, "time_s")

        radio_latest[(t, rnti)] = as_float(
            row,
            "sinr_db",
        )

    network_times = unique_sorted(
        as_float(row, "time_s")
        for row in network
    )

    rntis = unique_sorted(
        as_int(row, "rnti")
        for row in network
    )

    fig, ax = plt.subplots(
        figsize=(11, max(4, len(rntis) * 1.7))
    )

    ax.axis("off")

    text_lines = [
        "TEMPORAL CROSS-LAYER STATE",
        "",
    ]

    for t in network_times:
        text_lines.append(
            f"t = {t:.3f} s"
        )

        for rnti in rntis:
            matching = [
                row
                for row in network
                if (
                    as_int(row, "rnti") == rnti
                    and math.isclose(
                        as_float(row, "time_s"),
                        t,
                        abs_tol=1e-9,
                    )
                )
            ]

            if not matching:
                continue

            row = matching[-1]

            loss = as_float(
                row,
                "interval_packet_loss_pct",
            )
            delay = as_float(
                row,
                "interval_mean_delay_ms",
            )
            throughput = as_float(
                row,
                "interval_throughput_mbps",
            )

            sinr_candidates = [
                value
                for (rt, rr), value
                in radio_latest.items()
                if rr == rnti
                and abs(rt - t) <= 0.011
            ]

            sinr = (
                sinr_candidates[-1]
                if sinr_candidates
                else float("nan")
            )

            text_lines.append(
                "  "
                f"RNTI {rnti}: "
                f"SINR={sinr:.2f} dB, "
                f"loss={loss:.2f}%, "
                f"throughput={throughput:.2f} Mbps, "
                f"delay={delay:.2f} ms"
            )

        text_lines.append("")

    ax.text(
        0.01,
        0.99,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    fig.savefig(
        OUTPUT_DIR / "08_sdt_temporal_timeline.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary(radio, network):
    radio_latest = latest_radio_state(radio)
    network_latest = latest_network_state(network)

    summary_path = OUTPUT_DIR / "summary.txt"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SDT-LMM RESEARCH VISUALIZATION SUMMARY\n"
        )
        f.write("=" * 50 + "\n\n")

        for rnti in sorted(
            set(radio_latest)
            & set(network_latest)
        ):
            r = radio_latest[rnti]
            n = network_latest[rnti]

            f.write(
                f"RNTI {rnti}\n"
            )
            f.write(
                f"  UE IP: {n['ue_ip']}\n"
            )
            f.write(
                f"  SINR: {r['sinr_db']:.3f} dB\n"
            )
            f.write(
                f"  Packet loss: "
                f"{n['loss_pct']:.3f}%\n"
            )
            f.write(
                f"  Throughput: "
                f"{n['throughput_mbps']:.3f} Mbps\n"
            )
            f.write(
                f"  Delay: "
                f"{n['delay_ms']:.3f} ms\n"
            )
            f.write(
                f"  Jitter: "
                f"{n['jitter_ms']:.3f} ms\n"
            )
            f.write(
                f"  Position: "
                f"({r['ue_x']:.3f}, "
                f"{r['ue_y']:.3f}, "
                f"{r['ue_z']:.3f})\n\n"
            )


def main():
    print("==============================================")
    print("SDT RESEARCH VISUALIZATION")
    print("==============================================")

    radio, network = setup()

    print(f"Radio rows:   {len(radio)}")
    print(f"Network rows: {len(network)}")
    print(f"Output:       {OUTPUT_DIR}")

    plot_sinr(radio)
    plot_packet_loss(network)
    plot_throughput(network)
    plot_delay(network)
    plot_jitter(network)
    plot_cross_layer_state(radio, network)
    plot_temporal_state(network, radio)
    write_summary(radio, network)

    print()
    print("Generated:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(" ", path.name)

    print()
    print("==============================================")
    print("VISUALIZATION COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()
