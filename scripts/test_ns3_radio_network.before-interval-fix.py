import csv
from pathlib import Path

from sdt_llm.pipeline import (
    SDTLLMPipeline,
    PipelineConfig,
)


RADIO_CSV = (
    Path.home()
    / "ns-3-dev"
    / "sdt_radio_trace.csv"
)

NETWORK_CSV = (
    Path.home()
    / "ns-3-dev"
    / "sdt_network_trace.csv"
)


def classify_sinr(sinr_db):
    if sinr_db >= 25.0:
        return "strong_radio_link"
    if sinr_db >= 15.0:
        return "good_radio_link"
    if sinr_db >= 5.0:
        return "degraded_radio_link"
    return "poor_radio_link"


def main():

    if not RADIO_CSV.exists():
        raise FileNotFoundError(
            f"Radio CSV not found: {RADIO_CSV}"
        )

    if not NETWORK_CSV.exists():
        raise FileNotFoundError(
            f"Network CSV not found: {NETWORK_CSV}"
        )

    pipeline = SDTLLMPipeline(
        PipelineConfig(
            llm_backend="mock",
            vision_backend="mock",
        )
    )

    # ============================================================
    # 1. Read radio measurements
    # ============================================================

    radio_measurements = []

    with RADIO_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            radio_measurements.append(row)

    if not radio_measurements:
        raise RuntimeError(
            "Radio CSV contains no measurements."
        )

    # ============================================================
    # 2. Use the first radio timestamp as the simulation snapshot
    # ============================================================

    snapshot_time = float(
        radio_measurements[0]["time_s"]
    )

    # ============================================================
    # 3. Ingest radio measurements
    # ============================================================

    radio_count = 0

    for row in radio_measurements:

        pipeline.ingest_nr_sinr(
            timestamp=float(row["time_s"]),
            cell_id=int(row["cell_id"]),
            rnti=int(row["rnti"]),
            bwp_id=int(row["bwp_id"]),
            sinr_linear=float(
                row["sinr_linear"]
            ),
            sinr_db=float(
                row["sinr_db"]
            ),
            ue_x=float(row["ue_x"]),
            ue_y=float(row["ue_y"]),
            ue_z=float(row["ue_z"]),
            gnb_x=float(row["gnb_x"]),
            gnb_y=float(row["gnb_y"]),
            gnb_z=float(row["gnb_z"]),
        )

        radio_count += 1

    # ============================================================
    # 4. Read network measurements
    # ============================================================

    network_measurements = []

    with NETWORK_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            network_measurements.append(row)

    if not network_measurements:
        raise RuntimeError(
            "Network CSV contains no measurements."
        )

    # ============================================================
    # 5. Ingest network snapshot
    # ============================================================

    pipeline.ingest_network(
        timestamp=snapshot_time,
        measurements=network_measurements,
    )

    # ============================================================
    # 6. Display representative radio state
    # ============================================================

    print()
    print("==============================================")
    print("RADIO + NETWORK SDT INGESTION")
    print("==============================================")

    print()
    print("Radio measurements:", radio_count)
    print("Network flows:", len(network_measurements))
    print(
        "Network snapshot time:",
        snapshot_time,
        "s",
    )

    print()
    print("Representative radio measurements:")

    for index, row in enumerate(
        radio_measurements[:5],
        start=1,
    ):

        sinr_db = float(
            row["sinr_db"]
        )

        print(
            f"[{index}] "
            f"UE={row['rnti']} "
            f"SINR={sinr_db:.3f} dB "
            f"label={classify_sinr(sinr_db)} "
            f"UE=("
            f"{float(row['ue_x']):.3f}, "
            f"{float(row['ue_y']):.3f}, "
            f"{float(row['ue_z']):.3f})"
        )

    # ============================================================
    # 7. Display network state
    # ============================================================

    print()
    print("Network state:")

    for row in network_measurements:

        tx = int(row["tx_packets"])
        rx = int(row["rx_packets"])

        if tx > 0:
            observed_loss = (
                (tx - rx)
                / tx
                * 100.0
            )
        else:
            observed_loss = 0.0

        print(
            f"Flow {row['flow_id']} "
            f"{row['protocol']} "
            f"throughput="
            f"{float(row['throughput_mbps']):.3f} Mbps "
            f"delay="
            f"{float(row['mean_delay_ms']):.3f} ms "
            f"jitter="
            f"{float(row['mean_jitter_ms']):.3f} ms "
            f"observed_loss="
            f"{observed_loss:.2f}%"
        )

    # ============================================================
    # 8. Inspect SDT context
    # ============================================================

    context = pipeline.twin.query_context(
        k_history=4
    )

    print()
    print("==============================================")
    print("SDT CONTEXT")
    print("==============================================")

    print(
        "Fused tokens in context:",
        len(context),
    )

    for index, token in enumerate(
        context[-10:],
        start=1,
    ):
        print()
        print(
            f"[TOKEN {index}]"
        )
        print(
            token.describe()
        )

    print()
    print("==============================================")
    print("COMBINED INGESTION COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()
