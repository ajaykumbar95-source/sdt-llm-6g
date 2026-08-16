import csv
from bisect import bisect_right
from collections import defaultdict
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

TIME_TOLERANCE = 0.010


def classify_sinr(sinr_db: float) -> str:
    if sinr_db >= 25.0:
        return "strong_radio_link"

    if sinr_db >= 15.0:
        return "good_radio_link"

    if sinr_db >= 5.0:
        return "degraded_radio_link"

    return "poor_radio_link"


def read_radio_rows():
    rows = []

    with RADIO_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "time_s",
            "cell_id",
            "rnti",
            "bwp_id",
            "sinr_linear",
            "sinr_db",
            "ue_x",
            "ue_y",
            "ue_z",
            "gnb_x",
            "gnb_y",
            "gnb_z",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "Radio CSV missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            rows.append(
                {
                    "time_s": float(row["time_s"]),
                    "cell_id": int(row["cell_id"]),
                    "rnti": int(row["rnti"]),
                    "bwp_id": int(row["bwp_id"]),
                    "sinr_linear": float(
                        row["sinr_linear"]
                    ),
                    "sinr_db": float(
                        row["sinr_db"]
                    ),
                    "ue_x": float(row["ue_x"]),
                    "ue_y": float(row["ue_y"]),
                    "ue_z": float(row["ue_z"]),
                    "gnb_x": float(row["gnb_x"]),
                    "gnb_y": float(row["gnb_y"]),
                    "gnb_z": float(row["gnb_z"]),
                }
            )

    return rows


def read_network_rows():
    rows = []

    with NETWORK_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "time_s",
            "flow_id",
            "rnti",
            "ue_ip",
            "ue_x",
            "ue_y",
            "ue_z",
            "protocol",
            "source",
            "destination",
            "interval_tx_packets",
            "interval_rx_packets",
            "interval_lost_packets",
            "interval_packet_loss_pct",
            "interval_throughput_mbps",
            "interval_mean_delay_ms",
            "interval_mean_jitter_ms",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise RuntimeError(
                "Network CSV missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            rows.append(
                {
                    "time_s": float(
                        row["time_s"]
                    ),

                    "flow_id": int(
                        row["flow_id"]
                    ),

                    "rnti": int(
                        row["rnti"]
                    ),

                    "ue_ip": str(
                        row["ue_ip"]
                    ),

                    "ue_x": float(
                        row["ue_x"]
                    ),

                    "ue_y": float(
                        row["ue_y"]
                    ),

                    "ue_z": float(
                        row["ue_z"]
                    ),

                    "protocol": str(
                        row["protocol"]
                    ),

                    "source": str(
                        row["source"]
                    ),

                    "destination": str(
                        row["destination"]
                    ),

                    "interval_tx_packets": int(
                        row["interval_tx_packets"]
                    ),

                    "interval_rx_packets": int(
                        row["interval_rx_packets"]
                    ),

                    "interval_lost_packets": int(
                        row["interval_lost_packets"]
                    ),

                    "interval_packet_loss_pct": float(
                        row["interval_packet_loss_pct"]
                    ),

                    "interval_throughput_mbps": float(
                        row["interval_throughput_mbps"]
                    ),

                    "interval_mean_delay_ms": float(
                        row["interval_mean_delay_ms"]
                    ),

                    "interval_mean_jitter_ms": float(
                        row["interval_mean_jitter_ms"]
                    ),
                }
            )

    return rows

def latest_radio_per_ue(
    radio_rows,
    target_time,
):
    """
    For a given 10 ms network timestamp, select the latest
    radio measurement available for each UE within the
    preceding TIME_TOLERANCE window.
    """

    latest = {}

    for row in radio_rows:

        t = row["time_s"]

        if t > target_time:
            continue

        if target_time - t > TIME_TOLERANCE:
            continue

        rnti = row["rnti"]

        previous = latest.get(rnti)

        if (
            previous is None
            or row["time_s"] > previous["time_s"]
        ):
            latest[rnti] = row

    return list(latest.values())


def main():

    if not RADIO_CSV.exists():
        raise FileNotFoundError(
            RADIO_CSV
        )

    if not NETWORK_CSV.exists():
        raise FileNotFoundError(
            NETWORK_CSV
        )

    radio_rows = read_radio_rows()
    network_rows = read_network_rows()

    if not radio_rows:
        raise RuntimeError(
            "No radio measurements found."
        )

    if not network_rows:
        raise RuntimeError(
            "No network measurements found."
        )

    pipeline = SDTLLMPipeline(
        PipelineConfig(
            llm_backend="mock",
            vision_backend="mock",
        )
    )

    # ----------------------------------------------------------
    # Group network rows by timestamp.
    # ----------------------------------------------------------

    network_by_time = defaultdict(list)

    for row in network_rows:
        network_by_time[
            round(row["time_s"], 6)
        ].append(row)

    snapshot_times = sorted(
        network_by_time.keys()
    )

    total_radio_tokens = 0
    total_network_tokens = 0

    print()
    print("==============================================")
    print("TEMPORALLY ALIGNED RADIO + NETWORK SDT")
    print("==============================================")

    for snapshot_time in snapshot_times:

        # ------------------------------------------------------
        # Latest radio state for each UE
        # ------------------------------------------------------

        radio_snapshot = latest_radio_per_ue(
            radio_rows,
            snapshot_time,
        )

        tokens = []

        for row in radio_snapshot:

            radio_token = (
                pipeline.nr_sinr_encoder
                .encode_measurement(
                    timestamp=snapshot_time,
                    cell_id=row["cell_id"],
                    rnti=row["rnti"],
                    bwp_id=row["bwp_id"],
                    sinr_linear=row["sinr_linear"],
                    sinr_db=row["sinr_db"],
                    ue_x=row["ue_x"],
                    ue_y=row["ue_y"],
                    ue_z=row["ue_z"],
                    gnb_x=row["gnb_x"],
                    gnb_y=row["gnb_y"],
                    gnb_z=row["gnb_z"],
                )
            )

            tokens.append(radio_token)
            total_radio_tokens += 1

        # ------------------------------------------------------
        # Network state at THIS exact timestamp
        # ------------------------------------------------------

        for row in network_by_time[
            snapshot_time
        ]:

            network_token = (
                pipeline.network_encoder
                .encode_measurement(
                    timestamp=snapshot_time,
                    flow_id=row["flow_id"],
                    rnti=row["rnti"],
                    ue_ip=row["ue_ip"],
                    ue_x=row["ue_x"],
                    ue_y=row["ue_y"],
                    ue_z=row["ue_z"],
                    protocol=row["protocol"],
                    source=row["source"],
                    destination=row["destination"],
                    tx_packets=row[
                        "interval_tx_packets"
                    ],
                    rx_packets=row[
                        "interval_rx_packets"
                    ],
                    throughput_mbps=row[
                        "interval_throughput_mbps"
                    ],
                    mean_delay_ms=row[
                        "interval_mean_delay_ms"
                    ],
                    mean_jitter_ms=row[
                        "interval_mean_jitter_ms"
                    ],
                    packet_loss_pct=row[
                        "interval_packet_loss_pct"
                    ],
                )
            )

            tokens.append(network_token)
            total_network_tokens += 1

        if not tokens:
            continue

        # ------------------------------------------------------
        # ONE SDT INGESTION PER TIME SNAPSHOT
        #
        # This is the critical difference from the previous
        # experiment.
        # ------------------------------------------------------

        record = pipeline.twin.ingest(
            tokens,
            snapshot_time,
        )

        context = pipeline.twin.query_context(
            k_history=1
        )

        print()
        print(
            f"TIME {snapshot_time:.3f}s"
        )

        print(
            f"  Radio tokens:   {len(radio_snapshot)}"
        )

        print(
            f"  Network tokens: "
            f"{len(network_by_time[snapshot_time])}"
        )

        print(
            f"  SDT clusters:   "
            f"{len(record.fused_clusters)}"
        )

        for token in context:

            print(
                "   - "
                f"{token.modality}: "
                f"{token.label}"
            )

    print()
    print("==============================================")
    print("TEMPORAL SDT INGESTION COMPLETE")
    print("==============================================")

    print(
        "Radio tokens:",
        total_radio_tokens,
    )

    print(
        "Network tokens:",
        total_network_tokens,
    )

    print(
        "SDT history records:",
        len(pipeline.twin.history),
    )

    print(
        "Tracked entities:",
        len(pipeline.twin.all_track_ids()),
    )

    print("==============================================")


if __name__ == "__main__":
    main()
