import csv
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


def read_radio():
    rows = []

    with RADIO_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

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


def read_network():
    rows = []

    with NETWORK_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "time_s": float(row["time_s"]),
                    "flow_id": int(row["flow_id"]),
                    "rnti": int(row["rnti"]),
                    "ue_ip": row["ue_ip"],
                    "ue_x": float(row["ue_x"]),
                    "ue_y": float(row["ue_y"]),
                    "ue_z": float(row["ue_z"]),
                    "protocol": row["protocol"],
                    "source": row["source"],
                    "destination": row["destination"],
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


def main():

    radio_rows = read_radio()
    network_rows = read_network()

    pipeline = SDTLLMPipeline(
        PipelineConfig(
            llm_backend="hf_local",
            llm_kwargs={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "device": "cpu",
                "dtype": "float32",
            },
            vision_backend="mock",
            k_history=4,
        )
    )

    network_by_time = defaultdict(list)

    for row in network_rows:
        network_by_time[
            round(row["time_s"], 6)
        ].append(row)

    # ----------------------------------------------------------
    # Feed the same temporally aligned multimodal data that
    # we already proved works.
    # ----------------------------------------------------------

    for snapshot_time in sorted(
        network_by_time
    ):

        # Latest radio state for each UE
        latest_radio = {}

        for row in radio_rows:

            if row["time_s"] > snapshot_time:
                continue

            if snapshot_time - row["time_s"] > 0.010:
                continue

            latest_radio[
                row["rnti"]
            ] = row

        tokens = []

        # Radio tokens
        for row in latest_radio.values():

            token = (
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

            tokens.append(token)

        # Network tokens
        for row in network_by_time[
            snapshot_time
        ]:

            token = (
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

            tokens.append(token)

        if tokens:
            pipeline.twin.ingest(
                tokens,
                snapshot_time,
            )

    # ----------------------------------------------------------
    # Inspect the latest fused state.
    # ----------------------------------------------------------

    print()
    print("==============================================")
    print("LATEST FUSED SDT STATE")
    print("==============================================")

    latest = pipeline.twin.query_context(
        k_history=1
    )

    for token in latest:
        print()
        print(token.describe())

    # ----------------------------------------------------------
    # Build an actual LLM prompt.
    # ----------------------------------------------------------

    question = (
        "Which UE is currently experiencing the strongest "
        "cross-layer degradation? Explain using its radio "
        "condition and network condition, including SINR, "
        "packet loss, throughput and delay."
    )

    prompt = (
        pipeline.answer.__self__
        if False
        else None
    )

    # Build and display exactly what the LLM receives.
    from sdt_llm.llm.prompt_builder import build_prompt

    llm_prompt = build_prompt(
        query=question,
        twin=pipeline.twin,
        k_history=4,
    )

    print()
    print("==============================================")
    print("LLM PROMPT")
    print("==============================================")
    print(llm_prompt)

    # ----------------------------------------------------------
    # Generate a grounded answer through the SDTLLMPipeline.
    #
    # This ensures the simulator-grounded evidence validator,
    # retry logic, and deterministic fallback are applied.
    # ----------------------------------------------------------

    answer = pipeline.answer(
        question,
        k_history=4,
    )

    print()
    print("==============================================")
    print("LLM ANSWER")
    print("==============================================")
    print(answer)

    print()
    print("==============================================")
    print("END-TO-END SDT -> LLM TEST COMPLETE")
    print("==============================================")


if __name__ == "__main__":
    main()
