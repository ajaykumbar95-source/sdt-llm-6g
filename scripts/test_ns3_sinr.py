import csv
from pathlib import Path

from sdt_llm.pipeline import (
    SDTLLMPipeline,
    PipelineConfig,
)


NS3_CSV = (
    Path.home()
    / "ns-3-dev"
    / "sdt_radio_trace.csv"
)


def classify_sinr(sinr_db: float) -> str:
    if sinr_db >= 25.0:
        return "strong_radio_link"

    if sinr_db >= 15.0:
        return "good_radio_link"

    if sinr_db >= 5.0:
        return "degraded_radio_link"

    return "poor_radio_link"


def main():
    if not NS3_CSV.exists():
        raise FileNotFoundError(
            f"ns-3 CSV not found: {NS3_CSV}"
        )

    pipeline = SDTLLMPipeline(
        PipelineConfig(
            llm_backend="mock",
            vision_backend="mock",
        )
    )

    count = 0

    with NS3_CSV.open(
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
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

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Missing columns in ns-3 radio CSV: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            timestamp = float(row["time_s"])
            cell_id = int(row["cell_id"])
            rnti = int(row["rnti"])
            bwp_id = int(row["bwp_id"])

            sinr_linear = float(
                row["sinr_linear"]
            )

            sinr_db = float(
                row["sinr_db"]
            )

            ue_x = float(row["ue_x"])
            ue_y = float(row["ue_y"])
            ue_z = float(row["ue_z"])

            gnb_x = float(row["gnb_x"])
            gnb_y = float(row["gnb_y"])
            gnb_z = float(row["gnb_z"])

            pipeline.ingest_nr_sinr(
                timestamp=timestamp,
                cell_id=cell_id,
                rnti=rnti,
                bwp_id=bwp_id,
                sinr_linear=sinr_linear,
                sinr_db=sinr_db,
                ue_x=ue_x,
                ue_y=ue_y,
                ue_z=ue_z,
                gnb_x=gnb_x,
                gnb_y=gnb_y,
                gnb_z=gnb_z,
            )

            count += 1

            if count <= 10:
                label = classify_sinr(sinr_db)

                distance = (
                    (
                        (ue_x - gnb_x) ** 2
                        + (ue_y - gnb_y) ** 2
                        + (ue_z - gnb_z) ** 2
                    )
                    ** 0.5
                )

                print(
                    f"[{count}] "
                    f"t={timestamp:.9f} "
                    f"UE={rnti} "
                    f"SINR={sinr_db:.3f} dB "
                    f"label={label} "
                    f"UE_pos=("
                    f"{ue_x:.3f},"
                    f"{ue_y:.3f},"
                    f"{ue_z:.3f}) "
                    f"gNB_pos=("
                    f"{gnb_x:.3f},"
                    f"{gnb_y:.3f},"
                    f"{gnb_z:.3f}) "
                    f"distance={distance:.3f} m"
                )

    print()
    print("========================================")
    print("NR SINR ingestion complete")
    print("Measurements:", count)
    print("========================================")


if __name__ == "__main__":
    main()
