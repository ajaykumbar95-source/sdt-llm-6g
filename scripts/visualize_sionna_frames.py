#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
NS3_DIR = Path.home() / "ns-3-dev"

FRAME_DIR = NS3_DIR / "sionna-rt-images"
TIMELINE_CSV = FRAME_DIR / "sionna-frame-timeline.csv"

OUTPUT_DIR = PROJECT_DIR / "results" / "visualizations"

MIN_FRAME_DURATION_MS = 80


def frame_number(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)

    if not match:
        raise ValueError(
            f"Invalid Sionna frame filename: {path.name}"
        )

    return int(match.group(1))


def load_timeline():
    if not TIMELINE_CSV.exists():
        raise FileNotFoundError(
            f"Missing Sionna timeline: {TIMELINE_CSV}"
        )

    rows = []

    with TIMELINE_CSV.open(
        newline="",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):
            rows.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "time_s": float(row["simulation_time_s"]),
                    "filename": row["image_filename"],
                }
            )

    rows.sort(key=lambda x: x["frame_id"])

    if not rows:
        raise RuntimeError(
            "Sionna timeline CSV contains no frames."
        )

    return rows


def annotate_frame(
    source: Path,
    output: Path,
    frame_id: int,
    time_s: float,
):
    image = Image.open(source).convert("RGB")

    draw = ImageDraw.Draw(image)

    label = (
        f"Sionna RT | Frame {frame_id} | "
        f"t = {time_s:.6f} s"
    )

    # White background behind timestamp.
    draw.rectangle(
        (12, 12, 430, 50),
        fill=(255, 255, 255),
    )

    draw.text(
        (20, 21),
        label,
        fill=(0, 0, 0),
    )

    image.save(
        output,
        quality=95,
    )


def build_animation(rows):
    images = []
    durations = []

    for index, row in enumerate(rows):
        path = FRAME_DIR / row["filename"]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Sionna frame: {path}"
            )

        image = Image.open(path).convert("RGB")
        images.append(image)

        if index < len(rows) - 1:
            delta_s = (
                rows[index + 1]["time_s"]
                - row["time_s"]
            )

            duration_ms = max(
                MIN_FRAME_DURATION_MS,
                int(delta_s * 1000),
            )
        else:
            duration_ms = MIN_FRAME_DURATION_MS

        durations.append(duration_ms)

    output = (
        OUTPUT_DIR
        / "sionna_rt_scene_animation.gif"
    )

    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )

    return output


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_timeline()

    # ---------------------------------------------------------
    # Preserve the authoritative timeline in the project.
    # ---------------------------------------------------------

    timeline_output = (
        OUTPUT_DIR
        / "sionna_frame_timeline.csv"
    )

    with timeline_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame_id",
                "simulation_time_s",
                "image_filename",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row["frame_id"],
                    f'{row["time_s"]:.9f}',
                    row["filename"],
                ]
            )

    # ---------------------------------------------------------
    # Select representative frames by ACTUAL simulation time.
    # ---------------------------------------------------------

    first = rows[0]

    last = rows[-1]

    middle = rows[len(rows) // 2]

    for row, name in [
        (first, "sionna_scene_early.png"),
        (middle, "sionna_scene_middle.png"),
        (last, "sionna_scene_final.png"),
    ]:

        source = FRAME_DIR / row["filename"]

        output = OUTPUT_DIR / name

        annotate_frame(
            source,
            output,
            row["frame_id"],
            row["time_s"],
        )

    animation = build_animation(rows)

    print("==============================================")
    print("SIONNA RT VISUALIZATION COMPLETE")
    print("==============================================")
    print(f"Frames:       {len(rows)}")
    print(
        f"Time range:   "
        f"{rows[0]['time_s']:.6f}s -> "
        f"{rows[-1]['time_s']:.6f}s"
    )
    print(
        f"Early frame:  "
        f"{first['frame_id']} @ "
        f"{first['time_s']:.6f}s"
    )
    print(
        f"Middle frame: "
        f"{middle['frame_id']} @ "
        f"{middle['time_s']:.6f}s"
    )
    print(
        f"Final frame:  "
        f"{last['frame_id']} @ "
        f"{last['time_s']:.6f}s"
    )
    print(f"Timeline:     {timeline_output}")
    print(f"Animation:    {animation}")
    print("==============================================")


if __name__ == "__main__":
    main()
