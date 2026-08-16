"""
Streaming bridge for connecting a LIVE ns-3 simulation to the SDT+LLM
pipeline, instead of a one-shot synthetic dataset or a batch Python script.

Architecture
------------
ns-3 (C++, inside SionnaRtChannelModel's periodic update) is expected to
APPEND one line per multipath component to a plain CSV file every time it
resolves the channel (i.e. every `UpdatePeriod`). This module TAILS that
file the way `tail -f` does — reading new lines as they're written, without
waiting for the simulation to finish — groups consecutive rows that share a
timestamp into one `RadioScene`, and feeds each completed scene into the SDT
pipeline as soon as it's available. That's what gives you "real-time-ish"
behaviour without needing sockets, shared memory, or embedding Python inside
the ns-3 process itself: the C++ side only ever does simple, unglamorous
`std::ofstream` appends, which is about as low-risk as C++ integration gets.

This is the PYTHON side of the bridge and is fully tested (see the
self-test at the bottom of this file, and tests/test_ns3_stream_bridge.py).
The C++ side (the part that writes the CSV from inside
SionnaRtChannelModel / SionnaRtSpectrumPropagationLossModel) is NOT included
here yet — see the chat message this file was delivered with for why, and
for exactly what to send so it can be written precisely instead of guessed.

CSV schema (one line per multipath component; every row is self-contained —
no cross-row state needed to parse a single row, only to *group* rows into
a RadioScene). Positions are given for BOTH transmitter and receiver
separately (a real ns-3 NR link is bistatic — gNB and UE are not
co-located), and angles are written in Sionna's own *zenith* convention
(0 = straight up), not this project's elevation-from-horizontal convention
— the conversion happens once, here, at read time, so the C++ writer can
just dump Sionna's raw numbers with zero physics/unit conversion on that
side (keeping the harder-to-debug C++ layer as mechanical as possible):

    timestamp,tx_x,tx_y,tx_z,rx_x,rx_y,rx_z,carrier_hz,delay_s,doppler_hz,aoa_az_rad,aoa_zenith_rad,aod_az_rad,aod_zenith_rad,path_gain_re,path_gain_im

No header row expected (simpler for C++ to just always append data rows;
pass `has_header=True` to `tail_csv_stream` if you add one). Comment lines
starting with '#' are ignored, which is a handy way for the C++ side to log
e.g. "# no valid paths this update" without breaking the parser.

If tx and rx are at the same position (a dedicated monostatic ISAC sensor,
as opposed to a real bistatic gNB<->UE link), that's a perfectly normal
special case of this same schema — just write identical tx_*/rx_* columns.
`bistatic_scatterer_location()` in synthetic_radio.py handles both exactly.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence

import numpy as np

from sdt_llm.data.synthetic_radio import MultipathComponent, RadioScene

CSV_FIELDS = [
    "timestamp", "tx_x", "tx_y", "tx_z", "rx_x", "rx_y", "rx_z", "carrier_hz",
    "delay_s", "doppler_hz", "aoa_az_rad", "aoa_zenith_rad", "aod_az_rad", "aod_zenith_rad",
    "path_gain_re", "path_gain_im",
]


def _parse_row(row: Sequence[str]) -> Optional[tuple]:
    if not row or row[0].startswith("#"):
        return None
    if len(row) != len(CSV_FIELDS):
        return None  # malformed/partial row (e.g. a torn write) -> skip, don't crash the stream
    try:
        vals = [float(x) for x in row]
    except ValueError:
        return None
    return tuple(vals)


def tail_csv_stream(
    path: str,
    poll_interval_s: float = 0.2,
    stop_after_idle_s: Optional[float] = None,
    from_start: bool = True,
) -> Iterator[tuple]:
    """
    Yield parsed CSV rows (as tuples of floats, matching CSV_FIELDS order) as
    they're appended to `path`, similarly to `tail -f`. Blocks (polling)
    between reads. Robust to a partial/torn last line (waits for the next
    poll rather than yielding a broken row) and to the file not existing yet
    (waits for it to be created).

    stop_after_idle_s: if set, stop iterating after this many seconds with
    no new complete rows (useful for demos/tests; omit for a truly
    long-running live session).
    """
    p = Path(path)
    while not p.exists():
        time.sleep(poll_interval_s)

    with open(p, "r", newline="") as f:
        if not from_start:
            f.seek(0, 2)  # jump to current end of file
        buffer = ""
        idle_for = 0.0
        while True:
            chunk = f.read()
            if chunk:
                buffer += chunk
                idle_for = 0.0
                *complete_lines, buffer = buffer.split("\n")  # keep last (possibly partial) segment
                for line in complete_lines:
                    line = line.strip()
                    if not line:
                        continue
                    row = next(csv.reader([line]))
                    parsed = _parse_row(row)
                    if parsed is not None:
                        yield parsed
            else:
                time.sleep(poll_interval_s)
                idle_for += poll_interval_s
                if stop_after_idle_s is not None and idle_for >= stop_after_idle_s:
                    return


def rows_to_scenes(rows: Iterator[tuple]) -> Iterator[RadioScene]:
    """
    Group consecutive same-timestamp rows into one RadioScene, yielding each
    scene as soon as a row with a NEW timestamp arrives (i.e. as early as
    possible without needing to know the stream has ended).
    """
    current_ts: Optional[float] = None
    current_tx = current_rx = None
    current_carrier = None
    current_paths: List[MultipathComponent] = []

    for row in rows:
        (ts, tx_x, tx_y, tx_z, rx_x, rx_y, rx_z, fc,
         delay, dop, aoa_az, aoa_zenith, aod_az, aod_zenith, gre, gim) = row
        if current_ts is not None and ts != current_ts:
            yield RadioScene(timestamp=current_ts, sensor_pos=current_rx, tx_pos=current_tx,
                              carrier_hz=current_carrier, paths=current_paths)
            current_paths = []
        current_ts = ts
        current_tx, current_rx, current_carrier = (tx_x, tx_y, tx_z), (rx_x, rx_y, rx_z), fc
        current_paths.append(MultipathComponent(
            delay_s=delay, doppler_hz=dop,
            # Sionna's angles are zenith (0=up); this project's *_el_rad is
            # elevation-from-horizontal (0=horizontal, +pi/2=up) -- convert once, here.
            aoa_az_rad=aoa_az, aoa_el_rad=(np.pi / 2 - aoa_zenith),
            aod_az_rad=aod_az, aod_el_rad=(np.pi / 2 - aod_zenith),
            path_gain_re=gre, path_gain_im=gim,
        ))
    if current_ts is not None:
        yield RadioScene(timestamp=current_ts, sensor_pos=current_rx, tx_pos=current_tx,
                          carrier_hz=current_carrier, paths=current_paths)


def write_csv_row(
    f, scene_timestamp: float, tx_pos, rx_pos, carrier_hz: float, comp: MultipathComponent,
) -> None:
    """Convenience for testing / for a Python-side producer standing in for
    the eventual C++ writer. Mirrors exactly what the C++ side should emit —
    including converting this project's elevation convention back to
    Sionna's raw zenith convention, so a file written by this function is
    byte-for-byte the same shape the real C++ patch will produce."""
    aoa_zenith = np.pi / 2 - comp.aoa_el_rad
    aod_zenith = np.pi / 2 - comp.aod_el_rad
    f.write(
        f"{scene_timestamp},{tx_pos[0]},{tx_pos[1]},{tx_pos[2]},"
        f"{rx_pos[0]},{rx_pos[1]},{rx_pos[2]},{carrier_hz},"
        f"{comp.delay_s},{comp.doppler_hz},{comp.aoa_az_rad},{aoa_zenith},"
        f"{comp.aod_az_rad},{aod_zenith},{comp.path_gain_re},{comp.path_gain_im}\n"
    )
    f.flush()


@dataclass
class LiveBridgeConfig:
    csv_path: str
    poll_interval_s: float = 0.2
    stop_after_idle_s: Optional[float] = None
    ask_every_n_scenes: int = 1          # query the LLM after every N ingested scenes; 0 disables auto-asking
    default_query: str = "Summarize what the sensing system currently understands about the environment."


def run_live(
    pipeline,
    config: LiveBridgeConfig,
    on_scene: Optional[Callable[[RadioScene, "TimestampRecord"], None]] = None,  # noqa: F821
    on_answer: Optional[Callable[[str, str], None]] = None,
) -> None:
    """
    Blocking loop: tails config.csv_path, ingests each completed RadioScene
    into `pipeline` (an SDTLLMPipeline) as it arrives, and optionally asks
    the LLM every `ask_every_n_scenes` scenes. Runs until the stream goes
    idle for `stop_after_idle_s` (or forever if that's None — Ctrl-C to stop).

    `on_scene(scene, record)` and `on_answer(query, answer)` are optional
    callbacks (e.g. to print, log, or push to a UI) — if omitted, results
    are just printed.
    """
    rows = tail_csv_stream(config.csv_path, poll_interval_s=config.poll_interval_s,
                            stop_after_idle_s=config.stop_after_idle_s)
    n = 0
    for scene in rows_to_scenes(rows):
        record = pipeline.ingest_radio(scene)
        if on_scene:
            on_scene(scene, record)
        else:
            labels = [fc.fused_token.label for fc in record.fused_clusters]
            print(f"[t={scene.timestamp}] {len(scene.paths)} paths -> {len(record.fused_clusters)} clusters {labels}")

        n += 1
        if config.ask_every_n_scenes and n % config.ask_every_n_scenes == 0:
            result = pipeline.ask(config.default_query, k_history=4)
            if on_answer:
                on_answer(config.default_query, result["answer"])
            else:
                print(f"  Q: {config.default_query}\n  A: {result['answer']}\n")


if __name__ == "__main__":
    # Self-test: spin up a producer thread that writes a few fake scenes to a
    # temp CSV, and confirm the consumer side picks them up live. No sionna
    # or ns-3 needed -- this only exercises the Python bridge.
    import tempfile
    import threading

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.close()

    def producer():
        time.sleep(0.3)
        with open(tmp.name, "a") as f:
            for t in range(3):
                for i in range(2):
                    comp = MultipathComponent(
                        delay_s=1e-8 * (i + 1), doppler_hz=float(t * 10),
                        aoa_az_rad=0.1 * i, aoa_el_rad=0.0, aod_az_rad=0.1 * i, aod_el_rad=0.0,
                        path_gain_re=0.01, path_gain_im=0.0,
                    )
                    write_csv_row(f, float(t), (0.0, 0.0, 2.6), (0.0, 0.0, 2.6), 28e9, comp)
                time.sleep(0.3)

    threading.Thread(target=producer, daemon=True).start()
    print("Tailing", tmp.name)
    for scene in rows_to_scenes(tail_csv_stream(tmp.name, poll_interval_s=0.1, stop_after_idle_s=1.5)):
        print(f"  got scene t={scene.timestamp} with {len(scene.paths)} paths")
    print("Self-test complete.")
