#!/usr/bin/env python3
"""
Watches a CSV file for live-streamed multipath data (see
data/ns3_stream_bridge.py for the schema) and runs it through the SDT+LLM
pipeline AS IT ARRIVES — this is what you point at the file your ns-3
SionnaRtChannelModel patch will append to, once that's in place.

Usage:
    # in one terminal: your ns-3 simulation, writing to /tmp/sdt_radio_stream.csv
    # in another terminal:
    python scripts/run_ns3_live_demo.py --csv /tmp/sdt_radio_stream.csv --llm-backend mock

    # to try it right now, with no ns-3 involved yet, using the built-in
    # fake producer (proves the plumbing works end-to-end):
    python scripts/run_ns3_live_demo.py --demo-producer
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sdt_llm.data.ns3_stream_bridge import LiveBridgeConfig, run_live, write_csv_row  # noqa: E402
from sdt_llm.data.synthetic_radio import MultipathComponent  # noqa: E402
from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402


def _launch_demo_producer(csv_path: str) -> None:
    """Writes a short fake stream so you can see the live loop work without ns-3."""
    def run():
        time.sleep(0.3)
        with open(csv_path, "a") as f:
            for t in range(6):
                wall = MultipathComponent(delay_s=2 * 4.5 / 3e8, doppler_hz=0.0,
                                           aoa_az_rad=0.6, aoa_el_rad=-0.1, aod_az_rad=0.6, aod_el_rad=-0.1,
                                           path_gain_re=0.04, path_gain_im=0.0)
                write_csv_row(f, float(t), (0.1, 0.1, 2.6), (0.1, 0.1, 2.6), 28e9, wall)
                if t >= 2:
                    person = MultipathComponent(
                        delay_s=2 * (3.0 - 0.2 * t) / 3e8, doppler_hz=180.0,
                        aoa_az_rad=-0.4 + 0.05 * t, aoa_el_rad=0.05, aod_az_rad=-0.4 + 0.05 * t, aod_el_rad=0.05,
                        path_gain_re=0.03, path_gain_im=0.01,
                    )
                    write_csv_row(f, float(t), (0.1, 0.1, 2.6), (0.1, 0.1, 2.6), 28e9, person)
                time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path to the live CSV stream (see ns3_stream_bridge.py for schema)")
    ap.add_argument("--llm-backend", default="mock",
                     choices=["mock", "hf_local", "anthropic_api", "openai_compatible_api"])
    ap.add_argument("--ask-every", type=int, default=2, help="query the LLM every N ingested scenes (0=never)")
    ap.add_argument("--idle-timeout", type=float, default=None,
                     help="stop after this many seconds of no new data (omit to run forever, Ctrl-C to stop)")
    ap.add_argument("--demo-producer", action="store_true",
                     help="also launch a built-in fake writer so you can see this work with zero ns-3 setup")
    args = ap.parse_args()

    csv_path = args.csv
    if args.demo_producer:
        if csv_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            tmp.close()
            csv_path = tmp.name
        print(f"[demo] launching a fake background writer -> {csv_path}")
        _launch_demo_producer(csv_path)
        if args.idle_timeout is None:
            args.idle_timeout = 4.0

    if csv_path is None:
        ap.error("--csv is required unless --demo-producer is set")

    print(f"Watching {csv_path} ... (Ctrl-C to stop)" if args.idle_timeout is None
          else f"Watching {csv_path} for up to {args.idle_timeout}s of new data...")

    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=256, k=3, cluster_ratio=0.7),
        llm_backend=args.llm_backend,
    ))
    try:
        run_live(pipeline, LiveBridgeConfig(
            csv_path=csv_path, stop_after_idle_s=args.idle_timeout, ask_every_n_scenes=args.ask_every,
        ))
    except KeyboardInterrupt:
        print("\nStopped.")

    print(f"\nSession summary: {len(pipeline.twin.history)} scenes ingested, "
          f"tracks seen: {pipeline.twin.all_track_ids()}")


if __name__ == "__main__":
    main()
