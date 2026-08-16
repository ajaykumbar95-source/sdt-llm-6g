#!/usr/bin/env python3
"""
Demo 2 — YOUR idea: swap the vision branch for 6G radio sensing entirely.

    6G multipath components (synthetic today; ns-3 + Sionna RT tomorrow)
        -> RadioChannelEncoder (Sec 2.2, T^c: range/velocity/angle -> "obstacle"/"movement"/"interference")
        -> SemanticDigitalTwin (Sec 2.3: DPC-KNN + fusion + tracking, SAME code as the vision demo)
        -> prompt_builder (Sec 3.2) -> LLM -> answer

No camera is used anywhere in this script — this is the
"6G networks -> SDT -> LLM inference" pipeline you described, validated on
synthetic data shaped exactly like real Sionna RT `Paths` output (see
data/synthetic_radio.py's module docstring for the exact field mapping —
that's your on-ramp to Step 2, plugging in real ns-3 + Sionna RT).

Usage:
    python scripts/run_radio_sdt_llm_demo.py [--llm-backend mock|hf_local|anthropic_api|openai_compatible_api]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sdt_llm.data.synthetic_radio import indoor_isac_scenario  # noqa: E402
from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-backend", default="mock",
                     choices=["mock", "hf_local", "anthropic_api", "openai_compatible_api"])
    args = ap.parse_args()

    print("=" * 78)
    print("DEMO 2: 6G radio (ISAC/CSI) -> SDT -> LLM   <-- your idea, no camera involved")
    print("=" * 78)

    print("\n[1/4] Simulating a 6G monostatic-sensing scenario (Sionna-RT-shaped multipath):")
    print("      a person walks across a room; a static obstacle + box sit out of any camera's view.")
    radio_scenes = indoor_isac_scenario()
    print(f"      generated {len(radio_scenes)} timestamps, "
          f"{sum(len(s.paths) for s in radio_scenes)} total multipath components")

    print(f"\n[2/4] Building pipeline (llm_backend={args.llm_backend})...")
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=256, k=3, cluster_ratio=0.7),
        llm_backend=args.llm_backend,
    ))

    print("\n[3/4] Ingesting radio scenes into the Semantic Digital Twin...")
    for scene in radio_scenes:
        record = pipeline.ingest_radio(scene)
        summary = [(fc.track_id, fc.fused_token.label) for fc in record.fused_clusters]
        print(f"      t={scene.timestamp:.0f}: {len(scene.paths)} raw MPCs -> "
              f"{len(record.raw_tokens)} detected tokens -> {len(record.fused_clusters)} fused clusters {summary}")

    persistent = [tid for tid in pipeline.twin.all_track_ids() if len(pipeline.twin.query_track(tid)) >= 4]
    print(f"\n      Persistent tracks (seen >=4/{len(radio_scenes)} timestamps -> confidently real, not clutter): {persistent}")

    print("\n[4/4] Querying the LLM through the SDT (purely from radio-derived semantics)...\n")
    for query, recall in [
        ("Is it safe for an autonomous cart to cross the room right now?", ["obstacle"]),
        ("Summarize what the 6G sensing system currently understands about the room.", None),
        ("Where was the nearest static obstacle located?", ["obstacle"]),
    ]:
        result = pipeline.ask(query, k_history=3, recall_labels=recall)
        print(f"Q: {query}")
        print(f"A ({result['llm_backend']}): {result['answer']}\n")

    print("Next step (your Step 2): replace `indoor_isac_scenario()` with multipath components")
    print("read from real ns-3 + Sionna RT ray tracing — RadioChannelEncoder doesn't need to change,")
    print("it only needs MultipathComponent objects with the same fields (see data/synthetic_radio.py).")


if __name__ == "__main__":
    main()
