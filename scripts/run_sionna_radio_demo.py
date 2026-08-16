#!/usr/bin/env python3
"""
Demo 2b — same as run_radio_sdt_llm_demo.py, but the radio scenes come from
REAL Sionna RT ray tracing (data/sionna_bridge.py) instead of the synthetic
generator (data/synthetic_radio.py). Everything downstream — RadioChannelEncoder,
SemanticDigitalTwin, prompt_builder, LLM — is UNCHANGED; only the data source
differs. That's the whole point of how this project was structured.

Requires: pip install -r requirements-sionna.txt --break-system-packages
(see that file and the accompanying chat message for details/hardware notes).

Usage:
    python scripts/run_sionna_radio_demo.py [--llm-backend mock|hf_local|...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-backend", default="mock",
                     choices=["mock", "hf_local", "anthropic_api", "openai_compatible_api"])
    ap.add_argument("--variant", default="llvm_ad_mono_polarized",
                     help="mitsuba variant: 'llvm_ad_mono_polarized' (CPU, no GPU needed) "
                          "or 'cuda_ad_mono_polarized' (needs an NVIDIA GPU)")
    args = ap.parse_args()

    try:
        from sdt_llm.data.sionna_bridge import MovingScatterer, simulate_moving_scenario
        import sionna.rt.scene as scene_mod
    except ImportError as e:
        print(f"This demo needs sionna-rt installed: pip install -r requirements-sionna.txt "
              f"--break-system-packages\n(original error: {e})")
        return

    print("=" * 78)
    print("DEMO 2b: REAL Sionna RT ray tracing -> SDT -> LLM")
    print("=" * 78)

    print("\n[1/3] Ray tracing a person walking through a built-in 'floor_wall' scene "
          f"(mitsuba variant={args.variant})...")
    scenes = simulate_moving_scenario(
        base_scene_path=scene_mod.floor_wall,
        sensor_pos=(0.1, 0.1, 2.6),
        scatterer_tracks={
            "person": [
                (0.0, 1.0, 1.0, 1.0, 0.8, 0.0, 0.0),
                (1.0, 1.8, 1.0, 1.0, 0.8, 0.0, 0.0),
                (2.0, 2.6, 1.0, 1.0, 0.8, 0.0, 0.0),
                (3.0, 3.4, 1.0, 1.0, 0.8, 0.0, 0.0),
            ],
        },
        scatterer_specs=[MovingScatterer(name="person", radius_m=0.4)],
        variant=args.variant,
    )
    print(f"      ray-traced {len(scenes)} timestamps")

    print(f"\n[2/3] Building pipeline (llm_backend={args.llm_backend})...")
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=256, k=3, cluster_ratio=0.7),
        llm_backend=args.llm_backend,
    ))

    print("\n      Ingesting real ray-traced scenes into the Semantic Digital Twin...")
    for scene in scenes:
        record = pipeline.ingest_radio(scene)
        labels = [fc.fused_token.label for fc in record.fused_clusters]
        print(f"      t={scene.timestamp}: {len(scene.paths)} real multipath components -> "
              f"{len(record.fused_clusters)} fused clusters {labels}")

    print("\n[3/3] Querying the LLM through the SDT (grounded in real ray-traced physics)...\n")
    result = pipeline.ask("What does the sensing system currently understand about the room?", k_history=4)
    print(f"Q: What does the sensing system currently understand about the room?")
    print(f"A ({result['llm_backend']}): {result['answer']}")


if __name__ == "__main__":
    main()
