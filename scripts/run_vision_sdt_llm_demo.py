#!/usr/bin/env python3
"""
Demo 1 — replicates the paper's own pipeline: vision -> SDT -> LLM inference.

    camera frames (synthetic) -> VisionSensorEncoder (Sec 2.1, T^s)
                              -> SemanticDigitalTwin (Sec 2.3: DPC-KNN + fusion + tracking)
                              -> prompt_builder (Sec 3.2) -> LLM -> answer

Usage:
    python scripts/run_vision_sdt_llm_demo.py [--llm-backend mock|hf_local|anthropic_api|openai_compatible_api]
                                               [--vision-backend mock|clip]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sdt_llm.data.synthetic_vision import generate_and_save  # noqa: E402
from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-backend", default="mock",
                     choices=["mock", "hf_local", "anthropic_api", "openai_compatible_api"])
    ap.add_argument("--vision-backend", default="mock", choices=["mock", "clip"])
    ap.add_argument("--out-dir", default=str(ROOT / "data/synthetic/vision"))
    args = ap.parse_args()

    print("=" * 78)
    print("DEMO 1: vision -> SDT -> LLM  (paper's own pipeline, Sec 2.1 + 2.3 + 3.2)")
    print("=" * 78)

    print("\n[1/4] Generating synthetic camera scenario (person enters, sits, reads)...")
    scene_paths = generate_and_save(args.out_dir)
    print(f"      wrote {len(scene_paths)} frames to {args.out_dir}")

    print(f"\n[2/4] Building pipeline (vision_backend={args.vision_backend}, llm_backend={args.llm_backend})...")
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=256, k=3, cluster_ratio=0.6),
        vision_backend=args.vision_backend,
        llm_backend=args.llm_backend,
    ))

    print("\n[3/4] Ingesting frames into the Semantic Digital Twin...")
    for i, (img_path, _json_path) in enumerate(scene_paths):
        record = pipeline.ingest_vision(img_path, timestamp=float(i))
        labels = [fc.fused_token.label for fc in record.fused_clusters]
        print(f"      t={i}: {img_path.name} -> {len(record.raw_tokens)} raw tokens -> "
              f"{len(record.fused_clusters)} fused clusters {labels}")

    print("\n[4/4] Querying the LLM through the SDT...\n")
    for query in [
        "What is happening in the monitored area right now?",
        "Has the person been stationary or moving over the observed period?",
    ]:
        result = pipeline.ask(query, k_history=5)
        print(f"Q: {query}")
        print(f"A ({result['llm_backend']}): {result['answer']}\n")

    print("Tip: rerun with --llm-backend hf_local (after `pip install -r requirements-full.txt`,")
    print("     on a machine with internet access) for real LLM reasoning instead of the mock stand-in.")


if __name__ == "__main__":
    main()
