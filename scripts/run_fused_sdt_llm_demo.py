#!/usr/bin/env python3
"""
Demo 3 — the paper's actual full method: vision AND radio fused into ONE SDT
(Sec 2.1 T^s + Sec 2.2 T^c -> Sec 2.3 joint clustering/fusion). This is what
you'd run if you want to ADD 6G sensing alongside a camera rather than
replace it — e.g. so the twin still knows an object's location once it
leaves the camera's field of view, using radio to fill the gap.

Same room, same objects, same timestamps, observed by BOTH a camera and a
6G monostatic sensor at once. When a cluster ends up containing both a
vision token and a radio token for the same physical person/object, its
label becomes e.g. "person:reading+movement" and its modality becomes
"fused" — that combination *is* the paper's central contribution.

Usage:
    python scripts/run_fused_sdt_llm_demo.py [--llm-backend mock|hf_local|anthropic_api|openai_compatible_api]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sdt_llm.data.synthetic_radio import simulate_radio_scene  # noqa: E402
from sdt_llm.data.synthetic_vision import book_reading_scenario, save_scene  # noqa: E402
from sdt_llm.pipeline import PipelineConfig, SDTLLMPipeline  # noqa: E402
from sdt_llm.sdt.digital_twin import SDTConfig  # noqa: E402

RADIO_SENSOR_POS = (0.1, 0.1, 2.6)  # a 6G BS mounted in a corner; camera sits elsewhere (see VisionScene.camera_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-backend", default="mock",
                     choices=["mock", "hf_local", "anthropic_api", "openai_compatible_api"])
    ap.add_argument("--out-dir", default=str(ROOT / "data/synthetic/vision"))
    args = ap.parse_args()

    print("=" * 78)
    print("DEMO 3: vision + 6G radio, FUSED into one SDT  (paper's full method, Sec 2.1+2.2+2.3)")
    print("=" * 78)

    vision_scenes = book_reading_scenario()

    print(f"\n[1/3] Building pipeline (llm_backend={args.llm_backend})...")
    pipeline = SDTLLMPipeline(PipelineConfig(
        sdt=SDTConfig(embed_dim=256, k=4, cluster_ratio=0.55, location_weight=0.7),
        llm_backend=args.llm_backend,
    ))

    print("\n[2/3] Ingesting BOTH modalities per timestamp (same room, same objects, same instant)...")
    out_dir = Path(args.out_dir)
    for i, vscene in enumerate(vision_scenes):
        img_path, _ = save_scene(vscene, out_dir, f"fused_{i:02d}")
        rscene = simulate_radio_scene(vscene.objects, timestamp=vscene.timestamp, sensor_pos=RADIO_SENSOR_POS)
        record = pipeline.ingest_multimodal(timestamp=vscene.timestamp, image_path=img_path, radio_scene=rscene)

        n_vision = sum(1 for t in record.raw_tokens if t.modality == "vision")
        n_radio = sum(1 for t in record.raw_tokens if t.modality == "radio")
        fused_modal = [fc.fused_token.modality for fc in record.fused_clusters]
        labels = [fc.fused_token.label for fc in record.fused_clusters]
        print(f"      t={i}: {n_vision} vision tok + {n_radio} radio tok -> "
              f"{len(record.fused_clusters)} clusters | modalities={fused_modal} | labels={labels}")

    n_cross_modal = sum(
        1 for rec in pipeline.twin.history for fc in rec.fused_clusters if fc.fused_token.modality == "fused"
    )
    print(f"\n      {n_cross_modal} cluster(s) across the session fused vision+radio tokens together "
          f"(modality=\"fused\") — this is the paper's Sec 2.3 mechanism in action.")

    print("\n[3/3] Querying the LLM through the combined SDT...\n")
    for query, recall in [
        ("What is the person doing, and is there independent radio confirmation of their presence?", None),
        ("If the camera view were blocked, would the system still know where the person is?", ["person"]),
    ]:
        result = pipeline.ask(query, k_history=5, recall_labels=recall)
        print(f"Q: {query}")
        print(f"A ({result['llm_backend']}): {result['answer']}\n")


if __name__ == "__main__":
    main()
