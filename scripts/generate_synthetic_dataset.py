#!/usr/bin/env python3
"""Regenerate all synthetic data (vision PNG/JSON + radio JSON) used by the demos.

Usage:
    python scripts/generate_synthetic_dataset.py [--out-dir data/synthetic]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdt_llm.data import synthetic_radio, synthetic_vision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/synthetic")
    args = ap.parse_args()

    vision_paths = synthetic_vision.generate_and_save(f"{args.out_dir}/vision")
    print(f"[vision] wrote {len(vision_paths)} scenes to {args.out_dir}/vision/")

    radio_paths = synthetic_radio.generate_and_save(f"{args.out_dir}/radio")
    print(f"[radio]  wrote {len(radio_paths)} scenes to {args.out_dir}/radio/")


if __name__ == "__main__":
    main()
