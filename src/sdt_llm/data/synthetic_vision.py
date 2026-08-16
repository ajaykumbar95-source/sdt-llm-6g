"""
Synthetic "camera" data — stands in for Section 2.1's raw vision sensor.

We don't have a 3-D renderer available, so scenes are rendered as a simple
2-D camera-schematic (perspective-ish: farther objects are smaller and sit
higher near a horizon line) using only PIL primitives. This is enough for
(a) a human glancing at the PNG to sanity-check the scenario, and (b) the
default rule-based "mock" vision encoder to do real, if simple, pixel-level
object detection (colour-blob finding) on actual image data rather than
just reading back the ground truth. If you install `torch`+`transformers`
(see requirements-full.txt) you can instead point a real CLIP model at
these same PNGs — see encoders/vision_encoder.py.

All positions are in the SAME world frame (metres) used by the radio branch
(data/synthetic_radio.py), so a vision-derived token and a radio-derived
token that describe the same physical object end up at (nearly) the same
`location`, which is what lets the fusion stage cluster them together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

Color = Tuple[int, int, int]

# Deterministic colour per category — lets the simple mock CV detector key
# off colour, the way a classic (pre-deep-learning) vision pipeline would.
CATEGORY_COLOR: dict[str, Color] = {
    "person": (222, 90, 90),
    "book": (90, 140, 222),
    "chair": (150, 150, 150),
    "table": (120, 90, 60),
    "box": (222, 180, 70),
    "wall_obstacle": (80, 80, 90),
}

ROOM_W_M, ROOM_D_M, ROOM_H_M = 6.0, 6.0, 3.0
IMG_W, IMG_H = 512, 320
HORIZON_Y = int(IMG_H * 0.55)


@dataclass
class SyntheticObject:
    object_id: str
    category: str                     # key into CATEGORY_COLOR
    world_pos: Tuple[float, float, float]   # (x, y, z) metres, room frame
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # m/s, used by the radio branch too
    size_m: float = 0.4
    action: Optional[str] = None      # e.g. "reading", "walking" -> feeds the semantic label

    @property
    def color(self) -> Color:
        return CATEGORY_COLOR.get(self.category, (200, 200, 200))

    def label(self) -> str:
        return f"{self.category}:{self.action}" if self.action else self.category


@dataclass
class VisionScene:
    timestamp: float
    objects: List[SyntheticObject]
    camera_pos: Tuple[float, float, float] = (ROOM_W_M / 2, 0.2, 1.5)

    def to_manifest(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "camera_pos": self.camera_pos,
            "objects": [asdict(o) for o in self.objects],
        }


def _project(obj: SyntheticObject, cam: Tuple[float, float, float]) -> Tuple[int, int, float]:
    """Very small pinhole-ish projection: depth = distance along +y from camera."""
    cx, cy, cz = cam
    ox, oy, oz = obj.world_pos
    depth = max(0.3, oy - cy)
    scale = 3.5 / depth  # perspective scale factor
    screen_x = IMG_W / 2 + (ox - cx) * scale * 40
    screen_y = HORIZON_Y - (oz) * scale * 55
    return int(screen_x), int(screen_y), scale


def render_scene(scene: VisionScene) -> Image.Image:
    img = Image.new("RGB", (IMG_W, IMG_H), (235, 240, 245))
    draw = ImageDraw.Draw(img)
    # floor / horizon
    draw.rectangle([0, HORIZON_Y, IMG_W, IMG_H], fill=(210, 205, 195))
    draw.line([0, HORIZON_Y, IMG_W, HORIZON_Y], fill=(180, 175, 165), width=2)

    # draw back-to-front (far objects first) for correct occlusion
    for obj in sorted(scene.objects, key=lambda o: -o.world_pos[1]):
        sx, sy, scale = _project(obj, scene.camera_pos)
        r = max(4, int(obj.size_m * scale * 30))
        color = obj.color
        if obj.category == "person":
            # head + body so it's visually distinct from box-like objects
            draw.ellipse([sx - r * 0.5, sy - r * 1.7, sx + r * 0.5, sy - r * 0.7], fill=color)
            draw.rectangle([sx - r * 0.6, sy - r * 0.7, sx + r * 0.6, sy + r * 0.6], fill=color)
        else:
            draw.rectangle([sx - r, sy - r, sx + r, sy + r], fill=color)
        draw.text((sx - r, sy + r + 2), obj.label(), fill=(30, 30, 30))
    return img


def save_scene(scene: VisionScene, out_dir: Path, name: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{name}.png"
    json_path = out_dir / f"{name}.json"
    render_scene(scene).save(img_path)
    json_path.write_text(json.dumps(scene.to_manifest(), indent=2))
    return img_path, json_path


def book_reading_scenario() -> List[VisionScene]:
    """
    Mirrors the paper's running example (Fig. 4-ish): a person enters, sits
    at a table, and picks up a book to read. 5 timestamps, 1s apart.
    """
    table = SyntheticObject("table_1", "table", (3.0, 3.0, 0.4), size_m=0.9)
    chair = SyntheticObject("chair_1", "chair", (3.0, 2.6, 0.45), size_m=0.5)
    scenes = []
    walk_xs = [1.0, 1.8, 2.6, 3.0, 3.0]
    actions = ["walking", "walking", "sitting", "sitting", "reading"]
    for t, (x, act) in enumerate(zip(walk_xs, actions)):
        person = SyntheticObject(
            "person_1", "person", (x, 2.5, 0.0),
            velocity=(0.8, 0.0, 0.0) if act == "walking" else (0.0, 0.0, 0.0),
            size_m=0.5, action=act,
        )
        objs = [table, chair, person]
        if act == "reading":
            objs.append(SyntheticObject("book_1", "book", (x, 2.3, 0.9), size_m=0.15, action="held"))
        scenes.append(VisionScene(timestamp=float(t), objects=objs))
    return scenes


def generate_and_save(out_dir: str = "data/synthetic/vision") -> List[Tuple[Path, Path]]:
    out = Path(out_dir)
    paths = []
    for i, scene in enumerate(book_reading_scenario()):
        paths.append(save_scene(scene, out, f"scene_{i:02d}"))
    return paths


if __name__ == "__main__":
    for img_p, json_p in generate_and_save():
        print("wrote", img_p, json_p)
