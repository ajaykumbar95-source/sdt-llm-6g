"""
Vision sensor encoder — Section 2.1 "Semantic Sensor Data".

  "the LLM processes the raw sensor data captured, e.g. by a camera, to
  extract semantic information... encoding attributes such as their
  location and movement into more abstract semantic tokens T^s"

Two backends behind one interface (`VisionSensorEncoder.encode`):

  * "mock" (default, always available): a genuine, if simple, pixel-level
    detector — finds connected blobs of each known category colour in the
    rendered PNG and turns each blob into a semantic token. This is *not*
    reading the ground-truth JSON; it is really looking at pixels, which is
    the honest way to smoke-test the pipeline without a multi-GB model
    download.
  * "clip" (optional, requires `pip install -r requirements-full.txt` and
    internet access to Hugging Face Hub on your own machine — not available
    in this build/test sandbox): zero-shot-tags the image against a
    candidate label vocabulary using a real pretrained CLIP model and uses
    CLIP's own image embedding as the token embedding. This is the closest
    thing to a real pretrained component in the vision branch — there is no
    pretrained "SDT" checkpoint to download (see README), but CLIP is a
    reasonable, genuinely-pretrained stand-in for "an LLM/VLM extracts
    semantic tokens from a camera frame".
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from sdt_llm.encoders.base import BaseSensorEncoder, SeededLinearProjection
from sdt_llm.tokens import SemanticToken
from sdt_llm.data.synthetic_vision import CATEGORY_COLOR, IMG_W, IMG_H, HORIZON_Y, ROOM_W_M

_COLOR_TOL = 28  # per-channel tolerance when matching a pixel to a known category colour


def _blob_detect(img: np.ndarray, target_rgb: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
    """Very small connected-component finder for pixels close to `target_rgb`.
    Returns a list of bounding boxes (xmin, ymin, xmax, ymax). No external CV
    dependency (no opencv/skimage) needed — just numpy flood-fill via BFS."""
    mask = np.all(np.abs(img.astype(int) - np.array(target_rgb)) <= _COLOR_TOL, axis=-1)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    boxes = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        # BFS flood fill
        stack = [(y0, x0)]
        visited[y0, x0] = True
        ymin, ymax, xmin, xmax = y0, y0, x0, x0
        count = 0
        while stack:
            y, x = stack.pop()
            count += 1
            ymin, ymax = min(ymin, y), max(ymax, y)
            xmin, xmax = min(xmin, x), max(xmax, x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if count >= 12:  # discard 1-2px noise
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _screen_to_world(cx: float, cy: float, box_h: int) -> Tuple[float, float, float]:
    """Inverse of synthetic_vision._project — approximate world (x, y, z) from
    a detected blob's screen position/size. Depth is estimated from apparent
    box height (bigger on screen == closer), matching the renderer's model."""
    scale = max(box_h, 1) / 30.0  # inverse of `r = size_m * scale * 30` for size_m~1
    depth = float(np.clip(3.5 / max(scale, 1e-3), 0.3, ROOM_W_M))
    world_x = float(np.clip((cx - IMG_W / 2) / (scale * 40) + ROOM_W_M / 2, 0.0, ROOM_W_M))
    world_z = float(np.clip((HORIZON_Y - cy) / (scale * 55), 0.0, 3.0))
    return world_x, depth, world_z


class VisionSensorEncoder(BaseSensorEncoder):
    modality = "vision"

    def __init__(self, embed_dim: int = 256, backend: str = "mock", clip_model_name: str = "openai/clip-vit-base-patch32", seed: int = 11):
        self.embed_dim = embed_dim
        self.backend = backend
        self._proj = SeededLinearProjection(in_dim=16, out_dim=embed_dim, seed=seed)
        self._clip = None
        self._clip_labels = list(CATEGORY_COLOR.keys()) + ["reading", "walking", "sitting", "held"]
        if backend == "clip":
            self._init_clip(clip_model_name)

    def _init_clip(self, model_name: str) -> None:
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = CLIPModel.from_pretrained(model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            self._clip = True
        except Exception as e:  # pragma: no cover - depends on optional heavy deps / network
            print(
                f"[VisionSensorEncoder] Could not load CLIP backend ({e!r}); "
                f"falling back to the 'mock' colour-blob detector. "
                f"Install requirements-full.txt and ensure Hugging Face Hub is reachable to use CLIP."
            )
            self.backend = "mock"

    # -- mock backend -----------------------------------------------------
    def _encode_mock(self, image_path: Path, timestamp: float) -> List[SemanticToken]:
        img = np.array(Image.open(image_path).convert("RGB"))
        tokens = []
        for category, rgb in CATEGORY_COLOR.items():
            for (xmin, ymin, xmax, ymax) in _blob_detect(img, rgb):
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                box_h = ymax - ymin
                wx, wy, wz = _screen_to_world(cx, cy, box_h)
                feat = np.array([
                    cx / IMG_W, cy / IMG_H, box_h / IMG_H, (xmax - xmin) / IMG_W,
                    wx / 6.0, wy / 6.0, wz / 3.0,
                    rgb[0] / 255, rgb[1] / 255, rgb[2] / 255,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ], dtype=np.float32)
                embedding = self._proj(feat)
                tokens.append(SemanticToken(
                    embedding=embedding,
                    label=category,
                    modality="vision",
                    timestamp=timestamp,
                    location=(wx, wy, wz),
                    attributes={"pixel_bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]},
                    confidence=0.9,
                ))
        return tokens

    # -- real CLIP backend --------------------------------------------------
    def _encode_clip(self, image_path: Path, timestamp: float) -> List[SemanticToken]:  # pragma: no cover
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = self._clip_processor(text=self._clip_labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = self._clip_model(**inputs)
        image_embed = out.image_embeds[0].numpy()
        probs = out.logits_per_image.softmax(dim=-1)[0].numpy()
        top = int(probs.argmax())
        label = self._clip_labels[top]
        # project CLIP's (512 or 768-d) embedding into our shared token space
        proj = SeededLinearProjection(in_dim=image_embed.shape[0], out_dim=self.embed_dim, seed=99)
        embedding = proj(image_embed)
        return [SemanticToken(
            embedding=embedding, label=label, modality="vision", timestamp=timestamp,
            location=(ROOM_W_M / 2, 2.0, 1.0),  # CLIP gives no localisation; placeholder centre-room
            attributes={"clip_prob": float(probs[top])}, confidence=float(probs[top]),
        )]

    def encode(self, raw_input: Path | str, timestamp: float) -> List[SemanticToken]:
        image_path = Path(raw_input)
        if self.backend == "clip" and self._clip:
            return self._encode_clip(image_path, timestamp)
        return self._encode_mock(image_path, timestamp)
