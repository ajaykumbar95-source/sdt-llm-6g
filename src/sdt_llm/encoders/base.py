"""
Base interface all sensor -> semantic-token encoders implement, plus a
shared helper for projecting heterogeneous per-modality feature vectors into
the common D-dimensional token space that the fusion stage (Sec. 2.3) operates
in ("an additional neural network projection will be employed to align
[tokens from different modalities]").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sdt_llm.tokens import TokenSet


class BaseSensorEncoder(ABC):
    """Common interface: raw sensor reading -> List[SemanticToken] (Sec. 2.1 / 2.2)."""

    modality: str = "unknown"

    @abstractmethod
    def encode(self, raw_input: Any, timestamp: float) -> TokenSet:
        ...


class SeededLinearProjection:
    """
    A fixed, seeded (non-learned) linear projection R^{in_dim} -> R^{out_dim}.

    Stands in for the "additional neural network projection" the paper
    mentions for aligning tokens from different modalities/sources into a
    shared embedding space. It is deterministic and reproducible but *not
    trained* — exactly like the fusion transformer (see token_fusion.py),
    this should eventually be learned jointly with the rest of the pipeline
    on real data. Kept intentionally simple (single linear layer) so it's
    trivial to swap for a learned `nn.Linear` later.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = x @ self.W + self.b
        # L2-normalise: keeps embeddings on the unit hypersphere so cosine
        # similarity/distance behaves consistently across modalities/encoders.
        norm = np.linalg.norm(out) + 1e-12
        return (out / norm).astype(np.float32)
