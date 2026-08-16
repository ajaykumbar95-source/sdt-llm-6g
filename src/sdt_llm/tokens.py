"""
Core data structures.

Paper mapping
-------------
Section 2.1  "Semantic Sensor Data"          -> semantic token T^s   (modality="vision", or any non-radio sensor)
Section 2.2  "Tokenized Radio Channel
              Measurement"                    -> semantic token T^c   (modality="radio")
Section 2.3  "Semantic Digital Twin
              Representation"                 -> SemanticToken carries a timestamp AND a location,
                                                  "creating a three-dimensional representation of the
                                                  environment that encompasses time, space, and semantics"
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Vector = np.ndarray
Location = Tuple[float, float, float]  # (x, y, z) meters in the digital-twin world frame

_id_counter = itertools.count()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{next(_id_counter):06d}-{uuid.uuid4().hex[:6]}"


@dataclass(eq=False)  # numpy `embedding` field breaks auto-generated __eq__/__hash__; use token_id for identity
class SemanticToken:
    """
    One semantic token t_i.

    `embedding` lives in the *shared* D-dimensional semantic space that both the
    vision branch (T^s) and the radio branch (T^c) are projected into, so that
    tokens from different modalities can be clustered/fused together (Sec. 2.3).
    """

    embedding: Vector                      # shape (D,), float32
    label: str                             # human-readable semantic concept, e.g. "person:reading", "obstacle"
    modality: str                          # "vision" | "radio" | "fused"
    timestamp: float                       # seconds, monotonically increasing per scene/session
    location: Location                     # (x, y, z) meters — the "location stamp" from Sec. 2.3
    attributes: Dict[str, Any] = field(default_factory=dict)   # e.g. velocity, material, confidence, aoa_deg...
    confidence: float = 1.0
    token_id: str = field(default_factory=lambda: _new_id("tok"))
    source_ids: Tuple[str, ...] = field(default_factory=tuple)  # provenance: which raw token(s) this was fused from

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "token_id": self.token_id,
            "label": self.label,
            "modality": self.modality,
            "timestamp": round(float(self.timestamp), 3),
            "location": tuple(round(float(x), 3) for x in self.location),
            "attributes": self.attributes,
            "confidence": round(float(self.confidence), 3),
        }
        return d

    def describe(self) -> str:
        """One-line human-readable rendering used by the prompt builder (Sec. 3.2)."""
        loc = f"({self.location[0]:.1f}, {self.location[1]:.1f}, {self.location[2]:.1f})"
        attrs = ", ".join(f"{k}={_fmt(v)}" for k, v in self.attributes.items())
        base = f"t={self.timestamp:.2f}s @ {loc} [{self.modality}] {self.label}"
        if attrs:
            base += f" ({attrs})"
        if self.confidence < 0.999:
            base += f" conf={self.confidence:.2f}"
        return base


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


TokenSet = List[SemanticToken]


@dataclass
class FusedCluster:
    """
    Output of the token-fusion step (Sec. 2.3): one fused token T̃_n representing
    a whole feature cluster, plus bookkeeping needed for cross-timestamp pairing
    (the "clusters will only match if similarity distance between centers < d_c"
    rule) and for tracking an object's identity over time.
    """

    fused_token: SemanticToken
    member_ids: Tuple[str, ...]           # token_ids of the raw tokens fused into this cluster
    center_embedding: Vector              # embedding of the DPC-KNN cluster center (pre-fusion)
    center_location: Location             # location of that same DPC-KNN center token (for temporal gating)
    track_id: Optional[str] = None        # persistent identity across timestamps, assigned during temporal alignment

    def as_dict(self) -> Dict[str, Any]:
        d = self.fused_token.as_dict()
        d["track_id"] = self.track_id
        d["n_members"] = len(self.member_ids)
        return d
