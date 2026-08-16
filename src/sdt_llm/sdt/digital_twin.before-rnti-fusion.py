"""
SemanticDigitalTwin — Section 2.3 "Semantic Digital Twin Representation".

  "Every piece of information within the digital twin is tagged with a
  timestamp and location stamp, creating a three-dimensional representation
  of the environment that encompasses time, space, and semantics... The SDT
  is continuously updated as new sensor data and radio channel measurements
  become available."

This class is the stateful container that, on every `ingest()` call:
  1. DPC-KNN clusters the incoming token set (fusion/dpc_knn.py, Eq. 1-3)
  2. runs the transformer fusion block over each cluster (fusion/token_fusion.py)
  3. temporally aligns the new fused clusters against the previous timestamp's
     (fusion/temporal_alignment.py) so the same real-world object/event keeps
     a stable `track_id` across time — this is what gives the twin "memory"
     (Sec. 3.2's example of recalling where an out-of-view object was last seen).

Tokens fed into a single `ingest()` call may come from ONE modality (pure
vision, pure radio) or from BOTH at once — in the latter case a cluster that
ends up containing both a vision token and a radio token for the same
physical entity *is* the fusion the paper describes (e.g. a vision
"person:reading" token and a radio "movement" token for the same person
merge into one fused token with modality="fused").
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from sdt_llm.fusion.dpc_knn import DPCKNNResult, dpc_knn_cluster
from sdt_llm.fusion.temporal_alignment import align_clusters_across_time
from sdt_llm.fusion.token_fusion import TokenFusionBlock, TokenFusionConfig, fuse_all_clusters
from sdt_llm.tokens import FusedCluster, SemanticToken, TokenSet


def _aggregate_label(labels: List[str], weights: np.ndarray, max_join: int = 3) -> str:
    uniq: Dict[str, float] = {}
    for lbl, w in zip(labels, weights):
        uniq[lbl] = uniq.get(lbl, 0.0) + float(w)
    ranked = sorted(uniq.items(), key=lambda kv: -kv[1])
    return "+".join(lbl for lbl, _ in ranked[:max_join])


def _aggregate_modality(modalities: List[str]) -> str:
    uniq = set(modalities)
    return modalities[0] if len(uniq) == 1 else "fused"


def _aggregate_attributes(members: List[SemanticToken]) -> dict:
    out: dict = {}
    for m in members:
        out.update(m.attributes)
    if len({m.modality for m in members}) > 1:
        out["_fused_modalities"] = sorted({m.modality for m in members})
    return out


@dataclass
class TimestampRecord:
    timestamp: float
    raw_tokens: TokenSet
    fused_clusters: List[FusedCluster]
    dpc_result: Optional[DPCKNNResult] = None


@dataclass
class SDTConfig:
    embed_dim: int = 256
    k: int = 4
    n_clusters: Optional[int] = None
    cluster_ratio: float = 0.55
    dpc_metric: str = "cosine_distance"
    d_c: float = 0.45                   # temporal-match threshold (Sec 2.3)
    location_weight: float = 0.8        # blend of physical-location vs embedding distance for clustering/track matching
    # Normalises metre distances before blending with cosine distance (~[0,2]).
    # Deliberately set to a *characteristic frame-to-frame displacement*
    # (a person walking at ~1-2 m/s between samples), NOT the room size —
    # with a small room, normalising by the room diagonal makes almost every
    # pair of objects look "close", which isn't discriminative enough to
    # stop identity swaps between, say, a person and a nearby static object.
    location_scale_m: float = 2.5
    fusion_seed: int = 42


class SemanticDigitalTwin:
    def __init__(self, config: Optional[SDTConfig] = None):
        self.cfg = config or SDTConfig()
        self.fusion_block = TokenFusionBlock(TokenFusionConfig(dim=self.cfg.embed_dim, seed=self.cfg.fusion_seed))
        self.history: List[TimestampRecord] = []
        self._track_counter = itertools.count()
        self._track_registry: Dict[str, List[FusedCluster]] = {}

    def _new_track_id(self) -> str:
        return f"track-{next(self._track_counter):04d}"

    def ingest(self, tokens: TokenSet, timestamp: float, n_clusters: Optional[int] = None) -> TimestampRecord:
        if not tokens:
            record = TimestampRecord(timestamp=timestamp, raw_tokens=[], fused_clusters=[])
            self.history.append(record)
            return record

        embeddings = np.stack([t.embedding for t in tokens]).astype(np.float32)
        raw_locs = np.array([t.location for t in tokens], dtype=np.float64)
        dpc = dpc_knn_cluster(
            embeddings, k=self.cfg.k,
            n_clusters=n_clusters if n_clusters is not None else self.cfg.n_clusters,
            cluster_ratio=self.cfg.cluster_ratio, metric=self.cfg.dpc_metric,
            locations=raw_locs, location_weight=self.cfg.location_weight,
            location_scale_m=self.cfg.location_scale_m,
        )
        clusters = dpc.clusters()
        fused_embeds, attn_weights = fuse_all_clusters(embeddings, clusters, self.fusion_block)
        center_embeds = embeddings[dpc.center_indices]
        center_locs = np.array([tokens[i].location for i in dpc.center_indices], dtype=np.float64)

        prev_record = self.history[-1] if self.history else None
        prev_centers = prev_locs = None
        if prev_record is not None and prev_record.fused_clusters:
            prev_centers = np.stack([fc.center_embedding for fc in prev_record.fused_clusters])
            prev_locs = np.array([fc.center_location for fc in prev_record.fused_clusters], dtype=np.float64)
        matches = align_clusters_across_time(
            prev_centers, center_embeds, d_c=self.cfg.d_c,
            prev_locations=prev_locs, curr_locations=center_locs,
            location_weight=self.cfg.location_weight, location_scale_m=self.cfg.location_scale_m,
        )

        fused_clusters: List[FusedCluster] = []
        for n, idx in enumerate(clusters):
            members = [tokens[i] for i in idx]
            w = attn_weights[n]
            label = _aggregate_label([m.label for m in members], w)
            modality = _aggregate_modality([m.modality for m in members])
            locs = np.array([m.location for m in members], dtype=np.float64)
            loc = tuple((w[:, None] * locs).sum(axis=0))

            match = matches[n]
            track_id = None
            if match.prev_index is not None and prev_record is not None:
                track_id = prev_record.fused_clusters[match.prev_index].track_id
            if track_id is None:
                track_id = self._new_track_id()

            fused_tok = SemanticToken(
                embedding=fused_embeds[n], label=label, modality=modality, timestamp=timestamp,
                location=(float(loc[0]), float(loc[1]), float(loc[2])),
                attributes=_aggregate_attributes(members),
                confidence=float(np.mean([m.confidence for m in members])),
                source_ids=tuple(m.token_id for m in members),
            )
            fc = FusedCluster(
                fused_token=fused_tok, member_ids=tuple(m.token_id for m in members),
                center_embedding=center_embeds[n], center_location=tuple(center_locs[n]), track_id=track_id,
            )
            fused_clusters.append(fc)
            self._track_registry.setdefault(track_id, []).append(fc)

        record = TimestampRecord(timestamp=timestamp, raw_tokens=list(tokens), fused_clusters=fused_clusters, dpc_result=dpc)
        self.history.append(record)
        return record

    # ---- query interface used by the LLM prompt builder (Sec. 3.2) ------
    def query_context(self, k_history: int = 3) -> TokenSet:
        """Most recent `k_history` timestamps' fused tokens, oldest first."""
        toks: TokenSet = []
        for record in self.history[-k_history:]:
            toks.extend(fc.fused_token for fc in record.fused_clusters)
        return toks

    def query_track(self, track_id: str) -> List[FusedCluster]:
        return self._track_registry.get(track_id, [])

    def last_seen(self, label_substring: str) -> Optional[FusedCluster]:
        """Most recent fused cluster anywhere in history whose label contains
        `label_substring` — the "twin remembers what's no longer visible"
        query pattern from Sec. 3.2's context-aware-prediction example."""
        for record in reversed(self.history):
            for fc in record.fused_clusters:
                if label_substring in fc.fused_token.label:
                    return fc
        return None

    def all_track_ids(self) -> List[str]:
        return list(self._track_registry.keys())
