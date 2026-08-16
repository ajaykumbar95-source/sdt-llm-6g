"""
SemanticDigitalTwin — Section 2.3 "Semantic Digital Twin Representation".

The SDT is a stateful container that:

1. clusters/fuses incoming semantic tokens,
2. temporally aligns fused clusters across timestamps,
3. maintains stable track IDs,
4. provides context to the LLM.

For the ns-3 / 5G-LENA / Sionna RT integration, tokens carrying
the same RNTI are treated as representing the same UE. This gives
radio and network observations an explicit cross-modal identity
before semantic fusion.
"""

from __future__ import annotations

import itertools

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sdt_llm.fusion.dpc_knn import (
    DPCKNNResult,
    dpc_knn_cluster,
)

from sdt_llm.fusion.temporal_alignment import (
    align_clusters_across_time,
)

from sdt_llm.fusion.token_fusion import (
    TokenFusionBlock,
    TokenFusionConfig,
    fuse_all_clusters,
)

from sdt_llm.tokens import (
    FusedCluster,
    SemanticToken,
    TokenSet,
)


def _aggregate_label(
    labels: List[str],
    weights: np.ndarray,
    max_join: int = 3,
) -> str:
    """
    Aggregate labels according to fusion attention weights.
    """

    uniq: Dict[str, float] = {}

    for lbl, weight in zip(labels, weights):
        uniq[lbl] = (
            uniq.get(lbl, 0.0)
            + float(weight)
        )

    ranked = sorted(
        uniq.items(),
        key=lambda item: -item[1],
    )

    return "+".join(
        label
        for label, _ in ranked[:max_join]
    )


def _aggregate_modality(
    modalities: List[str],
) -> str:
    """
    Return a single modality when all members share one
    modality, otherwise return fused.
    """

    uniq = set(modalities)

    return (
        modalities[0]
        if len(uniq) == 1
        else "fused"
    )


def _aggregate_attributes(
    members: List[SemanticToken],
) -> dict:
    """
    Merge attributes from all members.
    """

    output: dict = {}

    for member in members:
        output.update(member.attributes)

    modalities = {
        member.modality
        for member in members
    }

    if len(modalities) > 1:
        output["_fused_modalities"] = sorted(
            modalities
        )

    return output


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

    d_c: float = 0.45

    location_weight: float = 0.8

    location_scale_m: float = 2.5

    fusion_seed: int = 42


class SemanticDigitalTwin:
    """
    Stateful Semantic Digital Twin.

    RNTI-aware behavior:

    If incoming tokens contain RNTI values from multiple
    modalities, tokens with the same RNTI are explicitly grouped
    before semantic fusion.

    This is used for ns-3 / 5G-LENA / Sionna RT UE identity,
    where RNTI is a simulator-grounded identity shared by
    radio and network observations.
    """

    def __init__(
        self,
        config: Optional[SDTConfig] = None,
    ):
        self.cfg = (
            config
            if config is not None
            else SDTConfig()
        )

        self.fusion_block = TokenFusionBlock(
            TokenFusionConfig(
                dim=self.cfg.embed_dim,
                seed=self.cfg.fusion_seed,
            )
        )

        self.history: List[TimestampRecord] = []

        self._track_counter = itertools.count()

        self._track_registry: Dict[
            str,
            List[FusedCluster],
        ] = {}

    def _new_track_id(self) -> str:
        return (
            f"track-{next(self._track_counter):04d}"
        )

    # ============================================================
    # RNTI-aware grouping
    # ============================================================

    @staticmethod
    def _has_cross_modal_rnti_identity(
        tokens: TokenSet,
    ) -> bool:
        """
        Return True when the incoming token set contains
        RNTI identity information from multiple modalities.

        This path is intentionally specific to UE-aware
        radio/network fusion.
        """

        rnti_values = [
            token.attributes.get("rnti")
            for token in tokens
            if token.attributes.get("rnti")
            is not None
        ]

        modalities = {
            token.modality
            for token in tokens
            if token.attributes.get("rnti")
            is not None
        }

        return (
            len(rnti_values) > 0
            and len(modalities) > 1
        )

    @staticmethod
    def _rnti_clusters(
        tokens: TokenSet,
    ) -> List[List[int]]:
        """
        Group tokens by RNTI.

        Tokens without an RNTI are placed into their own
        singleton cluster.
        """

        grouped: Dict[
            object,
            List[int],
        ] = {}

        for index, token in enumerate(tokens):

            rnti = token.attributes.get(
                "rnti"
            )

            if rnti is None:
                key = (
                    "_no_rnti",
                    index,
                )
            else:
                key = (
                    "_rnti",
                    int(rnti),
                )

            grouped.setdefault(
                key,
                [],
            ).append(index)

        return list(grouped.values())

    # ============================================================
    # Main ingestion
    # ============================================================

    def ingest(
        self,
        tokens: TokenSet,
        timestamp: float,
        n_clusters: Optional[int] = None,
    ) -> TimestampRecord:

        if not tokens:

            record = TimestampRecord(
                timestamp=timestamp,
                raw_tokens=[],
                fused_clusters=[],
            )

            self.history.append(record)

            return record

        embeddings = np.stack(
            [
                token.embedding
                for token in tokens
            ]
        ).astype(
            np.float32
        )

        raw_locs = np.array(
            [
                token.location
                for token in tokens
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------------
        # Choose clustering strategy.
        #
        # For UE-aware radio/network data, RNTI is an explicit
        # identity key and is therefore used before DPC.
        #
        # For all other data, retain the original DPC-KNN path.
        # --------------------------------------------------------

        dpc_result = None

        if self._has_cross_modal_rnti_identity(
            tokens
        ):

            clusters = self._rnti_clusters(
                tokens
            )

            # Choose the highest-confidence member of each
            # identity group as the representative center.
            center_indices = []

            for cluster in clusters:

                representative = max(
                    cluster,
                    key=lambda idx:
                    tokens[idx].confidence,
                )

                center_indices.append(
                    representative
                )

        else:

            dpc_result = dpc_knn_cluster(
                embeddings,
                k=self.cfg.k,
                n_clusters=(
                    n_clusters
                    if n_clusters is not None
                    else self.cfg.n_clusters
                ),
                cluster_ratio=(
                    self.cfg.cluster_ratio
                ),
                metric=self.cfg.dpc_metric,
                locations=raw_locs,
                location_weight=(
                    self.cfg.location_weight
                ),
                location_scale_m=(
                    self.cfg.location_scale_m
                ),
            )

            clusters = dpc_result.clusters()

            center_indices = (
                dpc_result.center_indices
            )

        # --------------------------------------------------------
        # Transformer fusion for each identity/cluster.
        # --------------------------------------------------------

        fused_embeds, attn_weights = (
            fuse_all_clusters(
                embeddings,
                clusters,
                self.fusion_block,
            )
        )

        center_embeds = embeddings[
            center_indices
        ]

        center_locs = np.array(
            [
                tokens[index].location
                for index in center_indices
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------------
        # Temporal tracking.
        # --------------------------------------------------------

        prev_record = (
            self.history[-1]
            if self.history
            else None
        )

        prev_centers = None
        prev_locs = None

        if (
            prev_record is not None
            and prev_record.fused_clusters
        ):

            prev_centers = np.stack(
                [
                    fc.center_embedding
                    for fc in
                    prev_record.fused_clusters
                ]
            )

            prev_locs = np.array(
                [
                    fc.center_location
                    for fc in
                    prev_record.fused_clusters
                ],
                dtype=np.float64,
            )

        matches = align_clusters_across_time(
            prev_centers,
            center_embeds,
            d_c=self.cfg.d_c,
            prev_locations=prev_locs,
            curr_locations=center_locs,
            location_weight=(
                self.cfg.location_weight
            ),
            location_scale_m=(
                self.cfg.location_scale_m
            ),
        )

        # --------------------------------------------------------
        # Construct fused clusters.
        # --------------------------------------------------------

        fused_clusters: List[FusedCluster] = []

        for cluster_index, member_indices in enumerate(
            clusters
        ):

            members = [
                tokens[index]
                for index in member_indices
            ]

            weights = attn_weights[
                cluster_index
            ]

            label = _aggregate_label(
                [
                    member.label
                    for member in members
                ],
                weights,
            )

            modality = _aggregate_modality(
                [
                    member.modality
                    for member in members
                ]
            )

            locs = np.array(
                [
                    member.location
                    for member in members
                ],
                dtype=np.float64,
            )

            loc = tuple(
                (
                    weights[:, None]
                    * locs
                ).sum(
                    axis=0
                )
            )

            match = matches[
                cluster_index
            ]

            track_id = None

            if (
                match.prev_index is not None
                and prev_record is not None
            ):

                track_id = (
                    prev_record
                    .fused_clusters[
                        match.prev_index
                    ]
                    .track_id
                )

            if track_id is None:
                track_id = (
                    self._new_track_id()
                )

            fused_token = SemanticToken(
                embedding=(
                    fused_embeds[
                        cluster_index
                    ]
                ),

                label=label,

                modality=modality,

                timestamp=timestamp,

                location=(
                    float(loc[0]),
                    float(loc[1]),
                    float(loc[2]),
                ),

                attributes=(
                    _aggregate_attributes(
                        members
                    )
                ),

                confidence=float(
                    np.mean(
                        [
                            member.confidence
                            for member
                            in members
                        ]
                    )
                ),

                source_ids=tuple(
                    member.token_id
                    for member
                    in members
                ),
            )

            fused_cluster = FusedCluster(
                fused_token=fused_token,

                member_ids=tuple(
                    member.token_id
                    for member in members
                ),

                center_embedding=(
                    center_embeds[
                        cluster_index
                    ]
                ),

                center_location=(
                    tuple(
                        center_locs[
                            cluster_index
                        ]
                    )
                ),

                track_id=track_id,
            )

            fused_clusters.append(
                fused_cluster
            )

            self._track_registry.setdefault(
                track_id,
                [],
            ).append(
                fused_cluster
            )

        record = TimestampRecord(
            timestamp=timestamp,
            raw_tokens=list(tokens),
            fused_clusters=fused_clusters,
            dpc_result=dpc_result,
        )

        self.history.append(record)

        return record

    # ============================================================
    # Query interface
    # ============================================================

    def query_context(
        self,
        k_history: int = 3,
    ) -> TokenSet:
        """
        Return fused tokens from the most recent timestamps.
        """

        tokens: TokenSet = []

        for record in self.history[
            -k_history:
        ]:

            tokens.extend(
                cluster.fused_token
                for cluster
                in record.fused_clusters
            )

        return tokens

    def query_track(
        self,
        track_id: str,
    ) -> List[FusedCluster]:

        return self._track_registry.get(
            track_id,
            [],
        )

    def last_seen(
        self,
        label_substring: str,
    ) -> Optional[FusedCluster]:
        """
        Find the most recent fused cluster whose label
        contains the requested substring.
        """

        for record in reversed(
            self.history
        ):

            for cluster in record.fused_clusters:

                if (
                    label_substring
                    in cluster.fused_token.label
                ):
                    return cluster

        return None

    def all_track_ids(
        self,
    ) -> List[str]:

        return list(
            self._track_registry.keys()
        )
