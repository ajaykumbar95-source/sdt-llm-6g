"""
DPC-KNN feature clustering — Section 2.3, Equations (1)-(3).

The paper clusters a token set T = {T^s, T^c} using "a variant of the density
peaks clustering based on k-nearest neighbors (DPC-KNN) algorithm", picking
cluster centers as tokens that are simultaneously (a) locally dense and
(b) far from any *other* token of higher density — the classic Rodriguez &
Laio (2014) density-peaks idea, adapted to use cosine similarity between
token embeddings instead of raw Euclidean distance.

--------------------------------------------------------------------------
A note on the published equations (read this before you trust the numbers)
--------------------------------------------------------------------------
As transcribed on the Huawei tech page, Eq. (1) defines KNN(t_i) with a
"<=" comparison against the k-th neighbour's similarity, and Eq. (2)/(3)
plug raw cosine *similarity* directly into exp(-mean(...)) to get the local
density rho_i. Taken completely literally:

  * Eq (1) with "<=" would select *all but* the (k-1) closest tokens as the
    "k nearest neighbours", which contradicts the definition of KNN and is
    almost certainly a sign flip introduced when the equation was converted
    out of the original paper's typesetting (this is a very common failure
    mode for inequality symbols in PDF/MathML -> web-text conversion).
  * Eq (2)/(3) plugging in raw cosine *similarity* rather than a *distance*
    inverts the usual density-peaks semantics: two near-duplicate tokens
    (similarity ~1) would get a *lower* rho than two dissimilar tokens
    (similarity ~0), i.e. "dense" regions would score as low density.

Standard DPC-KNN (e.g. as used for visual-token clustering in Xie et al./
TCFormer-style work, which this section's method is visibly modelled on)
defines density from a *distance*, not a similarity:

    rho_i  = exp( -(1/k) * sum_{j in KNN(i)} d(t_i, t_j) )
    delta_i = min_{j: rho_j > rho_i} d(t_i, t_j)   (or max_j d(t_i,t_j) if t_i is the global mode)
    s_i     = rho_i * delta_i                       (cluster centers = highest-scoring tokens)

We implement exactly this, standard, well-behaved version by default
(`metric="cosine_distance"`, i.e. d = 1 - cosine_similarity), which is what
actually gives you density peaks. For transparency / so you can reproduce
the equations exactly as published and compare, we also expose
`metric="cosine_similarity_literal"`, which plugs raw cosine similarity into
the same slots verbatim. Both modes share the *same* KNN(t_i) = "k closest
tokens by the chosen metric" definition (i.e. we treat Eq (1)'s comparator
as "<=" -> a garbled ">=", since a KNN set that excludes the nearest points
isn't a sensible reading under any metric).

This is exactly the kind of ambiguity worth flagging rather than quietly
"fixing" — see README.md, section "Equation interpretation notes".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np

Metric = Literal["cosine_distance", "cosine_similarity_literal"]


@dataclass
class DPCKNNResult:
    rho: np.ndarray                 # (N,) local density, Eq. (2)
    delta: np.ndarray               # (N,) relative distance, Eq. (3)
    score: np.ndarray               # (N,) s_i = rho_i * delta_i
    center_indices: np.ndarray      # (n_clusters,) indices of chosen cluster centers, ranked by score desc
    assignment: np.ndarray          # (N,) index into center_indices each token is assigned to
    k: int
    metric: Metric

    @property
    def n_clusters(self) -> int:
        return len(self.center_indices)

    def clusters(self) -> List[np.ndarray]:
        """Return, for each cluster, the array of token indices assigned to it."""
        return [np.where(self.assignment == c)[0] for c in range(self.n_clusters)]


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normed = embeddings / norms
    return normed @ normed.T


def _distance_matrix(
    embeddings: np.ndarray,
    metric: Metric,
    locations: Optional[np.ndarray] = None,
    location_weight: float = 0.0,
    location_scale_m: float = 2.5,
) -> np.ndarray:
    sim = _cosine_similarity_matrix(embeddings)
    if metric == "cosine_distance":
        dist = 1.0 - sim
    elif metric == "cosine_similarity_literal":
        dist = sim.copy()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    if location_weight > 0 and locations is not None:
        # Blend in physical proximity (see temporal_alignment.py for the same
        # idea applied across time). Justified directly by the paper's own
        # framing of the SDT as jointly "time, space, and semantics" (Sec.
        # 2.3) — and, practically, an *untrained* random projection (see
        # encoders/base.py) has no guarantee that spatially/physically
        # related tokens end up nearby in embedding space, so relying on
        # embedding distance alone can merge unrelated detections that
        # happen to project close together, or split one real object's
        # tokens apart. A trained projection would eventually make this
        # blend unnecessary.
        loc_dist = np.linalg.norm(locations[:, None, :] - locations[None, :, :], axis=-1) / location_scale_m
        dist = (1.0 - location_weight) * dist + location_weight * loc_dist
    np.fill_diagonal(dist, np.inf)  # a token is never its own neighbour
    return dist


def dpc_knn_cluster(
    embeddings: np.ndarray,
    k: int = 5,
    n_clusters: Optional[int] = None,
    cluster_ratio: float = 0.4,
    metric: Metric = "cosine_distance",
    locations: Optional[np.ndarray] = None,
    location_weight: float = 0.0,
    location_scale_m: float = 2.5,
) -> DPCKNNResult:
    """
    Cluster `embeddings` (N, D) with DPC-KNN, Eqs. (1)-(3).

    Parameters
    ----------
    k : number of nearest neighbours used to estimate local density (Eq. 1-2).
    n_clusters : explicit number of clusters/cluster-centers to keep. If None,
        derived from `cluster_ratio * N` (paper does not specify a fixed rule;
        DPC-style methods usually leave this as a hyperparameter).
    cluster_ratio : used only when n_clusters is None. 0.4 means roughly 40% of
        tokens become their own cluster centers, i.e. clusters average ~2.5
        tokens each — a reasonable default for small per-timestamp token sets.
    metric : see module docstring.
    locations, location_weight, location_scale_m : optional physical-location
        blend, see `_distance_matrix`. location_weight=0 (default) reproduces
        pure embedding-space DPC-KNN as specified by the paper's equations.
    """
    n = embeddings.shape[0]
    if n == 0:
        return DPCKNNResult(
            rho=np.zeros(0), delta=np.zeros(0), score=np.zeros(0),
            center_indices=np.zeros(0, dtype=int), assignment=np.zeros(0, dtype=int),
            k=k, metric=metric,
        )
    if n == 1:
        return DPCKNNResult(
            rho=np.ones(1), delta=np.zeros(1), score=np.zeros(1),
            center_indices=np.array([0]), assignment=np.array([0]),
            k=k, metric=metric,
        )

    k_eff = int(np.clip(k, 1, n - 1))
    dist = _distance_matrix(embeddings, metric, locations, location_weight, location_scale_m)

    # --- Eq. (1): KNN(t_i) = k closest tokens under the chosen metric ---
    knn_idx = np.argsort(dist, axis=1)[:, :k_eff]                # (N, k)
    knn_dist = np.take_along_axis(dist, knn_idx, axis=1)          # (N, k)

    # --- Eq. (2): local density ---
    rho = np.exp(-knn_dist.mean(axis=1))

    # --- Eq. (3): relative distance to the nearest token of higher density ---
    delta = np.empty(n, dtype=float)
    order_desc = np.argsort(-rho)
    global_mode = order_desc[0]
    for i in range(n):
        higher = np.where(rho > rho[i])[0]
        if higher.size > 0:
            delta[i] = dist[i, higher].min()
        else:
            # i is the (a) global density mode -> max distance to anyone else
            finite = dist[i, np.isfinite(dist[i])]
            delta[i] = finite.max() if finite.size else 0.0
    # guarantee the true global mode is well defined even under ties
    if not np.isfinite(delta[global_mode]):
        delta[global_mode] = np.nanmax(delta[np.isfinite(delta)]) if np.isfinite(delta).any() else 0.0

    score = rho * delta

    # --- cluster centers = top-scoring tokens ---
    if n_clusters is None:
        n_clusters = max(1, round(n * cluster_ratio))
    n_clusters = int(np.clip(n_clusters, 1, n))
    center_indices = np.argsort(-score)[:n_clusters]

    # --- assign every token (centers included) to its nearest center ---
    center_dist = dist[:, center_indices].copy()
    # a center's distance to itself is inf (diagonal); fix so it maps to itself
    for pos, c in enumerate(center_indices):
        center_dist[c, pos] = -np.inf
    assignment = np.argmin(center_dist, axis=1)

    return DPCKNNResult(
        rho=rho, delta=delta, score=score,
        center_indices=center_indices, assignment=assignment,
        k=k_eff, metric=metric,
    )
