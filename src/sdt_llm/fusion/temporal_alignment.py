"""
Temporal alignment — Section 2.3: "For feature clusters identified at
different timestamps, the pairing of these clusters is based on the
similarity distances between their respective cluster centers... clusters
will only be matched if the similarity distance between their centers is
less than a predefined threshold d_c."

This is what lets the digital twin keep a persistent identity ("track_id")
for the same real-world object/event across time — e.g. the paper's example
of a robot remembering *where* the biscuits were even once they're no longer
in view.

We solve the cross-timestamp matching as a linear assignment problem
(scipy.optimize.linear_sum_assignment) restricted to pairs whose distance is
below d_c, rather than plain greedy matching — this avoids a common greedy
failure mode where an early, mediocre match blocks a better one from being
made later in the same pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TemporalMatch:
    prev_index: Optional[int]   # index into previous timestamp's cluster centers, or None if newly appeared
    curr_index: int             # index into current timestamp's cluster centers
    distance: Optional[float]   # cosine distance between matched centers, or None if newly appeared


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between rows of a (Na,D) and b (Nb,D) -> (Na,Nb)."""
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - an @ bn.T


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def align_clusters_across_time(
    prev_centers: Optional[np.ndarray],
    curr_centers: np.ndarray,
    d_c: float = 0.35,
    prev_locations: Optional[np.ndarray] = None,
    curr_locations: Optional[np.ndarray] = None,
    location_weight: float = 0.0,
    location_scale_m: float = 2.5,
) -> List[TemporalMatch]:
    """
    Match current-timestamp cluster centers to previous-timestamp cluster
    centers under threshold d_c.

    Parameters
    ----------
    prev_centers : (Np, D) array of previous timestamp's cluster-center
        embeddings, or None if this is the first timestamp (-> everything is "new").
    curr_centers : (Nc, D) array of current timestamp's cluster-center embeddings.
    d_c : maximum *blended* distance for a match to be allowed (paper's threshold
        on "similarity distances between cluster centers").
    prev_locations, curr_locations : optional (N, 3) world-frame locations
        (metres) for the same clusters, used to blend in physical continuity.
        The paper's own Sec. 2.3 point is that every token carries a location
        stamp precisely so the SDT can reason about "time, space, and
        semantics" jointly — an untrained semantic projection alone is not a
        reliable identity cue frame-to-frame (a weak, ambiguous detection can
        sit numerically close to the wrong object's embedding), whereas real
        objects cannot teleport between consecutive frames. Blending in
        spatial distance is a physically-motivated way to stabilise track
        identity; set location_weight=0 to recover pure semantic matching.
    location_weight : 0 = ignore location (pure embedding match), 1 = ignore
        embedding (pure nearest-location match). Ignored if locations aren't given.
    location_scale_m : divides raw metre distances so they sit on a comparable
        numeric scale to cosine distance (which lives in [0, 2]) before blending.
    """
    nc = curr_centers.shape[0]
    if prev_centers is None or prev_centers.shape[0] == 0 or nc == 0:
        return [TemporalMatch(prev_index=None, curr_index=i, distance=None) for i in range(nc)]

    dist = _cosine_distance(curr_centers, prev_centers)   # (Nc, Np)
    if location_weight > 0 and prev_locations is not None and curr_locations is not None:
        loc_dist = _euclidean_distance(curr_locations, prev_locations) / location_scale_m
        dist = (1.0 - location_weight) * dist + location_weight * loc_dist

    # Forbid matches at/above threshold by making them very expensive, then
    # solve optimal assignment; afterwards drop any assignment that still
    # violates the threshold (can happen when one side has surplus items).
    BIG = 1e6
    cost = np.where(dist < d_c, dist, BIG)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_prev = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < BIG:
            matched_prev[r] = (c, float(dist[r, c]))

    results = []
    for i in range(nc):
        if i in matched_prev:
            prev_idx, d = matched_prev[i]
            results.append(TemporalMatch(prev_index=int(prev_idx), curr_index=i, distance=d))
        else:
            results.append(TemporalMatch(prev_index=None, curr_index=i, distance=None))
    return results
