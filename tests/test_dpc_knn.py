import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdt_llm.fusion.dpc_knn import dpc_knn_cluster  # noqa: E402


def test_recovers_well_separated_blobs():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8)) * 5
    pts = np.concatenate([c + rng.normal(size=(6, 8)) * 0.2 for c in centers], axis=0)

    res = dpc_knn_cluster(pts, k=4, n_clusters=3, metric="cosine_distance")
    assert res.n_clusters == 3
    # every cluster should be "pure": all members from the same true blob
    for members in res.clusters():
        true_blob_ids = {m // 6 for m in members}
        assert len(true_blob_ids) == 1


def test_edge_cases_n0_n1():
    assert dpc_knn_cluster(np.zeros((0, 8)), k=4).n_clusters == 0
    r1 = dpc_knn_cluster(np.random.randn(1, 8), k=4)
    assert r1.n_clusters == 1
    assert list(r1.center_indices) == [0]


def test_literal_metric_runs_without_error():
    pts = np.random.default_rng(1).normal(size=(12, 6))
    res = dpc_knn_cluster(pts, k=3, n_clusters=4, metric="cosine_similarity_literal")
    assert res.n_clusters == 4
    assert res.assignment.shape == (12,)


def test_cluster_ratio_default_produces_fewer_clusters_than_tokens():
    pts = np.random.default_rng(2).normal(size=(20, 16))
    res = dpc_knn_cluster(pts, k=4, cluster_ratio=0.4)
    assert 0 < res.n_clusters < 20


def test_location_blend_changes_assignment_when_appropriate():
    """Two embedding-identical points should still be split into different
    clusters if they are physically far apart and location_weight is high."""
    emb = np.array([[1.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 1.0, 0.0, 0.0]] * 2, dtype=np.float32)
    # first pair far apart in space, second pair close together
    locs = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [5.0, 5.0, 5.0], [5.1, 5.1, 5.1]])
    res = dpc_knn_cluster(
        emb, k=2, n_clusters=2, locations=locs, location_weight=0.9, location_scale_m=2.5
    )
    # token 0 and 1 share an embedding but are far apart -> should end up in different clusters
    assert res.assignment[0] != res.assignment[1]
    # token 2 and 3 are close together -> should end up in the same cluster
    assert res.assignment[2] == res.assignment[3]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
