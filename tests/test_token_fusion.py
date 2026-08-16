import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdt_llm.fusion.dpc_knn import dpc_knn_cluster  # noqa: E402
from sdt_llm.fusion.token_fusion import TokenFusionBlock, TokenFusionConfig, fuse_all_clusters  # noqa: E402


def test_single_token_passthrough():
    block = TokenFusionBlock(TokenFusionConfig(dim=16, n_heads=4, seed=1))
    tok = np.random.default_rng(0).normal(size=(1, 16)).astype(np.float32)
    fused, weights = block.forward(tok)
    assert np.allclose(fused, tok[0])
    assert weights.shape == (1,)
    assert np.isclose(weights.sum(), 1.0)


def test_attention_weights_sum_to_one():
    block = TokenFusionBlock(TokenFusionConfig(dim=32, n_heads=4, seed=2))
    toks = np.random.default_rng(1).normal(size=(7, 32)).astype(np.float32)
    fused, weights = block.forward(toks)
    assert fused.shape == (32,)
    assert weights.shape == (7,)
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert (weights >= 0).all()


def test_deterministic_given_seed():
    toks = np.random.default_rng(3).normal(size=(5, 24)).astype(np.float32)
    b1 = TokenFusionBlock(TokenFusionConfig(dim=24, n_heads=4, seed=42))
    b2 = TokenFusionBlock(TokenFusionConfig(dim=24, n_heads=4, seed=42))
    f1, _ = b1.forward(toks)
    f2, _ = b2.forward(toks)
    assert np.allclose(f1, f2)


def test_fuse_all_clusters_shapes():
    pts = np.random.default_rng(4).normal(size=(10, 32)).astype(np.float32)
    res = dpc_knn_cluster(pts, k=3, n_clusters=3)
    block = TokenFusionBlock(TokenFusionConfig(dim=32, n_heads=4, seed=5))
    fused, weights = fuse_all_clusters(pts, res.clusters(), block)
    assert fused.shape == (3, 32)
    assert len(weights) == 3
    for w, members in zip(weights, res.clusters()):
        assert w.shape == (len(members),)


def test_dim_not_divisible_by_heads_raises():
    with pytest.raises(AssertionError):
        TokenFusionBlock(TokenFusionConfig(dim=30, n_heads=4))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
