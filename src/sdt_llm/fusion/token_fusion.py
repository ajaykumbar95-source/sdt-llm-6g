"""
Token fusion — Section 2.3: "A transformer block is applied to each feature
cluster to capture the semantic relationships and information interaction
between different tokens in the same feature cluster... resulting in fused
token clusters T̃_n."

Implementation notes
---------------------
* Pure NumPy, zero heavy dependencies, so the synthetic-data pipeline you
  asked for runs instantly on any machine (no torch/CUDA download needed).
* This is an *untrained* transformer block: weights are Xavier-initialised
  from a fixed seed, not learned. That's an honest limitation, not a bug —
  the paper doesn't publish weights (there is no public pretrained "SDT
  model" to download; see README). The architecture is real and faithful
  to the paper's description; a training loop is the natural next step
  once you have a task loss (e.g. reconstruction, or downstream QA
  accuracy) to backprop through it. `fusion/token_fusion_torch.py`
  provides a drop-in, differentiable/trainable version of the exact same
  architecture in PyTorch for when you get there.
* After the transformer, each cluster's token sequence is pooled into a
  single fused token T̃_n via attention-pooling (a learned query attends
  over the transformed tokens) — a standard way to turn "N tokens in, 1
  token out" while still letting the model weigh members unequally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _xavier(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def _layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class TokenFusionConfig:
    dim: int = 256
    n_heads: int = 4
    ffn_mult: int = 4
    seed: int = 42


class TokenFusionBlock:
    """
    Self-attention transformer encoder layer + attention pooling.

    forward(tokens: (M, D)) -> fused: (D,), attn_weights: (M,)
    Works for any cluster size M >= 1 (M == 1 is a no-op passthrough, since a
    single-token "cluster" has nothing to attend to).
    """

    def __init__(self, config: TokenFusionConfig):
        self.cfg = config
        d, h = config.dim, config.n_heads
        assert d % h == 0, "dim must be divisible by n_heads"
        self.head_dim = d // h
        rng = np.random.default_rng(config.seed)

        # self-attention projections
        self.Wq = _xavier(rng, (d, d)); self.bq = np.zeros(d, dtype=np.float32)
        self.Wk = _xavier(rng, (d, d)); self.bk = np.zeros(d, dtype=np.float32)
        self.Wv = _xavier(rng, (d, d)); self.bv = np.zeros(d, dtype=np.float32)
        self.Wo = _xavier(rng, (d, d)); self.bo = np.zeros(d, dtype=np.float32)

        # feed-forward
        ff = d * config.ffn_mult
        self.W1 = _xavier(rng, (d, ff)); self.b1 = np.zeros(ff, dtype=np.float32)
        self.W2 = _xavier(rng, (ff, d)); self.b2 = np.zeros(d, dtype=np.float32)

        # attention-pooling query (learned "CLS"-style probe used to reduce M tokens -> 1)
        self.pool_q = _xavier(rng, (1, d))

    def _mha(self, x: np.ndarray) -> np.ndarray:
        m, d = x.shape
        h, hd = self.cfg.n_heads, self.head_dim
        q = (x @ self.Wq + self.bq).reshape(m, h, hd).transpose(1, 0, 2)  # (h, m, hd)
        k = (x @ self.Wk + self.bk).reshape(m, h, hd).transpose(1, 0, 2)
        v = (x @ self.Wv + self.bv).reshape(m, h, hd).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)                 # (h, m, m)
        attn = _softmax(scores, axis=-1)
        out = attn @ v                                                    # (h, m, hd)
        out = out.transpose(1, 0, 2).reshape(m, d)
        return out @ self.Wo + self.bo

    def forward(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(tokens, dtype=np.float32)
        m = tokens.shape[0]
        if m == 1:
            return tokens[0].copy(), np.array([1.0], dtype=np.float32)

        x = tokens
        x = x + self._mha(_layernorm(x))                 # pre-norm residual self-attention
        h = _layernorm(x)
        ff = _gelu(h @ self.W1 + self.b1) @ self.W2 + self.b2
        x = x + ff                                        # pre-norm residual FFN
        transformed = x                                   # (M, D) — "fused token clusters" pre-pool

        # attention-pooling: a learned query attends over the M transformed tokens
        q = self.pool_q                                   # (1, D)
        scores = (q @ transformed.T) / np.sqrt(self.cfg.dim)   # (1, M)
        weights = _softmax(scores, axis=-1)[0]             # (M,)
        fused = weights @ transformed                       # (D,)
        return fused.astype(np.float32), weights.astype(np.float32)


def fuse_all_clusters(
    embeddings: np.ndarray,
    clusters: List[np.ndarray],
    block: TokenFusionBlock,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Run the fusion block over every cluster.

    Returns
    -------
    fused_embeddings : (n_clusters, D)
    attn_weights_per_cluster : list of (M_n,) arrays, one per cluster
    """
    fused = np.zeros((len(clusters), embeddings.shape[1]), dtype=np.float32)
    weights_out: List[np.ndarray] = []
    for n, idx in enumerate(clusters):
        member_tokens = embeddings[idx]
        f, w = block.forward(member_tokens)
        fused[n] = f
        weights_out.append(w)
    return fused, weights_out
