#!/usr/bin/env python
"""Shared helpers for the fSRD depth-segmentation experiment (INVESTIGATION.md section 5).

Provides: harvest loading, (prompt, position) pair sampling, depth-matrix
construction (dropping hidden-state index 0, the raw embedding), row
standardization, and a logit-lens decoder (final RMSNorm + tied-embedding LM
head) that never materializes more than a small batch of the 248,320-wide
logits array.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "Qwen3.5-0.8B")

# Hidden-state layout (harvest_hidden.py): axis 1 index 0 is the raw embedding
# output; index i>0 is the output of decoder block i-1. Depth matrices drop
# index 0 per the spec, so their columns are layers 1..24 (1-based block outputs).
N_LAYERS = 24
HIDDEN = 1024


def load_harvest(prefix: str):
    """Return (states_memmap, mask, meta) for a harvest prefix like 'states/wt2'."""
    path = prefix if os.path.isabs(prefix) else os.path.join(HERE, prefix)
    h = np.load(path + ".npy", mmap_mode="r")
    mask = np.load(path + "_mask.npy")
    with open(path + "_meta.json") as fh:
        meta = json.load(fh)
    return h, mask, meta


def sample_pairs(mask: np.ndarray, n: int, seed: int = 0) -> list[tuple[int, int]]:
    """Sample n distinct (prompt, position) pairs with mask True, deterministic."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(mask)
    sel = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
    return [tuple(map(int, idx[i])) for i in sel]


def depth_matrix(h: np.ndarray, p: int, t: int) -> np.ndarray:
    """(HIDDEN, N_LAYERS) float64 depth matrix for prompt p, position t.

    Column j is the layer-(j+1) output; hidden-state index 0 (raw embedding)
    is dropped per spec step 2.
    """
    return np.ascontiguousarray(h[p, 1:, t, :].T).astype(np.float64)


def row_standardize(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=1, keepdims=True)
    sd = a.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (a - mu) / sd


class LogitLens:
    """Final RMSNorm + tied-embedding LM head, applied in small batches.

    The 0.8B stand-in has tie_word_embeddings=True, so the LM head weight is
    model.language_model.embed_tokens.weight (248320 x 1024). Logits for a
    batch of B states are (B, 248320) — kept only long enough to reduce to
    argmax / KL, per the harness rules.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        import torch
        from safetensors import safe_open

        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tcfg = cfg.get("text_config", cfg)
        self.eps = float(tcfg.get("rms_norm_eps", 1e-6))
        st_file = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
        with safe_open(st_file, framework="pt") as f:
            self.norm_w = f.get_tensor("model.language_model.norm.weight").to(torch.float32)
            self.emb = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.float32)
        self.torch = torch

    def _norm(self, x):
        torch = self.torch
        v = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return v * self.norm_w

    def argmax(self, states: np.ndarray, batch: int = 512) -> np.ndarray:
        """Top-1 token ids for an (N, HIDDEN) array of final-residual states."""
        torch = self.torch
        out = np.empty(len(states), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(states), batch):
                x = torch.from_numpy(np.ascontiguousarray(states[i:i + batch])).to(torch.float32)
                logits = self._norm(x) @ self.emb.T
                out[i:i + batch] = logits.argmax(-1).numpy()
                del logits
        return out

    def log_softmax(self, states: np.ndarray, batch: int = 256):
        """Yield (start, log_probs) chunks; caller must reduce immediately."""
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(states), batch):
                x = torch.from_numpy(np.ascontiguousarray(states[i:i + batch])).to(torch.float32)
                logits = self._norm(x) @ self.emb.T
                yield i, torch.log_softmax(logits, dim=-1)
                del logits


def fsrd_col_boundaries(res) -> list[int]:
    """Distinct interior column-split positions of an fSRD result.

    A returned value c means a boundary between column c-1 and column c of the
    input matrix; for depth matrices (col j = layer j+1) that is a boundary
    between layer c and layer c+1 in 1-based block numbering.
    """
    ncols = res.reconstruction.shape[1]
    cuts = set()
    for reg in res.regions:
        _, _, c0, c1 = reg.bounding_box
        if c0 > 0:
            cuts.add(int(c0))
        if c1 < ncols:
            cuts.add(int(c1))
    return sorted(cuts)


def rel_err(a: np.ndarray, rec: np.ndarray) -> float:
    """Relative Frobenius reconstruction error; NaN if rec is non-finite."""
    if not np.all(np.isfinite(rec)):
        return float("nan")
    return float(np.linalg.norm(a - rec) / np.linalg.norm(a))
