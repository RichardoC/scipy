#!/usr/bin/env python
"""INVESTIGATION2.md section 5 (arm A), pipeline step 1 + baselines B0/B1:
per-tensor block-statistic maps, per-bit-level quantization error tables, and
the uniform baselines, for the 8 fetched Qwen3.6-27B tensors.

For each tensor W (rows = output channels, cols = input dims; fp16 on disk
from inv2a_fetch.py) with imatrix importance w_j = sqrt(in_sum2_j/counts):

  * sort columns by importance (desc) and rows by weighted RMS (desc);
    permutations are metadata charged to the adaptive methods' bit budgets;
  * quantize the *sorted* tensor at every level b in {2,3,4,5,6,8} with the
    Q_K-style codec (32-element blocks, asymmetric scale+min quantized to
    6 bits against per-256-superblock fp16 references => exactly b + 0.5
    bits/weight) and store the per-(row, 32-col-block) imatrix-weighted SSE
    table for each level -- these tables make every later allocation exact,
    not predicted;
  * build the fSRD block-statistic maps Z (log RMS of the importance-weighted
    32-wide block; weighted and unweighted variants) on the sorted axes;
  * B0 (status quo): uniform 4-bit on the UNSORTED tensor -> weighted
    rel-RMSE + unweighted rel-Frobenius (sanity: must land near GGUF's own
    Q4_K error, ~7.4% measured on real tensors);
  * B1 (sort only): uniform 4-bit on the sorted tensor.

Outputs: states/inv2a/<name>.tables.npz (large, gitignored) and
results/inv2a_prep.json (summary).
"""
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "states", "inv2a")
IM = os.path.join(BASE, "results", "inv2a_imatrix.npz")
OUT = os.path.join(BASE, "results", "inv2a_prep.json")

LEVELS = [2, 3, 4, 5, 6, 8]
BLK = 32          # elements per quantization block
SUP = 8           # blocks per superblock
SLAB = 4096       # rows per processing slab

TARGETS = [
    "blk.0.ffn_down.weight", "blk.20.ffn_down.weight", "blk.40.ffn_down.weight",
    "blk.0.ffn_gate.weight", "blk.20.ffn_gate.weight", "blk.40.ffn_gate.weight",
    "blk.3.attn_q.weight", "blk.1.attn_qkv.weight",
]


def log(*a):
    print(*a, flush=True)


def quant_dequant(X, b):
    """Q_K-style quantize+dequantize a (R, C) slab at b bits (C % 256 == 0).

    32-elem blocks: scale sc=(max-min)/(2^b-1) and min-magnitude m=-min, each
    quantized to 6 bits against per-superblock fp16 references d=max(sc)/63,
    dmin=max(m,0)/63 (the Q4_K layout, sans llama.cpp's iterative weighted
    search). Returns the dequantized slab.
    """
    R, C = X.shape
    nb = C // BLK
    ns = nb // SUP
    Xb = X.reshape(R, nb, BLK)
    mn = Xb.min(axis=2)
    mx = Xb.max(axis=2)
    qmax = (1 << b) - 1
    sc = (mx - mn) / qmax                     # (R, nb)
    m = -mn                                   # Q4_K min-magnitude convention
    d = sc.reshape(R, ns, SUP).max(axis=2) / 63.0
    dm = np.maximum(m.reshape(R, ns, SUP).max(axis=2), 0.0) / 63.0
    d = d.astype(np.float16).astype(np.float32)
    dm = dm.astype(np.float16).astype(np.float32)
    d_b = np.repeat(d, SUP, axis=1)           # per-block reference
    dm_b = np.repeat(dm, SUP, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        qsc = np.clip(np.round(np.where(d_b > 0, sc / d_b, 0.0)), 0, 63)
        qm = np.clip(np.round(np.where(dm_b > 0, m / dm_b, 0.0)), 0, 63)
    Sc = (qsc * d_b)[:, :, None]              # effective scale
    M = (qm * dm_b)[:, :, None]               # effective -min
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(Sc > 0, np.round((Xb + M) / Sc), 0.0)
    q = np.clip(q, 0, qmax)
    return (Sc * q - M).reshape(R, C)


def sse_table(X, w2, b):
    """Per-(row, block) weighted SSE of the level-b codec on (R, C) X.

    w2: per-column squared importance (length C). Also returns unweighted SSE
    total (for the B0 sanity check).
    """
    R, C = X.shape
    nb = C // BLK
    tab = np.empty((R, nb), dtype=np.float64)
    usse = 0.0
    for r0 in range(0, R, SLAB):
        sl = slice(r0, min(r0 + SLAB, R))
        Xs = X[sl].astype(np.float32)
        E = Xs - quant_dequant(Xs, b)
        usse += float((E.astype(np.float64) ** 2).sum())
        Ew = (E.astype(np.float64) ** 2) * w2[None, :]
        tab[sl] = Ew.reshape(Xs.shape[0], nb, BLK).sum(axis=2)
    return tab, usse


def main():
    im = np.load(IM)
    summary = {}
    for name in TARGETS:
        path = os.path.join(DIR, name + ".fp16.npy")
        if not os.path.exists(path):
            log(f"MISSING {path}; run inv2a_fetch.py first")
            continue
        t0 = time.time()
        W = np.load(path).astype(np.float32)
        R, C = W.shape
        # sanity anchor (binding rule 7): GGUF's own Q4_K dequant of this very
        # tensor, fetched in the earlier probe -- our B0 must land near it.
        q4k = os.path.join(BASE, "states", "gguf", name + ".Q4_K.npy")
        if os.path.exists(q4k):
            Q = np.load(q4k).astype(np.float64)
            relf = float(np.linalg.norm(W.astype(np.float64) - Q)
                         / np.linalg.norm(W))
            summary[f"_sanity_gguf_q4k_{name}_rel_frob"] = relf
            log(f"  sanity: GGUF Q4_K vs BF16 rel-Frobenius on {name} = {relf:.5f}")
            del Q
        w = im[name].astype(np.float64)
        assert w.size == C, (name, W.shape, w.size)
        w2 = w ** 2

        # ---- permutations (descending importance / weighted row RMS) ----
        col_perm = np.argsort(-w, kind="stable")
        Wc = W[:, col_perm]
        ws = w[col_perm]
        ws2 = ws ** 2
        rrms = np.sqrt(((W.astype(np.float64) * w[None, :]) ** 2).mean(axis=1))
        row_perm = np.argsort(-rrms, kind="stable")
        Wsrt = np.ascontiguousarray(Wc[row_perm])

        denom = float(((W.astype(np.float64) ** 2) * w2[None, :]).sum())
        ufro = float((W.astype(np.float64) ** 2).sum())

        # ---- B0: uniform 4-bit, unsorted (the status quo) ----
        tab0, usse0 = sse_table(W, w2, 4)
        b0_wsse = float(tab0.sum())
        del tab0

        # ---- per-level SSE tables on the sorted tensor ----
        tables = np.empty((len(LEVELS), R, C // BLK), dtype=np.float32)
        b1_wsse = None
        for li, b in enumerate(LEVELS):
            tab, _ = sse_table(Wsrt, ws2, b)
            tables[li] = tab.astype(np.float32)
            if b == 4:
                b1_wsse = float(tab.sum())
        del W, Wc

        # ---- Z maps (sorted axes; weighted and unweighted) ----
        nb = C // BLK
        Wd = Wsrt.astype(np.float64)
        zw = 0.5 * np.log(((Wd * ws[None, :]) ** 2)
                          .reshape(R, nb, BLK).mean(axis=2) + 1e-24)
        zu = 0.5 * np.log((Wd ** 2).reshape(R, nb, BLK).mean(axis=2) + 1e-24)
        del Wd, Wsrt

        np.savez(os.path.join(DIR, name + ".tables.npz"),
                 levels=np.array(LEVELS), tables=tables,
                 zmap_weighted=zw.astype(np.float32),
                 zmap_unweighted=zu.astype(np.float32),
                 col_perm=col_perm, row_perm=row_perm,
                 w_sorted=ws, denom=denom)
        # the exact SSE tables fully determine every later evaluation, so the
        # fetched tensor can be deleted now (binding disk rule: transient, not
        # resident). Set INV2A_KEEP=1 to retain for debugging.
        if not os.environ.get("INV2A_KEEP"):
            os.remove(path)
            log(f"  deleted {path}")

        ent = {
            "shape": [R, C], "n_weights": R * C,
            "b0_rel_rmse_weighted": float(np.sqrt(b0_wsse / denom)),
            "b0_rel_frob_unweighted": float(np.sqrt(usse0 / ufro)),
            "b1_rel_rmse_weighted": float(np.sqrt(b1_wsse / denom)),
            "denom": denom, "seconds": round(time.time() - t0, 1),
        }
        summary[name] = ent
        log(f"{name} {R}x{C}: B0 wRMSE={ent['b0_rel_rmse_weighted']:.5f} "
            f"(unw relF={ent['b0_rel_frob_unweighted']:.5f}) "
            f"B1 wRMSE={ent['b1_rel_rmse_weighted']:.5f} "
            f"[{ent['seconds']}s]")
        with open(OUT, "w") as f:      # checkpoint after every tensor
            json.dump(summary, f, indent=1)

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=1)
    log(f"DONE -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
