#!/usr/bin/env python3
"""Stream ffn_gate.weight for blocks 0..63 from the remote Q6_K GGUF, reduce each
to compact per-layer signatures, and delete the tensor.  Never holds more than
~0.7 GB.

Signatures (shared seeded projections, identical for every layer):
  raw[p]   = W_l @ q_p           (17408,)   p = 0..NPROBE-1   -- exact under W<-A W
  gram     = (W_l Q)^T (W_l Q)/n_ff   (K,K) -- invariant to neuron-axis perm/rotation

Also records, streaming, the consecutive-layer similarity of the *full* tensors.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_fetch as gf

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = "Qwen3.6-27B-Q6_K.gguf"
NPROBE = 4
KGRAM = 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-63")
    ap.add_argument("--kind", default="ffn_gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    out = a.out or os.path.join(HERE, "results",
                                f"pilot_layer_sig_{a.kind}_s{a.seed}.npz")

    client = gf.RangeClient(gf.HF_URL.format(repo=gf.DEFAULT_REPO, path=FILE))
    hdr = gf.fetch_header(client, os.path.join(HERE, ".gguf_cache"))
    byname = {t.name: t for t in hdr.tensors}

    t0 = gf.TensorInfo  # noqa: F841
    first = byname[f"blk.{layers[0]}.{a.kind}.weight"]
    n_ff, n_embd = first.np_shape
    print(f"[cfg] {a.kind} np_shape={first.np_shape} type={first.ggml_type.name} "
          f"bytes/tensor={first.n_bytes:,}", flush=True)

    # two independent projection seeds computed in the SAME streaming pass, so the
    # projection-seed robustness check costs no extra download.
    SEEDS = [a.seed, a.seed + 1]
    Qs, Ps = [], []
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        Qs.append(rng.standard_normal((n_embd, KGRAM)).astype(np.float32) / np.sqrt(n_embd))
        Ps.append(rng.standard_normal((n_embd, NPROBE)).astype(np.float32) / np.sqrt(n_embd))
    Q, P = Qs[0], Ps[0]

    raw = np.zeros((len(SEEDS), len(layers), NPROBE, n_ff), np.float32)
    gram = np.zeros((len(SEEDS), len(layers), KGRAM, KGRAM), np.float32)
    rownorm = np.zeros((len(layers), n_embd), np.float32)   # per-residual-dim col norm
    fro = np.zeros(len(layers))
    cos_next = np.full(len(layers), np.nan)     # cos(vec W_l, vec W_{l+1})
    rel_next = np.full(len(layers), np.nan)     # ||W_{l+1}-W_l||/||W_l||
    dl = np.zeros(len(layers))

    prev = None
    prev_idx = None
    tstart = time.time()
    for i, L in enumerate(layers):
        ti = byname[f"blk.{L}.{a.kind}.weight"]
        arr, got, secs = gf.fetch_tensor(client, ti, dtype="fp32")
        dl[i] = got
        assert arr.shape == (n_ff, n_embd), arr.shape

        for si in range(len(SEEDS)):
            raw[si, i] = (arr @ Ps[si]).T
            WQ = arr @ Qs[si]
            gram[si, i] = (WQ.T @ WQ) / n_ff
            del WQ
        rownorm[i] = np.linalg.norm(arr, axis=0)
        fro[i] = float(np.linalg.norm(arr))

        if prev is not None and prev_idx == L - 1:
            dot = 0.0
            dif = 0.0
            for s in range(0, n_ff, 2048):
                pc = prev[s:s + 2048].astype(np.float32)
                cc = arr[s:s + 2048]
                dot += float(np.sum(pc.astype(np.float64) * cc.astype(np.float64)))
                dif += float(np.sum((cc.astype(np.float64) - pc.astype(np.float64)) ** 2))
            cos_next[i - 1] = dot / (fro[i - 1] * fro[i])
            rel_next[i - 1] = np.sqrt(dif) / fro[i - 1]

        del prev
        prev = arr.astype(np.float16)
        prev_idx = L
        del arr
        gc.collect()
        print(f"[{i+1}/{len(layers)}] blk.{L} {got/2**20:.0f} MiB {secs:.1f}s "
              f"fro={fro[i]:.3f} cos_prev={cos_next[i-1] if i else float('nan'):+.4f} "
              f"elapsed={time.time()-tstart:.0f}s", flush=True)

    np.savez_compressed(out, layers=np.array(layers), raw=raw, gram=gram,
                        rownorm=rownorm, fro=fro, cos_next=cos_next,
                        rel_next=rel_next, seeds=np.array(SEEDS), kind=a.kind,
                        n_ff=n_ff, n_embd=n_embd)
    print(f"[done] {out}  downloaded={dl.sum()/2**30:.2f} GiB "
          f"in {time.time()-tstart:.0f}s", flush=True)
    print(json.dumps({
        "mean_cos_consecutive": float(np.nanmean(cos_next)),
        "min_cos": float(np.nanmin(cos_next)), "max_cos": float(np.nanmax(cos_next)),
        "mean_rel_next": float(np.nanmean(rel_next)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
