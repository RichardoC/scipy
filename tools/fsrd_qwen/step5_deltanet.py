#!/usr/bin/env python
"""Spec step 5 (Rank-2 optional arm): DeltaNet vs full-attention token-time fits.

For 3 DeltaNet layers (block outputs 5, 9, 13) and 3 full-attention layers
(block outputs 4, 8, 12 — full_attention_interval=4, blocks 3/7/11 are full
attention), fit fSRD on the (1024, 128) token-time output matrix of the first
128 real tokens of 20 WikiText-2 prompts each, and compare in-window relative
error distributions DeltaNet-vs-attention at matched region count.

Two matched comparisons are recorded:
- max_depth=0 (global DMD, exactly 1 region — trivially matched), and
- max_depth=2 (theta=3.0, smoothness=0.03, oblique=False), grouped by the
  region count the solver actually returned.

Deliverable: per-layer error distributions + a boxplot (plots/step5_deltanet.png)
and a yes/no on "linear-attention outputs are measurably more piecewise-linear".

Usage:
    python step5_deltanet.py --states states/wt2 --out results/step5.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from exp_common import HERE, load_harvest, rel_err

DELTANET_LAYERS = [5, 9, 13]   # hidden-state indices (block outputs)
FULLATTN_LAYERS = [4, 8, 12]
N_PROMPTS = 20
T_WIN = 128

_G: dict = {}


def _init(states_prefix):
    h, mask, _ = load_harvest(states_prefix)
    _G["h"], _G["mask"] = h, mask
    import fsrd_standalone
    _G["fsrd"] = fsrd_standalone.fsrd


def _one(job):
    layer, p = job
    h, mask, fsrd = _G["h"], _G["mask"], _G["fsrd"]
    t_idx = np.where(mask[p])[0][:T_WIN]
    a = np.ascontiguousarray(h[p, layer, t_idx, :].T).astype(np.float64)  # (1024, T)
    rec = {"layer": layer, "p": p, "T": int(a.shape[1])}
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        r_g = fsrd(a, oblique=False, max_depth=0, theta=3.0, smoothness=0.03)
        r_m = fsrd(a, oblique=False, max_depth=2, theta=3.0, smoothness=0.03)
        rec["fit_s"] = round(time.perf_counter() - t0, 2)
    rec["err_glob"] = rel_err(a, r_g.reconstruction)
    rec["err_multi"] = rel_err(a, r_m.reconstruction)
    rec["n_regions"] = r_m.n_regions
    rec["n_warnings"] = len(wl)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="states/wt2")
    ap.add_argument("--out", default="results/step5.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(l, p) for l in DELTANET_LAYERS + FULLATTN_LAYERS for p in range(N_PROMPTS)]
    with mp.Pool(args.workers, initializer=_init, initargs=(args.states,)) as pool:
        recs = pool.map(_one, jobs, chunksize=1)

    def summarize(layers, key):
        errs = [r[key] for r in recs if r["layer"] in layers and np.isfinite(r[key])]
        return {"median": float(np.median(errs)), "mean": float(np.mean(errs)),
                "min": float(np.min(errs)), "max": float(np.max(errs)), "n": len(errs)}

    summary = {
        "deltanet_glob": summarize(DELTANET_LAYERS, "err_glob"),
        "fullattn_glob": summarize(FULLATTN_LAYERS, "err_glob"),
        "deltanet_multi": summarize(DELTANET_LAYERS, "err_multi"),
        "fullattn_multi": summarize(FULLATTN_LAYERS, "err_multi"),
    }
    out = {"records": recs, "summary": summary,
           "deltanet_layers": DELTANET_LAYERS, "fullattn_layers": FULLATTN_LAYERS}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(summary, indent=1))

    # boxplot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, key, title in [(axes[0], "err_glob", "global DMD (1 region)"),
                               (axes[1], "err_multi", "fSRD max_depth=2")]:
            data, labels = [], []
            for l in DELTANET_LAYERS + FULLATTN_LAYERS:
                data.append([r[key] for r in recs if r["layer"] == l and np.isfinite(r[key])])
                kind = "DN" if l in DELTANET_LAYERS else "FA"
                labels.append(f"{kind}{l}")
            ax.boxplot(data, tick_labels=labels)
            ax.set_title(title)
            ax.set_ylabel("in-window relative error")
        fig.suptitle("Token-time fSRD misfit: DeltaNet (DN) vs full-attention (FA) layer outputs")
        fig.tight_layout()
        os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
        fig.savefig(os.path.join(HERE, "plots", "step5_deltanet.png"), dpi=110)
        print("wrote plots/step5_deltanet.png")
    except ImportError:
        print("matplotlib unavailable; skipped plot")


if __name__ == "__main__":
    main()
