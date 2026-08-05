#!/usr/bin/env python
"""Spec step 2: fSRD depth fits + misfit gate + hyperparameter sweep.

For N sampled (prompt, position) pairs with mask=1, build the depth matrix
a = states[p, 1:, t, :].T of shape (1024, 24) (hidden-state index 0 — the raw
embedding — dropped), and fit, per sweep configuration,

    fsrd(a, oblique=False, max_depth=2, theta=THETA, smoothness=MU)   # multi
    fsrd(a, oblique=False, max_depth=0, theta=THETA, smoothness=MU)   # global DMD

over the pre-registered sweep theta in {1.5, 3.0} x smoothness in {0.03, 0.05}
x rows in {raw, standardized} (8 configs). The primary (gate) configuration is
theta=3.0, smoothness=0.03, standardized rows — exactly the call written in
INVESTIGATION.md section 5 step 2.

Records per fit: in-window relative Frobenius error (NaN when the blended
reconstruction is non-finite — counted as a solver event), interior column
boundaries, region count, both BICs, dominant-mode |Im(omega)|/2pi
(cycles/layer, descriptive only), and RuntimeWarning counts.

Kill gate (pre-registered, judged by the caller from this script's output):
median multi-region in-window relative error > 0.15 => H0 accepted.

Usage:
    python step2_depth_fsrd.py --states states/wt2 --n-pairs 400 --seed 0 \
        --out results/step2_wt2.json [--configs primary]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings

# Must precede the numpy import: OpenBLAS reads the thread count at load time,
# and multi-threaded BLAS in 4 forked workers on 4 cores spin-waits itself into
# a ~10x slowdown on these tiny (1024 x 24) SVDs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from exp_common import (HERE, depth_matrix, fsrd_col_boundaries, load_harvest,
                        rel_err, row_standardize, sample_pairs)

SWEEP = [
    # (theta, smoothness, standardize); first entry is the primary/gate config
    (3.0, 0.03, True),
    (3.0, 0.03, False),
    (3.0, 0.05, True),
    (3.0, 0.05, False),
    (1.5, 0.03, True),
    (1.5, 0.03, False),
    (1.5, 0.05, True),
    (1.5, 0.05, False),
]

_G: dict = {}


def _init(states_prefix: str):
    h, mask, meta = load_harvest(states_prefix)
    _G["h"] = h
    import fsrd_standalone
    _G["fsrd"] = fsrd_standalone.fsrd


def _dominant_cycles(res) -> float:
    """|Im(omega)|/2pi of the largest-amplitude mode across regions (descriptive)."""
    best_amp, best = -1.0, 0.0
    for reg in res.regions:
        if len(reg.amplitudes) == 0:
            continue
        k = int(np.argmax(np.abs(reg.amplitudes)))
        a = float(np.abs(reg.amplitudes[k]))
        if a > best_amp and np.isfinite(reg.eigenvalues[k]):
            best_amp, best = a, float(np.abs(np.imag(reg.eigenvalues[k])) / (2 * np.pi))
    return best


def _one(job):
    (p, t), (theta, mu, std) = job
    fsrd = _G["fsrd"]
    a = depth_matrix(_G["h"], p, t)
    if std:
        a = row_standardize(a)
    rec = {"p": p, "t": t, "theta": theta, "smoothness": mu, "std": std}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        r_m = fsrd(a, oblique=False, max_depth=2, theta=theta, smoothness=mu)
        r_g = fsrd(a, oblique=False, max_depth=0, theta=theta, smoothness=mu)
    rec["fit_s"] = round(time.perf_counter() - t0, 3)
    rec["err_multi"] = rel_err(a, r_m.reconstruction)
    rec["err_glob"] = rel_err(a, r_g.reconstruction)
    rec["n_regions"] = r_m.n_regions
    rec["bic_multi"] = float(r_m.bic)
    rec["bic_glob"] = float(r_g.bic)
    rec["col_bounds"] = fsrd_col_boundaries(r_m)
    rec["dom_cycles"] = _dominant_cycles(r_m)
    rec["n_warnings"] = len(wlist)
    rec["warn_types"] = sorted({w.category.__name__ for w in wlist})
    rec["nonfinite_recon"] = not np.isfinite(rec["err_multi"])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", choices=["all", "primary"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    _, mask, _ = load_harvest(args.states)
    pairs = sample_pairs(mask, args.n_pairs, seed=args.seed)
    configs = SWEEP if args.configs == "all" else SWEEP[:1]
    jobs = [(pr, cfg) for cfg in configs for pr in pairs]
    print(f"{len(pairs)} pairs x {len(configs)} configs = {len(jobs)} fits")

    t0 = time.perf_counter()
    with mp.Pool(args.workers, initializer=_init, initargs=(args.states,)) as pool:
        recs = pool.map(_one, jobs, chunksize=16)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    out = {"states": args.states, "n_pairs": len(pairs), "seed": args.seed,
           "sweep": [{"theta": c[0], "smoothness": c[1], "std": c[2]} for c in configs],
           "records": recs}
    os.makedirs(os.path.dirname(os.path.join(HERE, args.out)), exist_ok=True)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(out, fh)

    # Per-config summary table
    print(f"\n{'theta':>5} {'mu':>5} {'rows':>4} | {'med_err':>8} {'nan%':>5} "
          f"{'bic_win%':>8} {'med_nreg':>8} {'warn':>5}")
    for th, mu, std in configs:
        rs = [r for r in recs if r["theta"] == th and r["smoothness"] == mu and r["std"] == std]
        errs = np.array([r["err_multi"] for r in rs])
        nan_frac = np.mean(~np.isfinite(errs))
        med = np.median(np.where(np.isfinite(errs), errs, np.inf))
        bic_win = np.mean([r["bic_multi"] < r["bic_glob"] for r in rs])
        med_nreg = np.median([r["n_regions"] for r in rs])
        warn = sum(r["n_warnings"] for r in rs)
        print(f"{th:5.1f} {mu:5.2f} {'std' if std else 'raw':>4} | {med:8.3g} "
              f"{100 * nan_frac:5.1f} {100 * bic_win:8.1f} {med_nreg:8.1f} {warn:5d}")


if __name__ == "__main__":
    main()
