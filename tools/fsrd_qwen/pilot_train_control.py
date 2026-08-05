"""PILOT T4: negative control on fSRD temporal-boundary informativeness.

Are the cross-seed-consistent boundaries data-dependent, or forced by fSRD's
fixed split-fraction lattice (_FRACS = 0.35/0.5/0.65 of each node's extent)?
"""
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsrd_standalone import fsrd
from pilot_train_fsrd import temporal_boundaries, load, HERE

REAL = [46, 169, 355]          # the 5/5-consistent real-data boundaries
TOL = 25


def fit(a, tag):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fsrd(a, dt=1.0, max_depth=3, oblique=False)
    b, nt, ns = temporal_boundaries(r, a.shape[1])
    rel = float(np.linalg.norm(np.real(r.reconstruction) - a) / np.linalg.norm(a))
    hits = sum(1 for x in REAL if min(abs(x - y) for y in b) <= TOL) if b else 0
    print(f"{tag}: nreg={r.n_regions} bnds={b} inwin={rel:.3g} "
          f"real-hits={hits}/3", flush=True)
    return dict(tag=tag, n_regions=int(r.n_regions), boundaries=b,
                inwin=rel, real_hits=hits)


def main():
    res = []
    a0 = load(0)["a"]
    T = a0.shape[1]

    # (i) column-permuted real data
    for sd in (1, 2, 3):
        rs = np.random.RandomState(sd)
        res.append(fit(a0[:, rs.permutation(T)], f"shuffled_cols_rs{sd}"))

    # (ii) structureless surrogate: smooth monotone drift + iid noise,
    #      per-row mean/scale matched to the real matrix, no regimes at all.
    mu = a0.mean(axis=1, keepdims=True)
    d = a0[:, -1:] - a0[:, :1]
    step_sd = np.diff(a0, axis=1).std(axis=1, keepdims=True)
    ramp = np.linspace(0.0, 1.0, T)[None, :]
    for sd in (1, 2, 3):
        rs = np.random.RandomState(100 + sd)
        s = mu + d * (ramp - 0.5) + step_sd * rs.standard_normal((a0.shape[0], T))
        res.append(fit(s, f"surrogate_rs{sd}"))

    # (iii) drift + brownian motion, still no regime change -- the closest
    #       structureless analogue of an optimiser trajectory
    for sd in (1, 2):
        rs = np.random.RandomState(200 + sd)
        w = np.cumsum(step_sd * rs.standard_normal((a0.shape[0], T)), axis=1)
        s = mu + d * (ramp - 0.5) + w
        res.append(fit(s, f"randomwalk_rs{sd}"))

    hits = [r["real_hits"] for r in res]
    out = dict(real_boundaries=REAL, tol=TOL, fits=res,
               max_real_hits=max(hits), median_real_hits=float(np.median(hits)),
               t4_consistency_void=bool(max(hits) >= 3))
    with open(os.path.join(HERE, "pilot_train_control.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("max_real_hits", "median_real_hits",
                       "t4_consistency_void")}, indent=1))


if __name__ == "__main__":
    main()
