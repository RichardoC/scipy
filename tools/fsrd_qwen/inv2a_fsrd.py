#!/usr/bin/env python
"""INVESTIGATION2.md section 5 (arm A), pipeline step 2: the fSRD fits.

For each of the 8 tensors, run fsrd(Z_sorted, oblique=False, theta=3,
smoothness=0.03) over the pre-registered sweep -- max_depth in {2, 3}, both
sort orders (ascending/descending on both axes = a 180-degree flip of the
map), weighted/unweighted block-statistic map -- 8 configs x 8 tensors = 64
fits. Consumes ONLY region bounding boxes + BIC region count, exactly as the
spec limits (the sweep is the entire hyperparameter allowance; nothing else
is tuned).

Leaf boxes are recorded in ascending-sort coordinates. Run AFTER arm B has
finished: a fit at this shape is 68 s uncontended but 486 s under CPU
contention (measured, inv2_fsrd_timing.log).

Output: results/inv2a_fsrd.json
"""
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from fsrd_standalone import fsrd  # noqa: E402

DIR = os.path.join(BASE, "states", "inv2a")
OUT = os.path.join(BASE, "results", "inv2a_fsrd.json")

TARGETS = [
    "blk.0.ffn_down.weight", "blk.20.ffn_down.weight", "blk.40.ffn_down.weight",
    "blk.0.ffn_gate.weight", "blk.20.ffn_gate.weight", "blk.40.ffn_gate.weight",
    "blk.3.attn_q.weight", "blk.1.attn_qkv.weight",
]
THETA, MU = 3.0, 0.03


def log(*a):
    print(*a, flush=True)


def main():
    results = {}
    for name in TARGETS:
        npz = np.load(os.path.join(DIR, name + ".tables.npz"))
        results[name] = {}
        for wkey, zname in (("weighted", "zmap_weighted"),
                            ("unweighted", "zmap_unweighted")):
            Zasc = npz[zname].astype(np.float64)
            R, C = Zasc.shape
            for order in ("asc", "desc"):
                Z = Zasc if order == "asc" else Zasc[::-1, ::-1].copy()
                for depth in (2, 3):
                    cfg = f"{wkey}_{order}_d{depth}"
                    t0 = time.time()
                    res = fsrd(Z, oblique=False, max_depth=depth,
                               theta=THETA, smoothness=MU)
                    dt = time.time() - t0
                    boxes = []
                    for reg in res.regions:
                        r0, r1, c0, c1 = map(int, reg.bounding_box)
                        if order == "desc":     # map back to asc coordinates
                            r0, r1 = R - r1, R - r0
                            c0, c1 = C - c1, C - c0
                        boxes.append([r0, r1, c0, c1, int(reg.level)])
                    rec = np.asarray(res.reconstruction, dtype=np.float64)
                    finite = bool(np.all(np.isfinite(rec)))
                    relerr = (float(np.linalg.norm(Z - rec) / np.linalg.norm(Z))
                              if finite else None)
                    results[name][cfg] = {
                        "n_regions": int(res.n_regions), "bic": float(res.bic),
                        "boxes_asc": boxes, "map_shape": [R, C],
                        "recon_rel_err": relerr, "recon_finite": finite,
                        "seconds": round(dt, 1),
                    }
                    log(f"{name} {cfg}: regions={res.n_regions} "
                        f"bic={res.bic:.1f} relerr={relerr} [{dt:.0f}s]")
                    with open(OUT, "w") as f:   # checkpoint after every fit
                        json.dump(results, f, indent=1)
    log(f"DONE -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
