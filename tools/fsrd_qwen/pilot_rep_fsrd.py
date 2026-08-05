#!/usr/bin/env python
"""PILOT step 2/3: sliding-window fSRD spectra + cheap baselines, per pre-registration."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fsrd_standalone import fsrd  # noqa: E402

DATA = os.path.join(HERE, "pilot_rep_data")
W = 32          # window columns (>= 24 as registered)
K = 16          # POD rank -> fSRD rows
LAYERS = [12, 24]


def osc_score(res):
    """Registered oscillation score: amplitude-weighted sustained-oscillation evidence."""
    best = 0.0
    for reg in res.regions:
        ev = np.atleast_1d(reg.eigenvalues)
        if ev.size == 0:
            continue
        modes = np.atleast_2d(reg.modes)
        amps = np.atleast_1d(reg.amplitudes)
        n = min(ev.size, amps.size, modes.shape[1] if modes.ndim == 2 else 1)
        for i in range(n):
            w = ev[i]
            im = abs(w.imag)
            if not (0.05 <= im <= np.pi):
                continue
            nm = np.linalg.norm(modes[:, i]) if modes.ndim == 2 else 1.0
            a = abs(amps[i]) * nm * np.exp(-abs(w.real) * W)
            if np.isfinite(a) and a > best:
                best = float(a)
    return best


def spectra_summary(res):
    out = []
    for reg in res.regions:
        ev = np.atleast_1d(reg.eigenvalues)
        out.append(dict(bbox=[int(x) for x in reg.bounding_box],
                        re=[float(x) for x in ev.real],
                        im=[float(x) for x in ev.imag]))
    return out


def main():
    # argv: <meta.json> <tag> [shard] [nshards]
    metafile = sys.argv[1] if len(sys.argv) > 1 else "meta.json"
    tag = sys.argv[2] if len(sys.argv) > 2 else "r1"
    shard = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nsh = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    meta = json.load(open(os.path.join(DATA, metafile)))
    meta = [m for i, m in enumerate(meta) if i % nsh == shard]
    recs = []
    tfit = {"fsrd": [], "gdmd": [], "pod": []}
    for m in meta:
        gi, plen = m["gi"], m["plen"]
        ent = np.load(os.path.join(DATA, f"g{gi:03d}_ent.npy"))[plen:]
        maxp = np.load(os.path.join(DATA, f"g{gi:03d}_maxp.npy"))[plen:]
        ids = m["new_ids"]
        T = len(ids)

        # --- baseline (i): n-gram repeat count, output tokens only
        ngram = np.zeros(T)
        for t in range(T):
            best = 0
            for n in range(1, 9):
                if t + 1 - n < 0:
                    break
                g = tuple(ids[t + 1 - n:t + 1])
                c = sum(1 for s in range(0, t + 1 - n) if tuple(ids[s:s + n]) == g)
                if c > 0:
                    best = max(best, n * c)
            ngram[t] = best

        per_layer = {}
        for lay in LAYERS:
            h = np.load(os.path.join(DATA, f"g{gi:03d}_L{lay}.npy"))[plen:]  # (T, 1024)
            raw = h.T.astype(np.float64)                                     # (1024, T)
            mu = raw.mean(1, keepdims=True)
            sd = raw.std(1, keepdims=True) + 1e-8
            std = (raw - mu) / sd

            # --- baseline (ii): cosine self-similarity (raw and standardized)
            def cos_self(X):
                Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
                S = Xn.T @ Xn
                out = np.full(T, -1.0)
                for t in range(T):
                    lo = max(0, t - W)
                    hi = t - 1
                    if hi >= lo:
                        out[t] = S[t, lo:hi + 1].max()
                return out
            cos_raw = cos_self(raw)
            cos_std = cos_self(std)

            fs, gd, nreg, bnd = {}, {}, {}, {}
            spec = {}
            for t in range(W - 1, T):
                blk = std[:, t - W + 1:t + 1]
                t0 = time.perf_counter()
                U, s, Vt = np.linalg.svd(blk, full_matrices=False)
                a = (U[:, :K].T @ blk)
                tfit["pod"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                r2 = fsrd(a, dt=1.0, max_depth=2, oblique=False)
                tfit["fsrd"].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                r0 = fsrd(a, dt=1.0, max_depth=0, oblique=False)
                tfit["gdmd"].append(time.perf_counter() - t0)

                fs[t] = osc_score(r2)
                gd[t] = osc_score(r0)
                nreg[t] = int(r2.n_regions)
                # temporal region boundaries in absolute generated-token index
                cuts = set()
                for reg in r2.regions:
                    bb = reg.bounding_box
                    cs, ce = int(bb[2]), int(bb[3])
                    if cs > 0:
                        cuts.add(t - W + 1 + cs)
                    if ce < W:
                        cuts.add(t - W + 1 + ce)
                bnd[t] = sorted(cuts)
                if t in (W - 1, T - 1) or t % 32 == 0:
                    spec[t] = spectra_summary(r2)

            per_layer[str(lay)] = dict(
                cos_raw=cos_raw.tolist(), cos_std=cos_std.tolist(),
                fsrd={str(k): v for k, v in fs.items()},
                gdmd={str(k): v for k, v in gd.items()},
                nreg={str(k): v for k, v in nreg.items()},
                bounds={str(k): v for k, v in bnd.items()},
                spectra={str(k): v for k, v in spec.items()},
            )
        recs.append(dict(gi=gi, kind=m["kind"], label=m["label"], onset=m["onset"],
                         period=m["period"], T=T, ngram=ngram.tolist(),
                         entropy=ent.tolist(), maxprob=maxp.tolist(), layers=per_layer))
        print(f"[{gi:03d}] {m['label']} onset={m['onset']} done "
              f"({T-W+1} windows/layer)", flush=True)

    cost = {k: dict(mean_ms=float(np.mean(v) * 1e3), median_ms=float(np.median(v) * 1e3),
                    n=len(v)) for k, v in tfit.items()}
    json.dump(dict(W=W, K=K, layers=LAYERS, cost=cost, recs=recs),
              open(os.path.join(HERE, f"pilot_rep_raw_{tag}_{shard}.json"), "w"))
    print("cost:", json.dumps(cost, indent=1))


if __name__ == "__main__":
    sys.exit(main())
