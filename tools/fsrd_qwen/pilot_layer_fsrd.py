#!/usr/bin/env python3
"""fSRD on the layer-index sequence of ffn_gate weight signatures.

Held-out protocol (pre-registered):
  for T0 in CUTS: train on layers 0..T0-1, forecast H=4, score layers T0..T0+3
  metric: ||pred-true||_F / ||true||_F in the ORIGINAL signature space
  methods: copy, mean, global DMD (max_depth=0), fSRD (max_depth=2), period-4 split,
           thirds split
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsrd_standalone import fsrd

HERE = os.path.dirname(os.path.abspath(__file__))
CUTS = [40, 48, 56, 60]
H = 4
RANK = 16
FSRD_KW = dict(dt=1.0, theta=1.5, rcond=1e-4, eta=1e-4, smoothness=0.05,
               oblique=False, prune=True)


def relerr(pred, true):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def pod(train, rank):
    """Left-singular basis of the training columns (no test leakage)."""
    mu = train.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(train - mu, full_matrices=False)
    r = min(rank, U.shape[1])
    energy = float((s[:r] ** 2).sum() / (s ** 2).sum())
    return U[:, :r], mu, energy


def dmd_forecast(z, h, max_depth):
    """fsrd on reduced coords z (r, t); returns forecast block (r, h) and info."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fsrd(z, max_depth=max_depth, forecast=h, **FSRD_KW)
    rec = np.asarray(res.reconstruction)
    if np.iscomplexobj(rec):
        rec = rec.real
    return rec[:, -h:], res


def region_boundaries(res, t):
    """Column boundaries of the fitted regions, in column index units."""
    b = []
    for reg in res.regions:
        bb = np.asarray(reg.bounding_box, dtype=float).ravel()
        b.append(bb.tolist())
    return b


def eval_signature(X, name, out):
    """X: (D, L) signature matrix, columns = layers in order."""
    D, L = X.shape
    rows = []
    for T0 in CUTS:
        train = X[:, :T0]
        true = X[:, T0:T0 + H]
        U, mu, energy = pod(train, RANK)
        z = U.T @ (train - mu)

        def lift(zf):
            return U @ zf + mu

        rec = {}
        rec["copy"] = relerr(np.repeat(train[:, -1:], H, axis=1), true)
        rec["mean"] = relerr(np.repeat(train.mean(axis=1, keepdims=True), H, axis=1), true)

        info = {}
        for tag, depth in (("dmd_global", 0), ("fsrd_d1", 1), ("fsrd_d2", 2)):
            t0 = time.time()
            try:
                zf, res = dmd_forecast(z, H, depth)
                rec[tag] = relerr(lift(zf), true)
                info[tag] = dict(n_regions=int(res.n_regions), bic=float(res.bic),
                                 secs=round(time.time() - t0, 2),
                                 boxes=region_boundaries(res, T0),
                                 lam_absmax=float(max(
                                     (np.abs(np.asarray(r.eigenvalues)).max()
                                      for r in res.regions if
                                      np.asarray(r.eigenvalues).size), default=np.nan)))
            except Exception as e:                      # noqa: BLE001
                rec[tag] = float("nan")
                info[tag] = dict(error=f"{type(e).__name__}: {e}")

        # period-4 split: separate global DMD per residue class l mod 4
        pred = np.zeros_like(true)
        ok = True
        for ph in range(4):
            idx = [i for i in range(T0) if i % 4 == ph]
            tgt = [j for j in range(H) if (T0 + j) % 4 == ph]
            if not tgt:
                continue
            sub = X[:, idx]
            Us, mus, _ = pod(sub, RANK)
            zs = Us.T @ (sub - mus)
            steps = max((T0 + j - idx[-1]) // 4 for j in tgt)
            try:
                zf, _ = dmd_forecast(zs, steps, 0)
                for j in tgt:
                    k = (T0 + j - idx[-1]) // 4
                    pred[:, j] = (Us @ zf[:, k - 1] + mus.ravel())
            except Exception:                            # noqa: BLE001
                ok = False
        rec["period4_dmd"] = relerr(pred, true) if ok else float("nan")

        # thirds split: global DMD on the last contiguous third of training only
        third = T0 - T0 // 3
        sub = X[:, third:]
        Us, mus, _ = pod(sub, RANK)
        try:
            zf, _ = dmd_forecast(Us.T @ (sub - mus), H, 0)
            rec["thirds_dmd"] = relerr(Us @ zf + mus, true)
        except Exception:                                # noqa: BLE001
            rec["thirds_dmd"] = float("nan")

        rows.append(dict(T0=T0, pod_energy=energy, err=rec, info=info))
        print(f"[{name}] T0={T0} pod_energy={energy:.3f} " +
              " ".join(f"{k}={v:.4f}" for k, v in rec.items()) +
              "  nreg=" + ",".join(str(info[t].get("n_regions", "x"))
                                   for t in ("dmd_global", "fsrd_d1", "fsrd_d2")),
              flush=True)
    out[name] = rows
    return rows


def summarize(rows):
    keys = list(rows[0]["err"])
    return {k: float(np.nanmean([r["err"][k] for r in rows])) for k in keys}


def main() -> int:
    npz = np.load(os.path.join(HERE, "results", "pilot_layer_sig_ffn_gate_s0.npz"))
    layers = npz["layers"]
    out = {"config": dict(cuts=CUTS, horizon=H, rank=RANK, kind=str(npz["kind"]),
                          seeds=npz["seeds"].tolist(), n_layers=int(len(layers)),
                          fsrd_kw={k: v for k, v in FSRD_KW.items()})}

    cos_next = npz["cos_next"]
    rel_next = npz["rel_next"]
    out["consecutive_similarity"] = dict(
        mean_cos=float(np.nanmean(cos_next)),
        median_cos=float(np.nanmedian(cos_next)),
        min_cos=float(np.nanmin(cos_next)), max_cos=float(np.nanmax(cos_next)),
        mean_rel_frob=float(np.nanmean(rel_next)),
        cos_by_layer=[None if np.isnan(v) else round(float(v), 5) for v in cos_next],
        fro=npz["fro"].tolist(),
    )
    print("[sim] mean cos(vec W_l, vec W_l+1) = "
          f"{out['consecutive_similarity']['mean_cos']:.5f}  "
          f"range [{out['consecutive_similarity']['min_cos']:.5f}, "
          f"{out['consecutive_similarity']['max_cos']:.5f}]  "
          f"mean rel-Frob step = {out['consecutive_similarity']['mean_rel_frob']:.3f}",
          flush=True)

    raw = npz["raw"]      # (nseed, L, nprobe, n_ff)
    gram = npz["gram"]    # (nseed, L, K, K)
    nseed, L, nprobe, n_ff = raw.shape

    # signature-space consecutive cosine (Gram), reported alongside
    g = gram[0].reshape(L, -1).T
    gc_ = [float(np.dot(g[:, i], g[:, i + 1]) /
                 (np.linalg.norm(g[:, i]) * np.linalg.norm(g[:, i + 1])))
           for i in range(L - 1)]
    gcen = g - g.mean(axis=1, keepdims=True)
    gcc = [float(np.dot(gcen[:, i], gcen[:, i + 1]) /
                 (np.linalg.norm(gcen[:, i]) * np.linalg.norm(gcen[:, i + 1])))
           for i in range(L - 1)]
    out["consecutive_similarity"]["gram_mean_cos"] = float(np.mean(gc_))
    out["consecutive_similarity"]["gram_mean_cos_centered"] = float(np.mean(gcc))
    print(f"[sim] Gram signature consecutive cos = {np.mean(gc_):.4f} "
          f"(mean-centered {np.mean(gcc):.4f})", flush=True)

    res = {}
    for si in range(nseed):
        eval_signature(gram[si].reshape(L, -1).T, f"gram_seed{si}", res)
        for p in range(nprobe):
            eval_signature(raw[si, :, p, :].T, f"raw_seed{si}_probe{p}", res)

    out["runs"] = res
    out["summary"] = {k: summarize(v) for k, v in res.items()}

    # aggregate raw over probes per seed
    agg = {}
    for si in range(nseed):
        for fam, pat in (("gram", f"gram_seed{si}"), ("raw", f"raw_seed{si}_probe")):
            names = [n for n in res if n.startswith(pat)]
            keys = list(res[names[0]][0]["err"])
            agg[f"{fam}_seed{si}"] = {
                k: float(np.nanmean([r["err"][k] for n in names for r in res[n]]))
                for k in keys}
    out["aggregate"] = agg

    # degeneracy check
    nreg = [r["info"]["fsrd_d2"].get("n_regions") for v in res.values() for r in v]
    nreg = [n for n in nreg if n is not None]
    out["degeneracy"] = dict(n_fits=len(nreg), frac_single_region=
                             float(np.mean([n == 1 for n in nreg])),
                             counts={str(k): int(sum(1 for n in nreg if n == k))
                                     for k in sorted(set(nreg))})

    # verdict against pre-registered thresholds
    verdict = {}
    for tag, a in agg.items():
        base = min(a["copy"], a["mean"], a["dmd_global"])
        verdict[tag] = dict(fsrd_d2=a["fsrd_d2"], fsrd_d1=a["fsrd_d1"],
                            best_baseline=base, ratio=a["fsrd_d2"] / base,
                            passes_go=bool(a["fsrd_d2"] <= 0.75 * base and
                                           min(base, a["fsrd_d2"]) < 0.5))
    out["verdict"] = verdict

    p = os.path.join(HERE, "results", "pilot_layer_weights.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== aggregate (mean rel-L2 over cuts x probes) ===")
    for k, v in agg.items():
        print(f"  {k:14s} " + " ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()))
    print("\n=== verdict ===")
    print(json.dumps(dict(degeneracy=out["degeneracy"], verdict=verdict), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
