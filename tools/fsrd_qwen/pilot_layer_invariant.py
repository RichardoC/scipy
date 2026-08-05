#!/usr/bin/env python3
"""Basis-invariant per-layer signature track (no refetch: reuses the .npz).

Signatures, all invariant to any rotation/permutation of the FFN-neuron axis (the
axis with no cross-layer canonical alignment):
  spec      log eigenvalues of the projected Gram (W_l Q)^T(W_l Q)/n_ff  -> 64 values
            (a random-projection estimate of the top singular-value spectrum)
  quant     log of 65 fixed quantiles of the per-residual-dim column norms of W_l
  composite z-scored spec  ++  z-scored quant  ++  log ||W_l||_F
Same held-out protocol / baselines / thresholds as pilot_layer_fsrd.py.
Also reports whether fSRD's region boundaries align with the period-4 structure.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_layer_fsrd as P

HERE = os.path.dirname(os.path.abspath(__file__))


def zs(a):
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True) + 1e-12
    return (a - m) / s


def consec_cos(X):
    c = [float(np.dot(X[:, i], X[:, i + 1]) /
               (np.linalg.norm(X[:, i]) * np.linalg.norm(X[:, i + 1])))
         for i in range(X.shape[1] - 1)]
    Xc = X - X.mean(axis=1, keepdims=True)
    cc = [float(np.dot(Xc[:, i], Xc[:, i + 1]) /
                (np.linalg.norm(Xc[:, i]) * np.linalg.norm(Xc[:, i + 1])))
          for i in range(X.shape[1] - 1)]
    return float(np.mean(c)), float(np.mean(cc))


def main() -> int:
    npz = np.load(os.path.join(HERE, "results", "pilot_layer_sig_ffn_gate_s0.npz"))
    gram, rownorm, fro = npz["gram"], npz["rownorm"], npz["fro"]
    nseed, L = gram.shape[0], gram.shape[1]
    qs = np.linspace(0.0, 1.0, 65)

    quant = np.log(np.quantile(rownorm, qs, axis=1) + 1e-20)          # (65, L)
    out = {"config": dict(cuts=P.CUTS, horizon=P.H, rank=P.RANK, n_layers=int(L),
                          note="basis-invariant track; blocks 0..63 of ffn_gate")}
    res, sig_cos = {}, {}
    for si in range(nseed):
        ev = np.zeros((gram.shape[2], L))
        for i in range(L):
            G = gram[si, i].astype(np.float64)
            w = np.linalg.eigvalsh((G + G.T) / 2)[::-1]
            ev[:, i] = np.log(np.clip(w, 1e-12, None))
        comp = np.vstack([zs(ev), zs(quant), np.log(fro)[None, :]])
        for nm, X in (("spec", ev), ("quant", quant), ("composite", comp)):
            tag = f"{nm}_seed{si}"
            sig_cos[tag] = consec_cos(X)
            print(f"[sim] {tag}: consecutive cos {sig_cos[tag][0]:.4f} "
                  f"(mean-centered {sig_cos[tag][1]:.4f})", flush=True)
            P.eval_signature(X, tag, res)

    out["signature_consecutive_cos"] = {k: dict(raw=v[0], centered=v[1])
                                        for k, v in sig_cos.items()}
    out["runs"] = res
    out["summary"] = {k: P.summarize(v) for k, v in res.items()}
    nreg = [r["info"]["fsrd_d2"].get("n_regions") for v in res.values() for r in v]
    nreg = [n for n in nreg if n is not None]
    out["degeneracy"] = dict(n_fits=len(nreg),
                             frac_single_region=float(np.mean([n == 1 for n in nreg])),
                             counts={str(k): int(sum(1 for n in nreg if n == k))
                                     for k in sorted(set(nreg))})
    verdict = {}
    for tag, a in out["summary"].items():
        base = min(a["copy"], a["mean"], a["dmd_global"])
        verdict[tag] = dict(fsrd_d2=a["fsrd_d2"], fsrd_d1=a["fsrd_d1"],
                            best_baseline=base, ratio=a["fsrd_d2"] / base,
                            period4_dmd=a["period4_dmd"],
                            passes_go=bool(a["fsrd_d2"] <= 0.75 * base
                                           and min(base, a["fsrd_d2"]) < 0.5))
    out["verdict"] = verdict

    # region-boundary alignment with the period-4 full_attention structure
    bnds = []
    for v in res.values():
        for r in v:
            for t in ("fsrd_d1", "fsrd_d2"):
                for bx in r["info"][t].get("boxes", []):
                    c0 = int(round(bx[2]))
                    if c0 > 0:
                        bnds.append(c0)
    mods = np.array([b % 4 for b in bnds])
    out["boundary_period4"] = dict(
        n_boundaries=len(bnds),
        mod4_hist={str(m): int((mods == m).sum()) for m in range(4)},
        frac_mod4_eq3=float((mods == 3).mean()),
        chance=0.25, boundaries=sorted(set(bnds)))
    print("[bnd] " + json.dumps(out["boundary_period4"]["mod4_hist"]) +
          f"  frac==3: {out['boundary_period4']['frac_mod4_eq3']:.3f} (chance 0.25)")

    p = os.path.join(HERE, "results", "pilot_layer_invariant.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== invariant track: mean rel-L2 over the 4 cuts ===")
    for k, a in out["summary"].items():
        print(f"  {k:16s} " + " ".join(f"{kk}={vv:.4f}" for kk, vv in a.items()))
    print("\n" + json.dumps(dict(degeneracy=out["degeneracy"], verdict=verdict), indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
