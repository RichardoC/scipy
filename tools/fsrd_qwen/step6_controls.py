#!/usr/bin/env python
"""Spec step 6: forecast negative controls (Rank-6 and Rank-5 falsification arms).

(a) Depth forecast: for N sampled (prompt, position) pairs, fit fSRD on the
    raw depth matrix restricted to layers 1-16 with forecast=8, take the
    forecast column for layer 24, and compare to the true h_24 (cosine and
    decoded-argmax agreement) against the identity baseline = logit-lens at
    layer 16 (i.e. using h_16 as the estimate of h_24).

(b) Token-time first-draft-token control: for 20 held-out WikiText-2 prompts
    (states/wt2ho — never seen by any ridge fit), 3 windows per prompt, fit
    fSRD on the (1024, 128) last-layer token-time window with forecast=4 and
    decode the first forecast column via final norm + LM head. Acceptance =
    the drafted token equals the full model's argmax at the true next position.
    Baselines: (i) identity (copy the window's last state), (ii) a corpus
    ridge next-state map h_t -> h_{t+1} trained on the main states/wt2 harvest
    (layer 24, all prompts).

All RuntimeWarnings and zero-tail (no-support forecast) events are caught and
counted. Pre-registered expectation: fSRD <= identity < ridge; an fSRD win
over ridge would force a Rank-5 revisit.

Usage:
    python step6_controls.py --out results/step6.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import warnings

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from exp_common import (HERE, LogitLens, depth_matrix, load_harvest, rel_err,
                        sample_pairs)
from step4_ridge_head import apply_ridge, fit_ridge

_G: dict = {}


def _init(prefix_main, prefix_ho):
    _G["h"], _G["mask"], _ = load_harvest(prefix_main)
    _G["hho"], _G["mho"], _ = load_harvest(prefix_ho)
    import fsrd_standalone
    _G["fsrd"] = fsrd_standalone.fsrd


def _depth_one(pair):
    p, t = pair
    fsrd = _G["fsrd"]
    a = depth_matrix(_G["h"], p, t)          # (1024, 24), layers 1..24
    a16 = np.ascontiguousarray(a[:, :16])    # layers 1..16
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        r = fsrd(a16, oblique=False, max_depth=2, theta=3.0, smoothness=0.03,
                 forecast=8)
    fc = r.reconstruction[:, 16:]            # forecast of layers 17..24
    zero_tail = bool(np.allclose(fc[:, -1], 0.0))
    return {"p": p, "t": t,
            "h24_hat": fc[:, -1].astype(np.float32),
            "h16": a[:, 15].astype(np.float32),
            "h24": a[:, 23].astype(np.float32),
            "finite": bool(np.all(np.isfinite(fc))),
            "zero_tail": zero_tail,
            "n_warnings": len(wl),
            "warn_types": sorted({w.category.__name__ for w in wl})}


def _tt_one(job):
    p, start = job
    fsrd = _G["fsrd"]
    hho, mho = _G["hho"], _G["mho"]
    t_idx = np.where(mho[p])[0]
    win = t_idx[start:start + 128]
    nxt = t_idx[start + 128]
    a = np.ascontiguousarray(hho[p, 24, win, :].T).astype(np.float64)  # (1024,128)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        r = fsrd(a, oblique=False, max_depth=2, theta=3.0, smoothness=0.03,
                 forecast=4)
    fc1 = r.reconstruction[:, 128]           # first forecast column
    return {"p": p, "start": int(start),
            "err_in_window": rel_err(a, r.reconstruction[:, :128]),
            "fsrd_hat": fc1.astype(np.float32),
            "h_last": a[:, -1].astype(np.float32),
            "h_true_next": hho[p, 24, nxt, :].astype(np.float32),
            "finite": bool(np.all(np.isfinite(fc1))),
            "zero_tail": bool(np.allclose(fc1, 0.0)),
            "n_warnings": len(wl),
            "warn_types": sorted({w.category.__name__ for w in wl})}


def cos(a, b):
    n = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    n[n == 0] = 1.0
    return (a * b).sum(-1) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="states/wt2")
    ap.add_argument("--holdout", default="states/wt2ho")
    ap.add_argument("--n-depth", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/step6.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    h, mask, _ = load_harvest(args.states)
    hho, mho, _ = load_harvest(args.holdout)

    depth_pairs = sample_pairs(mask, args.n_depth, seed=args.seed + 42)
    tt_jobs = [(p, s) for p in range(hho.shape[0]) for s in (0, 32, 64)]

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.states, args.holdout)) as pool:
        depth_recs = pool.map(_depth_one, depth_pairs, chunksize=4)
        tt_recs = pool.map(_tt_one, tt_jobs, chunksize=1)

    # ---- corpus ridge next-state map (trained on main harvest only) -------
    Xs, Ys = [], []
    for p in range(h.shape[0]):
        t_idx = np.where(mask[p])[0]
        s = h[p, 24, t_idx, :].astype(np.float64)
        Xs.append(s[:-1])
        Ys.append(s[1:])
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    n_val = len(X) // 10
    Wbest, best = None, None
    for lam in (1e-8, 1e-6, 1e-4, 1e-2):
        W = fit_ridge(X[:-n_val], Y[:-n_val], lam)
        mse = float(np.mean((apply_ridge(W, X[-n_val:]) - Y[-n_val:]) ** 2))
        if best is None or mse < best[1]:
            Wbest, best = W, (lam, mse)
    W = fit_ridge(X, Y, best[0])
    print(f"ridge next-state map: lam {best[0]:g}, n_train {len(X)}")

    lens = LogitLens()

    # ---- (a) depth control -------------------------------------------------
    Hhat = np.stack([r["h24_hat"] for r in depth_recs])
    H16 = np.stack([r["h16"] for r in depth_recs])
    H24 = np.stack([r["h24"] for r in depth_recs])
    truth = lens.argmax(H24)
    ok = np.array([r["finite"] and not r["zero_tail"] for r in depth_recs])
    Hhat_safe = np.where(np.isfinite(Hhat), Hhat, 0.0)
    depth_out = {
        "n": len(depth_recs),
        "n_nonfinite": int(sum(not r["finite"] for r in depth_recs)),
        "n_zero_tail": int(sum(r["zero_tail"] for r in depth_recs)),
        "n_warnings": int(sum(r["n_warnings"] for r in depth_recs)),
        "warn_types": sorted({t for r in depth_recs for t in r["warn_types"]}),
        "fsrd_cos_mean": float(np.mean(cos(Hhat_safe, H24))),
        "fsrd_cos_mean_valid": float(np.mean(cos(Hhat_safe, H24)[ok])) if ok.any() else None,
        "identity_cos_mean": float(np.mean(cos(H16, H24))),
        "fsrd_argmax_agree": float(np.mean(lens.argmax(Hhat_safe) == truth)),
        "identity_argmax_agree": float(np.mean(lens.argmax(H16) == truth)),
    }

    # ---- (b) token-time control -------------------------------------------
    F = np.stack([r["fsrd_hat"] for r in tt_recs])
    L = np.stack([r["h_last"] for r in tt_recs])
    T = np.stack([r["h_true_next"] for r in tt_recs])
    R = apply_ridge(W, L.astype(np.float64)).astype(np.float32)
    truth_tt = lens.argmax(T)
    F_safe = np.where(np.isfinite(F), F, 0.0)
    tt_out = {
        "n": len(tt_recs),
        "n_nonfinite": int(sum(not r["finite"] for r in tt_recs)),
        "n_zero_tail": int(sum(r["zero_tail"] for r in tt_recs)),
        "n_warnings": int(sum(r["n_warnings"] for r in tt_recs)),
        "warn_types": sorted({t for r in tt_recs for t in r["warn_types"]}),
        "in_window_err_median": float(np.nanmedian([r["err_in_window"] for r in tt_recs])),
        "fsrd_accept": float(np.mean(lens.argmax(F_safe) == truth_tt)),
        "identity_accept": float(np.mean(lens.argmax(L) == truth_tt)),
        "ridge_accept": float(np.mean(lens.argmax(R) == truth_tt)),
        "fsrd_cos_mean": float(np.mean(cos(F_safe, T))),
        "identity_cos_mean": float(np.mean(cos(L, T))),
        "ridge_cos_mean": float(np.mean(cos(R, T))),
        "ridge_lambda_rel": best[0],
    }

    out = {"depth_forecast": depth_out, "token_time": tt_out}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
