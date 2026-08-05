#!/usr/bin/env python
"""Spec step 4: closed-form ridge exit head, base model frozen.

Trains W in R^(1025 x 1024) (ridge regression with bias): h_l -> h_24 on
harvested positions from the training prompts, for each requested attach layer
l. The metric is top-1 agreement of lm_head(final_norm(W h_l)) with the full
model's next-token argmax lm_head(final_norm(h_24)) on ~5k held-out positions.

fSRD's only contribution to this step is the choice of one attach layer (the
modal depth boundary l* from step 2); everything else is ordinary ridge
regression, identical across layers.

Design guards (binding rule 6):
- The train/test split is by *prompt* (no position of a test prompt ever enters
  the ridge fit), controlled by --split-seed for the required re-run.
- The ridge lambda is selected on an inner validation split of the training
  prompts only.
- Logits are computed in batches; only argmax is kept.

Usage:
    python step4_ridge_head.py --states states/wt2 --layers 6 12 16 18 --lstar 13 \
        --split-seed 0 --out results/step4_seed0.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from exp_common import HERE, HIDDEN, LogitLens, load_harvest


def gather_positions(h, mask, prompts, layer, max_n=None, rng=None):
    """(N, 1024) states at `layer` (1-based block output) and at layer 24."""
    xs, ys = [], []
    for p in prompts:
        t_idx = np.where(mask[p])[0]
        xs.append(h[p, layer, t_idx, :])
        ys.append(h[p, 24, t_idx, :])
    X = np.concatenate(xs).astype(np.float64)
    Y = np.concatenate(ys).astype(np.float64)
    if max_n is not None and len(X) > max_n:
        sel = (rng or np.random.default_rng(0)).choice(len(X), max_n, replace=False)
        X, Y = X[sel], Y[sel]
    return X, Y


def fit_ridge(X, Y, lam_rel):
    """Ridge with bias; lam is lam_rel * mean eigenvalue scale of X'X."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    G = Xb.T @ Xb
    scale = np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam_rel * scale * np.eye(G.shape[0]), Xb.T @ Y)
    return W  # (1025, 1024)


def apply_ridge(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="states/wt2")
    ap.add_argument("--layers", type=int, nargs="+", required=True,
                    help="attach layers to evaluate (1-based block outputs)")
    ap.add_argument("--lstar", type=int, required=True,
                    help="the fSRD-chosen layer (must be in --layers; recorded)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    h, mask, _ = load_harvest(args.states)
    P = h.shape[0]
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(P)
    test_prompts = sorted(int(x) for x in perm[:20])
    train_prompts = sorted(int(x) for x in perm[20:])
    # inner validation split for lambda selection (train prompts only)
    val_prompts = train_prompts[-10:]
    fit_prompts = train_prompts[:-10]

    lens = LogitLens()
    layers = sorted(set(args.layers) | {args.lstar})

    # held-out evaluation positions (same for every layer)
    rng_eval = np.random.default_rng(args.split_seed + 1000)
    _, Y_test_full = gather_positions(h, mask, test_prompts, 24)
    n_test = min(args.n_test, len(Y_test_full))
    eval_sel = rng_eval.choice(len(Y_test_full), n_test, replace=False)
    truth = lens.argmax(Y_test_full[eval_sel].astype(np.float32))

    results = {}
    lam_grid = [1e-8, 1e-6, 1e-4, 1e-2]
    for layer in layers:
        Xf, Yf = gather_positions(h, mask, fit_prompts, layer)
        Xv, Yv = gather_positions(h, mask, val_prompts, layer)
        best = None
        for lam in lam_grid:
            W = fit_ridge(Xf, Yf, lam)
            mse = float(np.mean((apply_ridge(W, Xv) - Yv) ** 2))
            if best is None or mse < best[1]:
                best = (lam, mse)
        lam = best[0]
        # final fit on all training prompts with the selected lambda
        Xt, Yt = gather_positions(h, mask, train_prompts, layer)
        W = fit_ridge(Xt, Yt, lam)

        Xe, Ye = gather_positions(h, mask, test_prompts, layer)
        Xe, Ye = Xe[eval_sel], Ye[eval_sel]
        pred_states = apply_ridge(W, Xe).astype(np.float32)
        pred = lens.argmax(pred_states)
        agree = float(np.mean(pred == truth))
        # descriptive extras: raw logit-lens agreement at this layer, cosine
        lens_pred = lens.argmax(Xe.astype(np.float32))
        cosine = float(np.mean(
            (pred_states * Ye).sum(-1)
            / (np.linalg.norm(pred_states, axis=-1) * np.linalg.norm(Ye, axis=-1))))
        results[layer] = {
            "lambda_rel": lam, "val_mse": best[1],
            "top1_agreement": agree,
            "logit_lens_agreement": float(np.mean(lens_pred == truth)),
            "cosine_to_h24": cosine,
            "n_train": len(Xt), "n_test": int(n_test),
        }
        print(f"layer {layer:2d}: ridge-head top1 {agree:.4f} | raw lens "
              f"{results[layer]['logit_lens_agreement']:.4f} | lam {lam:g}", flush=True)

    out = {"states": args.states, "split_seed": args.split_seed,
           "lstar": args.lstar, "train_prompts": train_prompts,
           "test_prompts": test_prompts, "results": results}
    os.makedirs(os.path.dirname(os.path.join(HERE, args.out)), exist_ok=True)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
