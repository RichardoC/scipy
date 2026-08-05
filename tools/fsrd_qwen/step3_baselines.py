#!/usr/bin/env python
"""Spec step 3: free depth-changepoint baselines (no fSRD involved).

For the same sampled (prompt, position) pairs as step 2 (same seed), compute
per token:

(i)  cosine-similarity changepoint: c_l = cos(h_l, h_{l+1}) for l = 1..23;
     the changepoint is the transition with the minimum similarity (the layer
     pair where the residual stream changes direction the most).
(ii) logit-lens KL knee: KL(p_final || p_l) for l = 1..23 with p_l =
     softmax(lm_head(final_norm(h_l))); the knee is the transition l -> l+1
     with the largest one-step drop in KL.

Both changepoints are reported in the same coordinate as step 2's fSRD column
boundaries: value b means "between layer b and layer b+1" (1-based blocks).

Logits are computed in small batches (final norm + tied-embedding head) and
reduced to KL / argmax immediately — the full (N, 248320) array is never
materialized.

Usage:
    python step3_baselines.py --states states/wt2 --n-pairs 400 --seed 0 \
        --out results/step3_wt2.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from exp_common import HERE, LogitLens, load_harvest, sample_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    h, mask, _ = load_harvest(args.states)
    pairs = sample_pairs(mask, args.n_pairs, seed=args.seed)
    N = len(pairs)

    # (N, 24, 1024) stack of layer outputs 1..24 for the sampled pairs
    states = np.stack([h[p, 1:, t, :] for p, t in pairs]).astype(np.float32)

    # --- (i) cosine changepoints ------------------------------------------
    a = states[:, :-1, :]
    b = states[:, 1:, :]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    # transition index j (0-based) is between layer j+1 and j+2 -> boundary b = j+1
    cos_cp = cos.argmin(axis=1) + 1

    # --- (ii) logit-lens KL knees -----------------------------------------
    import torch
    lens = LogitLens()
    final = states[:, -1, :]  # layer 24
    # final log-probs, kept in fp32 (N x 248320 = ~400 MB for N=400)
    lp_final = torch.empty((N, lens.emb.shape[0]), dtype=torch.float32)
    for i, chunk in lens.log_softmax(final, batch=128):
        lp_final[i:i + chunk.shape[0]] = chunk
    p_final = lp_final.exp()

    kl = np.zeros((N, 24), dtype=np.float64)  # KL(final || layer l), l = 1..24
    for li in range(24):
        for i, chunk in lens.log_softmax(states[:, li, :], batch=128):
            sl = slice(i, i + chunk.shape[0])
            kl[sl, li] = (p_final[sl] * (lp_final[sl] - chunk)).sum(-1).numpy()
        print(f"  KL layer {li + 1}/24 done", flush=True)
    # largest one-step KL drop: transition j (0-based) between layer j+1 and j+2
    dkl = kl[:, 1:] - kl[:, :-1]
    kl_cp = dkl.argmin(axis=1) + 1

    def hist_and_mode(vals):
        vals = np.asarray(vals)
        cnt = np.bincount(vals, minlength=25)
        # mode under +-1 window
        mass = np.array([cnt[max(0, v - 1):v + 2].sum() for v in range(25)])
        mode = int(mass.argmax())
        return {"hist": cnt.tolist(), "mode": mode,
                "mode_mass_pm1": float(mass[mode] / len(vals))}

    out = {
        "states": args.states, "n_pairs": N, "seed": args.seed,
        "pairs": pairs,
        "cosine_changepoints": cos_cp.tolist(),
        "kl_changepoints": kl_cp.tolist(),
        "cosine": hist_and_mode(cos_cp),
        "kl": hist_and_mode(kl_cp),
        "kl_curve_mean": kl.mean(axis=0).tolist(),
        "cos_curve_mean": cos.mean(axis=0).tolist(),
    }
    os.makedirs(os.path.dirname(os.path.join(HERE, args.out)), exist_ok=True)
    with open(os.path.join(HERE, args.out), "w") as fh:
        json.dump(out, fh)
    print("cosine:", out["cosine"]["mode"], out["cosine"]["mode_mass_pm1"])
    print("kl    :", out["kl"]["mode"], out["kl"]["mode_mass_pm1"])


if __name__ == "__main__":
    main()
