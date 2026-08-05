#!/usr/bin/env python
"""Plots for the fSRD depth experiment (steps 2-4).

Reads results/step2_*.json, results/step3_*.json, results/step4_*.json and
writes small PNGs to plots/:

- step2_boundaries.png : fSRD depth-boundary histograms per dataset (primary
  config) overlaid with the free cosine / logit-lens KL changepoint modes.
- step2_misfit.png     : sweep table of median in-window relative error.
- step4_head.png       : ridge exit-head top-1 agreement vs attach layer.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P = os.path.join(HERE, "plots")
os.makedirs(P, exist_ok=True)


def load(name):
    path = os.path.join(R, name)
    return json.load(open(path)) if os.path.exists(path) else None


def boundary_hist(recs, theta=3.0, mu=0.03, std=True):
    cnt = np.zeros(25)
    n_tok = 0
    for r in recs:
        if r["theta"] == theta and r["smoothness"] == mu and r["std"] == std:
            n_tok += 1
            for b in r["col_bounds"]:
                cnt[b] += 1
    return cnt, n_tok


def main() -> None:
    s2w, s2t = load("step2_wt2.json"), load("step2_ts.json")
    s3w, s3t = load("step3_wt2.json"), load("step3_ts.json")

    # --- boundaries ---------------------------------------------------------
    if s2w:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ax, s2, s3, title in [(axes[0], s2w, s3w, "WikiText-2"),
                                  (axes[1], s2t, s3t, "TinyStories")]:
            if not s2:
                continue
            # primary config collapses to 1 region on std rows; show raw too
            for std, color, label in [(True, "#2b6cb0", "std rows (primary)"),
                                      (False, "#dd6b20", "raw rows")]:
                cnt, n = boundary_hist(s2["records"], std=std)
                ax.bar(np.arange(25) + (0.0 if std else 0.35), cnt / max(n, 1),
                       width=0.35, color=color, label=f"fSRD bounds, {label}")
            if s3:
                ax.axvline(s3["cosine"]["mode"], color="green", ls="--",
                           label=f"cosine mode ({s3['cosine']['mode']})")
                ax.axvline(s3["kl"]["mode"] + 0.15, color="red", ls=":",
                           label=f"KL-knee mode ({s3['kl']['mode']})")
            ax.set_xlabel("boundary b (between layer b and b+1)")
            ax.set_ylabel("boundaries per token")
            ax.set_title(title)
            ax.legend(fontsize=7)
        fig.suptitle("fSRD depth boundaries (theta=3, mu=0.03, max_depth=2) vs free baselines")
        fig.tight_layout()
        fig.savefig(os.path.join(P, "step2_boundaries.png"), dpi=110)
        print("wrote plots/step2_boundaries.png")

    # --- misfit sweep --------------------------------------------------------
    if s2w:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels, meds = [], []
        for cfg in s2w["sweep"]:
            rs = [r for r in s2w["records"]
                  if r["theta"] == cfg["theta"] and r["smoothness"] == cfg["smoothness"]
                  and r["std"] == cfg["std"]]
            errs = np.array([r["err_multi"] for r in rs])
            meds.append(np.median(np.where(np.isfinite(errs), errs, np.inf)))
            labels.append(f"th{cfg['theta']}/mu{cfg['smoothness']}/"
                          f"{'std' if cfg['std'] else 'raw'}")
        ax.bar(range(len(meds)), meds, color="#2b6cb0")
        ax.axhline(0.15, color="red", ls="--", label="kill gate 0.15")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("median in-window rel. error")
        ax.set_title("Step 2 sweep, WikiText-2 depth matrices (1024 x 24)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(P, "step2_misfit.png"), dpi=110)
        print("wrote plots/step2_misfit.png")

    # --- ridge head -----------------------------------------------------------
    s4 = load("step4_seed0.json")
    s4b = load("step4_seed1.json")
    if s4:
        fig, ax = plt.subplots(figsize=(7, 4))
        for data, marker, label in [(s4, "o", "split seed 0"),
                                    (s4b, "s", "split seed 1")]:
            if not data:
                continue
            layers = sorted(int(k) for k in data["results"])
            ag = [data["results"][str(l)]["top1_agreement"] for l in layers]
            ax.plot(layers, ag, marker + "-", label=f"ridge head ({label})")
        layers = sorted(int(k) for k in s4["results"])
        lens_ag = [s4["results"][str(l)]["logit_lens_agreement"] for l in layers]
        ax.plot(layers, lens_ag, "^--", color="gray", label="raw logit-lens (no head)")
        ax.axvline(s4["lstar"], color="red", ls=":", label=f"fSRD l* = {s4['lstar']}")
        ax.set_xlabel("attach layer")
        ax.set_ylabel("top-1 agreement with full model")
        ax.set_title("Ridge exit head h_l -> h_24: agreement on held-out positions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(P, "step4_head.png"), dpi=110)
        print("wrote plots/step4_head.png")


if __name__ == "__main__":
    main()
