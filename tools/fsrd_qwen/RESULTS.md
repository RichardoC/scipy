# fSRD depth-segmentation experiment on Qwen3.5-0.8B — results

**Date:** 2026-08-05. **Spec:** `INVESTIGATION.md` section 5 ("The smallest
experiment"), executed as pre-registered. **Model:** `Qwen/Qwen3.5-0.8B`
(architecturally faithful stand-in for Qwen3.6-27B; **not** Qwen3.6 weights —
every conclusion below transfers to the 27B as a hypothesis, not a
measurement). **Hardware:** 4 CPU cores, 15 GB RAM, no GPU.

## What was run

| step | script | data | scale |
|---|---|---|---|
| 1. harvest | `harvest_hidden.py` (+`step0_prompts.py`) | WikiText-2 train, TinyStories valid | wt2: 128 prompts x 256 tok (3.35 GB); ts: 64 x 256 (1.68 GB); wt2ho holdout: 20 x 256 (0.52 GB) |
| 2. depth fits + gate | `step2_depth_fsrd.py` | 400 (prompt,pos) pairs per dataset | 8-config sweep x (multi + global) = 6,400 fits/dataset, 68-70 s each dataset |
| 3. free baselines | `step3_baselines.py` | same 400 pairs per dataset | cosine changepoints + logit-lens KL knees |
| 4. ridge exit head | `step4_ridge_head.py` | wt2, split by prompt | 27,648 train / 5,000 held-out positions, 2 split seeds |
| 5. DeltaNet arm | `step5_deltanet.py` | wt2, layers {5,9,13} vs {4,8,12} | 120 token-time fits (1024 x 128) |
| 6. forecast controls | `step6_controls.py` | wt2 + wt2ho holdout | 100 depth forecasts; 60 token-time drafts |

Depth matrices are `states[p, 1:, t, :].T` — (1024, 24), hidden-state index 0
(raw embedding) dropped. All sequence lengths are 256 = 4x64 (above the T<64
linear-attention kernel cliff). Logits (vocab 248,320) were always computed in
batches and reduced to argmax/KL immediately.

## Pre-registered decision criteria

| criterion (threshold) | verdict | measured |
|---|---|---|
| **Kill gate:** median multi-region in-window rel. error <= 0.15 | **MET** | primary config (theta=3, mu=0.03, std rows): **4.57e-15** (wt2), 4.55e-15 (ts); raw rows: **0.0373** (wt2), 0.0375 (ts) — all 8 sweep configs pass |
| **Segmentation is real:** multi-region BIC < global BIC on >= 70% of tokens AND boundary-histogram mode holds >= 40% mass within +-1 layer, on both datasets | **NOT MET** | BIC prong fails everywhere: primary config **0.0%** (both datasets, model prunes to 1 region); best sweep config **65.8%** (wt2, theta=1.5/mu=0.05/raw) and **59.2%** (ts) — never >= 70%. Mode prong alone passes: **0.799** (wt2) / **0.746** (ts) at boundary 8 |
| **fSRD adds information:** modal boundary differs from both free baselines' modes by > 1 layer on >= 1 dataset | **MET** (with caveat) | fSRD mode **8** vs cosine mode **22** and KL-knee mode **22**, both datasets — 14 layers apart. Caveat: with T=24 and the implementation's deterministic split grid (fractions 0.35/0.5/0.65), root boundaries can essentially only land near columns 8/12/16, so the *placement resolution* is coarse; the information content is "8, not 12 or 16" |
| **Placement is useful:** ridge head at l\* beats best uniform placement by >= 2 points top-1 | **NOT MET** | l\*=9: **29.4%** / 29.2% (seeds 0/1) vs best uniform (layer 18): **49.9%** / 51.6% — the fSRD placement is **20.6 / 22.4 points worse** |
| **Hard-negative clause** (misfit > 0.15, or no >= 40% mode, or BIC wins < 50%) | not triggered | misfit passes, mode exists, BIC wins 55-66% (raw) — the depth arm dies on the 70% BIC prong and the placement test, not on the hard-kill clause |
| **Token-time control expectation:** fSRD <= identity < ridge; fSRD beating ridge would force a Rank-5 revisit | ridge >> fSRD confirmed | acceptance: fSRD **8.3%**, identity 0.0%, ridge **18.3%**; cosine: ridge 0.776 > identity 0.657 > fSRD 0.523. fSRD does not beat ridge; Rank-5 verdict stands. (fSRD's 8.3% vs identity's 0.0% argmax acceptance is the one departure from the expected ordering, on n=60; on the cosine metric the expected ordering holds) |

**Bottom-line verdict: H0.** The pre-registered success condition for Rank-1
was the conjunction of all four positive criteria; two of the four fail,
including the only one that matters for deployment (placement). fSRD's
BIC-selected depth boundary confers no placement advantage — the opposite:
exit-head quality increases monotonically with depth and the fSRD boundary
(layer 9) is 20+ points below the deepest uniform comparator.

## Per-step results

### Step 1 — harvests

wt2: 32,768 tokens forward in 271 s (8.3 ms/token), peak RSS 8.24 GB.
ts and wt2ho ran CPU-contended (63.7 / 104.2 ms/token). Total ~5.6 GB in
`states/` (gitignored).

### Step 2 — fSRD depth fits (full sweep tables)

WikiText-2, 400 depth matrices (1024 x 24), median over fits with non-finite
reconstructions counted as +inf:

| theta | mu | rows | med rel err | nonfinite % | BIC multi<glob % | med regions | RuntimeWarnings |
|---|---|---|---|---|---|---|---|
| 3.0 | 0.03 | std | 4.57e-15 | 0.0 | 0.0 | 1 | 0 |
| 3.0 | 0.03 | raw | 0.0373 | 14.0 | 59.8 | 3 | 115 |
| 3.0 | 0.05 | std | 4.57e-15 | 0.0 | 0.0 | 1 | 0 |
| 3.0 | 0.05 | raw | 0.0370 | 12.2 | 62.3 | 3 | 100 |
| 1.5 | 0.03 | std | 4.57e-15 | 0.0 | 0.0 | 1 | 0 |
| 1.5 | 0.03 | raw | 0.0362 | 14.0 | 63.7 | 4 | 115 |
| 1.5 | 0.05 | std | 4.57e-15 | 0.0 | 0.0 | 1 | 0 |
| 1.5 | 0.05 | raw | 0.0362 | 12.0 | 65.8 | 4 | 98 |

TinyStories: same pattern — std 4.55e-15 / 0% / 0.0% / 1 region; raw medians
0.0366-0.0377, nonfinite 19.2-21.8%, BIC wins 55.0-59.2%, 3-4 regions,
155-175 warnings.

Reading, honestly stated:

- **Standardized rows (the primary, spec-written config) are degenerate at
  this shape.** With M=1024 >> T=24, X1 has full column rank after
  standardization flattens the spectrum, so one global DMD reconstructs the
  window to machine precision (4.6e-15) and BIC correctly refuses every split:
  1 region, no boundaries, nothing to segment. The gate is passed trivially
  and vacuously.
- **Raw rows pass the gate non-trivially** (median 0.0373, global-DMD median
  0.0451 — both far under 0.15 and under the 0.267 token-time figure), but
  12-22% of multi-region fits return non-finite reconstructions: small
  (3-column) early-depth regions fit an eigenvalue on the negative real axis /
  at zero, and the ln(lambda) conversion NaNs those columns (the counted
  RuntimeWarnings: "invalid value encountered in log"). Median over finite
  fits only: 0.0358 (wt2).
- Boundary histograms (raw, theta=3, mu=0.03) are extremely concentrated —
  wt2: {3: 58, 8: 115, 9: 115} over 400 tokens; ts: {3: 88, 8: 135, 9: 135, else 4}.
  (A single fuzzy split contributes both column 8 and 9 — sibling boxes
  overlap by one column; the split is between layers ~8-10, l\* = 9.) The
  secondary boundary at column 3 (between layers 3 and 4) appears in ~15-22%
  of fits. Descriptive: the dominant-mode |Im(omega)|/2pi was 0.0 in the
  median — no period-4 (0.25 cycles/layer) eigenvalue signature from the
  3:1 DeltaNet/full-attention layout was observed.

### Step 3 — free baselines

Per-token changepoints, same 400 pairs, boundary coordinate "between layer b
and b+1":

| baseline | wt2 mode (+-1 mass) | ts mode (+-1 mass) |
|---|---|---|
| cosine of successive layers (argmin) | 22 (0.935) | 22 (0.930) |
| logit-lens KL(final \|\| layer) largest drop | 22 (0.748) | 22 (0.768) |

Both free baselines put the change at the end of the stack (the final-norm /
prediction-forming layers 22-24), where the mean KL curve drops 3.56 -> 2.26
-> 0 and successive-layer cosine dips to 0.612. fSRD's boundary (8) is 14
layers away — it is seeing something different (early-to-mid-stack transition),
not reproducing the free curves.

### Step 4 — ridge exit head (the deployment-relevant test)

Ridge W (1025 x 1024, bias, lambda by inner validation = 1e-2 relative) from
h_l to h_24; top-1 agreement of `lm_head(final_norm(W h_l))` with the full
model's argmax on 5,000 held-out positions (split by prompt; test prompts
never entered any fit):

| attach layer | top-1 agree (seed 0) | top-1 agree (seed 1) | raw logit-lens | cosine(Wh, h24) |
|---|---|---|---|---|
| 6 | 0.2894 | 0.2878 | 0.0050 | 0.803 |
| **9 = l\*** | **0.2936** | **0.2920** | 0.0038 | 0.804 |
| 12 | 0.3164 | 0.3134 | 0.0050 | 0.811 |
| 16 | 0.4226 | 0.4308 | 0.0218 | 0.859 |
| 18 | 0.4994 | 0.5156 | 0.0556 | 0.892 |

Placement criterion: **fails by 20.6 points (seed 0) / 22.4 points (seed 1)**.
The confound check pre-registered for a *win* (is l\* just deeper?) is moot for
a loss this size, but its logic cuts the other way and explains the result:
agreement is monotone in depth, so placement quality is governed by depth, not
by any regime boundary — an fSRD boundary at layer 9 is simply an expensive
way to pick a bad layer. Both split seeds agree; no artifact check is needed
for a negative.

### Step 5 — DeltaNet vs full-attention (Rank-2 arm)

120 token-time fits (1024 x 128), theta=3, mu=0.03, oblique=False. Median
in-window relative error:

| | global DMD (1 region) | fSRD max_depth=2 (median 4 regions) |
|---|---|---|
| DeltaNet outputs (layers 5, 9, 13) | 0.608 | 0.440 |
| full-attention outputs (layers 4, 8, 12) | 0.616 | 0.453 |

**Answer: no** — linear-attention layer outputs are not measurably more
piecewise-linear than full-attention outputs (differences ~0.01, within the
per-layer spread; boxplot in `plots/step5_deltanet.png`). Both are badly
non-linear along token time (44-62% misfit), consistent with and worse than
the 0.267 mid-stack figure that motivated the pessimistic prior.

### Step 6 — forecast negative controls

(a) **Depth forecast** (fit layers 1-16 raw, forecast=8, compare layer-24
prediction; n=100): fSRD cosine to true h_24 **0.260** vs identity(h_16)
**0.290**; decoded-argmax agreement fSRD **0.00** vs identity **0.03**.
8 RuntimeWarnings, 0 zero-tails. fSRD extrapolation across depth is no better
than copying layer 16 — as pre-registered (Rank-6 stays dead).

(b) **Token-time first-draft-token** (20 held-out prompts x 3 windows of 128,
forecast=4; ridge next-state map trained only on the main wt2 harvest):

| drafter | acceptance | cosine to true h |
|---|---|---|
| fSRD forecast | 0.083 | 0.523 |
| identity (copy h_T) | 0.000 | 0.657 |
| corpus ridge h_t -> h_{t+1} | **0.183** | **0.776** |

In-window median misfit of the 60 fitted windows: **0.274** (independently
confirms the investigation's 0.267 token-time figure). 6/60 forecasts were
zero-tail (no-support) events; 7 RuntimeWarnings. The plain trained ridge map
dominates fSRD on both metrics, so the pre-registered Rank-5 trigger ("fSRD
beats ridge") did not fire. One nuance reported as measured: on argmax
acceptance fSRD (8.3%) beat identity (0.0%) — identity decodes to the *previous*
position's prediction and almost never matches — while on the state-space
(cosine) metric the expected fSRD < identity ordering held.

## Deviations from the spec

1. **No subagent tool existed in this environment** (the meta-task asked for
   `Agent`-tool delegation); independent strands (harvests, steps 2/3, 5, 6)
   were instead run as concurrent background processes on the 4 cores.
2. **BLAS threading bug, one re-run:** the first step-2 launch imported numpy
   before pinning `OMP_NUM_THREADS`, so 3 workers x 4 OpenBLAS threads
   spin-waited a 68-second job into 100+ CPU-minutes; it was killed, the env
   pin moved above the numpy import, and the sweep re-run from scratch. No
   results from the aborted run were used.
3. **l\* definition:** the modal fuzzy split reports overlapping sibling boxes
   (columns 8 and 9), i.e. a boundary between layers ~8-10; l\* = 9 (the
   overlap layer) was used for step 4. Sensitivity is moot given the head at
   6, 9 and 12 differs by < 2.5 points and all lose to 16/18 by 11+ points.
4. **Step 6b used 60 trials** (3 windows per held-out prompt) rather than one
   per prompt, and a dedicated 20-prompt holdout harvest (`states/wt2ho`)
   disjoint from every ridge fit.
5. **Sweep scope:** the 8-config sweep was run on both datasets (the spec
   listed it once); the gate/config table reports both.
6. `matplotlib` (plots) and `py-spy` (debugging deviation 2) were installed
   into the venv; nothing else was installed and nothing re-downloaded.
7. TinyStories prompts came from `roneneldan/TinyStories` validation split
   (the "TinyStories-valid" the spec names), filtered to >= 256 tokens.

## Plots

- `plots/step2_boundaries.png` — fSRD boundary histograms vs free-baseline modes
- `plots/step2_misfit.png` — sweep medians vs the 0.15 kill gate
- `plots/step4_head.png` — exit-head agreement vs attach layer (both seeds)
- `plots/step5_deltanet.png` — DeltaNet vs full-attention misfit boxplots

## Honest bottom line

On the architecturally faithful 0.8B stand-in, the depth arm passed its misfit
gate (raw-row median 0.0373, and trivially at machine precision on the
spec's standardized-row config, which turns out to be rank-degenerate at
1024 x 24) and produced strikingly consistent depth boundaries (mode at layer
8/9 carrying 75-80% of boundary mass on both datasets, 14 layers away from
where the free cosine/KL baselines put their change) — but it failed both
pre-registered tests that would have made those boundaries *mean* something:
the multi-region model wins the method's own BIC comparison on only 55-66% of
tokens (threshold: 70%), and the ridge exit head attached at the fSRD boundary
is 20+ points *worse* than the same head at the best uniform layer, under two
train/test splits, because head quality is simply monotone in depth. The
DeltaNet arm answered "no" (linear-attention outputs are not more
piecewise-linear than attention outputs), and both forecast controls failed
exactly as pre-registered (fSRD depth forecast <= identity; fSRD token-time
drafting well below a plain corpus ridge map, with 10% zero-output events and
a 0.274 in-window misfit confirming the earlier 0.267 measurement). Net:
**H0 is accepted for the deployment-relevant claim** — fSRD's depth
segmentation of this model family's residual stream, while cheap, numerically
consistent, and not merely a reproduction of free baselines, does not identify
useful attachment points for a linear exit head; and the forecast-shaped uses
remain dead on measurement. For Qwen3.6-27B all of this transfers as
hypothesis only (same `qwen3_5` architecture family and 3:1 hybrid layout, 64
layers vs 24, T=65 depth columns vs 24, different weights); the one observation
that might merit a look at 27B scale — where T=65 would give fSRD real
boundary resolution — is the stable early-stack boundary at layer 8/9, but
nothing measured here suggests it would change the placement verdict, and the
idea is noted as future work, not as a finding.
