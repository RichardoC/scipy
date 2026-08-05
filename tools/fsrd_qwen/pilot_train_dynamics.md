# PILOT: fSRD on neural-network training dynamics

Target: a snapshot matrix whose columns are successive *training steps* of a small
transformer LM, rows are coordinates of a fixed low-dimensional summary of the
parameter vector. This is the one candidate with a genuine autonomous ordered time
axis (step count, fixed update rule) and genuine regimes (warmup, descent, LR-drop,
plateau) — exactly the object fSRD's temporal-split + BIC machinery is built for.

Status: **PRE-REGISTRATION WRITTEN BEFORE ANY FIT WAS RUN.** Results appended below.

---

## 0. Setup (fixed before any result was seen)

- Model: char-level decoder-only transformer, 4 layers, d_model 96, 4 heads,
  block 48, vocab = chars of `prompts_wt2.txt` (WikiText-2 excerpt, 185 KB).
  ~363 K parameters. CPU, `OMP_NUM_THREADS=1`.
- Optimiser: AdamW, lr 3e-3, linear warmup 40 steps, constant, then **x0.1 at
  step 350**, constant to the end. 700 steps total.
  Two *independently known* ground-truth events: step 40 (warmup end), step 350
  (LR drop). Plus data-driven events (loss knee, grad-norm knee).
- Per step recorded: loss, grad L2 norm, per-layer weight norms, and the
  **state vector** = 256-dim CountSketch (sparse Johnson–Lindenstrauss: fixed
  seeded hash + sign, seed 12345, identical across training seeds) of the full
  flattened parameter vector. Snapshot matrix `a` = (256, T).
- Seeds: 5 training seeds (0..4).
- fSRD called with `oblique=False` (rows are unordered sketch coordinates, so
  oblique/row-migrating splits are semantically void here; also oblique regions
  are never extrapolated, which would silently void the forecast test).

---

## 1. Pre-registered thresholds

### T1 — Forecasting the parameter state

Fit fSRD on columns `[0, k)`, forecast `h` columns, compare column `k+h-1`
against truth. Fit points `k in {150, 250, 350, 450, 550}` (straddling the LR
drop), horizons **h = 10 (primary) and h = 40 (secondary)**.

Metric: relative L2 error `||x̂ - x|| / ||x - x_ref||` where `x_ref` is the
last observed column `a[:, k-1]` — i.e. **skill relative to persistence**:
`skill = 1 - ||x̂-x||/||x_persist-x||`. Positive = better than copying.

Baselines: (i) persistence, (ii) linear extrapolation from the last 5 columns
(least-squares line per row), (iii) **global DMD = `fsrd(max_depth=0)`** — the
gate that matters.

> **PRE-REGISTERED GATE (T1).** fSRD (`max_depth=3`) counts as signal only if its
> median relative forecast error over all (seed, k) pairs at h=10 is **≥20 %
> lower than global DMD's** (i.e. `err_fSRD / err_globalDMD ≤ 0.80`) *and* it
> beats persistence and linear extrapolation. If `0.80 < ratio < 1.25` the
> verdict is **"regions add nothing"** and the pilot is NO-GO regardless of how
> fSRD does against persistence.

> **PRE-REGISTERED GATE (T1b, parameter jump — conditional).** Only run if T1
> passes. Jump the real parameters along the fSRD-forecast displacement (in an
> invertible SVD basis of the stored trajectory), then measure held-out loss.
> Counts as signal only if the jump reaches a loss that normal training needs
> **≥ 1.2 x h extra steps** to reach, on ≥4/5 seeds, *and* the jump is not
> simply a scaled ordinary step (checked by the cosine of the jump direction
> against the mean recent update direction and against the best scalar-rescaled
> ordinary step).

### T2 — Phase / regime detection

Extract fSRD's temporal region boundaries (pure-time splits, `v[1]==0`) from a
full-window fit on `a` (all 700 columns) per seed.

Free baselines: changepoints of the loss curve and of the grad-norm curve, both
by the same exhaustive-search piecewise-constant/linear cost (PELT-style
binary segmentation, same number of changepoints as fSRD produced, so it is a
like-for-like comparison).

> **PRE-REGISTERED GATE (T2).** fSRD boundaries count as added information only if:
> (a) **consistency** — at least **4 of 5 seeds** place a boundary within
>     **±25 steps** (≈3.5 % of the run) of a common location, for at least one
>     boundary that is not the trivial run start/end;
> AND (b) **novelty** — at least one such consistent boundary is **> 40 steps**
>     away from the nearest loss-curve changepoint *and* the nearest grad-norm
>     changepoint (else the free baselines already gave it);
> AND (c) **groundedness** — the consistent boundaries land within ±25 steps of a
>     known event (step 40 warmup end, step 350 LR drop) at a rate better than
>     chance for that many boundaries (chance = 2 x 51 / 700 ≈ 14.6 % per
>     boundary).
> Failing (a) → NO-GO on T2. Passing (a) but failing (b) → "fSRD recovers what
> the loss curve already tells you for free" = NO-GO on T2.

### T3 — Degeneracy auto-fail (checked explicitly)

> **PRE-REGISTERED AUTO-FAIL.** If `n_regions == 1` on **more than half** of the
> fits (across seeds x fit windows) with `max_depth=3`, fSRD is declaring there
> is nothing to segment. That is an immediate NO-GO for the whole candidate,
> reported as such, no further metrics.
> Also recorded: fraction of *temporal* splits among all splits. If fSRD splits
> only along rows (sketch coordinates) and never along time, that is the same
> degeneracy in disguise and is reported as a **soft auto-fail**.

### Overall verdict rule

GO only if T3 passes AND (T1 passes OR T2 passes fully). Anything else = NO-GO.

---

## 2. Results

(appended by `pilot_train_run.py` / `pilot_train_fsrd.py` after the fact)

---

## 1b. Third test, PRE-REGISTERED after seeing T1/T2 region structure but BEFORE running it

Motivation: T2's cross-seed boundary agreement turned out to be near-exact
(boundaries 46/169/355 identical to the step across all 5 seeds). fSRD places
splits on a **fixed deterministic fraction grid** (`_FRACS = (0.35, 0.5, 0.65)`
of each node's extent), so for a fixed matrix shape and tree depth the set of
*achievable* boundary positions is a small lattice that does not depend on the
data values at all. Cross-seed consistency may therefore be forced by the
algorithm, not discovered in the data. This is a validity check on the T2
measurement, not a new success metric.

### T4 — negative control on boundary informativeness

Fit fSRD with identical settings on:
 (i) `a` with its **columns randomly permuted** (temporal structure destroyed,
     row marginals preserved);
 (ii) a **structureless surrogate**: smooth monotone drift + iid noise, per-row
     scale matched to `a`, containing no warmup, no LR drop, no plateau.

> **PRE-REGISTERED THRESHOLD (T4).** If the control fits reproduce **>= 3** of
> the real-data temporal boundaries to within **+-25 steps**, then the observed
> cross-seed consistency is an artifact of the deterministic split grid and
> T2 gate (a) is declared **void** (not merely failed).

---

# 2. RESULTS — verdict: **NO-GO**

Deciding number: **fSRD's h=10 forecast of the parameter state has median
skill -0.070 against last-value persistence** (i.e. it is *worse* than copying
the previous step), and **-0.283 at h=40**. It beats persistence in 7/25
(seed, k) pairs at h=10 and 1/25 at h=40. The pre-registered T1 gate therefore
fails, and T1b (the parameter jump) was not run, exactly as pre-registered.

Raw numbers: `pilot_train_results.json`, `pilot_train_control.json`.
Runs: 5 seeds x 700 steps, 480 K parameters, ~40 s/seed wall, loss 5.2 -> 2.10
(single-batch val 5.2 -> 2.1-2.3). 50 fSRD forecast fits + 5 full-window fits +
8 control fits, all under 2 s each at (256, T<=700).

## T3 — degeneracy: **PASSED** (this target is genuinely not degenerate)

| quantity | value |
|---|---|
| fits with `n_regions == 1` (max_depth=3) | **0 / 50** |
| region counts observed | 3, 5, 6, 7, 8 (mean 6.84) |
| splits that are *temporal* | **342 / 342 = 100 %** |
| in-window rel. error, fSRD | **5.1e-4** (median) |
| in-window rel. error, global DMD | 2.7e-3 (median) |

This is the healthiest fSRD behaviour anywhere in this project: 5.3x better
in-window reconstruction than global DMD, every split along the time axis, and
5e-4 relative error — inside the method's validated competence envelope
(vs 8e-2 to 2.7e-1 on transformer token-axis activations, INVESTIGATION §3.9).
**A training trajectory really is the kind of object fSRD models well.** The
no-go is not a fitting failure.

## T1 — forecasting: **FAILED** (gate: ratio<=0.80 vs global DMD AND beat persistence+linear)

Median error at h=10 over 25 (seed, k) pairs:

| predictor | median err | median skill vs persistence |
|---|---|---|
| persistence (copy step k-1) | **0.443** | 0 (reference) |
| linear extrapolation (last 5) | 3.770 | -0.195 |
| global DMD (`max_depth=0`) | 3.913 | -0.381 |
| **fSRD (`max_depth=3`)** | 2.156 | **-0.070** |

- **Region structure DID clear its own margin: median `err_fSRD/err_gDMD` =
  0.753 at h=10 and 0.712 at h=40, inside the pre-registered <=0.80.** So the
  answer is *not* "regions add nothing" — the regions measurably help. The whole
  DMD family is simply beaten by copying the last column.
- The failure is worst exactly where it should be best. At `k=350`, the fit
  window ends at the LR drop; persistence error collapses to 0.44 while fSRD
  extrapolates the fast pre-drop dynamics forward and scores **skill -3.22 to
  -5.04 on all 5 seeds**. A regime-aware forecaster failing hardest at the one
  genuine regime transition is the decisive negative: fSRD's regions describe
  the *past* window, and its last region's autonomous linear operator carries no
  information that the regime it just characterised is about to end.
- Consistent with INVESTIGATION §3.5: forecasting is an unvalidated extension,
  never evaluated beyond a 20-step clean linear rotation.

## T2 — phase detection: **FAILED on novelty** (consistency and groundedness passed)

fSRD temporal boundaries, full-window fits (`max_depth=3, oblique=False`):

| seed | n_regions | temporal boundaries (steps) |
|---|---|---|
| 0 | 8 | 46, 169, 355, 616 |
| 1 | 8 | 46, 169, 355, 616 |
| 2 | 8 | 46, 169, 355, 546 |
| 3 | 8 | 46, 169, 355, 546 |
| 4 | 8 | 45, 188, 355, 616 |

| cluster | seeds | dist. to nearest loss changepoint | to grad-norm changepoint | to known event |
|---|---|---|---|---|
| 45.8 | 5/5 | **2.2** | 3.2 | 5.8 (warmup end, 40) |
| 172.8 | 5/5 | 40.2 | **13.8** | 132.8 |
| 355.0 | 5/5 | **0.0** | 4.0 | 5.0 (**LR drop, 350**) |
| 616.0 | 3/5 | 248.0 | 21.0 | 266.0 |
| 546.0 | 2/5 | 178.0 | 31.0 | 196.0 |

- Gate (a) consistency: **PASS** — 3 clusters at 5/5 seeds, two of them
  identical to the step.
- Gate (c) groundedness: **PASS** — 2 of 3 land within +-6 steps of a known
  event (warmup end, LR drop); 2/3 vs 14.6 % chance.
- Gate (b) novelty: **FAIL, 0 of 3 novel.** Every consistent boundary sits
  within 40 steps of a free loss-curve or grad-norm changepoint; the two
  grounded ones sit within **2.2 and 0.0 steps** of a loss changepoint. Reading
  the loss curve gives you the same segmentation for zero cost.
  Per the pre-registered rule this is "fSRD recovers what the loss curve already
  tells you for free" = NO-GO on T2.

## T4 — negative control (registered before running): **boundaries are data-dependent, but consistency is worthless as evidence**

| control | n_regions | boundaries | hits on real [46,169,355] (+-25) |
|---|---|---|---|
| column-shuffled real data, 3x | **1** | none | 0/3 |
| drift + iid noise surrogate, 3x | **1** | none | 0/3 |
| drift + random-walk surrogate, 2x | **8** | 91, 239, 406, 608 (both) | **0/3** |

`t4_consistency_void = False`: the real boundaries are *not* reproduced by any
control, so they are genuinely data-dependent, and BIC correctly returns 1
region when temporal structure is destroyed. That is a real point in fSRD's
favour and T2 gate (a) stands.

But the random-walk control — a trajectory with **no warmup, no LR drop and no
plateau whatsoever** — still yields 8 regions with boundaries repeatable to the
step across surrogate seeds. So "boundaries agree across seeds" is worth
nothing on its own: fSRD's deterministic split lattice (`_FRACS =
0.35/0.5/0.65` of each node's extent) makes cross-run agreement nearly free.
Only boundary *location* against an independent event carries information — and
that is precisely what the loss curve supplies for free.

## Overall

T3 pass; T1 fail; T2 fail on novelty. Per the pre-registered rule:
**NO-GO.** fSRD models a training trajectory well (5e-4 in-window, regions
genuinely beat global DMD by 25 %), but neither downstream use survives a free
baseline: forecasts lose to copying the last column and collapse at the one real
regime change, and its segmentation duplicates the loss curve.
