# PILOT: fSRD as a *compressor* of a training-checkpoint sequence

Target: the checkpoint matrix `A ∈ R^(P×N)` — P parameters (rows), N successive
training checkpoints (columns). Storing N checkpoints naively costs N·P. The
hypothesis is that fSRD represents the trajectory with a handful of regions
whose modes span the whole run, so storage falls to ≈ (total rank R)·P plus
metadata, a win of ≈ N/R×, **using reconstruction (fSRD's validated strength),
not forecasting (its established failure, see `pilot_train_dynamics.md`).**

Basis in fSRD's favour, from the completed sibling pilot: on a parameter
trajectory fSRD reaches in-window relative reconstruction error **5.1e-4** vs
global DMD's 2.7e-3, **0/50 fits degenerate to one region**, **342/342 splits
temporal**. That pilot's NO-GO was on forecasting and on segmentation novelty,
neither of which this candidate uses.

Status: **PRE-REGISTRATION WRITTEN AND COMMITTED BEFORE ANY FIT WAS RUN.**
Results appended in §4.

---

## 0. Setup, fixed before any result was seen

### 0.1 Data

- Model + optimiser + corpus: **identical to `pilot_train_run.py`** (reused, not
  rewritten): char-level decoder-only transformer, 4 layers, d_model 96,
  4 heads, block 48, vocab = chars of `prompts_wt2.txt`; **P ≈ 4.7e5
  parameters**; AdamW lr 3e-3, linear warmup 40 steps, ×0.1 LR drop at step 350,
  700 steps total. CPU, `OMP_NUM_THREADS=1`.
- **Full parameter vectors** are stored (not the 256-dim sketch), because
  storage ratios are meaningless at reduced dimensionality.
- **Step subsampling, and why**: 700 steps × 4.7e5 params × 4 B = 1.3 GB per
  seed, over the memory budget for two seeds. Every **2nd** step is stored →
  **350 checkpoints/seed, 658 MB fp32/seed**, 1.32 GB for two seeds. fp32 is
  kept (not fp16) so that the naive baseline `N·P·4 B` is the honest thing the
  compressor is competing against; fp16 halves *every* method equally and is
  reported as a note, not as a separate arm.
- **Fit / held-out split, and why it preserves fSRD's assumptions**: DMD
  requires *uniformly spaced* columns. Removing scattered columns would break
  that and unfairly handicap fSRD. So the 350 stored checkpoints are
  interleaved: **even positions (original steps 0, 4, 8, …, 696; N_fit = 175)
  are the only columns any method may see**, and **odd positions (steps 2, 6,
  …, 694; N_ho = 174) are held out**. Both sets are uniformly spaced; each
  held-out checkpoint sits exactly at fit-unit index i + 0.5, i.e. **strict
  interpolation, never extrapolation** (the final odd column, which would need
  half a step of extrapolation, is dropped). N_fit = 175 ≥ 64. ✔
- Seeds: **0 and 1** (a positive result must survive the second seed).

### 0.2 How fSRD is applied at P = 4.7e5 rows (and why this is lossless)

fSRD cannot take a 4.7e5 × 175 matrix (8.2e7 cells > its 5e7 cap; and cost
scales with row count). The trajectory has **rank ≤ N_fit = 175 by
construction**, so an *exact* orthonormal basis of its column space exists.
Procedure:

1. Subtract the mean fit checkpoint `μ` (charged to every low-rank method
   below, identically, as P extra floats — it is needed for numerical
   conditioning, since the raw checkpoint matrix is dominated by a rank-1
   constant).
2. Thin SVD of the centred fit matrix, **all components kept** down to
   σ/σ₀ > 1e-7: `A_c = U C`, `U` (P×K) orthonormal, `C = ΣVᵀ` (K×N_fit).
   Residual `‖A_c − UC‖/‖A_c‖` is reported; it bounds every method equally.
3. **fSRD is run on `C`** (K×175). Because U is orthonormal, *every Frobenius
   error measured in coefficient space equals the parameter-space error
   exactly*, and a coefficient-space mode φ maps to the parameter-space mode
   `Uφ` — **P floats per rank, exactly as for SVD**. Nothing is discarded and
   fSRD sees the entire trajectory. This is also the documented
   "PCA-project first" guidance (INVESTIGATION §3.8).
4. `oblique=False`, matching the sibling pilot: rows are unordered PCA
   coordinates, so oblique/row-migrating splits are semantically void.
   `theta=1.5, smoothness=0.05, prune=True` (defaults).
5. Held-out columns are produced by evaluating the *fitted* regions'
   Vandermonde at fractional index i + 0.5 with the same fuzzy membership
   blend as `fsrd`'s own reconstruction. The re-implementation of the blend is
   **verified to reproduce `res.reconstruction` exactly on the integer grid**
   before any held-out number is trusted.

### 0.3 Storage ledger (itemised, in bytes, fp32 unless stated)

- **naive**: `N·P·4` for the columns represented.
- **fSRD**: Σ over regions of `rows_i · r_i · 4` for **realified** modes
  (a complex-conjugate mode pair of a real matrix is exactly two real vectors,
  so a rank-r region needs r real P-vectors — fSRD is *not* charged double for
  complex arithmetic; any unpaired complex mode found is charged double and
  reported) + `4·r_i·4` for eigenvalues and amplitudes + bounding box
  (4×int32) + the split path (per level: 3 floats `v`, 1 float `tau`, 1 int
  `side`) + `P·4` for μ.
- **global SVD (rank R)**: `P·R·4` (U) + `R·N_fit·4` (coefficients) + `P·4` (μ).
- **per-segment SVD (m segments, rank r each)**: Σ `P·r·4` + `r·q_seg·4`
  + boundaries + `P·4` (μ).
- **subsampling (n_keep kept checkpoints)**: `n_keep·P·4`, no μ.

At each budget, comparators are given the **largest rank/segment-rank/keep-count
that fits inside fSRD's own byte total** (floor). Every number is reported as
achieved bytes, not nominal rank.

### 0.4 Metrics (all three registered now)

For each held-out checkpoint j:

- `frob_rel[j] = ‖x̂_j − x_j‖ / ‖x_j‖` (raw relative Frobenius error).
- `frob_rel_c[j] = ‖x̂_j − x_j‖ / ‖x_j − μ‖` (relative to the *variation*; the
  raw version is expected to be tiny for every method because the checkpoint
  matrix is dominated by a constant, so it discriminates poorly — this is
  stated in advance, not after seeing results).
- **`dloss[j] = valloss(x̂_j) − valloss(x_j)`** — the **primary** fidelity
  metric. The reconstructed parameter vector is loaded back into the network
  and evaluated on a **fixed, deterministic** validation token set (first 64
  non-overlapping 48-token windows of the held-out 20 000-character tail of
  `prompts_wt2.txt`, 3072 tokens). Identical tokens for every arm, so all
  comparisons are paired and deterministic. Evaluated at **8 held-out
  checkpoints** spread uniformly over the run (including one before the warmup
  end and one after the LR drop).

Summary statistic per arm: **mean `dloss` over the 8 checkpoints** (median also
reported).

---

## 1. Pre-registered thresholds

Let `D_x` = mean `dloss` of arm x at a given matched byte budget.
Arms: `fsrd`, `svd` (global), `useg` (uniform-boundary per-segment SVD, same
segment count as fSRD), `sub` (uniform checkpoint subsampling + linear
interpolation), `gdmd` (`max_depth=0`).

> **PRE-REGISTERED GATE (G1, primary — piecewise vs the hard SVD ceiling).**
> `D_fsrd ≤ 0.70 · D_svd` (≥30 % reduction in excess validation loss) at
> **≥3 of the 4 byte budgets tested, on both seeds.**
> *Justification of the margin*: rank-R SVD is the Frobenius-optimal rank-R
> representation, so fSRD starts behind on the metric SVD optimises; a claim
> that piecewise structure pays for itself needs a margin clearly larger than
> the sibling pilot's observed 25 % in-window advantage, hence 30 %. The
> loss evaluation is deterministic and paired, so the margin is not absorbing
> measurement noise.

> **PRE-REGISTERED GATE (G2, primary — adaptive vs arbitrary boundaries).**
> `D_fsrd ≤ 0.80 · D_useg` at ≥3 of 4 budgets on both seeds.
> *Justification*: uniform boundaries are free. Adaptive placement is the only
> thing fSRD uniquely contributes over "segment it and SVD each piece", so it
> must show a real but not necessarily large increment; 20 % is the same margin
> the sibling pilot pre-registered for "regions add something".

> **PRE-REGISTERED GATE (G3, primary — the dumb baseline).**
> `D_fsrd ≤ 0.80 · D_sub` at ≥3 of 4 budgets on both seeds.

> **PRE-REGISTERED GATE (G4, usefulness floor).** A win between two useless
> reconstructions is not a storage win. At **some** budget with storage ratio
> ≥ 4× versus naive, fSRD must reach **mean `dloss` ≤ 0.05 nats**
> (≈2 % of the ≈2.1-nat converged loss). Without this, the verdict is at best
> PARTIAL regardless of ratios.

> **PRE-REGISTERED AUTO-FAIL (G0, degeneracy).** If `n_regions == 1` on **more
> than half** of the `max_depth=3` fits, fSRD is declaring there is nothing to
> segment → immediate NO-GO, region counts reported, no further metrics
> interpreted. (This has happened repeatedly in this project.)

### Verdict rule

- **GO (PASS)**: G0 passes AND G1, G2, G3, G4 all pass.
- **PARTIAL**: G0 and G1 and G3 pass but **G2 fails** → verdict
  *"segmentation helps; fSRD's adaptive placement is ceremony"*.
- **NO-GO (FAIL)**: `D_svd ≤ D_fsrd` or `D_sub ≤ D_fsrd` at ≥2 of 4 budgets
  (using a 10 % tolerance: comparator ≤ 1.10·fsrd counts as matching), or G4
  fails at every budget, or G0 fires.

### Stated in advance: the most likely failure mode

A training trajectory is plausibly **dominated by a single smooth global
drift**: a couple of PCA directions carrying almost all the motion, with
smooth, monotone-ish coefficient curves. That is *precisely* the object a
global truncated SVD represents optimally, and it leaves piecewise structure
almost nothing to add. If so, the expected outcome is `D_svd ≲ D_fsrd` and a
clean NO-GO. A second likely outcome is that **plain subsampling wins outright**,
because linear interpolation between two nearby checkpoints of a smooth
trajectory is extremely accurate and costs no metadata at all.

### Sanity checks required of any positive result

1. Held-out columns never enter any fit (asserted in code: fits touch only
   even-indexed columns; the basis `U`, `μ`, and every rank/segment choice are
   derived from those columns alone).
2. The blend re-implementation reproduces `fsrd`'s own reconstruction on the
   integer grid (asserted).
3. The whole result reproduces on training seed 1.

---

## 2. Scripts

- `pilot_ckpt_train.py SEED` — trains, dumps every 2nd step's full parameter
  vector to `pilot_ckpt_data/`.
- `pilot_ckpt_eval.py SEED` — basis, fSRD fits, all baselines, byte ledger,
  reconstructed-model validation loss; writes `pilot_ckpt_results_seed{S}.json`.

---

## 3. Results

(appended after the fact — see §4 below)

---

# 4. RESULTS

Scripts: `pilot_ckpt_train.py` (training + fp32 checkpoint memmap),
`pilot_ckpt_eval.py` (basis, fSRD fits, baselines, ledger, dloss),
`pilot_ckpt_control.py` (positive control). Raw numbers:
`pilot_ckpt_results_seed{0,1}.json`, `pilot_ckpt_control.json`.

## 4.0 Notes on the design (implementation decisions forced by gaps; no
pre-registered threshold, margin, metric or split rule was changed)

- **The four byte budgets.** §1 references "the 4 byte budgets tested" but §0
  never names their generator. Budgets are defined as fSRD's own achieved
  byte totals at `rcond ∈ {1e-1, 1e-2, 1e-3, 1e-4}` (rcond is the only
  rank/size knob the solver exposes; `max_depth=3` per §1's G0), fixed before
  any fit was run. Comparators are sized into each budget by §0.3's floor rule.
- **Held-out decompression of the SVD-family arms.** §0 does not state how
  `svd`/`useg`/`sub` produce a column at fit-index i+0.5 from their stored
  bytes. All three use linear interpolation of their stored/reconstructed
  columns (midpoint average of columns i and i+1) — the natural full-strength
  decompressor; `sub` interpolates between its two nearest kept checkpoints.
  `fsrd`/`gdmd` evaluate the fitted Vandermonde at i+0.5 as §0.2(5) requires.
- **`sub` in coefficient space.** `sub` errors are computed in coefficient
  space + out-of-span residual, neglecting interpolation of the basis's own
  fit-residual (bounded by the reported SVD truncation residual, ~1e-7
  relative — nil).
- **Blend verification against a re-fit.** The fit pipeline is deterministic
  for a fixed input buffer, but two `fsrd()` calls on separately allocated
  copies of the same matrix can diverge on a near-tie split (BLAS
  kernel/alignment rounding), which changes the tree, not the blend. The
  binding §0.2(5) check is therefore asserted against the library's own
  `_reconstruct` applied to the *same* fitted leaves that are evaluated (that
  is `res.reconstruction` of the evaluated model, bit-for-bit); the diff
  against an independent public `fsrd()` re-fit is recorded as a diagnostic
  (`public_refit_recon_maxdiff`).
- **8 dloss checkpoints.** Held-out ordinals j ∈ {9, 32, 55, 78, 101, 124,
  147, 170} (training steps 4j+2 = 38, 130, 222, 314, 406, 498, 590, 682):
  uniform spread, step 38 < 40 (before warmup end), steps 406+ after the LR
  drop. Fixed before any fit.
- **naive bytes** = 349·P·4 (175 fit + 174 held-out columns actually
  represented; the 350th stored column is dropped by §0.1's split rule).
- **Objection recorded (design not changed):** under §0.3 every mode of every
  region is charged P floats, so fSRD's byte total is ≈ (Σ region ranks)·P·4
  and the temporal-coefficient savings DMD offers over SVD (4r floats vs
  R·N_fit) is ~0.07 % of a budget at P=4.8e5. fSRD can therefore only win
  via *better interpolation per stored P-vector*, not via its coefficient
  economy. This is a fair reading of the ledger, but it makes G1's 30 %
  margin very demanding a priori.

## 4.1 Positive control — planted piecewise-linear regimes: **DETECTED**

Synthetic trajectory, same shape (350 × 4.7e5, fp32): 3 regimes, each a
12-dim rotation/decay system in its own disjoint 12-dim subspace (global
rank exactly 36), switches planted at stored positions 111 and 236 =
fit-units 55.5 and 118.0 (off fSRD's split lattice); rotation rates kept
below fit-grid Nyquist. Pipeline identical to the real run; dloss does not
apply (synthetic parameters are not a network).

- Basis: K = 36 recovered exactly; ‖UᵀU−I‖_F = 1.6e-15 (after one Cholesky
  re-orthonormalisation; 3.9e-11 before); SVD truncation residual 2.8e-7.
- Blend verification: max abs discrepancy **0.0** at all four budgets.
- Regions: 4–5 per fit, splits 13/14 temporal. Planted switch 55.5 found at
  57/60/57/59 (within 1.5–4.5 fit-units) in 4/4 fits; switch 118.0 found at
  112/116/112 (within 2–6) in 3/4 fits (worst fit: 98, off 20).
- Held-out error at matched bytes (frob_rel_c, median over 174 columns):
  fSRD **0.0011–0.052** vs global SVD 0.025, useg 0.025–0.10, sub
  0.12–0.35, gdmd 0.53–0.75. fSRD beats the Frobenius-optimal global SVD by
  8–22× in the median at 3 of 4 budgets (means: fsrd 0.037–0.088 vs svd
  0.051 — the mean is dominated by the fuzzy-boundary columns, where the
  blend of two regimes is soft; fSRD's mean still beats svd at 3/4 budgets).

The protocol detects planted structure: a null on the real data is
therefore attributable to the data, not the pipeline.

## 4.2 Real data — setup and mandatory checks

Training (`pilot_ckpt_train.py`, seeds 0 and 1): P = 480 000, loss
5.24→2.11 / 5.19→2.10, ~44 s/seed; 350 fp32 checkpoints = 672 MB/seed,
deleted after evaluation. naive = 349·P·4 = 670 080 000 B.

| check (per §0.2 / task) | seed 0 | seed 1 |
|---|---|---|
| K (all σ/σ₀ > 1e-7 kept) | 174 | 174 |
| ‖UᵀU−I‖_F (after 1 Cholesky pass; before) | 4.7e-15 (2.6e-9) | 4.8e-15 (2.4e-9) |
| SVD truncation residual ‖A_c−UC‖/‖A_c‖ | 1.1e-15 | 9.7e-16 |
| blend verification, max abs diff, worst of 4 fits (recon scale ≈ 26) | 6.7e-15 | 1.95e-14 |
| independent public `fsrd()` re-fit recon maxdiff | 0.0 (all 4) | 0.0 (all 4) |
| unpaired complex modes (charged double) | 0 | 0 |
| held-out columns in any fit | none (asserted) | none (asserted) |

The four budgets (fSRD achieved bytes; ratio vs naive):

| budget | seed 0 | seed 1 |
|---|---|---|
| B1 (rcond 1e-1) | 38.4 MB (17.4×), 6 regions, ranks [7,4,2,2,2,2] | 40.3 MB (16.6×), 8 regions, ranks [1,1,1,2,7,4,2,2] |
| B2 (rcond 1e-2) | 188.2 MB (3.56×), 8 regions | 307.2 MB (2.18×), 8 regions |
| B3 (rcond 1e-3) | 345.6 MB (1.94×), 3 regions | 330.2 MB (2.03×), 3 regions |
| B4 (rcond 1e-4) | 334.1 MB (2.01×), **1 region** | 334.1 MB (2.01×), **1 region** |

All splits axis-aligned; 12/14 (seed 0) and 13/16 (seed 1) of unique splits
temporal. B1 boundaries (steps): seed 0 = 44, 124, 240, 340, 464; seed 1 =
44, 124, 240, 340, 492, 568, 632 — again within a few steps of the warmup
end (40) and LR drop (350), duplicating the loss curve as in the sibling
pilot. In-window relative error of the fits: 0.017–0.091 (cf. 5.1e-4 on the
256-dim sketch in the sibling pilot; the full-P coefficient trajectory is a
much harder object).

**Degeneracy of the metric regime, stated before the gate table.** Every
arm's mean `dloss` is *negative* at every budget on both seeds (except
gdmd): reconstructed checkpoints have *lower* validation loss than the true
checkpoints, because low-rank/averaged reconstructions denoise SGD noise —
indeed μ alone (the mean fit checkpoint) scores mean dloss −0.038 / −0.034,
better than every arm at every budget. The pre-registered ratio gates were
written for positive "excess loss"; with negative D they are applied
literally as inequalities, and G1 in particular can "pass" at a budget
where fSRD is numerically *worse* than SVD (seed 0, B2). Flagged here,
gates unchanged.

Also: at B3/B4 (and seed 1 B2) the budget exceeds the cost of storing all
K=174 coefficients, so svd/useg/sub saturate (lossless on fit columns,
midpoint-interpolation error only, identical dloss −0.00584 / −0.00783);
fSRD at those budgets still carries 1.7–3.7 % in-window error.

## 4.3 Gate table (mean dloss over the 8 held-out checkpoints, nats)

Seed 0:

| budget (ratio) | fsrd | svd | useg | sub | gdmd | G1 (≤0.7·svd) | G2 (≤0.8·useg) | G3 (≤0.8·sub) |
|---|---|---|---|---|---|---|---|---|
| B1 (17.4×) | **−0.01553** | −0.01375 | **−0.02261** | −0.01038 | +0.04810 | pass | FAIL | pass |
| B2 (3.6×) | −0.00555 | **−0.00627** | −0.00572 | **−0.00764** | +0.06838 | pass | pass | FAIL |
| B3 (1.9×) | **−0.00643** | −0.00584 | −0.00584 | −0.00584 | +0.00924 | pass | pass | pass |
| B4 (2.0×) | −0.00059 | **−0.00584** | −0.00584 | −0.00584 | −0.00059 | FAIL | FAIL | FAIL |

Seed 1:

| budget (ratio) | fsrd | svd | useg | sub | gdmd | G1 | G2 | G3 |
|---|---|---|---|---|---|---|---|---|
| B1 (16.6×) | −0.01704 | −0.01627 | **−0.02751** | −0.01078 | +0.04671 | pass | FAIL | pass |
| B2 (2.2×) | **−0.00868** | −0.00783 | −0.00787 | −0.00783 | +0.06305 | pass | pass | pass |
| B3 (2.0×) | **−0.01107** | −0.00783 | −0.00782 | −0.00783 | +0.02026 | pass | pass | pass |
| B4 (2.0×) | −0.00304 | **−0.00783** | −0.00783 | −0.00783 | −0.00304 | FAIL | FAIL | FAIL |

(Medians in the JSONs; same picture. gdmd = global DMD, max_depth=0, at
its own achieved bytes: 11.5–334 MB.)

Gate verdicts (a gate passes only if it passes at ≥3 of 4 budgets on BOTH
seeds):

- **G0 (degeneracy auto-fail): PASS** — n_regions == 1 on 2 of 8
  max_depth=3 fits (both B4), not more than half.
- **G1 (vs global SVD): PASS — but only via the sign degeneracy.**
  3/4 budgets on each seed satisfy the literal inequality. Note: at seed 0
  B2 fSRD is *worse* than SVD (−0.00555 vs −0.00627) yet the inequality
  −0.00555 ≤ 0.7·(−0.00627) holds; with positive excess losses this gate
  would not be meaningful here.
- **G2 (vs uniform-boundary segmentation): FAIL** — 2/4 budgets on both
  seeds. At the only ≥4× budget (B1), useg beats fSRD outright on both
  seeds (−0.0226 vs −0.0155; −0.0275 vs −0.0170) and is the best arm
  overall: *uniform* boundaries with per-segment SVD beat fSRD's adaptive
  placement.
- **G3 (vs subsampling+linear interpolation): FAIL** — 3/4 on seed 1 but
  2/4 on seed 0 (sub beats fSRD at seed 0 B2: −0.00764 vs −0.00555).
- **G4 (usefulness floor): PASS** — at B1 (ratio ≥ 4× on both seeds), mean
  dloss −0.0155 / −0.0170 ≤ 0.05 nats. (Trivially satisfied in the
  negative-dloss regime; even μ alone would pass it.)
- **Explicit NO-GO clause: FIRES on seed 0** — `D_svd ≤ D_fsrd` at 2 of 4
  budgets (B2, B4) and `D_sub ≤ D_fsrd` at 2 of 4 (B2, B4), strict
  inequality, no tolerance needed.

Secondary Frobenius metrics (registered): raw `frob_rel` is 5e-4–9e-3 for
every arm at every budget — it discriminates poorly, exactly as §0.4
predicted. On `frob_rel_c` (median): fSRD loses to matched-bytes global
SVD at B2–B4 on both seeds (up to 7×: 0.0211 vs 0.0029 at seed 1 B2) and
roughly ties at B1 (0.064 vs 0.062; 0.071 vs 0.062); useg and sub are
also ahead of fSRD nearly everywhere. gdmd is far behind everyone
(frob_rel_c median up to 0.99) — the regions genuinely help over a global
operator, as in the sibling pilot, and still lose to classical baselines.

## 4.4 VERDICT: **NO-GO**

Applying §1's verdict rule literally:

- GO requires G1∧G2∧G3∧G4: **no** (G2, G3 fail).
- PARTIAL requires G0∧G1∧G3 with G2 failing: **no** (G3 fails on seed 0).
- NO-GO: the explicit clause (`D_svd ≤ D_fsrd` or `D_sub ≤ D_fsrd` at ≥2 of
  4 budgets) fires on seed 0; a positive result was required to survive
  both seeds. **NO-GO.**

Deciding facts: (1) at the only byte budget with a ≥4× storage ratio,
plain uniform-boundary per-segment SVD is the best method and fSRD's
adaptive boundary placement subtracts value — "segmentation helps; fSRD's
placement is ceremony" describes B1, though the pre-registered PARTIAL
verdict is unavailable because G3 also fails; (2) at the three finer
budgets fSRD's byte total buys the comparators effectively lossless
storage of the fit columns, so it cannot win there except through the
denoising accident; (3) the dloss metric itself collapsed into a
denoising contest (all arms negative, μ alone best of all), which no
method "wins" in the pre-registered sense. The eighth application of this
solver in this project falls to the same pattern as the previous seven:
the cheap classical baseline wins wherever it exists.
