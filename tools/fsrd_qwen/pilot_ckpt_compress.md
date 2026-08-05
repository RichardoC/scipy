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

(appended after the fact)
