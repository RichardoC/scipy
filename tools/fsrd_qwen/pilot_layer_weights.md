# PILOT — fSRD on the sequence of weight tensors across layer index

Candidate: **weight matrices across depth**. For one tensor kind, treat
`W_0, W_1, ..., W_63` (layer index as the ordered axis) as a trajectory and ask whether
there is an operator `A` with `W_{l+1} ≈ A W_l` over *bands* of layers, with fSRD choosing
the bands. Distinct from the two failed prior experiments (fSRD on *activations* across
depth; fSRD on a *single static* weight matrix for quantization).

Model: `unsloth/Qwen3.6-27B-MTP-GGUF`, file `Qwen3.6-27B-Q6_K.gguf` (Q6_K chosen over
Q4_K_M because Q4 dequantization error ~2-3% sits at the ±3% noise level where the fSRD
paper's own factorial study shows partition discovery starts to fail; Q6_K is ~0.3%).
Tensor kind: `blk.L.ffn_gate.weight`, numpy shape (17408, 5120) = (n_ff, n_embd),
blocks L = 0..63 (block 64 is the MTP head — excluded). `ffn_gate` exists in every block,
so the sequence is homogeneous in tensor kind and contiguous in layer index; that is what
lets the period-4 `full_attention_interval` check be meaningful.

---

## PRE-REGISTRATION (written before any numbers were produced)

### Reductions and what they can hide

A full tensor is 356 MB in fp32; 64 of them cannot be held. Two per-layer signatures are
computed in a single streaming pass (fetch → reduce → delete), both from **shared, seeded**
random projections so that every layer is reduced identically:

1. **RAW probe signature** `x_l = W_l q`, with `q ∈ R^5120` a fixed seeded unit Gaussian
   (one probe per replicate). Right-multiplication by a shared `q` is *exactly* compatible
   with a left-acting operator: `W_{l+1} = A W_l ⟹ x_{l+1} = A x_l`. So this signature
   tests the literal premise without deforming it. It **hides** structure that a single
   probe direction misses (mitigated by running 4 independent probe replicates) and it
   cannot see a right-acting operator `W_{l+1} ≈ W_l B`.
2. **Basis-invariant Gram signature** `g_l = vec( (W_l Q)^T (W_l Q) / n_ff )` with
   `Q ∈ R^(5120×64)` shared and seeded → 4096-dim. This is invariant to any orthogonal
   transform or permutation of the FFN-neuron axis (the axis that has **no** canonical
   alignment across layers), while living entirely in the residual-stream basis (dim 5120),
   which *is* canonically aligned across layers. It **hides** everything that lives purely
   in the neuron axis, and it is quadratic in `W`, so a positive result here would be
   evidence of layer-to-layer dynamical structure but *not* directly a scheme for
   regenerating `W`.

Both signatures are then projected onto the top-`r = 16` POD (left-singular) directions of
the **training columns only** (no test leakage). fSRD does no dimensionality reduction of
its own and §3 of INVESTIGATION.md explicitly says to PCA-project first. All errors are
reported in the **original signature space** after lifting back through the POD basis, so
out-of-span mass in held-out layers is charged as error, and every method (including copy
and mean, which are in-span by construction) is judged on the same footing.

### Held-out protocol

For cut points `T0 ∈ {40, 48, 56, 60}`: train on layers `0..T0-1`, forecast `h = 4`,
score against true layers `T0..T0+3`. Metric: relative L2
`||pred - true||_F / ||true||_F` over the forecast block, averaged over the 4 cut points
(and over the 4 probe replicates for signature 1).

Methods compared at every cut:
- **copy**: repeat the last training column.
- **mean**: mean of the training columns.
- **global DMD**: `fsrd(..., max_depth=0, forecast=h)` — one operator, no regions.
- **fSRD**: `fsrd(..., max_depth=2, oblique=False, forecast=h)` — BIC-chosen regions.
- **period-4 split**: fit a separate global DMD per residue class `l mod 4` (the known
  `full_attention_interval: 4` structure) and forecast each — the trivial-prior baseline.
- **thirds split**: separate global DMD on early/middle/late contiguous thirds of training.

### Thresholds (fixed now)

- **GO requires**: fSRD's mean held-out forecast relative L2 ≤ **0.75 ×** the best of
  {copy, mean, global DMD}, on at least one of the two signature types, **and** the best
  method's absolute error < **0.5** (if nothing predicts held-out layers at all, regions
  are moot).
- **Robustness requirement for any GO**: re-running with a different projection seed must
  still show fSRD ≤ **0.85 ×** best baseline. Otherwise the positive is a reduction artifact
  and the verdict is NO-GO.
- **Degeneracy auto-fail**: if BIC returns `n_regions == 1` in ≥ **50%** of the
  `max_depth=2` fits, fSRD is adding nothing over global DMD by construction — report that
  as a NO-GO on the region-discovery claim regardless of error numbers.
- **Consecutive-layer similarity, interpreted in advance**: mean
  `cos(vec W_l, vec W_{l+1})` over full tensors. If > **0.9**, the premise is trivially
  true and uninteresting (copy baseline will be near-perfect and there is no science here).
  If < **0.05**, successive raw tensors are near-orthogonal — no cross-layer basis
  alignment exists — and the raw-probe premise is hopeless a priori, leaving the
  basis-invariant Gram signature as the only meaningful test. This number is reported first
  and may settle the pilot on its own.
- **Region-boundary check**: report fSRD's split column indices and whether they land on
  layer indices ≡ 3 (mod 4) more often than chance, and whether the trivial period-4 split
  baseline matches or beats fSRD.

### Rules

A clean no-go is a successful pilot. No metrics are added after seeing results.

---

## RESULTS

**Verdict: NO-GO on both tracks.** Raw-tensor track dies on the pre-registered
consecutive-similarity kill condition. Invariant track survives that condition easily but
fSRD loses to a trivial baseline by 2-14x on every signature and every seed.

Compute: 64 `ffn_gate` tensors streamed in 944 s / 4.36 GiB downloaded, peak RSS 736 MB, no
tensor ever written to disk. All fSRD fits are milliseconds at these shapes.

### Protocol validation (positive control) — `pilot_layer_control.py`, `results/pilot_layer_control.json`

Before trusting any null: a genuine 2-regime linear system planted in a signature space of
the same shape (D=4096, L=64, latent rank 12, 1% noise) is recovered by the same evaluator
at held-out rel-L2 **0.011-0.027** (cuts 48/56/60) versus 0.95-1.02 for mean/global-DMD —
aggregate ratio **0.395**, passes the GO gate — and the fitted region boundary lands at
column **30-32 for a true switch at 32**. The protocol has power and fSRD can find layer
bands when they exist. Every null below is therefore a statement about the weights.

### Track 1 — raw tensors: NO-GO, killed by consecutive similarity

| quantity | value |
|---|---|
| mean `cos(vec W_l, vec W_{l+1})`, full tensors | **0.00444** (range 0.00092 – 0.01191) |
| mean `‖W_{l+1}-W_l‖_F / ‖W_l‖_F` | **1.415** |

The pre-registered kill condition (`< 0.05`) fired: successive `ffn_gate` tensors are
effectively **orthogonal**, and "copy the previous layer" is *worse than predicting zero*
(1.415 > 1.0). Under a shared random projection inner products are preserved
(Johnson–Lindenstrauss), so this is a property of the weights, not of the reduction. The
cause is the one anticipated in the brief: the FFN-neuron axis has no canonical alignment
across layers, and nothing in training creates one.

Held-out forecast, mean rel-L2 over 4 cuts x 4 probes (`results/pilot_layer_weights.json`):

| signature | copy | mean | global DMD | period-4 | thirds | **fSRD d2** | ratio vs best |
|---|---|---|---|---|---|---|---|
| raw probe, seed 0 | 1.401 | 1.007 | 1.043 | 1.325 | 1.012 | **1.232** | 1.22 |
| raw probe, seed 1 | 1.399 | 1.005 | 1.033 | 1.324 | 1.013 | **1.168** | 1.16 |

Nothing beats predicting the training mean, which here is indistinguishable from predicting
nothing. POD energy at r=16 is only 0.35-0.48 — the raw sequence is not even low-rank.
Dead in the raw-tensor basis; not rescued, not re-run.

### Track 2 — basis-invariant signatures: NO-GO, fSRD loses to "copy the last layer"

Signatures invariant to any rotation/permutation of the neuron axis. Consecutive-layer
correlation of the *signature* is indeed high, exactly as expected — invariants drift
smoothly across depth:

| signature | consecutive cos | mean-centered |
|---|---|---|
| projected Gram `vec((W Q)^T(W Q))`, 4096-d | 0.9938 | 0.617 |
| log-spectrum (64 eigenvalues of that Gram) | 1.0000 | 0.840 |
| log column-norm quantiles (65) | 0.9989 | 0.854 |
| composite (z-scored spectrum ++ quantiles ++ log‖W‖) | 0.9373 | 0.861 |

So the premise's *precondition* holds here. It does not help. Held-out forecast, mean
rel-L2 over the 4 cuts (`results/pilot_layer_invariant.json`):

| signature | copy | mean | global DMD | period-4 | thirds | fSRD d1 | **fSRD d2** | ratio vs best |
|---|---|---|---|---|---|---|---|---|
| Gram, seed 0 | **0.1427** | 0.2053 | 0.1813 | 0.2955 | 0.1918 | 1.261 | 1.452 | **10.2** |
| Gram, seed 1 | **0.1462** | 0.2096 | 0.2128 | 0.3002 | 0.1964 | 4.376 | 2.102 | **14.4** |
| spectrum, seed 0 | **0.0070** | 0.0207 | 0.0206 | 0.0336 | 0.0223 | 0.0163 | 0.0143 | **2.06** |
| spectrum, seed 1 | **0.0069** | 0.0208 | 0.0208 | 0.0342 | 0.0578 | 0.0157 | 0.0246 | **3.58** |
| norm quantiles | **0.0814** | 0.2136 | 0.2120 | 0.3617 | 0.1836 | 0.2294 | 0.2690 | **3.30** |
| composite, seed 0 | **0.4561** | 0.9302 | 0.9114 | 1.7709 | 1.4813 | 1.349 | 0.714 | **1.57** |
| composite, seed 1 | **0.4510** | 0.9339 | 0.8970 | 1.7924 | 1.0249 | 1.614 | 0.876 | **1.94** |

Three facts, in order of importance:

1. **fSRD never beats the best baseline on any signature or seed** — ratios 1.57 to 14.4,
   against a GO threshold of ≤ 0.75. Failing in the same direction on both projection seeds
   makes this the opposite of a reduction artifact.
2. **Global DMD also loses to `copy`** on every signature (0.181 vs 0.143 on Gram; 0.0206 vs
   0.0070 on the spectrum). The layer axis is not a dynamical system here; it is a slow drift
   that a zeroth-order hold predicts better than any fitted linear operator.
3. **The failure mechanism is instability, not degeneracy.** Fitted region eigenvalues reach
   `|λ|max = 3.2 – 4.6`; a 4-step Vandermonde rollout of a `|λ|>1` mode blows up, which is why
   fSRD's error can exceed 1.0 on a signature whose consecutive correlation is 0.99. With only
   48-60 columns, extrapolating a fitted operator is worse than not extrapolating.

### Degeneracy check — did NOT fire

| track | depth-2 fits | 1 region | region counts |
|---|---|---|---|
| raw + Gram | 40 | 3 (**7.5%**) | 1:3, 2:13, 3:9, 4:15 |
| invariant | 24 | 0 (**0%**) | 3:1, 4:23 |

Unlike the two earlier experiments in this project, BIC did **not** collapse to one region.
fSRD happily splits the layer axis — it just splits it unprofitably. This is a distinct,
cleaner failure mode: the regions are real model structure, they simply do not generalise.

### Region boundaries vs the known period-4 structure

95 fitted boundaries over all fits, at columns
`12,13,14,16,18,20,23,25,27,30,32,36,38,39,41,43,46,49,52`. Residues mod 4:
`{0:31, 1:10, 2:39, 3:15}` — the fraction landing on the `full_attention` layers
(`l ≡ 3 mod 4`) is **0.158, below the 0.25 chance rate**. No alignment with
`full_attention_interval: 4`. The boundaries instead cluster at roughly 1/3, 1/2 and 2/3 of
whatever training window is used (20/40 at T0=40; 19/37/46 at T0=56; 20/40/52 at T0=60),
i.e. near-proportional bisections of the index grid rather than discovered regimes. And the
trivial period-4 split baseline is *worse than global DMD and worse than copy* on every
signature (e.g. 0.296 vs 0.143 on Gram, 0.0336 vs 0.0070 on the spectrum), so the known
periodicity is not exploitable for forecasting these signatures either — by fSRD or by hand.

### What the one positive-looking number is and is not

`copy` predicts the next 4 layers' log-spectrum to **0.7%** relative error. That is a real
fact — projected singular-value spectra drift very smoothly with depth — but it is a
statement about the *baseline*, not about fSRD, and it **cannot be a memory win**: knowing a
layer's spectrum (or its column-norm quantiles, or its projected Gram) does not let you
reconstruct the layer, because everything that was thrown away is exactly the basis
information the raw track just showed is unshared between layers. At best this is an
interpretability observation, and it needs no Koopman machinery to make it — a
zeroth-order hold is strictly better than every operator fitted here.

