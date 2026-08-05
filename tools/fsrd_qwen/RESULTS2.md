# The two pre-registered memory experiments — results

**Date:** 2026-08-05. **Spec:** `INVESTIGATION2.md` §5 (arm A:
fSRD-partitioned adaptive-precision weight quantization, on real
Qwen3.6-27B weights) and §5b (arm B: DeltaNet state truncation, on the
Qwen3.5-0.8B stand-in), executed as pre-registered. **Hardware:** 4 CPU
cores, 15 GB RAM, no GPU; 27B weights reached by HTTP range requests into
the BF16 shard of `unsloth/Qwen3.6-27B-MTP-GGUF` (1.17 GB fetched at
13–27 MiB/s; every tensor deleted as soon as its statistics were computed;
disk never fell below 15 GB free).

**Scope cap, restated before any result (spec §5):** arm A is an *offline
codec* comparison on an imatrix-weighted relative-RMSE proxy — the metric
family llama.cpp quant development itself uses — on 8 real Qwen3.6-27B
tensors. No 27B inference is possible on this box, so there is **no
end-to-end perplexity claim and no kernel-speed claim**; the deployable
subset of any win would be row-band regions served as split tensors. Arm B
runs entirely on the 0.8B stand-in (18 DeltaNet layers × 16 heads of
128×128 fp32 state, vs the 27B's 48 layers × 48 heads), so its conclusions
transfer to Qwen3.6-27B **as hypothesis only**. And arm B is explicitly
**not an fSRD experiment**: its tool is truncated SVD; it is reported
because it is the one testable memory idea (§5b registered it as exactly
that).

## Verdicts

- **Arm A (the fSRD experiment): FAIL.** fSRD-region allocation met the
  PASS conjunction (≥10% better than B0 *and* ≥5% better than B2) on
  **0 of 8** tensors. It is *worse* than the uniform baseline on 6 of 8
  (median margin vs B0 **−6.8%**), and worse than the 1-D sorted-band
  baseline B2 on **8 of 8** (median **−18.7%**). The auto-FAIL degeneracy
  clause did **not** fire (BIC returned >1 region on 8/8 tensors under the
  best config; even across all sweep configs no tensor was all-1-region) —
  the method segmented, and segmenting didn't help. Where adaptive bits do
  pay (attention projections, +47…+56% vs B0), the 20-line 1-D baseline
  collects far more of the same win (+75%, +64%), so not even the PARTIAL
  verdict's condition ("beats B0 by ≥10% on ≥6 of 8") is met: 2 of 8.
- **Arm B (not fSRD — plain truncated SVD): PASS on every gate.**
  At-rest rank-16 one-shot truncation: mean KL **0.00331 nats** (gate
  ≤ 0.05) and top-1 agreement **96.84%** (gate ≥ 95%) — an **8.0×**
  at-rest state compression is viable as engineering on the stand-in. The
  kill clause (rank-32 KL > 0.2) measured **0.00066** — not remotely
  triggered. In-loop re-truncation every 64 tokens: rank-32 perplexity
  delta **−0.14%** (gate ≤ +2%), rank-64 **−0.09%** (kill at > +5%) —
  cost indistinguishable from zero.

## Gates table

| gate (pre-registered) | threshold | measured | verdict |
|---|---|---|---|
| A PASS: fSRD vs B0 margin | ≥10% lower weighted rel-RMSE, on ≥6/8 | median **−6.8%**; ≥10% on **2/8** | **FAIL** |
| A PASS: fSRD vs B2 margin | ≥5% lower, on ≥6/8 | median **−18.7%**; ≥5% on **0/8** | **FAIL** |
| A auto-FAIL: BIC returns 1 region | on ≥4/8 tensors | **0/8** (best config; 0/8 all-config) | not triggered |
| A PARTIAL: beats B0 ≥10% (but not B2) | on ≥6/8 | **2/8** | not met |
| A sanity: B0 reproduces GGUF Q4_K error scale | ≈7.4% rel-Frobenius | B0 unweighted 7.86–7.91% on all 8; GGUF's own Q4_K on blk.3.attn_q = **7.38%** | **PASS** |
| B at-rest rank-16: mean KL | ≤ 0.05 nats | **0.00331** | **PASS** |
| B at-rest rank-16: top-1 agreement | ≥ 95% | **96.84%** | **PASS** |
| B kill: one-shot rank-32 KL | > 0.2 nats kills candidate | **0.00066** | not triggered |
| B in-loop rank-32: rel. perplexity | ≤ +2% | **−0.14%** (within noise, see below) | **PASS** |
| B in-loop rank-64: rel. perplexity | > +5% = in-loop dead | **−0.09%** (within noise) | not triggered |

## What was run

| stage | script | scale | wall clock |
|---|---|---|---|
| A fetch | `inv2a_fetch.py` | 8 BF16 tensors (1.17 GB) + imatrix vectors, HTTP range requests via `gguf_fetch.py` machinery | ~1 min (overlapped with arm B) |
| A prep | `inv2a_prep.py` | per-tensor exact per-(row, 32-block) weighted SSE tables at b∈{2,3,4,5,6,8}, Z maps, B0/B1, sanity | 6.3 min |
| A fSRD | `inv2a_fsrd.py` | 64 fits: 8 tensors × {depth 2,3} × {asc,desc} × {weighted,unweighted}, `fsrd(Z, oblique=False, theta=3, smoothness=0.03)` | 29.6 min (3.8–72.9 s/fit, uncontended) |
| A eval | `inv2a_eval.py` | B2 joint band+bits DP; exact (exhaustive) fSRD region allocation; ledgers; gates | ~2 min |
| B | `inv2b_state_trunc.py` | 20 streams × 1024 tokens, teacher-forced, fp32 CPU | 33.5 min |

Codec used everywhere in arm A (all methods, including both baselines):
Q_K-style — 32-element blocks, asymmetric scale+min each quantized to 6 bits
against per-256-superblock fp16 references — exactly **b + 0.5 bits/weight**.
The per-level SSE tables are produced by actually quantizing at every level,
so allocation is over *measured* error, not a model of it, and the final
numbers are exact. Budget for every method: B0's total, **4.5 bpw**.

## Arm A — per-tensor results (imatrix-weighted relative RMSE, equal bits)

fSRD column = best of the 8 pre-registered sweep configs per tensor (the
choice most favourable to H1; the sweep is the entire allowance and nothing
else was tuned).

| tensor (np shape) | B0 uniform | B1 sorted | B2 1-D bands | fSRD (best cfg, regions) | fSRD vs B0 | fSRD vs B2 | B2 vs B0 |
|---|---|---|---|---|---|---|---|
| blk.0.ffn_down (5120×17408) | 0.07779 | 0.08053 | 0.05675 | 0.08260 (w-asc-d3, 6) | −6.2% | −45.6% | **+27.0%** |
| blk.20.ffn_down (5120×17408) | 0.07845 | 0.07888 | 0.07790 | 0.08939 (u-asc-d3, 5) | −13.9% | −14.7% | +0.7% |
| blk.40.ffn_down (5120×17408) | 0.07879 | 0.07894 | 0.07883 | 0.08476 (w-desc-d3, 6) | −7.6% | −7.5% | −0.0% |
| blk.0.ffn_gate (17408×5120) | 0.08009 | 0.07916 | 0.07640 | 0.07664 (u-desc-d3, 8) | +4.3% | −0.3% | +4.6% |
| blk.20.ffn_gate (17408×5120) | 0.08131 | 0.07993 | 0.07665 | 0.08735 (w-desc-d3, 5) | −7.4% | −14.0% | +5.7% |
| blk.40.ffn_gate (17408×5120) | 0.07931 | 0.08187 | 0.06486 | 0.08618 (w-desc-d3, 6) | −8.7% | −32.9% | **+18.2%** |
| blk.3.attn_q (12288×5120) | 0.05979 | 0.08528 | **0.01488** | 0.03187 (w-asc-d3, 6) | **+46.7%** | −114.2% | **+75.1%** |
| blk.1.attn_qkv (10240×5120) | 0.13033 | 0.14703 | **0.04703** | 0.05765 (w-asc-d3, 6) | **+55.8%** | −22.6% | **+63.9%** |
| **median** | | | | | **−6.8%** | **−18.7%** | **+12.0%** |

(Positive margin = fSRD/B2 better, i.e. lower error. Full per-config
numbers, boxes and BIC values: `results/inv2a_fsrd.json`,
`results/inv2a_eval.json`.)

### Bit ledger (itemised, bits/weight; budget 4.5000 for every method)

| tensor | method | payload | scales (6+6b/32) | fp16 supers | region/band table | permutations | total |
|---|---|---|---|---|---|---|---|
| blk.0.ffn_down | B0 | 4.0000 | 0.3750 | 0.1250 | — | — | 4.5000 |
| | B2 (5 bands: 8/6/5/4/3-bit) | ≈4.121 | 0.3750 | ≈0.125 | 0.0000008 | 0.00293 | 4.4993 |
| | fSRD (6 reg: 4,4,4,4,4,3) | 3.9084 | 0.3750 | 0.1255 | 0.000005 | 0.00368 | 4.4126 |
| blk.20.ffn_down | fSRD (5 reg: 3,4,4,4,5) | 3.9734 | 0.3750 | 0.1250 | 0.000004 | 0.00368 | 4.4771 |
| blk.40.ffn_down | fSRD (6 reg: 4,4,3,4,4,4) | 3.9355 | 0.3750 | 0.1253 | 0.000005 | 0.00368 | 4.4395 |
| blk.0.ffn_gate | fSRD (8 reg: 3,4,4,4,4,4,5,5) | 3.9750 | 0.3750 | 0.1437 | 0.000006 | 0.00368 | 4.4974 |
| blk.20.ffn_gate | fSRD (5 reg: 4,4,4,3,4) | 3.9375 | 0.3750 | 0.1375 | 0.000004 | 0.00368 | 4.4537 |
| blk.40.ffn_gate | fSRD (6 reg: 4,4,3,4,4,4) | 3.9375 | 0.3750 | 0.1500 | 0.000005 | 0.00368 | 4.4662 |
| blk.3.attn_q | B2 (4 bands: 8/5/4/3-bit) | ≈4.113 | 0.3750 | ≈0.125 | 0.0000009 | 0.00106 | 4.4886 |
| | fSRD (6 reg: 6,6,5,3,3,3) | 3.8927 | 0.3750 | 0.1302 | 0.000006 | 0.00379 | 4.4018 |
| blk.1.attn_qkv | B2 (5 bands: 8/6/5/4/3-bit) | ≈4.113 | 0.3750 | ≈0.125 | 0.0000013 | 0.00127 | 4.4888 |
| | fSRD (6 reg: 8,4,3,4,4,3) | 3.9637 | 0.3750 | 0.1375 | 0.000008 | 0.00400 | 4.4802 |

Reading the ledger: metadata is *not* what kills fSRD — permutations cost
0.003–0.004 bpw and region tables are microscopic, so no win "vanished once
permutation metadata was charged". What kills it is structural: (i)
region-level allocation is coarse — six discrete bit levels over ~6 large
regions cannot spend the last ~0.1 bpw of budget, so fSRD tensors run at
4.40–4.48 bpw against B2's 4.49–4.50 (a real, honestly counted disadvantage
of the representation, not an accounting artifact); (ii) fSRD's split
positions come from its deterministic internal grid (fractions
≈0.35/0.5/0.65 recursively), so its column boundaries land far from the
importance cliffs that B2's freely-placed band edges find (B2 routinely
spends 8 bits on a *single* 32-column block of extreme-importance columns —
fSRD cannot express that); (iii) on tensors with no exploitable importance
skew (ffn_down/gate at layers 20/40, where B2 ≈ B0), sorting + regioning is
pure cost.

### Sanity anchors (binding rule 7)

- GGUF's own Q4_K dequant of `blk.3.attn_q` vs BF16: **7.38%**
  rel-Frobenius (re-measured here; the prior probe said ~7.4%).
- Our B0 codec at b=4 on the same tensors: **7.86–7.91%** unweighted
  rel-Frobenius across all 8 — the gap to 7.38% is llama.cpp's iterative
  weighted grid search, which we do not replicate; the scale is right, so
  margins against B0 are meaningful. At b=6 the codec gives 1.92% on real
  rows (GGUF Q6_K: ~1.8%).
- Caveat cutting *against* the B2 headline: our B0 is a naive min/max
  Q4_K, while shipped GGUFs already use imatrix-weighted block fitting.
  Some of B2's +75%/+64% on attention tensors overlaps with what
  llama.cpp's weighted search already captures inside each block; the
  B0/B1/B2/fSRD comparison is internally consistent (same codec
  everywhere), but a B2-vs-ecosystem margin would be smaller than the
  B2-vs-B0 margin here.

### Why the attention tensors are different

The imatrix importance skew is extreme there (max/median ≈ 350× for
attn_q, ≈ 200× for attn_qkv, vs ≈ 4–30× for the FFN tensors). Any method
that can give a few hundred high-importance input columns more bits wins
big on the *weighted* metric; B2 expresses this perfectly with 3–5 sorted
bands, fSRD only coarsely. This is also exactly the structure llama.cpp's
imatrix-guided quantization already targets.

## Arm B — DeltaNet state truncation on the 0.8B stand-in (not fSRD)

Protocol: 20 fixed streams × 1024 tokens (wt2-holdout text first, then wt2
text; nothing in this arm is fitted, so "held-out" means fixed in advance,
not tuned). Teacher-forced throughout — no sampling anywhere; truncated and
untouched runs consume byte-identical token windows, so the KL/ppl deltas
measure truncation and nothing else. Determinism control (binding rule 7):
an untouched re-run through the full compare path gives mean KL
**2×10⁻⁶ nats** and top-1 agreement **1.0000** — the protocol noise floor
for part 1 is ~3 orders of magnitude below the smallest truncation effect
measured.

### Part 1 — at-rest (prefix-cache resume): prefill 256, truncate every S once, continue 256

| condition | mean KL (nats) | top-1 agreement |
|---|---|---|
| untouched control | 0.000002 | 1.0000 |
| rank 8 | 0.009254 | 0.9494 |
| rank 16 | **0.003309** | **0.9684** |
| rank 32 | 0.000657 | 0.9850 |

Gate (rank 16: KL ≤ 0.05, top-1 ≥ 95%): **PASS**, by 15× on KL.
Rank 8 passes the KL threshold but misses the top-1 gate at 94.94% — an
earlier 2-stream pilot's 95.7% was small-sample optimism — so **rank 16 is
the operating point**. Honest compression accounting: a rank-16 factored
state stores both factors, U and V, 2 × 128×16 fp16 = 8,192 B per
head-state vs 128×128 fp32 = 65,536 B → **8.0×** (rank 8 would be 16×,
rank 32 is 4×). Applied to Qwen3.6-27B's 151 MB/seq DeltaNet state this is
151 → ~18.9 MB per cached prefix — **as hypothesis only** (the 27B has 48
heads/layer vs 16 here, and its spectra were never measured).

### Part 2 — in-loop: re-truncate every S every 64 tokens during a 1024-token stream

| condition | ppl (teacher-forced) | rel. delta vs untouched |
|---|---|---|
| untouched (same chunked protocol) | 21.919 | — |
| rank 16 | 21.941 | +0.10% |
| rank 32 | 21.888 | −0.14% |
| rank 64 | 21.901 | −0.09% |

Gate (rank 32 ≤ +2%): **PASS**. Kill (rank 64 > +5%): not triggered.
**Noise statement:** all three deltas are within ±0.15% and two are
*negative*, which genuine truncation damage cannot produce — the numbers
are consistent with zero cost at the measurement's noise level, and no
rank ordering should be read into them. Per-stream spread was not measured
for these runs: the per-stream accumulation code was added after the
20-stream job had started (a Python process reads its source once), and
re-running ~35 min of inference to put error bars on deltas this far below
the gate was judged not worth the wall-clock. The honest summary is:
*in-loop re-truncation at rank ≥16 costs nothing measurable here; spread
not measured.*

The in-loop result exceeds what §5b dared hope (the gate anticipated paying
up to 2% at rank 32); on the stand-in the recurrence simply does not notice
rank-32 truncation at 64-token cadence, so the §4 worry that truncation
error compounds through the recurrence is not visible at this scale and
cadence. Also note this arm's own framing once more: none of this involves
fSRD — the tool is per-head truncated SVD, exactly as INVESTIGATION2.md §4
Rank 2 predicted ("plausible engineering, zero fSRD content").

## Contention management (what was actually overlapped)

- Overlapped: arm B's 20-stream inference (compute, ~34 min, 3 threads)
  with arm A's downloads only (pure I/O, ~1 min as it turned out) plus
  zero-CPU script writing. Two tiny validation runs (a 64-row quantizer
  check, a 200×40 fSRD smoke test) ran nice-19 during arm B; both are
  seconds-scale.
- Not overlapped: arm A's prep (6 quantization passes × 8 tensors), the
  64 fSRD fits, and the eval all ran after arm B's process had exited,
  with the box to itself. Fit times confirm no contention: 3.8–72.9 s per
  fit against the measured 68 s uncontended / 486 s contended reference.
- One incident: a duplicate `inv2a_prep.py` chain (launched externally on
  the suspicion this session had stalled) raced this session's prep and
  died immediately on `FileNotFoundError` — prep deletes each tensor after
  computing its tables, and the first tensor was already processed. No
  data was lost and no results were affected; all tables come from the
  single surviving prep run.

## Deviations from the spec

1. **No Agent/Task tool** exists in this environment (same deviation as
   both prior rounds); independent strands ran as concurrent background
   processes under the contention rules above.
2. `blk.3.attn_q` BF16 was reused from the earlier probe's download
   (byte-identical CDN object, same ETag) instead of re-fetching.
3. "Delete each fetched tensor as soon as its statistics are computed" was
   implemented as delete-after-SSE-tables: the exact per-level error
   tables subsume the tensor for every later step (allocation and
   evaluation are then table lookups), so nothing downstream needed the
   weights again. Peak resident fetched data ≈1.3 GB fp16.
4. B1's ledger charges the column permutation only; the row sort is
   error-neutral for a within-row block codec and need not be stored. B1
   exceeds the budget by those perm bits and is reported as diagnostic
   only (the spec uses it to isolate the sort's contribution, not as a
   gate comparator).
5. fSRD leaf boxes overlap by one row/column at fuzzy splits; a hard
   partition was obtained by painting boxes in (level desc, area asc)
   order, and metadata is charged as that ordered box list. Coverage was
   complete on all 64 fits.
6. Allocators are *stronger* than the spec's greedy water-filling: fSRD
   regions get the exact optimum (exhaustive over ≤6⁸ level assignments),
   B2 gets a joint boundaries×levels Lagrangian DP with 50-step bisection.
   This removes allocator suboptimality as an excuse for the fSRD result
   and does not hobble B2 (binding rule 2). With the exact allocator
   fSRD's numbers improved slightly (e.g. blk.20.ffn_gate 0.0994 → 0.0874)
   and the verdict is unchanged.
7. Superblock fp16 references at region/band edges are charged
   ceil(width/8) per row per region — an identical, slightly conservative
   rule for B2 and fSRD.
8. Arm B part-2 per-stream spread not measured (per-stream code landed
   after the run started); reported as within-noise rather than re-run.
9. Arm B streams above 20×512 tokens required more text than the
   20-prompt holdout file contains; streams are contiguous windows of
   holdout text first, then wt2 text. Nothing in arm B fits parameters,
   so this affects no train/test boundary.

## Honest bottom line

**What fSRD did:** on the one candidate the whole investigation identified
as the only place its machinery could act (BIC-partitioned per-region bit
allocation), fSRD segmented every tensor (no degeneracy this time — the
auto-FAIL clause stayed silent) and then *lost*: worse than the uniform
grid on 6 of 8 tensors (median −6.8%), worse than the trivial 1-D
sorted-band baseline on 8 of 8 (median −18.7%), at honestly-equal bits
with metadata fully charged. The pre-registered FAIL condition ("B2 ≥
fSRD, or medians within ±10% of B0") is met on both prongs, with the
hyperparameter escape hatch closed by the registered sweep (fSRD's number
is already the best of its 8 configs per tensor). Combined with
`RESULTS.md` (H0 on prediction/placement) this closes the second and last
open line: **fSRD has no measured application to this model family** —
not prediction, not placement, and now not memory/codec work either. The
32-element scale grid plus 1-D importance banding absorbs what
heterogeneity exists; a 2-D BIC tree on the sorted statistic map adds
coarser boundaries and allocation-granularity loss, not signal.

**What plain 1-D adaptive allocation did (no fSRD):** B2 — sort columns by
imatrix importance, ≤8 contiguous bands, allocate bits — beats uniform by
+75%/+64% weighted rel-RMSE on the attention projections and by +27%/+18%
on two FFN tensors (median over 8: +12%). That is a real, 20-line
observation, with the stated caveat that shipped GGUFs already capture
part of it via imatrix-weighted block fitting inside the uniform grid. If
anyone pursues arm A's *idea* further, the thing to test on a bigger box
is B2-style banding against llama.cpp's actual weighted quantizer — fSRD
has no role in that follow-up.

**What plain truncated SVD did (arm B, no fSRD):** every §5b gate passed
with an order-of-magnitude margin on the 0.8B stand-in. At-rest rank-16
truncation of all 18 layers' DeltaNet states after a 256-token prefill
moves continuation KL by 0.0033 nats at 96.8% top-1 — 8× state compression
for prefix-cache snapshots (the Marconi niche INVESTIGATION2.md §2 flagged
as unclaimed) — and in-loop rank-32 re-truncation every 64 tokens costs
nothing measurable, which even licenses the factored-state decode idea
(state bytes-moved 0.30 → ~0.15 GB/token on the 27B) to a bigger-box test.
This is the one result from the whole investigation worth carrying
forward, and it is worth carrying forward precisely *because* it owes
nothing to fSRD: the next step is a 27B-scale replication (48 heads/layer,
real serving lengths, retrieval-heavy evals — the Frobenius-vs-recall
caveat from §4 still stands, softened but not settled by the clean KL
numbers here).

**Not established by this round** (scope cap, again): any 27B end-to-end
quality effect, any kernel-speed statement, any claim about arm B beyond
the 0.8B stand-in at 1024-token scale.
