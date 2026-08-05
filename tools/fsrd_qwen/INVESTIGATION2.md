# Can fSRD buy memory on Qwen3.6-27B? The bytes-moved question

**Date:** 2026-08-05. **Question:** the prior assessment (`INVESTIGATION.md`) and experiment
(`RESULTS.md`) closed the *prediction-quality* question negatively (H0 accepted: fSRD depth
segmentation is real but confers no exit-head placement advantage; forecast-shaped uses dead on
measurement). This is a different question: **is there a memory / bytes-moved application**, in
the spirit of DeepSeek's MLA — which was not a better predictor but a low-rank *representation*
change that cut KV bytes dramatically at near-equal quality. Target: `Qwen/Qwen3.6-27B`
([config](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json)); real weights reachable
here via HTTP range requests into
[unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
(verified below). Method under assessment: fSRD, C. Bokor, M. Cary, D. Morrey, F. Bonatesta,
[arXiv:2607.17990](https://arxiv.org/abs/2607.17990); solver `fsrd_standalone.py` /
`scipy.linalg.fsrd`.

**Settled facts carried over, not re-litigated:** fSRD is a batch, single-matrix, index-grid
decomposition (fuzzy regions on the (row, column) grid, per-region regularized exact-DMD
operator, region count by BIC); no input channel, no cross-input transfer, forecasting
out of scope per its authors. Measured on this model family: token-time in-window misfit
0.27–0.44, depth boundary stable at layer 8/9 but BIC-consistency 55–66% and placement value
negative, DeltaNet outputs no more piecewise-linear than attention outputs (0.440 vs 0.453).

**Process note:** the meta-task asked for subagent delegation; no Agent/Task tool exists in this
environment (same deviation recorded in `RESULTS.md`), so the strands — byte budget, MLA/NSA
literature (primary sources via web fetch), DeltaNet-state rank probe on the 0.8B stand-in
(`inv2_state_rank.py`), GGUF range-request feasibility probe on real 27B tensors
(`inv2_gguf_probe.py`), fSRD cost measurement at the experiment's shape — were run as concurrent
background processes and batched web fetches, synthesized here.

---

## 1. Where the memory actually goes (compute first, judge second)

All numbers exact from the verified config (64 layers = 48 linear-attention Gated DeltaNet + 16
full-attention; 4 KV heads × head_dim 256, bf16 KV; DeltaNet state per layer = 48 value heads ×
(d_k=128 × d_v=128), **float32** per `mamba_ssm_dtype`, shape verified in
`modeling_qwen3_5.py::torch_recurrent_gated_delta_rule` and GGUF metadata
`ssm.inner_size=6144, ssm.state_size=128, ssm.group_count=16`).

### 1.1 Static (per-replica) bytes

| item | bytes | share |
|---|---|---|
| weights, bf16 safetensors | **55.6 GB** | 100% |
| — of which embedding (248320×5120, untied) | 2.54 GB | 4.6% |
| — of which lm_head (untied twin) | 2.54 GB | 4.6% |
| — of which MTP extra block (`blk.64`) | ≈0.79 GB | 1.4% |
| weights, Q4_K_M GGUF (repo, measured) | 17.11 GB | ≈4.9 bits/weight |
| vision projector `mmproj-BF16.gguf` | 0.93 GB | separate file, skippable |

### 1.2 Per-sequence cache bytes

| item | formula | value |
|---|---|---|
| KV cache (16 full-attn layers only) | 16 × 2 × 4 heads × 256 × 2 B | **64 KiB/token** |
| DeltaNet recurrent state (48 layers) | 48 × 48 heads × 128×128 × 4 B | **151.0 MB/seq, ctx-independent** |
| DeltaNet conv state | 48 × 10240 ch × 4 taps × 4 B | 7.9 MB/seq (negligible) |
| crossover (KV = state) | 151.0 MB / 64 KiB | **2,304 tokens** |

KV by context: 0.13 GB @2k · 0.54 GB @8k · 2.15 GB @32k · 8.6 GB @131k · **17.18 GB @262k
(native max)** per sequence.

### 1.3 Regime verdict

- **Long-context (the regime the 262k window exists for):** full-attention KV dominates the
  cache totally — 96.6% of cache bytes at 64k, 99.1% at 262k. The DeltaNet state is a rounding
  error here. *This is the MLA-shaped term.*
- **High-batch, short-context serving (≤ ~2.3k tokens):** the fp32 DeltaNet state is the larger
  cache term (52.9% at 2k) — B=256 × 2k costs 73 GB of cache of which 38.7 GB is DeltaNet state.
  But the prize per sequence is capped at 151 MB forever; the trivial competitor (store the
  state bf16 instead of fp32) already halves it, and any cleverer scheme must beat that.
- **Bytes moved per decode token, single stream:** weight reads dominate (17–56 GB/token
  depending on quant) until context grows: at 262k with a ~17 GB Q4 quant, the 17.2 GB KV read
  *equals* the weight read — long-context decode is KV-read-bound even at batch 1. DeltaNet
  state r/w is a constant 0.30 GB/token regardless of context.
- **Already-taken wins:** the hybrid layout *is* Qwen's DeepSeek-flavoured memory move — a peer
  all-full-attention stack (e.g. Qwen3-32B-style: 64 layers × 8 KV heads × 128) carries
  256 KiB/token; Qwen3.6 carries 64 KiB/token, a 4× cut taken at pre-training time.

**Consequences for candidate ranking:** an idea aimed at the DeltaNet state attacks at most
151 MB/seq (and only matters below ~2.3k ctx); an idea aimed at KV attacks up to 17.2 GB/seq
but only at long context; an idea aimed at weights attacks 17–56 GB always, per replica — and
that term is already served by a mature 21-quant GGUF ecosystem, so only a *better codec* wins.

---

## 2. What MLA actually did, mechanically — and why it worked

From the DeepSeek-V2 paper ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434), §MLA,
eqs. 9–18, Table 1), verified against the HTML full text:

1. **Low-rank joint KV compression.** All heads' keys and values are generated from one latent:
   `c_t^KV = W^DKV h_t` with `d_c = 512 = 4·d_h` (n_h=128, d_h=128), then per-head up-projections
   `k^C = W^UK c`, `v^C = W^UV c`. Only `c_t^KV` is cached.
2. **Decompression absorbed into adjacent projections.** Because attention is bilinear in K and
   V, `W^UK` folds into the query projection and `W^UV` into the output projection — the paper:
   *"W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to
   compute keys and values out."* Compression is therefore **free at decode time** — no
   decompress step exists.
3. **RoPE decoupled.** RoPE's position-dependent rotation sits between `W^UK` and the score, so
   it would block absorption. Solution: a separate small rotary key `k^R` (`d_h^R = 64`,
   *shared across all heads*) carries position; cache per token = `d_c + d_h^R = 576` elements
   vs 32,768 for MHA (**≈57×**; the paper's headline "93.3%" is vs its 67B predecessor's GQA).
   Equivalent to "GQA with 2.25 groups" at stronger-than-MHA quality.
4. **Why low-rank was viable** (the paper shows it empirically; the structural reason is plain):
   every head's K and V are linear functions of the *same* `h_t`; the KV map's rank is bounded
   by `d_model` regardless of `n_h·d_h = 16,384`, and training-in the latent lets the network
   arrange the useful subspace. Post-hoc conversions prove most of that rank is redundant even
   in *already-trained* models: **TransMLA** ([arXiv:2502.07864](https://arxiv.org/abs/2502.07864))
   shows GQA is formally a special case of MLA, converts checkpoints by PCA on latent
   activations with RoPE concentrated into few dimensions (RoRoPE/FreqFold), and recovers
   quality with ~500M–6B fine-tune tokens at 68.75–93% KV cuts; **MHA2MLA**
   ([arXiv:2502.14837](https://arxiv.org/abs/2502.14837)) gets 92.19% KV reduction on
   Llama2-7B at −0.5% LongBench with 0.3–0.6% of pre-training data, via partial-RoPE removal +
   joint SVD of KV weights.

**Successors (the memory story after MLA):**

- **NSA** ([arXiv:2502.11089](https://arxiv.org/abs/2502.11089)): trainable hierarchical sparse
  attention (compressed coarse tokens + selected blocks + sliding window) — attacks attention
  *compute and bytes-read*, not cache storage; trained from scratch.
- **DSA / DeepSeek-V3.2** ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556),
  [vLLM writeup](https://vllm.ai/blog/2025-09-29-deepseek-v3-2)): a lightning indexer keeps a
  small 128-dim index key per token, scores queries, and feeds only the top-2048 latent KV
  entries to sparse MLA — O(L²)→O(Lk) attention *reads*; storage stays MLA's.
- **Hybrid/linear-attention state work:** **LoLA**
  ([arXiv:2505.23666](https://arxiv.org/abs/2505.23666)) — training-free, routes
  hard-to-memorize KV pairs *around* the linear-attention state rather than compressing the
  state matrix; **Marconi** ([arXiv:2411.19379](https://arxiv.org/abs/2411.19379)) — prefix
  caching for hybrid models, where per-sequence recurrent-state *snapshots* (exactly Qwen3.6's
  151 MB objects) are the storage pain point, managed by admission/eviction policies, **not
  compressed** — i.e. snapshot compression is an open, unclaimed niche.

**The transferable lesson, stated once:** MLA-class wins need three properties simultaneously —
(P1) a redundant representation whose components are projections of a shared source (rank bound
built into the architecture); (P2) linear/bilinear surroundings so the decompression can be
algebraically absorbed (no runtime decompress); (P3) a cheap way to make the network live in the
compressed representation (native training or short distillation). Note what is *absent* from
every one of these wins: dynamics. None of MLA/NSA/DSA models time evolution; they exploit
static linear-algebraic redundancy. That already forecasts where fSRD — whose distinctive core
is piecewise *dynamics* — can and cannot contribute.

---

## 3. Which Qwen3.6-27B structures share the MLA property

| structure | P1 shared-source redundancy | P2 absorbable | P3 cheaply trainable-in | prize (from §1) |
|---|---|---|---|---|
| full-attn KV (16 layers, GQA 4×256) | partial — GQA already merged 24→4 heads; per layer the KV map is 5120→2048, and **partial RoPE (only 64/256 dims rotary) means 75% of K dims are NoPE and absorbable** — Qwen accidentally ships DeepSeek's decoupled-RoPE structure | yes (bilinear, TransMLA construction applies directly) | needs 10⁸–10⁹ fine-tune tokens (GPU cluster) | up to ~2–3× of the dominant long-ctx term (17.2→~6–8 GB/seq) |
| DeltaNet recurrent state S (48×48 heads, 128×128 fp32) | S = Σ_t (decayed) rank-1 outer products β k v^T — *built* as a sum of projections; read is bilinear `o = S^T q` | **yes, exactly** — see §4 Rank 2: factored S=UV^T updates in factored form with rank growing 1/token | n/a in-loop (exact); truncation error compounds through recurrence | 151 MB/seq cap; dominant only ≤2.3k ctx or at-rest snapshots |
| weights (17–56 GB) | weak — weight matrices of trained LLMs are near-full-rank (that is why the ecosystem quantizes rather than SVDs them) | n/a (static) | n/a | the largest term, always; only a *better codec* wins |
| embedding + lm_head (5.09 GB, untied) | rows are token-indexed with extreme frequency skew | n/a | n/a | ≤9% of weights; GGUF already type-mixes these |

---

## 4. Candidates, ranked

Ranking axes are explicit, because they diverge: **(impact)** how much of §1's budget it attacks;
**(fSRD content)** whether fSRD's actual machinery (BIC-driven fuzzy index-grid partitioning +
per-region regularized low-rank operators) does the work, or a plain SVD does; **(testability)**
whether this box (no GPU, 4 cores, 15 GB RAM, 27 GB disk) can measure it on real 27B weights.
The rank order is by *what this investigation can honestly establish about fSRD*, with the
non-fSRD memory wins stated plainly rather than dressed up.

### Rank 1 — fSRD-partitioned adaptive-precision weight quantization (candidate C). **The only candidate where fSRD machinery does any work; testable on real 27B weights here; modest odds. Experiment in §5.**

**Mechanism.** GGUF K-quants use a fixed grid: 256-element superblocks, 32-element sub-blocks,
one scale+min per sub-block, uniform bit-width per tensor (unsloth's UD- dynamic quants vary
bits *per tensor*, and the shipped `imatrix_unsloth.gguf_file` supplies activation-aware column
importance). The hypothesis: weight tensors have *sub-tensor* heterogeneity — bands of output
channels / input columns with systematically different variance and importance — and a
BIC-selected partition of the (row, column) index grid, with **per-region bit-width** chosen
under a fixed total-bit budget, beats the uniform grid at equal bits. fSRD is used exactly as
its machinery permits: fit on a **block-statistic map** (rows = output channels, columns =
32-wide input blocks, entries = log-RMS importance-weighted magnitude, both axes **sorted** so
that contiguous index regions are meaningful — the sort permutation is ~11 KB metadata,
~0.001 bits/weight), `oblique=False`, consuming only **region boundaries + BIC region count** —
precisely the outputs `RESULTS.md` already validated as numerically stable and cheap.

**Honest characterisation, stated plainly:** this uses fSRD's *partitioning-and-model-selection*
machinery, not its Koopman/dynamics core. Eigenvalues and Vandermonde structure on a sorted
weight-statistic grid have no dynamical meaning; fSRD serves as a BIC-principled 2-D changepoint
detector. If that is not "fSRD applied to the model" in the intended sense, this candidate dies
by definition — we state it rather than oversell it.

**Counter-arguments, in strength order.** (1) GGUF's 32-element scales already adapt to local
magnitude at a granularity ~500× finer than any fSRD region — the residual heterogeneity a
region-level bit-width can exploit may be nearly zero; (2) a 1-D sorted-axis threshold search
(sort columns by imatrix importance, sweep a boundary, allocate 2 bit-levels) is a
few-lines-of-NumPy baseline that captures most of the same structure — fSRD must beat it, not
just beat uniform; (3) irregular 2-D regions break fixed-stride kernels — the only deployable
region shape is a **row-band** (contiguous output channels), which llama.cpp can serve today by
splitting one tensor into two stacked tensors with different quant types; column-regions would
need input gathers; (4) per-region metadata is cheap (~bytes) but the *bit-budget accounting*
must be exact or the comparison is fake.

**What would have to be true:** post-sort block-statistic maps must contain genuine 2-D
*region* structure (not just smooth 1-D gradients); fSRD's BIC must select >1 region on most
tensors; and the per-region allocation must cut imatrix-weighted reconstruction error ≥10% at
equal bits vs the uniform grid *and* ≥5% vs the 1-D sorted baseline. Feasibility is verified:
the GGUF tensor index (571 tensors, shard 1) was parsed via range requests and individual
tensors fetched and decoded correctly (`inv2_gguf_probe.py`, `results/inv2_gguf_index.json`;
a 5120×17408 bf16 FFN tensor is 178 MB — trivially fetchable); fSRD at the full 5120×544
statistic shape was timed on this box: **68 s per fit uncontended (depth 3, 6 regions), 486 s
under a 3-way-contended CPU (depth 2, 4 regions)** — and, encouragingly for the experiment's
premise, BIC selected multiple regions rather than collapsing to 1 on a synthetic banded map.

### Rank 2 — DeltaNet state, factored form and at-rest snapshot compression (candidate B). **The closest structural analogue of MLA in the whole model — and the analysis kills the in-loop version on prize size, while the surviving at-rest version needs no fSRD (plain SVD).**

**The absorption check the brief demanded, done first (analytical, from the modeling code).**
The verified update (`torch_recurrent_gated_delta_rule`, transformers 5.14.1) is
`S ← g_t·S; S ← S + β_t·k_t(v_t − S^T k_t)^T`, i.e. `S_t = g_t(I − β_t k_t k_t^T)S_{t-1} + β_t k_t v_t^T`,
with scalar-per-head gate `g_t`. For factored `S = U V^T` (U: 128×r, V: 128×r):

- decay: scale V by `g_t` — stays rank r;
- read: `S^T q = V(U^T q)` — never decompress (the MLA-absorption analogue, exact);
- update: `U' = [U | k_t]`, `V' = [g_t V | β_t(v_t − g_t V(U^T k_t))]` — **exact, rank r+1**.

So the delta-rule update *is* compatible with compressed form — but rank grows by 1 per token,
so a lossy re-truncation (incremental SVD) every ~r tokens is mandatory, whose error feeds back
into the recurrence and compounds (unlike MLA, where the latent is the *trained* representation
and nothing compounds). The model's own `mamba_ssm_dtype: float32` — the only fp32 island in a
bf16 model — is the designers flagging this recurrence as numerically sensitive.

**Empirical spectrum (measured on the 0.8B stand-in, `inv2_state_rank.py` →
`results/inv2_state_rank.json`; 18 DeltaNet layers × 16 heads, 6 WikiText prompt streams,
states snapshotted mid-prefill; medians over layers × heads × prompts; transfers to 27B as
hypothesis only):** the states are **far more low-rank than the delta-rule's design suggests**.
At T=256: median energy fraction of the top singular direction 0.89; median r90 = 2, r99 = 5;
best rank-16 approximation leaves 2.4% relative Frobenius error, rank-32 leaves 0.6% — and this
holds uniformly across all 18 layers (worst layer: r99 = 10, rank-32 error 2.7%; deeper layers
mildly higher-rank). Crucially, the spectrum **saturates rather than fills** as context grows:
at T=1024 (2 concatenated-document streams, `results/inv2_state_rank_long2.json`) the medians
are essentially unchanged — r90 = 2, r99 = 6, rank-16 error 2.45%, rank-32 error 0.75% (vs
2.35%/0.60% at T=256) — refuting the natural expectation that the delta rule progressively
fills its 128 association slots. The per-head gating decay evidently keeps the
effective associative memory much smaller than its 128 slots at these lengths. Caveats stated:
Frobenius energy is not retrieval quality — the delta rule's precise-recall property may live
exactly in the small tail components a truncation discards; and this is the 0.8B stand-in
(16 heads/layer vs the 27B's 48).

**Verdict.** In-loop factored state: **dead** — the prize is ≤151 MB/seq (§1: ≤50% of cache only
below ~2.3k ctx), best case ~2–4× on that term, i.e. tens of MB per sequence, against real
per-token re-truncation compute and compounding error, in fp32-flagged arithmetic. Nobody should
spend that risk for that prize. At-rest snapshot compression (the Marconi prefix-cache niche —
151 MB per cached prefix, no error compounding since resume is one-shot): **plausible
engineering, zero fSRD content** — truncated SVD on 128×128 matrices with unordered axes; fSRD's
index-grid partitioning has nothing to grip (both axes unordered, no oblique semantics), and a
per-head plain SVD strictly dominates any per-region DMD reconstruction at equal rank
(established in INVESTIGATION.md Rank-4, point 3). Killed *as an fSRD application*; noted as a
sensible ordinary-engineering idea.

**What would have to be true for the at-rest niche to be real:** (a) spectra at rank ≤ 16 with
≤ a few percent error, *stable as T grows* — **now measured true on the stand-in** (2.4% at both
T=256 and T=1024; the pre-registered expectation that the delta rule fills its 128 slots was
refuted); (b) a quality-vs-state-error transfer showing the recurrence tolerates truncation —
open, and exactly what the §5b experiment tests (the designers' fp32 island argues for caution);
(c) a serving regime dominated by ≤2.3k-token high-batch traffic or heavy prefix-snapshot
storage — a product question, not a research one. Even with (a) measured, none of this creates
a role for fSRD — every step cashes out as plain truncated SVD.

### Rank 3 — GQA→MLA retrofit of the 16 full-attention layers. **The truest DeepSeek analogue in this model; attacks the dominant long-context term; zero fSRD content; untestable in this box.**

Qwen3.6's partial RoPE (64 of 256 dims) hands over, for free, the decoupled-RoPE structure
DeepSeek had to engineer: 192 of 256 K-dims per head are position-free and absorbable. TransMLA's
construction (GQA ⊂ MLA) applies directly: merge the 4 KV heads into a joint latent (PCA on
calibration activations), keep 4×64 rotary dims apart, absorb up-projections into Q and O. A
latent of 512–768 would cut the 64 KiB/token to ~24–32 KiB — the 262k-context cache from 17.2 GB
to ~6–8 GB/seq. **But:** it needs activation calibration plus 10⁸–10⁹ tokens of recovery
fine-tuning (TransMLA: 500M–6B; MHA2MLA: 0.3–0.6% of pretraining data) on hardware this box does
not have; GQA at only 4 groups is already 6× merged, and TransMLA's own results show quality
drops steeply at high compression before fine-tuning; and nothing in it is fSRD — it is SVD/PCA
plus training. Ranked for completeness because it is the honest answer to "what is the MLA-like
move on this model"; not the answer to "where can fSRD be used". **What would have to be true:**
access to training hardware and a tokens budget (≥10⁸), plus evidence that 4-group GQA still
carries enough cross-head redundancy — a cheap advance signal obtainable on the 0.8B stand-in
(joint PCA of harvested K/V activations across its 2 KV heads), the natural follow-up *if*
someone owns GPUs.

### Rank 4 — embedding/lm_head row-band precision by token frequency. **Real but trivial; no fSRD needed.**

5.09 GB (9.1%) sits in two 248320×5120 matrices whose row relevance follows the token frequency
distribution. Sorting rows by corpus frequency and allocating precision in bands is a 1-D
sort-and-threshold exercise; GGUF already type-mixes these tensors (the ecosystem's Q4_K_M keeps
embeddings at higher effective precision). Any fSRD role collapses to the same 1-D changepoint
problem as candidate C's B2 baseline. Not pursued separately; folded into C's machinery if C
passes.

### Killed outright

- **Runtime fSRD anywhere in the decode loop** — settled by the prior investigation (batch
  single-matrix fit, no transfer, 10²–10³× cost overruns); nothing in the memory framing
  changes it.
- **fSRD on KV token-time sequences (cache compression by DMD modes)** — stays dead
  (INVESTIGATION.md Rank 4, five independent failures); the *static* variant of the idea is
  exactly Rank 3 above, where SVD/PCA does it without dynamics.
- **fSRD on DeltaNet state/output trajectories** — measured dead in RESULTS.md step 5
  (DeltaNet outputs no more piecewise-linear than attention: 0.440 vs 0.453 misfit).
- **Vision tower / mmproj** — 0.93 GB in a separate optional file; the byte budget says
  attacking it is pointless.
- **MTP block compression** — 0.79 GB, and unsloth deliberately holds `blk.64` at Q8_0 even in
  2-bit quants because draft-head quality gates the whole speculative speedup; compressing the
  one tensor everyone protects is anti-useful.

---

## 5. Pre-registered experiment — candidate C pilot on real Qwen3.6-27B weights

**Claim under test (H1):** BIC-selected adaptive regions with per-region bit-width beat GGUF's
uniform blocking at equal total bits on imatrix-weighted reconstruction error.
**H0:** the 32-element scale grid already absorbs the heterogeneity; region-level allocation
adds ≤10% — or a trivial 1-D sorted threshold does as well as fSRD's 2-D tree.

**Data (verified reachable).** From
`https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf`
via HTTP range requests (index at `results/inv2_gguf_index.json`, 571 tensors, data_start
10,977,440): 8 tensors ≈ 1.5 GB total download, deleted after processing —
`blk.{0,20,40}.ffn_down.weight` (17408×5120), `blk.{0,20,40}.ffn_gate.weight` (5120×17408),
`blk.3.attn_q.weight` (full-attn layer, 5120×12288, gate-fused per `attn_output_gate` —
verified in the index), `blk.1.attn_qkv.weight` (DeltaNet in-projection). Plus
`imatrix_unsloth.gguf_file` — **fetched and parsed during this investigation** (13.6 MB, GGUF
v3, 992 tensors named `blk.N.<tensor>.weight.in_sum2` + `.counts`, 77 chunks of unsloth's
calibration set): per-input-column second moments exist for every target tensor, so the
weighted-RMSE metric is fully specified (weight column j by `sqrt(in_sum2[j]/counts)`, the
same weighting `llama-quantize` uses). One naming detail: full-attention layers store
`attn_q/attn_k/attn_v` separately while DeltaNet layers fuse `attn_qkv`; the tensor index in
`results/inv2_gguf_index.json` resolves each case.

**Pipeline per tensor W.**
1. Block-statistic map `Z` (rows = output channels, cols = 32-wide input blocks; entry =
   log RMS of importance-weighted block); sort rows by RMS, columns by importance (permutations
   counted as metadata).
2. `fsrd(Z_sorted, oblique=False, max_depth ∈ {2,3}, theta=3, smoothness=0.03)` → region
   boundaries + BIC count. Measured at the full 5120×544 shape: 68 s/fit uncontended, 486 s
   contended; a 2×2-downsampled 2560×272 map is the fallback if the box is shared.
3. Per-region bit allocation `b_i ∈ {2,3,4,5,6,8}` bits (Q_K-style 32-block scale+min inside
   every region) by greedy water-filling on predicted weighted MSE, constrained to total bits
   (payload + scales + permutations + region table) ≤ the uniform baseline's budget.
4. Quantize, dequantize, measure **imatrix-weighted relative RMSE**.

**Baselines at exactly equal total bits:** **B0** uniform 4-bit 32-block grid (Q4_K-style,
6-bit scales) — the status quo; **B1** B0 on sorted axes (isolates the sort's contribution);
**B2** 1-D adaptive: sorted columns, exhaustive best split into K ≤ 8 contiguous bands, same
water-filling allocator — the "fSRD is unnecessary" comparator.

**Gates (numeric, pre-registered).**
- **PASS:** fSRD-region allocation ≥10% lower median weighted rel-RMSE than B0 **and** ≥5%
  lower than B2, at ≤ equal bits, on ≥6 of 8 tensors.
- **PARTIAL** (verdict: "adaptive allocation works; fSRD is ceremony"): beats B0 by ≥10% but
  not B2 by ≥5%.
- **FAIL:** neither margin; **auto-FAIL** if fSRD's BIC returns 1 region on ≥4 of 8 tensors —
  the method itself declaring there is nothing to segment (exactly the standardized-rows
  degeneracy already seen in RESULTS.md step 2 — this outcome has precedent and is live).
- **What the negative looks like, so it cannot be explained away:** medians within ±10% of B0,
  or B2 ≥ fSRD, or 1-region BICs — meaning fine-grained scales already do the work and the
  entire "adaptive blocking" idea (not just fSRD's version) is dead at this bit budget, with the
  step-2 sweep (two depths, both sort orders, weighted/unweighted) closing the hyperparameter
  escape hatch.
- **Scope cap stated in advance:** even a PASS proves an *offline codec* improvement on a
  weighted-RMSE proxy (the metric llama.cpp quant development itself uses), not end-to-end
  perplexity (no 27B inference is possible here) and not kernel-speed parity; the deployable
  subset is row-band regions served as split tensors. A PASS buys a follow-up on a bigger box,
  nothing more.

**Budget:** ≈1.5 GB transient disk (~16 GB currently free, measured), ≤2 GB RAM peak (one 356 MB fp32 tensor +
temporaries), wall-clock ≈ half a day (downloads ~1–2 h through the proxy; 8×2 fSRD fits
≈ 20–40 min at full 5120×544 resolution per the measured 68 s uncontended fit — budget ~2 h
if the box is contended; quant/eval ~1 h).

**Prior, honestly:** P(PASS) ≈ 0.15, P(PARTIAL) ≈ 0.3, P(FAIL incl. auto-FAIL) ≈ 0.55 — the
32-element scale grid is a strong incumbent, and the fSRD-vs-B2 margin is the harder gate by
design.

### 5b. Second arm (pre-registered, optional): DeltaNet state truncation quality test — the honest best *memory* experiment this box can run, with zero fSRD content

Motivated by the new measurement (§4 Rank 2): stand-in states are near-low-rank (median rank-16
error 2.4%, stable from T=256 to T=1024). The open question is whether truncation error
compounds through the
recurrence, and whether Frobenius error maps to quality. Test, on the prepared 0.8B stand-in:

1. **At-rest (prefix-cache resume):** prefill 256 tokens; replace every DeltaNet S by its best
   rank-r approximation (r ∈ {8, 16, 32}), once; continue decoding 256 tokens; measure mean KL
   of continuation logits vs the untouched run, and top-1 agreement, over 20 held-out prompts.
   **Gate:** rank-16 keeps mean KL ≤ 0.05 nats and top-1 agreement ≥ 95% → 8× at-rest state
   compression (rank-16 factored fp16 vs full fp32) is declared viable *as engineering*.
2. **In-loop (periodic re-truncation):** truncate to rank r every 64 tokens during a 1024-token
   prefill; measure perplexity delta on the same prompts. **Gate:** rank-32 costs ≤ 2% relative
   perplexity → factored-state decode (which also cuts state bytes-moved 0.30 → ~0.15 GB/token)
   graduates to a bigger-box test; > 5% at rank-64 → in-loop compression dead.
3. **Negative statement:** if even the one-shot rank-32 truncation moves continuation KL
   > 0.2 nats, the low-rank spectra were retrieval-irrelevant energy and the whole candidate
   (at-rest included) dies — Frobenius rank was the wrong lens, as pre-warned in §4 Rank 2.

Cost: ~2–4 h wall-clock, all local, no downloads. This arm is deliberately labelled *not fSRD*:
its tool is truncated SVD; it is registered because the brief's underlying goal is memory, and
it would be dishonest to bury the one testable memory idea because the method under assessment
contributes nothing to it.

---

## 6. Honest bottom line

The byte budget rules before any method question: **long-context cache is full-attention KV
(17.2 GB/seq at 262k, 99% of cache), always-on memory is weights (17–56 GB), and the DeltaNet
state everyone finds intriguing is capped at 151 MB/seq** — the least valuable target in the
model despite being the most "matrix-shaped".

On the actual question — *is there an MLA-like win fSRD can deliver here* — the answer splits:

1. **The true MLA analogues exist in Qwen3.6 but are not fSRD.** The 16 GQA layers convert to
   MLA form by TransMLA/MHA2MLA machinery (SVD/PCA + fine-tuning), helped by Qwen's partial
   RoPE, attacking the dominant long-context term; the DeltaNet state admits an *exact*
   MLA-style absorbed factored form (derived in §4 Rank 2, rank grows 1/token) and its at-rest
   snapshots are a real, unclaimed compression niche (Marconi manages, does not compress, them)
   — made concrete by this round's measurement that stand-in states sit at ~2.4% error at
   rank 16, *saturating rather than growing* from T=256 to T=1024 (≈8× at-rest compression if
   the §5b quality gate passes) — but the correct tool is truncated SVD both times. Every property that made MLA work (shared
   source, bilinear absorption, trainable latent) is static linear algebra; none of it needs —
   or rewards — piecewise Koopman dynamics.
2. **The one candidate where fSRD's own machinery (BIC tree + fuzzy index-grid partitioning)
   does the work is adaptive-precision weight quantization** — honestly characterised as using
   fSRD's partitioning core, not its dynamics core — and it is testable on real 27B tensors in
   this box today (range-request feasibility verified, tensor index in hand, solver cost at the
   required shape measured). It carries a pre-registered experiment (§5) whose most likely
   outcome (P ≈ 0.55) is failure against GGUF's fine-grained incumbent, and whose second most
   likely outcome (P ≈ 0.3) is "adaptive allocation works but a 20-line 1-D baseline does it
   without fSRD". That is the calibrated truth, published before the data: the experiment
   exists to kill the idea cheaply if it deserves killing.
3. **If the goal is the memory win itself, rather than a role for fSRD** — the priority order
   this analysis supports is: (a) ship the existing UD-quants (weights term, solved by the
   ecosystem); (b) bf16 the DeltaNet state at rest and add Marconi-style snapshot admission if
   serving many prefixes; (c) fund a TransMLA-style retrofit of the 16 full layers on real
   hardware if 262k-context serving is the product. fSRD appears in none of these; the §5
   experiment is the one place it might still earn a line.

---

## 7. Sources

- fSRD: C. Bokor, M. Cary, D. Morrey, F. Bonatesta,
  [arXiv:2607.17990](https://arxiv.org/abs/2607.17990); author Q&A (3 rounds, in scratchpad
  `zipwork/`); `fsrd_standalone.py` / `scipy/linalg/_fsrd.py`.
- Prior assessment and experiment: `INVESTIGATION.md`, `RESULTS.md` (this directory).
- DeepSeek-V2 / MLA: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (eqs. 9–18, Table 1).
- NSA: [arXiv:2502.11089](https://arxiv.org/abs/2502.11089). DSA / DeepSeek-V3.2:
  [arXiv:2512.02556](https://arxiv.org/abs/2512.02556),
  [vLLM blog](https://vllm.ai/blog/2025-09-29-deepseek-v3-2),
  [SGLang blog](https://www.lmsys.org/blog/2025-09-29-deepseek-V32/).
- TransMLA: [arXiv:2502.07864](https://arxiv.org/abs/2502.07864). MHA2MLA:
  [arXiv:2502.14837](https://arxiv.org/abs/2502.14837).
- LoLA: [arXiv:2505.23666](https://arxiv.org/abs/2505.23666). Marconi:
  [arXiv:2411.19379](https://arxiv.org/abs/2411.19379).
- Model: [Qwen/Qwen3.6-27B config](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json);
  [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) (file
  sizes measured from the repo tree; GGUF header and tensor index parsed by range request in
  `inv2_gguf_probe.py` → `results/inv2_gguf_index.json`).
- DeltaNet update rule: `transformers 5.14.1`,
  `models/qwen3_5/modeling_qwen3_5.py::torch_recurrent_gated_delta_rule` (read directly).
- New measurements this round: `inv2_state_rank.py` → `results/inv2_state_rank.json` (T=64/256,
  6 prompts) and `results/inv2_state_rank_long2.json` (T=1024, 2 concatenated streams) —
  DeltaNet state spectra on the 0.8B stand-in; `inv2_gguf_probe.py` (range-request
  feasibility + tensor index, real 27B weights; imatrix fetched and parsed); fSRD timing at
  5120×544 (`inv2_fsrd_timing.log`: depth-2 486 s contended / depth-3 68 s uncontended,
  4–6 regions).
