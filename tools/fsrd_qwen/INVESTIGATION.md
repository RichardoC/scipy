# Can fSRD be used on Qwen3.6-27B? A technical assessment

**Date:** 2026-08-04.
**Method under assessment:** fSRD — *Fuzzy Spectral Region Decomposition* — C. Bokor, M. Cary,
D. Morrey, F. Bonatesta, "fSRD: Fuzzy Spectral Region Decomposition — Automated Multi Operator
Koopman Representations via an Adaptive Spectral Learning Architecture",
[arXiv:2607.17990](https://arxiv.org/abs/2607.17990). Implementations examined: the NumPy
standalone (`fsrd_standalone.py`) and the SciPy branch (`scipy/linalg/_fsrd.py`,
`scipy.linalg.fsrd`). Three rounds of Q&A with the paper's authors were used as the primary
authority on the method's intended scope and failure modes.
**Target:** `Qwen/Qwen3.6-27B` and `unsloth/Qwen3.6-27B-MTP-GGUF`.

This assessment was produced by parallel research strands (target-model facts from primary HF/web
sources; method dossier from the author Q&A + supplementary LaTeX + code; SciPy-implementation
analysis including the eight recent correction commits; environment feasibility; and two
independent skeptical evaluations of the candidate applications), then synthesized.

---

## 1. Executive summary

fSRD is a **retrospective decomposition of a single spatio-temporal snapshot matrix**: it
partitions the *(row-index, column-index) grid* of one matrix into fuzzy regions, fits an
autonomous linear (exact-DMD) operator in each, and selects the region count by BIC. Its regions
are tied to that one matrix's index plane; **a fitted model does not transfer to new inputs**, its
local operators are **autonomous** (no input/control term), and the paper's authors explicitly
declared **forecasting beyond the input window out of scope** ("it would be dangerous to try and
jump to a solution for this" — author Q&A, round 1, Q11e).

Those three verified properties disqualify every "deploy fSRD inside the inference loop" idea for
an LLM — including the superficially attractive one, a Koopman draft head competing with the
model's MTP heads. A first run of fSRD on **real activations of this model family** reinforces
the verdict from the data side: in-window reconstruction error on Qwen3.5-0.8B token-time
matrices is 8–27%, versus 0.5% on synthetic data — the locally-linear-operator premise is
measurably strained before forecasting, transfer, or cost even enter. What survives is **offline
analysis**: fSRD is a legitimate, BIC-principled *changepoint/segmentation* tool for transformer
internals, and its most defensible use on this model family is **segmenting the residual stream
across layer depth** to decide *where* a cheap, conventionally-trained linear component
(early-exit probe or auxiliary draft projection) should attach — plus one sharp
architecture-specific probe of whether the model's own Gated-DeltaNet linear recurrences are
*effectively* piecewise-constant. Both are testable in half a day, on CPU, in the already
prepared Qwen3.5-0.8B environment, with a pre-registered misfit gate that can kill them cheaply.

---

## 2. Target-model facts (verified from primary sources)

### 2.1 Qwen/Qwen3.6-27B — architecture

Sources: [config.json](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json) (use the
`/raw/main/` form — the `/api/` config path 404s),
[model card](https://huggingface.co/Qwen/Qwen3.6-27B), HF API metadata (created 2026-04-21,
Apache-2.0). 27.78B parameters, 55.6 GB of safetensors. Despite the branding, `model_type` is
`qwen3_5` (`Qwen3_5ForConditionalGeneration`) — a **dense, multimodal, hybrid linear-attention**
model, not a plain transformer and not MoE:

| fact | value |
|---|---|
| layers | **64** = repeating [3 × Gated DeltaNet (linear attention) → 1 × full Gated Attention] → 48 linear-attn + 16 full-attn (`full_attention_interval: 4`) |
| hidden size | **5120** (`intermediate_size` 17408) |
| attention | 24 Q heads / **4 KV heads** (GQA 6:1), `head_dim` 256, partial RoPE (64 of 256 dims), `rope_theta` 1e7, `attn_output_gate: true` |
| DeltaNet layers | `linear_num_value_heads` 48 / `linear_num_key_heads` 16, key/value head dim 128, `linear_conv_kernel_dim` 4, `mamba_ssm_dtype: float32`, `output_gate_type: swish` — a gated-delta/mamba-style **linear recurrence over token position** with an explicitly materialised constant-size recurrent state; **no KV cache** on these 48 layers |
| vocab | **248,320** (padded), `tie_word_embeddings: false` |
| context | 262,144 native (YaRN-extensible ~1M) |
| MTP | `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` |
| vision tower | 27-layer SigLIP-like encoder, out width 5120 (skippable; text-only use fine) |

The MoE sibling is `Qwen/Qwen3.6-35B-A3B` (40 layers, hidden 2048, 256 experts / 8 active,
same MTP keys). **These two are the only open Qwen3.6 sizes** — there is no small Qwen3.6
([HF author search](https://huggingface.co/api/models?author=Qwen&search=Qwen3.6)).

### 2.2 What "MTP" means in this model

Verified from
[model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/model.safetensors.index.json)
(15 `mtp.*` tensors) and the model card:

- **One extra transformer block** (DeepSeek-V3 / Qwen3-Next "NextN"-style): the draft module
  takes `concat(RMSNorm(h_final), RMSNorm(embed(next_token)))` (2×5120 → 5120 via `mtp.fc`), runs
  **one full gated-attention layer** (not DeltaNet), applies `mtp.norm`, and reuses the **shared
  LM head** to predict one token ahead. No dedicated draft embeddings.
- The card says "MTP: trained with multi-steps" — the single layer is trained to be applied
  **recursively**, so serving stacks run it k times to draft k tokens: SGLang
  `--speculative-algo NEXTN --speculative-num-steps 3` (4-token verify), vLLM
  `{"method":"qwen3_next_mtp","num_speculative_tokens":2}` (both from the model card).
- Inference use is standard **self-speculative decoding**: MTP drafts k tokens; the main model
  verifies in one batched pass; acceptance is exact, so output distribution is unchanged.
  Measured performance: llama.cpp PR
  [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673) reports ~75%
  steady-state acceptance with 3 draft tokens and >2× speedup; a third-party RTX PRO 6000 run
  measured 65% acceptance, 1.73× end-to-end on the 27B (jarvislabs.ai write-up). Code/math prompts
  reach 80%+.
- **The critical design point for this assessment:** the MTP head needs (a) the freshly sampled
  token's embedding as an input at *every* draft step, (b) a full nonlinear attention layer, and
  (c) corpus-scale training — and with all three it gets 65–80% acceptance at ~1.7% of a forward
  pass.

### 2.3 The MTP-GGUF release, and what GGUF permits

Verified from the [unsloth repo file tree](https://huggingface.co/api/models/unsloth/Qwen3.6-27B-MTP-GGUF/tree/main)
and by range-downloading and parsing the GGUF header of `Qwen3.6-27B-UD-IQ2_XXS.gguf`:

- `general.architecture = qwen35`, **866 tensors, block_count = 65**: the 64 real layers plus
  **`blk.64` = the MTP layer**, advertised by `qwen35.nextn_predict_layers = 1`. MTP tensors are
  named `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight` plus the block's own
  attention/FFN tensors. Even in the 2-bit quant, `blk.64.nextn.eh_proj.weight` is stored at Q8_0
  — the MTP head is deliberately kept at high precision.
- llama.cpp has used these tensors for speculative decoding since PR #22673 (May 2026):
  `--spec-type draft-mtp --spec-draft-n-max 2` (single slot only, text-only for now — unsloth card).
- **You cannot train a GGUF.** Block-quantized integer weights are not differentiable parameters,
  llama.cpp mainline has no training path for this architecture, and quantization is one-way
  (dequantized Q4 ≠ original bf16). Any training — including of a small auxiliary head — starts
  from the **HF safetensors checkpoint** (bf16), with re-export via `convert_hf_to_gguf.py` +
  `llama-quantize` afterwards.

### 2.4 Small same-family models (for the CPU experiment)

- **No Qwen3.6 under 27B exists.** The smallest same-*architecture* (`qwen3_5`) models are the
  **Qwen3.5 series**; `Qwen/Qwen3.5-0.8B` (873M params, 24 layers = 18 linear-attn + 6 full-attn,
  hidden 1024, vocab 248,320, 1.75 GB bf16) **ships the same 15-tensor MTP head** — verified by
  parsing its safetensors header — and is **already downloaded and verified working** at
  `/home/user/scipy/tools/fsrd_qwen/models/` (loads as `Qwen3_5ForCausalLM`, vision tower
  dropped, 0.7524B used params, fp32 peak RSS 5.09 GB, prefill 7.3–17.5 ms/token at T ≥ 64).
- Fallback with plain attention: `Qwen/Qwen3-0.6B` — 28 layers, hidden 1024, vocab 151,936, tied
  embeddings, 1.5 GB bf16
  ([config](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)). No MTP head, and a
  different architecture generation — the 0.8B is the right stand-in.
- **Qwen3.6-27B cannot be touched in this environment, in any form**: 27.78B params, 55.6 GB of
  safetensors (54 GB bf16, 108 GB fp32) against 15 GB RAM and ~27 GB free disk — several times
  over both budgets, likewise the 35B-A3B. Training, loading, or even storing the 27B here is
  impossible; this is stated plainly and shapes the experiment design below.

---

## 3. What fSRD actually is (the facts that decide applicability)

From the author Q&A (rounds 1–2 answered, round 3 sent), the supplementary LaTeX, and the code:

1. **Input**: one snapshot matrix `a ∈ R^(M×T)` — M state coordinates (rows), T uniformly spaced
   time samples (columns). The paper rescales all data to [0,1]; the SciPy port is
   scale-invariant instead.
2. **Regions are index-grid objects.** Both axes are normalized to [0,1] *globally*
   (author-confirmed), and splits are sigmoids over `(u_row, u_col)`:
   temporal splits (contiguous time windows), row splits (blocks of coordinates), or oblique
   splits (a boundary migrating across coordinates over time). Regions are **not** state-space
   partitions and the same time window cannot get different operators depending on state.
3. **Local model** = exact DMD on the (topologically transformed, optionally whitened) block:
   autonomous linear evolution, reconstruction `Φ diag(b) Vandermonde(λ)`, amplitudes pinned to
   the region's **first snapshot** (`b = (WΛ)⁻¹x̃₀`, author-confirmed; the SciPy branch adopted
   this in commit `5b465d0`). No input/control term exists anywhere in the method.
4. **Model order** by whole-model BIC with a level-exponential complexity term
   `k = Σ rᵢ·Θ^Lᵢ`; split *placement* by WNRMSE (paper: PSO; SciPy: deterministic grid) —
   author-confirmed division of labor.
5. **Forecasting is not part of the validated method.** The paper defines nothing beyond the
   input window; asked directly, the authors deferred it as "dangerous". The SciPy `forecast=`
   keyword is an extension: only regions whose box reaches the last column are continued
   (Vandermonde rollout), **oblique regions are never extrapolated** (zeros + `RuntimeWarning`),
   and no forecast accuracy is validated anywhere in the material beyond a 20-step clean linear
   rotation.
6. **Noise**: the paper's own factorial study shows partition discovery starts failing at ±3%
   noise and mostly fails by ±5%; the per-region SVDe whitening the authors consider necessary
   for "real data" is **not implemented** in the standalone/SciPy port.
7. **Validation envelope**: Lorenz bifurcation 7500×60, Hankel Lorenz 903×700, Duffing ensembles
   — all 2–3-state ODEs inflated into tall matrices; largest matrix 450k elements; **every
   published number is in-window reconstruction, never a forecast**; nothing in any document
   touches ML or neural-network activations.
8. **Cost (measured on this machine, 4-core Xeon)**: at the shapes this investigation actually
   proposes, cost is a non-issue — measured fresh-process timings: M=1024, T=128, depth 2 →
   **7.9 s / 0.08 GB / 3 regions** (depth 3 → 8.6 s, 6 regions); M=2048, T=256, depth 2 →
   **15.2 s / 0.20 GB / 4 regions** (depth 3 → 25.5 s, and BIC pruned the extra level back to
   4 regions, so depth 2 is the better default). Cost climbs steeply with T and with
   depth/oblique: a defaults run (depth 3, `oblique=True`) at M=1024, T=512 took **244 s**, one
   SVD at 5120×2047 is ~7 s, and a defaults fit at M=5120, T=2048 extrapolates to **1.5–4+ hours**
   (legal — 10.5M cells is under the 5×10⁷ cap — but far outside the 61×840 benchmark regime).
   Sweet spot: M in the hundreds-to-low-thousands, T ≲ 256. With M=5120 raw hidden dims,
   **PCA-project first**; the routine does no dimensionality reduction of its own, and
   `oblique=False` is the documented large-grid speedup.
9. **The most important empirical signal — fSRD has now been run on real activations of this
   model family.** On (1024, 128) token-time snapshot matrices harvested from Qwen3.5-0.8B, fSRD
   is numerically well-behaved (3–4 s per fit, 3–4 regions, all finite, untroubled by layer-24
   outlier dims at absmax 35.6) — but the **in-window relative reconstruction error is 8.1×10⁻²
   at layer 1, 2.67×10⁻¹ at layer 12, and 1.21×10⁻¹ at layer 24, versus 4.9×10⁻³ on synthetic
   regime-switching data**. Reconstruction is the method's core, validated competence; a 27%
   in-window error mid-stack is direct evidence that real transformer activations along the
   token axis are substantially less low-rank / less locally-linear than anything in fSRD's
   validation envelope. This number is weighed against every candidate below rather than
   explained away.

Two structural mismatches with transformer internals, worth stating once:

- **Row order carries meaning in fSRD** (oblique splits and the topological transform interpolate
  along row indices). Transformer hidden dimensions are unordered, so oblique/row machinery is
  semantically void on raw hidden states → run `oblique=False`, or accept that row splits
  partition PCA components.
- **Massive-activation outlier dimensions** (ubiquitous in transformer residual streams) dominate
  every local SVD because no whitening is implemented → row-standardize (and report sensitivity,
  since shifting data changes the fitted dynamics).

---

## 4. Candidate applications, ranked

Two independent evaluator analyses (one covering layer-depth + KV candidates, one dedicated to
the token-time draft head) converged on the ordering below; a fifth candidate (the hybrid
architecture's own linear-attention recurrence) was added once the target's DeltaNet structure
and the real-activation measurements were in hand. The measured 27% mid-stack reconstruction
error (§3, item 9) is applied to every verdict.

### Rank 1 — Layer-depth regime segmentation (analysis) → placement of a trained linear head. **Workable, as analysis only.**

**Mechanism.** Snapshot matrix per token position: column ℓ = residual-stream state after layer ℓ.
`M = hidden_size` (5120 on 27B; 1024 on the 0.8B stand-in), `T = n_layers+1` (65 / 25 columns),
`dt = 1 layer`. A *temporal* split is a **depth boundary** — "layers 0–17 evolve under one
effective linear operator, layers 18–41 under another." Consumed outputs: region column
boundaries (`bounding_box`), per-region eigenvalues (growth/oscillation per layer), and the BIC
of the multi-region model vs `max_depth=0` (global DMD) — the method's own built-in null test.
Valid input choices: one token's trajectory per fit (aggregate boundary histograms over many
tokens), or row-stacking k tokens over the shared depth axis (M = k·d; a tested pattern —
`test_stacked_systems_with_row_splits` — that *forces* a shared depth segmentation, which is
exactly the claim under test). Concatenating tokens along the column axis is **invalid** (fake
transitions at every seam).

**Why it can work.** The residual stream is `x_{ℓ+1} = x_ℓ + f_ℓ(x)` — near-identity updates with
monotone norm growth; "stages of inference" phenomenology suggests a handful of depth phases; an
effective linear operator per phase is a defensible *descriptive* model, and DMD represents the
growth trend naturally. T = 65 matches the paper's own bifurcation matrix (T = 60) exactly.
T = 25 (the 0.8B stand-in) resolves at most ~3 segments (a region needs ≥ 8 columns to split) —
coarse but aligned with the small number of stages the literature claims.

**What breaks, honestly.** The depth map is different at every layer and depends on other tokens'
states through attention (and DeltaNet recurrent state on 3/4 of Qwen3.6's layers), so the fitted
operator is an *average*; conclusions are per-trajectory and must be aggregated. On Qwen3.6, the
period-4 layer pattern (3 DeltaNet + 1 attention) injects a built-in period-4 structure fSRD will
either soak into an oscillatory eigenvalue (~0.25 cycles/layer — itself a diagnostic to look for)
or waste regions on; the eigenvalue signature is worth checking explicitly in the 0.8B arm. And
the honest baselines are nearly
free: cosine-similarity-of-successive-layers changepoints and logit-lens KL knees. **If fSRD's
boundaries merely reproduce those knees, it adds ceremony, not information** — that is the null
hypothesis the experiment in §5 is built to test.

**Cost.** Per-token fit at 5120×65: seconds (min SVD dimension ≤ 64). At 1024×25: ~1–3 s
(measured: 1024×128 depth-2 takes 7.9 s; T=25 is much cheaper). Fitting hundreds of tokens is
under an hour on this box. Fully feasible — the only obstacle at 27B scale is harvesting the
states (needs a bigger machine; the *fits* are cheap).

**Deployment link.** The fSRD output (a BIC-selected boundary layer ℓ*) informs where to attach a
conventionally trained linear early-exit / draft projection `W: h_{ℓ*} → h_final`. W itself is
plain ridge regression trained across a corpus — deliberately *not* fSRD, because nothing fitted
by fSRD transfers across inputs. fSRD's role ends at choosing ℓ*.

**Caveat from the real-activation measurement.** The 8–27% errors of §3 item 9 were measured on
*token-time* matrices, not depth matrices, so they do not directly condemn this candidate — a
depth trajectory (25–65 near-identity residual updates) is a genuinely different, more
DMD-friendly object than 128 content-driven token steps. But they set the prior: if the depth
matrices come back with comparable in-window error, the piecewise-linear-in-depth premise fails
too, and the experiment in §5 measures exactly this before anything else.

### Rank 2 — The hybrid architecture's own linear-attention (Gated DeltaNet) dynamics. **The sharpest question; workable only as an analysis probe, and probably a negative.**

**Why this candidate exists.** 48 of Qwen3.6-27B's 64 layers (18 of 24 in the 0.8B stand-in) are
Gated DeltaNet layers: each maintains an explicitly materialised, constant-size recurrent state
and updates it with a **linear recurrence over token position** (delta-rule form
`S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, gated). A linear state-space recurrence is, on
its face, the most Koopman-shaped object in the entire model — far more natural a target than a
softmax-attention residual stream. Two readings must be weighed honestly:

- **Reading A (best case): fSRD as an effective-autonomy probe.** The DeltaNet recurrence is
  linear but its coefficients are *data-dependent* (gates, β, k vary per token). fSRD fits
  piecewise-**constant** autonomous operators, so fitting it to DeltaNet state/output
  trajectories asks a well-posed, novel question: *is the gated, time-varying recurrence
  effectively piecewise-constant over token regimes?* A "yes" (low in-window error with a few
  regions) would be evidence that stretches of the recurrence could be distilled into fixed
  operators — a real, if long-range, efficiency research direction. This is a legitimate use of
  exactly what fSRD does (BIC-selected segmentation + local spectra), in its validated
  in-window-reconstruction mode.
- **Reading B (deflationary): fSRD is redundant here.** The dynamics are already linear and
  *known in closed form* — there is nothing to identify that cannot be read off the weights and
  gates directly; and the only thing that makes the recurrence nontrivial is precisely its
  input-driven, token-varying coefficients, which is fSRD's excluded regime (autonomous,
  constant-A per region). System identification of a known system is analysis, not capability.

**Which reading does the evidence support?** The measured signal cuts against Reading A: the
8–27% in-window errors of §3 item 9 were obtained on hidden-state sequences of Qwen3.5-0.8B —
a model in which layers are 3:1 DeltaNet — i.e., the *outputs* of this very recurrence are
already demonstrably not well captured by a few constant linear operators. The state trajectories
themselves are worse targets dimensionally: the per-layer recurrent state is
(48 value heads × 128 × 128) ≈ 786k dims at 27B scale — unusable as M; only per-head slices
(M = 128) or layer outputs (M = hidden) are feasible inputs. Feasible and cheap, though: per-head
output matrices are 128×T, seconds per fit.

**Mechanism, if probed.** Per DeltaNet layer (and per value head for the state view): snapshot
matrix = layer output sequence (M = 1024/5120, T = 256) or per-head output slice (M = 128,
T = 256), `dt` = 1 token, `oblique=False`. Consumed outputs: in-window NRMSE at fixed region
count, region count at fixed error — **compared head-to-head against the same layer index's
full-attention siblings**. The falsifiable question: do linear-attention layers' outputs show
systematically lower fSRD reconstruction error (more effectively linear) than softmax-attention
layers'? That comparison is the one concrete thing this candidate can deliver that nothing else
in this document can.

**Verdict: workable-only-as-analysis, expected mostly negative.** No deployment story survives
Reading B (the recurrence is already linear and already implemented; approximating a known
time-varying linear system with unknown constant ones helps only if the effective-autonomy answer
is "yes", which the measured 27% figure makes unlikely). But as a one-day probe it is the
sharpest, most architecture-specific question fSRD can be asked of this model, and it is included
as an optional arm of the §5 experiment.

### Rank 3 — Token-time regime segmentation, offline (interpretability). **Marginal; directly weakened by the measured evidence.**

**Mechanism.** Per sequence: PCA-project hidden states to 64–128 dims (fSRD's sweet spot;
mandatory for cost and to kill outlier-dim domination), fit `fsrd(a)` with `a ∈ R^(64×T)`,
T = 256–1024 tokens, `oblique=False`. Regions = fuzzy token-position windows with distinct local
linear dynamics; consumed outputs: boundary positions (do they align with sentence/paragraph/
reasoning-phase boundaries?), per-region eigenvalues (local timescales/oscillations in
cycles-per-token). This genuinely exercises the full fSRD machinery (tree + BIC + blending) inside
its validated *reconstruction* competence — no forecasting involved.

**Verdict.** Legitimate exploratory interpretability in principle — but this is the candidate the
real-activation measurement (§3 item 9) hits squarely: on exactly these matrices (1024×128,
real Qwen3.5-0.8B states), in-window reconstruction error was 8–27%. A segmentation whose local
models leave a quarter of the variance on the table is a weak lens; region boundaries under that
much misfit are not trustworthy semantic markers. Must beat trivial text segmentation (sentence
boundaries) to be interesting; expected not to. No deployment story. Demoted accordingly.

### Rank 4 — KV-cache compression / low-rank summarization via DMD modes. **Will not work.**

**Mechanism (as proposed).** Per full-attention layer and KV head, `a` = K or V sequence
(`head_dim × T` = 256×T), fSRD regions replace raw cache with `(Φ, λ, b, box)` per region.

**Why not — five independent failures.**
1. *Scope, on this model*: 48 of 64 layers are DeltaNet and **have no KV cache**; the addressable
   object is 16 layers × 4 KV heads, already GQA-compressed 6:1.
2. *No dynamics*: consecutive keys jump with token content; the one genuinely autonomous-linear
   component — the RoPE rotation (unit-modulus modes DMD would lock onto beautifully) — covers
   only 64/256 dims here and is precisely the carrier attention *removes*; fSRD models the
   irrelevant part well and the relevant part badly.
3. *Dominated design*: per region, `Φ diag(b) Vandermonde(λ)` is a rank-r reconstruction with
   *extra* constraints (geometric temporal laws, amplitudes pinned to the region's first column).
   A per-segment truncated SVD at the same rank has strictly lower Frobenius error at a fraction
   of the cost; "BIC-segmented low-rank cache" is better built without fSRD.
4. *Causality*: the cache grows every token; fSRD is a batch fit tied to one matrix; refit of
   ~128 matrices at T=2048 costs 4–20 CPU-minutes against a ~30 ms/token generation budget —
   3–4 orders of magnitude over budget, with no amortization schedule that closes the gap.
5. *Objective mismatch*: KV compression must preserve `softmax(qKᵀ)V` (attention-aware baselines:
   H2O/SnapKV eviction, 2-bit KV quantization with guaranteed 8×); fSRD minimizes blended
   Frobenius error, blind to queries, and its fuzzy blending smears keys exactly at boundaries.

### Rank 5 — Koopman draft head over token positions (the MTP-adjacent idea). **Will not work.** *(Scrutinized hardest, as required.)*

**Mechanism (steel-manned).** During generation, form `H = [h_1..h_T]` (last-layer states,
M = 5120 or 1024), call `fsrd(H, forecast=k)`, read ĥ_{T+1..T+k} from the reconstruction tail,
push each through the frozen final norm + LM head to draft k tokens, verify like MTP speculative
decoding. In-window, this *will* reconstruct well (a large slowly-varying DC direction plus drift
is easy) — and that is precisely the trap: reconstruction is not prediction.

**Four independent kill shots** (each alone fatal; fixing any one leaves the others):

1. **Dynamics validity — fails at the definition.** `h_{t+1} = F(h_1..h_t, e_{t+1})` where
   `e_{t+1}` is the embedding of the *sampled* next token: input-driven, stochastic forcing.
   fSRD's local models are autonomous `x_{t+1} = A x_t`; there is no DMDc-style input channel and
   the authors never contemplate one (their only concession to non-ideal data is *measurement*
   -noise whitening on an autonomous system). The residual of `h_{t+1}` given `h_t` contains the
   entropy of the next token; if a linear autonomous operator could remove it, `lm_head ∘ A`
   would be a competitive linear language model at ~50 MFLOPs/token. The target model itself is
   the counter-argument: **Qwen3.6's MTP head feeds the sampled token's embedding into every
   draft step through a trained nonlinear attention layer precisely because `h_t` alone is
   insufficient** — the proposal deletes the input channel, the nonlinearity, and the training.
2. **Forecast semantics — it doesn't even condition on `h_T`.** Amplitudes are pinned to the
   trailing region's *first* snapshot (author-confirmed; commit `5b465d0`); the forecast is step
   `t+k` of a rollout anchored there, with all accumulated phase/drift error baked in — strictly
   worse than "apply any one-step map to the current state". The dominant fitted mode on hidden
   states is λ ≈ 1 (the DC direction), so the rollout collapses toward a frozen/mean state — an
   expensive approximation of the identity baseline. And with the default `oblique=True`, a
   trailing oblique region yields **literal zero vectors plus a RuntimeWarning** as the "draft".
3. **Refit semantics.** Regions live on the normalized index plane of one matrix; appending one
   column rescales the whole time axis and invalidates every boundary. Honest use = full refit
   per verify cycle. There is no train-once/apply-many mode — fSRD *cannot be a head*; it can
   only be a per-sequence fitting procedure.
4. **Cost — negative ROI even at the cheapest measured shapes.** Measured: 7.9 s per fit at
   M=1024, T=128 and 244 s at M=1024, T=512 (defaults) — i.e. ~10²–10³ token-times of the small
   model per refit on the same CPU; estimated minutes-to-hours at M=5120, T=2048, vs an MTP
   draft step at ~1.7% of a forward pass. Even at fantasy assumptions (refit every 64 tokens,
   k=4, 100% acceptance) the fit costs ≥25× the compute it saves. And the measured 8–27%
   *in-window* reconstruction error on real activations (§3 item 9) caps the ceiling before
   forecasting even starts: the rollout extrapolates a model that cannot even reproduce the data
   it was fitted to.

**Predicted acceptance ordering** (first draft token): MTP head 65–80% (measured) ≫ offline ridge
`h_t→h_{t+1}` ~10–25% (this is plain regression, *not* fSRD — and EAGLE-style ablations show
feature-only drafting without the sampled-token embedding degrades sharply) > n-gram lookup
(free, task-dependent) > identity/copy ~5–15% ≥ **fSRD forecast ≲5%** (an unconditioned,
stale-anchored rollout ≈ identity minus conditioning, plus drift, plus intermittent zero output).
The proposal is simultaneously the most expensive and least accurate drafter in the table.

**Salvage check.** "fSRD modes initialize a tiny trained draft head" degenerates: any
transferable drafter is one fixed map trained across sequences — plain ridge/DMD regression — and
the tree, BIC, and fuzzy blending contribute nothing that survives pooling. Initializing a
regression that will be trained anyway is at best a no-op. **Not an fSRD application.**
Hankel embedding doesn't help: delay coordinates fix observability of autonomous latent dynamics,
not exogenous stochastic input, which is the binding constraint.

### Rank 6 — Layer forecasting / early exit via `forecast=k` on shallow layers. **Will not work.**

Fit on layers 0..ℓ, forecast the remaining columns, decode with the LM head. Three kill shots:
(1) no transfer — a per-matrix fit refit per token costs ~5–30× more than just *running* the
layers it would skip (0.8B: ~1 s fit vs ~10–20 ms full forward; the ratio never wins at any
scale, and generation needs the true deep state for subsequent tokens anyway); (2) non-autonomy —
deeper-layer computation attends to *other positions'* states, information literally absent from
the input matrix; (3) it is self-refuting: it extrapolates across exactly the depth-regime
boundaries whose existence is the Rank-1 finding — if the boundaries are real, extrapolation
across them is invalid; if they aren't, global DMD suffices and fSRD is overhead. This candidate
survives only as the *forecast-degradation measurement* inside the Rank-1 experiment, where its
expected failure is itself evidence that the segmentation is real.

### Cost-scaling verdict (the question asked directly)

**M ≈ 5120 with T ≈ full sequence length is not an acceptable operating point for fSRD** — legal
under the implementation's caps, but hours of 4-core CPU per fit (`np_only`, no GPU path),
several GB of membership/blend temporaries, and 20× the element count of anything the authors
ever validated. The workable operating points for transformer data are comfortably cheap,
though — measured: 1024×128 in ~8 s, 2048×256 in ~15–25 s, under 0.25 GB — so **cost is not the
binding objection at analysis shapes** (depth matrices with min dimension = layer count;
raw-or-projected token windows of T ≤ 256). The binding objection is validity: the
linear-operator-per-region assumption is *plausible descriptively* for residual-stream evolution
across depth, *measured to be marginal* for token-time evolution on real activations of this
family (8–27% in-window error, §3 item 9), and *fatal* for anything input-driven and
forecast-shaped — which is exactly the split between the candidates that survived and the ones
that did not.

---

## 5. The smallest experiment ("start small")

**Hard constraints respected:** no GPU; 4 CPU cores; 15 GB RAM; ~27 GB free disk. **Training or
even loading Qwen3.6-27B here is impossible, in any form** — 27.78B params, 55.6 GB of
safetensors, several times over both the RAM and the disk budget; the same holds for
Qwen3.6-35B-A3B, and no smaller Qwen3.6 exists. The experiment therefore uses the
architecturally faithful small sibling and trains only a tiny auxiliary linear component with
the base model frozen (full fine-tuning of even the 0.75B stand-in needs ~12+ GB for gradients
plus optimizer state — tight-to-infeasible in 15 GB; the head below is closed-form ridge, so no
optimizer state at all). The environment is **already prepared** at
`/home/user/scipy/tools/fsrd_qwen/` — nothing further to install or download:

- `.venv/` — torch 2.13.0+cpu, transformers 5.14.1, numpy 2.4.6, datasets 5.0.1;
- `models/` — **Qwen/Qwen3.5-0.8B**, downloaded and verified working (873M params, 1.75 GB;
  loads via `AutoModelForCausalLM` as `Qwen3_5ForCausalLM`, vision tower dropped, 0.7524B used
  params, 24 layers, hidden 1024, `output_hidden_states` returns a 25-tuple of (1, T, 1024) fp32);
- `harvest_hidden.py` — hidden-state harvester (`--model --out --n-prompts --max-len
  --prompts-file --dtype --threads --no-pad-to-max`; writes `PREFIX.npy` float32 of shape
  `(n_prompts, n_layers+1, max_len, hidden)` via memmap, plus `PREFIX_mask.npy`,
  `PREFIX_meta.json`; right-pads to max-len, safe under causal attention);
- `fsrd_standalone.py` — the solver (same API as `scipy.linalg.fsrd`).

Measured on this box (fp32, 4 threads): model load 4.4–8.1 s; peak RSS 5.09 GB during inference;
prefill 7.3–17.5 ms/token at T ≥ 64. **Perf cliff: below 64 tokens the chunked linear-attention
kernel degrades to a sequential recurrence, ~70× slower per token (T=8 → 1126 ms/token vs T=128 →
8.9 ms/token); the CUDA fast kernels are unavailable. All sequence lengths below are therefore
multiples of 64.**

**Stand-in caveat, stated plainly:** Qwen3.5-0.8B is the same `model_type: qwen3_5`, the same
3:1 DeltaNet/full-attention hybrid with `full_attention_interval: 4`, same head_dim 256, same
vocab 248,320, same MTP config fields — but it is **not Qwen3.6 weights**. Every conclusion is
about the architecture family and transfers to Qwen3.6-27B as a hypothesis, not a measurement.

### What is being tested

**H1 (the Rank-1 claim):** fSRD's BIC-selected depth boundaries are (a) low-misfit (the
locally-linear premise holds in depth even though it measurably strains along token time),
(b) consistent across tokens and prompts, (c) not already given by free layer-similarity curves,
and (d) *useful* — a linear early-exit head trained at the fSRD boundary layer beats the same
head trained at uniformly chosen layers.
**H0 (the null):** depth-matrix misfit is as bad as the measured token-time misfit, or boundaries
are inconsistent, or reproduce the cosine-curve knees, or confer no placement advantage.
**Optional arm (Rank-2):** DeltaNet vs full-attention layer outputs — is the linear-attention
recurrence *effectively* piecewise-constant?
**Falsification arm (Rank-5 negative control):** the fSRD token-time forecast drafts tokens no
better than the identity baseline — pre-registered as the *expected* outcome, so a surprise here
is a genuine finding in either direction.

### Setup (everything already in place)

| item | choice |
|---|---|
| model | `Qwen/Qwen3.5-0.8B` at `/home/user/scipy/tools/fsrd_qwen/models/`, fp32, `torch.set_num_threads(4)` — 24 layers (18 DeltaNet + 6 full-attn), hidden 1024, verified 5.09 GB peak RSS |
| solver | `/home/user/scipy/tools/fsrd_qwen/fsrd_standalone.py` (same API as `scipy.linalg.fsrd`) |
| data | WikiText-2 raw ([train parquet, 6.4 MB, URL verified](https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet)) via the installed `datasets`; TinyStories-valid (19 MB) as a domain contrast |
| harvester | `/home/user/scipy/tools/fsrd_qwen/harvest_hidden.py` |
| env | `/home/user/scipy/tools/fsrd_qwen/.venv` (torch 2.13.0+cpu, transformers 5.14.1, numpy 2.4.6, datasets 5.0.1) |

### Steps

1. **Harvest** (~10–20 min, ~6.7 GB disk): from the repo tools dir, with the venv active:
   `python harvest_hidden.py --model models/<Qwen3.5-0.8B dir> --out states/wt2 --n-prompts 128
   --max-len 256 --prompts-file prompts_wt2.txt --dtype float32 --threads 4`
   (prompts file = 128 WikiText-2 paragraphs of ≥ 256 tokens; **max-len 256 = 4×64**, safely
   above the T<64 kernel cliff). Output `states/wt2.npy` is (128, 25, 256, 1024) fp32 = 3.35 GB
   + mask + meta. Repeat with 64 TinyStories prompts → `states/ts` (1.7 GB). Budget check:
   ~5 GB of 27 GB free. Harvest cost: 128×256 = 32,768 tokens at 7.3–17.5 ms/token ≈ **4–10 min**
   per dataset.
2. **Misfit gate + fSRD depth fits** (~30–90 min): for 400 sampled (prompt, position) pairs with
   mask=1, form the **depth matrix** `a = states[p, 1:, t, :].T` → shape (1024, 24) (dropping
   index 0, the raw embedding, to avoid the first-snapshot anchor pathology); row-standardize;
   fit `fsrd(a, oblique=False, max_depth=2, theta=3.0, smoothness=0.03)` (depth 2 per the
   measured BIC behaviour) and `max_depth=0` (global DMD) on the same matrix. Record: in-window
   relative error, region boundaries, eigenvalues, both BICs. Expected ~1–3 s per fit (well under
   the measured 7.9 s at T=128). **Gate:** if the median multi-region in-window relative error
   exceeds 0.15 — i.e. comparable to the measured token-time misfit — H0 is accepted at this step
   and steps 3–4 are reported as moot. Sweep theta ∈ {1.5, 3}, smoothness ∈ {0.03, 0.05},
   raw-vs-standardized rows (×8 fits ≈ still < 2 h; T=24 fits are cheap).
3. **Free baselines** (minutes, pure NumPy on the harvested array): per sampled token,
   changepoints of (i) cosine similarity of successive layer states, (ii) logit-lens
   KL(layer ℓ → final) knees (final norm + `lm_head` applied in 1024-position batches; keep only
   argmax/KL, never the full 248,320-wide logits array — 5 GB if materialized). Compare boundary
   histograms (mode agreement within ±1 layer).
4. **Train the auxiliary component** (~30–60 min, closed form, base frozen): ridge regression
   `W ∈ R^(1024×1024)`: `h_ℓ → h_24` on ~30k harvested positions (X'X is 1024² — trivial in RAM;
   no gradients, no optimizer state, consistent with the 15 GB budget), at
   ℓ ∈ {fSRD modal boundary ℓ*} ∪ {6, 12, 16, 18} (uniform placements). Metric: top-1 agreement
   of `lm_head(final_norm(W h_ℓ))` with the full model's next-token argmax on 5k held-out
   positions, evaluated in batches as in step 3. This is the trained-not-fSRD component: a
   Koopman-flavoured *linear* exit head whose *placement* is fSRD's sole contribution.
5. **Rank-2 optional arm** (~30 min): for 6 layers (3 DeltaNet, e.g. ℓ = 5, 9, 13; 3
   full-attention, ℓ = 4, 8, 12 — `full_attention_interval: 4`), fit fSRD on the (1024, 128)
   token-time output matrix of 20 prompts each; compare in-window NRMSE distributions
   DeltaNet-vs-attention at matched region count. Deliverable: one boxplot and a yes/no on
   "linear-attention outputs are measurably more piecewise-linear".
6. **Forecast negative control** (~20 min): (a) depth: fit on layers 1–16, `forecast=8`, compare
   forecast deep states to truth (cosine; decoded argmax) against the identity baseline
   (logit-lens at layer 16); (b) token-time: for 20 held-out prompts, fit (1024, 128) windows
   with `forecast=4`; decode ĥ via final norm + `lm_head`; measure first-draft-token acceptance
   vs (i) identity (copy h_T), (ii) corpus-trained ridge `h_t → h_{t+1}`. Catch `RuntimeWarning`s
   and count zero-tail (oblique/no-support) events.

### Decision criteria (pre-registered)

- **Locally-linear-in-depth premise holds** iff median multi-region in-window relative error
  ≤ 0.15 (against the measured 0.267 token-time figure at layer 12). Fails ⇒ Rank-1 dies here.
- **Segmentation is real** iff multi-region BIC < global-DMD BIC on ≥ 70% of tokens AND the
  boundary histogram has ≥ 1 mode holding ≥ 40% mass within ±1 layer, on both datasets.
- **fSRD adds information** iff its modal boundary differs from both free baselines' modal
  changepoints by > 1 layer on at least one dataset — otherwise report "reproduces free
  baselines" (a negative for added value even if segmentation is real).
- **Placement is useful** iff top-1 agreement of the ridge head at ℓ* exceeds the best uniform
  placement by ≥ 2 points. Otherwise placement value is null.
- **Negative result, stated so it can't be explained away:** median misfit > 0.15, OR boundaries
  scattered (no ≥ 40% mode), OR multi-region BIC wins on < 50% of tokens ⇒ depth dynamics are
  not usefully piecewise-linear at this scale, and the Rank-1 candidate dies with no appeal to
  "wrong hyperparameters" — the step-2 sweep already covers theta, smoothness, and
  standardization.
- The token-time control is *expected* to show fSRD ≤ identity < ridge ≪ MTP-class; if fSRD
  beats the ridge map, that is a genuine surprise and the Rank-5 verdict gets revisited.

### Wall-clock budget (4 cores, measured rates)

harvest 2×(4–10 min) + depth fits with sweeps ≤ 2 h + baselines ~15 min + ridge heads ~1 h +
optional Rank-2 arm ~30 min + controls ~20 min ≈ **half a working day end-to-end**; ~6 GB peak
disk of 27 GB free; < 8 GB peak RAM (model 5.1 GB + one memmapped harvest slice).

---

## 6. Risks and honest bottom line

**Risks in the experiment itself.** (1) Row standardization changes the fitted dynamics (the
docstring warns data shifts do); both raw and standardized runs are in the sweep. (2) T = 24
depth columns resolves at most ~3 segments (a region needs ≥ 8 columns to split in time) — a
real but coarse test; the 27B's T = 65 (which exactly matches the paper's own bifurcation-matrix
column count) would need externally harvested states from a bigger machine. (3) The BIC's data
term grows with M·T, so region counts on 1024-row matrices may inflate; theta = 3 (the paper's
own headline setting) is the guard, and the global-DMD comparison is scale-matched. (4) The
stand-in is Qwen3.5-0.8B, not Qwen3.6 weights: same architecture generation, same hybrid
layout, same MTP config — but every result transfers to the 27B as a hypothesis only.
(5) fSRD implementation caveats: no whitening stage, deterministic grid in place of the paper's
PSO, and the authors' round-3 weighted-vs-unweighted-fit contradiction was still unresolved —
region *counts* carry an implementation asterisk even where accuracy does not. (6) The measured
token-time misfit (8–27%) sets a pessimistic prior for the depth arm too; the step-2 gate exists
precisely so a bad answer kills the experiment early and cheaply.

**Bottom line.** fSRD is a batch, single-matrix, index-grid decomposition method whose authors
explicitly do not endorse forecasting; every inference-loop application to Qwen3.6-27B (Koopman
draft head, early-exit forecasting, KV compression) fails on multiple independent grounds —
non-autonomous input-driven dynamics, no cross-input transfer, unordered row axis, misaligned
objectives, and cost overruns against components (like the built-in MTP head: one trained
nonlinear layer, token-embedding-conditioned, 65–80% measured acceptance at 1.7% of a forward
pass) that already solve those problems well. The first contact with real data sharpened, not
softened, this verdict: **fSRD run on actual Qwen3.5-0.8B activations reconstructs its own
training window to only 8–27% relative error** (vs 0.5% on synthetic data), so the
locally-linear-operator premise is already strained *in-window, along token time* — before any
question of forecasting, transfer, or cost. The defensible residue is real but modest and now
carries a measured caveat: **fSRD as an offline, BIC-principled changepoint detector on
residual-stream depth trajectories, guiding the placement of a conventionally trained linear
exit/draft projection** — testable in half a day on the prepared Qwen3.5-0.8B environment, with
a pre-registered misfit gate (median in-window error ≤ 0.15) under which it, too, is declared
dead. Expectation calibration: the depth arm has a genuine but no better than even chance of
passing the gate; if it passes, segmentation will likely be real by BIC but has a substantial
chance of merely reproducing free cosine/logit-lens changepoints; the placement advantage is the
one genuinely open question. The forecast-based uses are expected — and designed — to fail on
the record, and the Rank-2 DeltaNet probe is the sharpest architecture-specific question fSRD
can be asked here, with the measured evidence already leaning negative.

---

## 7. Attribution and sources

- **Method:** C. Bokor, M. Cary, D. Morrey, F. Bonatesta, *fSRD: Fuzzy Spectral Region
  Decomposition — Automated Multi Operator Koopman Representations via an Adaptive Spectral
  Learning Architecture*, [arXiv:2607.17990](https://arxiv.org/abs/2607.17990). Author Q&A
  (three rounds, two answered) and the JMLR supplementary LaTeX were used with the authors'
  answers taken as authoritative over the printed text where they conflict.
- **Implementation:** `scipy/linalg/_fsrd.py` on branch `claude/fable-qwen-subagent-igfklg`
  (with tests, tutorial, benchmarks, and the eight correction commits `32187f8`…`1e16774`), and
  the NumPy standalone from the evaluation bundle.
- **Model:** [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (config.json, model
  card, safetensors index), [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
  [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) (file tree
  + parsed GGUF header of the IQ2_XXS quant), llama.cpp PR
  [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (via search snippets; GitHub direct
  fetch blocked), SGLang/vLLM invocations from the model card, third-party MTP benchmarks
  (jarvislabs.ai). Small models: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
  (MTP tensors verified in the safetensors header),
  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
- Qwen3.6 blog/tech report could not be fetched (JS-rendered); MTP *training* details beyond the
  model card's "trained with multi-steps" remain unverified.
- **Environment measurements** (model load/prefill timings, RAM footprints, the T<64
  linear-attention kernel cliff, fSRD wall-clock at (1024,128)/(2048,256), and the
  real-activation reconstruction errors of §3 item 9) were produced by the parallel
  environment-preparation agent on this machine and are reported as measured.
