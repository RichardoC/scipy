# PILOT: Detecting degenerate repetition in LLM generation with fSRD

Date: 2026-08-05. Model: `models/Qwen3.5-0.8B` (24 layers, hidden 1024, fp32 CPU).
Solver: `fsrd_standalone.fsrd`.

## Pre-registration (written BEFORE any results were seen)

### Task
Given the hidden-state trajectory of a greedy LLM generation along token positions, raise an
alarm before/at the token where the generation demonstrably enters a repetition loop.

### Ground truth (independent of every detector)
Token-level cycle detection on the *generated token ids only*. For each position `t`, find the
smallest period `p in [1, 24]` such that `tok[t-p:t] == tok[t-2p:t-p]` and the same period-`p`
cycle continues (allowing no mismatch) for at least `3` full repetitions. `onset` = index of the
first token of the first of those repetitions. A generation is POSITIVE iff such an onset exists
and the cycle runs to within 8 tokens of the end (i.e. it is a terminal loop, not a passing
coincidence); otherwise NEGATIVE (healthy). Labels are frozen before any spectral quantity is
computed. No detector output ever enters the label.

### Data / harvest
- Greedy decoding (`do_sample=False`), no repetition penalty, `max_new_tokens = 192`.
- Loop-prone prompts: short / low-information / repetitive, some seeded with a partially
  repeated phrase. Healthy prompts: informative, open-ended factual/narrative prompts.
- Target >= 10 positives and >= 10 negatives. Class balance reported honestly whatever it is.
- Hidden states obtained by one teacher-forced forward pass over `prompt + generation`
  (identical to decode-time states because the model is causal), sequence length padded to a
  multiple of 64 to avoid the measured <64-token perf cliff.
- Primary layer: **12** (mid-stack). Secondary layer reported for sensitivity: **24** (last).
  Every baseline is computed at the *same* layer as fSRD, so the comparison is apples-to-apples.

### Detector (fSRD)
Sliding window over generated token positions, `W = 32` columns, stride 1, `dt = 1` token.
Rows: hidden dims standardized using that generation's own per-dim mean/std (label-independent;
required because massive-activation dims otherwise dominate every local SVD -- INVESTIGATION.md
§3), then POD-projected onto the top `k = 16` left singular vectors of the (uncentered) window.
No per-window mean subtraction (centering a snapshot matrix makes DMD return spurious
Fourier-frequency eigenvalues, which would fake the exact signal being tested).
fSRD called with `max_depth=2, oblique=False`.
Window score = **oscillation score** := max over regions and modes of
`|b_i * ||phi_i||| * exp(-|Re w_i| * W) * 1[0.05 <= |Im w_i| <= pi]`, i.e. amplitude-weighted
evidence for a sustained (non-decaying) oscillatory mode. Score is assigned to the window's last
token. Secondary registered fSRD readout: does a region *temporal* boundary land within +-4
tokens of onset more often than chance.

### Baselines (all at the same layer, same windows)
1. **n-gram**: max over `n in {1..8}` of the count of the current `n`-gram earlier in the output.
2. **cosine self-similarity**: max cosine similarity between hidden state at `t` and states in
   `[t-W, t-2]`. (Nearly free; acknowledged as the strong bar.)
3. **entropy / max-prob** of the next-token distribution at `t`.
4. **global DMD**: identical pipeline with `max_depth=0` (one region, no fSRD partition).

### Decision thresholds (fixed now)
- **Signal at all** requires pooled window-level `AUC >= 0.80` (positive class = window whose
  last token is at or after onset in a positive generation; negative class = all windows of
  negative generations), AND positive median lead time at matched false-positive rate.
- **Lead time** measured at a threshold calibrated on NEGATIVE generations to a 5% window-level
  FPR. Lead = `onset - first_alarm_token` (positive = early).
- **WIN over cosine self-similarity** requires: median lead time at least **+3 tokens** better
  than cosine, OR `AUC` at least **+0.05** higher, with no worse FPR.
- **fSRD's regions add value** only if fSRD beats `max_depth=0` global DMD by **+3 tokens**
  median lead or **+0.05 AUC**. If not, the honest verdict is "DMD may help, fSRD's regions do
  not".
- **Degeneracy auto-fail check**: if `n_regions == 1` on **>50%** of windows at `max_depth=2`,
  fSRD's partition is inoperative here; report as such (not fatal for DMD, fatal for fSRD).
- **Cost bar**: per-window fit time must be below one decode forward pass (~14 ms/token measured
  on this box). A detector costing more than the model step is useless in serving; a fit slower
  than that is reported as a cost failure even if accurate.
- **GO** requires: signal threshold met AND a win over cosine self-similarity AND cost bar met.
  Anything else is **NO-GO**. A clean null is a successful pilot and will not be softened.

---

## Results

(filled in after the run; see `pilot_repetition.json` for raw numbers)
