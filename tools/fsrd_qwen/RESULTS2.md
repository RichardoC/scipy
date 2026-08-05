# fSRD memory experiments on Qwen3.6-27B weights / Qwen3.5-0.8B states — results

**Date:** 2026-08-05. **Spec:** `INVESTIGATION2.md` §5 (arm A) and §5b (arm B),
executed as pre-registered. **Hardware:** 4 CPU cores, 15 GB RAM, no GPU;
weights reached by HTTP range requests into
`unsloth/Qwen3.6-27B-MTP-GGUF` (BF16 shard), ~1.17 GB downloaded total.

**Scope cap (restated from the spec, §5):** arm A is an *offline codec*
comparison on an imatrix-weighted RMSE proxy — the same metric family
llama.cpp quant development uses — on 8 real Qwen3.6-27B weight tensors. No
27B inference is possible on this box, so there is **no end-to-end perplexity
and no kernel-speed claim**; the deployable subset of any win would be
row-band regions served as split tensors. Arm B runs entirely on the
Qwen3.5-0.8B stand-in (same architecture family, 18 DeltaNet layers, 16 heads
of 128×128 fp32 state per layer), so its conclusions transfer to Qwen3.6-27B
**as hypothesis only**. Arm B is explicitly **not an fSRD experiment**: its
tool is truncated SVD.

## Gates table

(filled in from measured results below)

## What was run

| stage | script | data | notes |
|---|---|---|---|
| A: fetch | `inv2a_fetch.py` | 8 BF16 tensors + imatrix from the GGUF repo | range requests via `gguf_fetch.py` machinery; header cache hit, ~1.17 GB total |
| A: prep | `inv2a_prep.py` | per-tensor exact SSE tables at b∈{2,3,4,5,6,8}, Z maps, B0/B1 | Q_K-style codec: 32-elem blocks, 6-bit scale/min vs fp16 superblock refs = b+0.5 bpw |
| A: fSRD | `inv2a_fsrd.py` | 8 tensors × 8 sweep configs (depth {2,3} × sort order × weighted/unweighted) | `fsrd(Z, oblique=False, theta=3, smoothness=0.03)`, uncontended |
| A: eval | `inv2a_eval.py` | B2 joint band+bits DP; fSRD region allocation; bit ledgers; gates | identical Lagrangian allocator for B2 and fSRD |
| B: truncation | `inv2b_state_trunc.py` | 20 streams × 1024 tokens (wt2-holdout text first, then wt2) | teacher-forced, deterministic; per-stream spreads recorded |

