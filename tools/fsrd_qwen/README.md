# fSRD on Qwen — exploratory scaffolding

Scratch experiment area investigating whether `fsrd` (the piecewise
Koopman/DMD solver added on this branch, `scipy/linalg/_fsrd.py`) is useful
for anything inside a transformer language model, with `Qwen/Qwen3.6-27B` and
its multi-token-prediction (MTP) releases as the notional target.

**This is exploratory research code, not a SciPy deliverable.** Nothing here is
part of the `scipy.linalg.fsrd` public API, and nothing here is wired into the
SciPy build or test suite.

The method is due to C. Bokor, M. Cary, D. Morrey, and F. Bonatesta, *"fSRD:
Fuzzy Spectral Region Decomposition"*, [arXiv:2607.17990](https://arxiv.org/abs/2607.17990).

## Findings so far

See `INVESTIGATION.md` for the full assessment, and **`RESULTS.md` for the
executed pre-registered experiment** (section 5 of the investigation): the
misfit gate passed, but the depth segmentation failed its BIC-consistency
threshold and the fSRD-placed ridge exit head lost to uniform deeper
placements by 20+ points — H0 accepted for the deployment-relevant claim. Two facts dominate the design:

- **Qwen3.6 cannot be run here.** The only official Qwen3.6 checkpoints are
  27B (55.6 GB of safetensors) and 35B-A3B; there is no small variant. Both
  exceed this machine's disk and RAM by a wide margin. Experiments therefore
  use `Qwen/Qwen3.5-0.8B` as an *architecturally faithful* stand-in — same
  `qwen3_5` model type, same 3:1 `linear_attention`/`full_attention` hybrid
  (`full_attention_interval: 4`), same `head_dim` 256, same 248320 vocab, same
  `mtp_num_hidden_layers: 1` — but it is **not** Qwen3.6 weights, and results
  do not automatically transfer.
- **fSRD is cheap, but real activations are not very low-rank.** Cost is a
  non-issue (15 s / 200 MB at M=2048, T=256 on 4 CPU cores). However, relative
  reconstruction error on *real* Qwen3.5-0.8B activations is 8.1e-2 (layer 1),
  2.67e-1 (layer 12), 1.21e-1 (layer 24), against 4.9e-3 on synthetic
  low-rank test signals. Transformer activations are substantially less
  locally-linear than the synthetic-data fit quality suggests.

## Environment

Not committed (see `.gitignore`); recreate with:

```sh
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch transformers huggingface_hub numpy safetensors datasets
```

CPU-only by design — this machine has no GPU. The fast linear-attention kernels
(`causal-conv1d`, `flash-linear-attention`) are CUDA-only, so the pure-torch
fallback is used.

`fsrd_standalone.py`, the NumPy-only single-file extract of the solver, is also
not committed since it duplicates `scipy/linalg/_fsrd.py`. Either build the
branch and `from scipy.linalg import fsrd`, or drop the standalone extract from
the fSRD evaluation bundle into this directory.

## Scripts

- `harvest_hidden.py` — runs the model over prompts on CPU and saves per-layer
  hidden states to `.npy`, in a layout documented in its module docstring
  (including ready-made `(M, T)` slicing recipes for fSRD, for both the
  token-position and layer-depth time axes).

## Gotchas

- **Use sequence length >= 64**, ideally a multiple of 64. Below 64 tokens the
  chunked linear-attention path degrades to a sequential recurrence and costs
  ~70x more per token (1126 ms/token at T=8 vs 8.9 ms/token at T=128).
- `harvest_hidden.py` right-pads to `--max-len`, which is safe under causal
  attention (real tokens' hidden states are unchanged) and avoids that cliff.
  Always slice with the emitted mask before analysis.
- Output size is `n_prompts * (n_layers+1) * max_len * hidden * 4` bytes, so
  64 prompts at 512 tokens is ~3.4 GB. Budget disk before large harvests.
- Qwen3.6-27B's config is `Qwen3_5ForConditionalGeneration` (vision + text).
  `AutoModelForCausalLM` drops the vision tower, but code assuming a flat
  `Qwen3ForCausalLM` config must handle the `config.text_config` nesting.
- Full fine-tuning even the 0.8B stand-in needs ~12+ GB for gradients and
  optimizer state, which is tight in 15 GB. Freeze the base model and train
  only a small auxiliary component, or use LoRA.
