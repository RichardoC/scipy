#!/usr/bin/env python
"""Harvest per-layer hidden states from a CPU-loaded HF causal LM into .npy files.

Purpose
-------
Produce activation "snapshot matrices" suitable for feeding into the fSRD solver
(see fsrd_standalone.py in this directory), which expects a real-valued 2-D array
of shape (M, T): M spatial/feature rows by T time/snapshot columns.

Outputs
-------
Given ``--out PREFIX`` (a path; the ``.npy`` suffix is optional and stripped),
three files are written:

1. ``PREFIX.npy``       float32, shape ``(n_prompts, n_layers + 1, max_len, hidden_size)``
   Axis 0 : prompt index, in the same order as the prompt list used.
   Axis 1 : layer index. Index 0 is the *embedding* output (before block 0);
            index i>0 is the output of decoder block i-1. Hence the length is
            ``n_layers + 1``. This is exactly HF's ``output_hidden_states=True``
            tuple, stacked.
   Axis 2 : token position, right-padded to ``max_len``. Padded positions are
            written as 0.0 and are flagged False in the mask (below). Sequences
            longer than ``max_len`` are truncated.
   Axis 3 : hidden/model dimension (``config.hidden_size``).

2. ``PREFIX_mask.npy``  bool, shape ``(n_prompts, max_len)``
   True where axis-2 of the main array holds a real token, False for padding.
   Always use this to slice before analysis, otherwise padding zeros will
   contaminate the statistics.

3. ``PREFIX_meta.json`` provenance: model path, dtype, prompt strings, real token
   count per prompt, resolved shape, layer count, hidden size, timings.

Converting to an fSRD snapshot matrix
-------------------------------------
fSRD wants (M, T) = (features, snapshots). For a single prompt p and layer l,
take the real tokens as snapshots and transpose so hidden dims are rows::

    h    = np.load("PREFIX.npy", mmap_mode="r")
    mask = np.load("PREFIX_mask.npy")
    a    = np.ascontiguousarray(h[p, l][mask[p]].T)   # -> (hidden_size, n_tokens)
    #      a.shape == (M=hidden_size, T=n_real_tokens)

To instead treat *layer depth* as the time axis for one token position t::

    a = np.ascontiguousarray(h[p, :, t, :].T)         # -> (hidden_size, n_layers+1)

Performance note
----------------
By default each prompt is right-padded to ``--max-len`` before the forward pass.
On hybrid linear-attention models (Qwen3.5 / Qwen3.6 ``qwen3_5``) the CPU
fallback kernel processes tokens in chunks of 64 and degrades to a slow
sequential recurrence for shorter inputs -- measured ~1000 ms/token at T=16
versus ~14 ms/token at T=64. Keep ``--max-len`` at 64 or more (ideally a
multiple of 64). Padding is safe because attention is causal, so real-token
hidden states match the unpadded run exactly; use the mask to drop pad columns.
Pass ``--no-pad-to-max`` to disable.

Memory note
-----------
The main array is float32 and its size is
``n_prompts * (n_layers+1) * max_len * hidden_size * 4`` bytes. It grows fast:
e.g. 64 prompts x 25 layers x 512 tokens x 2048 dims = 6.7 GB. It is written
incrementally to a ``np.lib.format.open_memmap`` so peak RAM stays at roughly
one prompt's worth, but check your free disk before requesting large runs.

Usage
-----
    python harvest_hidden.py --model ./models/Qwen3.5-0.8B --out ./hidden/run1 \
        --n-prompts 4 --max-len 64
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

# Built-in prompt pool. --n-prompts takes the first N (cycling if N exceeds the
# pool). Replace or extend freely, or pass --prompts-file for your own.
DEFAULT_PROMPTS = [
    "The capital of France is Paris, a city known for its architecture.",
    "In mathematics, a prime number is a natural number greater than one.",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
    "The mitochondrion is often described as the powerhouse of the cell.",
    "Gradient descent iteratively updates parameters to reduce a loss function.",
    "Shakespeare wrote both tragedies and comedies during the Elizabethan era.",
    "A singular value decomposition factors a matrix into three components.",
    "Photosynthesis converts light energy into chemical energy in plants.",
]


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def load_prompts(n: int, prompts_file: str | None) -> list[str]:
    if prompts_file:
        with open(prompts_file) as fh:
            pool = [ln.strip() for ln in fh if ln.strip()]
        if not pool:
            raise SystemExit(f"no non-empty lines in {prompts_file}")
    else:
        pool = DEFAULT_PROMPTS
    return [pool[i % len(pool)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Harvest per-layer hidden states to .npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="HF model id or local directory")
    ap.add_argument("--out", required=True, help="output path prefix (.npy suffix optional)")
    ap.add_argument("--n-prompts", type=int, default=4, help="number of prompts to run")
    ap.add_argument("--max-len", type=int, default=64, help="pad/truncate length in tokens")
    ap.add_argument("--prompts-file", default=None, help="one prompt per line; overrides built-ins")
    ap.add_argument("--no-pad-to-max", action="store_true",
                    help="feed each prompt at its true length instead of padding to --max-len. "
                         "Much slower for max_len<64 on hybrid linear-attention models, whose "
                         "chunked kernel degrades to a sequential loop below 64 tokens.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="torch compute dtype (output .npy is always float32)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)

    prefix = args.out[:-4] if args.out.endswith(".npy") else args.out
    out_dir = os.path.dirname(os.path.abspath(prefix))
    os.makedirs(out_dir, exist_ok=True)

    prompts = load_prompts(args.n_prompts, args.prompts_file)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)
    )
    model.eval()
    t_load = time.perf_counter() - t0

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)  # multimodal configs nest the LM config
    n_layers = tcfg.num_hidden_layers
    hidden = tcfg.hidden_size
    n_l = n_layers + 1
    P, L = len(prompts), args.max_len

    nbytes = P * n_l * L * hidden * 4
    print(f"model loaded in {t_load:.2f}s | layers={n_layers} hidden={hidden}")
    print(f"allocating {prefix}.npy shape=({P}, {n_l}, {L}, {hidden}) float32 "
          f"= {nbytes / 1e6:.1f} MB")

    arr = np.lib.format.open_memmap(
        prefix + ".npy", mode="w+", dtype=np.float32, shape=(P, n_l, L, hidden)
    )
    mask = np.zeros((P, L), dtype=bool)
    real_lens: list[int] = []

    t_fwd_total = 0.0
    pad_to_max = not args.no_pad_to_max
    if pad_to_max and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    for i, text in enumerate(prompts):
        # Right-pad to a fixed length so the model always sees L tokens. Because
        # attention is causal (both the full-attention and the recurrent
        # linear-attention layers), hidden states at real token positions are
        # bit-identical to the unpadded run; only the trailing pad columns differ,
        # and those are excluded by the mask. Fixed L >= 64 keeps the chunked
        # linear-attention kernel on its fast path.
        if pad_to_max:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=L,
                      padding="max_length")
            t = int(enc["attention_mask"][0].sum())
        else:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=L)
            t = enc["input_ids"].shape[1]
        real_lens.append(t)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        t_fwd_total += time.perf_counter() - t0

        # out.hidden_states: tuple of n_layers+1 tensors, each (1, seq, hidden)
        stacked = torch.stack(out.hidden_states, dim=0)[:, 0, :t]  # (n_l, t, hidden)
        arr[i, :, :t, :] = stacked.to(torch.float32).numpy()
        mask[i, :t] = True
        del out, stacked
        print(f"  [{i + 1}/{P}] tokens={t:<4d} '{text[:44]}...'")

    arr.flush()
    del arr
    np.save(prefix + "_mask.npy", mask)

    meta = {
        "model": os.path.abspath(args.model) if os.path.isdir(args.model) else args.model,
        "compute_dtype": args.dtype,
        "stored_dtype": "float32",
        "shape": [P, n_l, L, hidden],
        "axes": ["prompt", "layer(0=embeddings)", "token(right-padded)", "hidden"],
        "n_layers": n_layers,
        "hidden_size": hidden,
        "max_len": L,
        "n_prompts": P,
        "padded_to_max_len": pad_to_max,
        "prompts": prompts,
        "real_token_lens": real_lens,
        "load_time_s": round(t_load, 3),
        "forward_total_s": round(t_fwd_total, 3),
        "forward_s_per_token": round(t_fwd_total / max(sum(real_lens), 1), 5),
        "peak_rss_gb": round(peak_rss_gb(), 3),
    }
    with open(prefix + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {prefix}.npy, {prefix}_mask.npy, {prefix}_meta.json")
    print(f"forward total {t_fwd_total:.2f}s for {sum(real_lens)} tokens "
          f"({t_fwd_total / max(sum(real_lens), 1) * 1000:.1f} ms/token) | "
          f"peak RSS {peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
