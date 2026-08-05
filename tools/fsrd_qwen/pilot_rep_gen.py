#!/usr/bin/env python
"""PILOT step 1: induce degeneration, label it independently, harvest hidden states.

Greedy decoding, no repetition penalty. Ground-truth loop labels come from token-level
cycle detection on the generated ids only (never from any spectral quantity).
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

torch.set_num_threads(2)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "Qwen3.5-0.8B")
OUT = os.path.join(HERE, "pilot_rep_data")
LAYERS = [12, 24]
MAX_NEW = 160

# ---- prompts -------------------------------------------------------------
# loop-prone: short / low-information / partially-seeded repetition. Deliberately a mix of
# immediate loopers and mild ones, so that some onsets land mid-generation.
LOOPY = [
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
    "ha ha ha ha ha ha ha ha",
    "1, 1, 1, 1, 1, 1, 1,",
    "I don't know. I don't know. I don't know.",
    "Item 1: a\nItem 1: a\nItem 1: a\n",
    "..........",
    "the the the the the the",
    "Song: la la la la la la",
    "x = x = x = x = x =",
    "one two one two one two one two",
    "Count from one to twenty: one, two,",
    "Translate to French: hello. Translate to French: hello. Translate to French:",
    "A list of things to do:\n1.",
    "Once upon a time there was a very small village called",
]
HEALTHY = [
    "The mitochondrion is an organelle that",
    "In 1969, the Apollo 11 mission successfully",
    "Photosynthesis converts light energy into chemical energy by",
    "The main differences between TCP and UDP are that",
    "She opened the letter and read the first line, which said",
    "A brief history of the printing press begins with",
    "The theory of plate tectonics explains that the Earth's",
    "Machine learning models generalize better when the training data",
    "Quantum entanglement is a phenomenon in which two particles",
    "The causes of the French Revolution included widespread",
    "In computer science, a hash table is a data structure that",
    "The immune system defends the body against pathogens by",
    "Ocean currents are driven primarily by differences in",
    "Antibiotic resistance develops when bacteria",
]


def find_onset(ids, max_p=24, min_reps=3, tail_slack=8):
    """Independent, token-level cycle detection. Returns (onset, period) or (None, None).

    Scans left to right; at each candidate start s and period p, checks whether the block
    ids[s:s+p] repeats verbatim min_reps times and keeps repeating to within tail_slack of
    the end (terminal loop).
    """
    n = len(ids)
    best = None
    for s in range(n):
        for p in range(1, max_p + 1):
            if s + p * min_reps > n:
                continue
            blk = ids[s:s + p]
            reps = 1
            j = s + p
            while j + p <= n and ids[j:j + p] == blk:
                reps += 1
                j += p
            if reps >= min_reps and j >= n - tail_slack - p:
                if best is None or s < best[0]:
                    best = (s, p)
                break
        if best is not None:
            break
    return best if best is not None else (None, None)


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    prompts = [(p, "loopy") for p in LOOPY] + [(p, "healthy") for p in HEALTHY]
    meta = []
    for gi, (prompt, kind) in enumerate(prompts):
        t0 = time.time()
        enc = tok(prompt, return_tensors="pt")
        plen = enc.input_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=False,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                min_new_tokens=MAX_NEW, use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        full = gen[0].tolist()
        new_ids = full[plen:]
        t_gen = time.time() - t0

        onset, period = find_onset(new_ids)

        # teacher-forced forward pass for hidden states + next-token distributions
        t1 = time.time()
        L = len(full)
        pad_to = ((L + 63) // 64) * 64
        ids = torch.tensor([full + [tok.eos_token_id] * (pad_to - L)])
        att = torch.zeros((1, pad_to), dtype=torch.long)
        att[0, :L] = 1
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs = out.hidden_states
        for lay in LAYERS:
            h = hs[lay][0, :L].to(torch.float32).numpy()
            np.save(os.path.join(OUT, f"g{gi:03d}_L{lay}.npy"), h.astype(np.float32))
        logits = out.logits[0, :L].to(torch.float32)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = (-(p * logp).sum(-1)).numpy()
        maxp = p.max(-1).values.numpy()
        np.save(os.path.join(OUT, f"g{gi:03d}_ent.npy"), ent.astype(np.float32))
        np.save(os.path.join(OUT, f"g{gi:03d}_maxp.npy"), maxp.astype(np.float32))
        t_fwd = time.time() - t1

        meta.append(dict(
            gi=gi, kind=kind, prompt=prompt, plen=plen, n_new=len(new_ids),
            total_len=L, onset=onset, period=period,
            label="positive" if onset is not None else "negative",
            new_ids=new_ids, t_gen=t_gen, t_fwd=t_fwd,
            text=tok.decode(new_ids),
        ))
        print(f"[{gi:03d}] {kind:8s} onset={onset} p={period} "
              f"gen={t_gen:.1f}s fwd={t_fwd:.1f}s | {tok.decode(new_ids[:40])!r}",
              flush=True)

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)
    npos = sum(m["label"] == "positive" for m in meta)
    print(f"\npositives={npos} negatives={len(meta)-npos}")


if __name__ == "__main__":
    sys.exit(main())
