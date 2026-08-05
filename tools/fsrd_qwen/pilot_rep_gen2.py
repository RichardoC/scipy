#!/usr/bin/env python
"""PILOT step 1b: more generations from the informative/open-ended pool, which is the pool
that spontaneously loops MID-generation (onset >= W) and therefore the only pool that can
support a lead-time measurement.  Same greedy settings, same frozen labelling rule.
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
sys.path.insert(0, HERE)
from pilot_rep_gen import find_onset, MODEL, OUT, LAYERS, MAX_NEW  # noqa: E402

GI0 = 100

POOL = [
    "The steps of cellular respiration are as follows:",
    "A summary of the causes of the First World War:",
    "Explain in detail how a refrigerator works:",
    "The difference between weather and climate is that",
    "The periodic table is organised so that elements",
    "A list of the planets in order from the Sun:",
    "How does a vaccine train the immune system? First,",
    "Describe the water cycle step by step:",
    "The main organs of the digestive system are",
    "What are the advantages and disadvantages of nuclear power?",
    "The rules of chess state that each piece",
    "A short biography of Marie Curie:",
    "The layers of the Earth from the outside in are",
    "How is glass manufactured industrially? The process",
    "Explain the difference between RAM and a hard disk:",
    "The three branches of the United States government are",
    "A recipe for bread requires the following ingredients:",
    "The stages of mitosis in order are",
    "Why is the sky blue? The explanation involves",
    "The largest deserts in the world include",
    "Describe how an internal combustion engine converts fuel into motion:",
    "The functions of the liver include",
    "A brief timeline of the space race:",
    "How do noise-cancelling headphones work?",
    "The main types of clouds and what they indicate:",
    "Explain what inflation means in economics:",
    "The parts of a flower and their functions are",
    "How does GPS determine your position on Earth?",
    "The differences between mitosis and meiosis are",
    "A list of common logical fallacies with examples:",
    "Explain how a suspension bridge carries load:",
    "The main causes of coral reef bleaching are",
    "How is electricity generated in a hydroelectric dam?",
    "The stages of human sleep and what happens in each:",
    "Describe the structure of an atom:",
    "The advantages of public transport over private cars include",
    "How do plants transport water from roots to leaves?",
    "The major biomes of the world are",
    "Explain the greenhouse effect in simple terms:",
    "The differences between a virus and a bacterium are",
    "A summary of how the internet routes a packet:",
    "The most important inventions of the 19th century were",
    "How does a microwave oven heat food?",
    "The main muscle groups of the human body are",
    "Explain what a black hole is and how one forms:",
    "The steps involved in making cheese are",
    "How do birds navigate during migration?",
    "The properties that distinguish metals from non-metals are",
    "Describe the life cycle of a star:",
    "The main features of Gothic architecture include",
    "How does a camera lens form an image?",
    "The reasons that languages change over time include",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    meta_path = os.path.join(OUT, "meta2.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    done = {m["gi"] for m in meta}

    for j, prompt in enumerate(POOL):
        gi = GI0 + j
        if gi in done:
            continue
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

        t1 = time.time()
        L = len(full)
        pad_to = ((L + 63) // 64) * 64
        ids = torch.tensor([full + [tok.eos_token_id] * (pad_to - L)])
        att = torch.zeros((1, pad_to), dtype=torch.long)
        att[0, :L] = 1
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
        for lay in LAYERS:
            h = out.hidden_states[lay][0, :L].to(torch.float32).numpy()
            np.save(os.path.join(OUT, f"g{gi:03d}_L{lay}.npy"), h.astype(np.float32))
        logits = out.logits[0, :L].to(torch.float32)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        np.save(os.path.join(OUT, f"g{gi:03d}_ent.npy"),
                (-(p * logp).sum(-1)).numpy().astype(np.float32))
        np.save(os.path.join(OUT, f"g{gi:03d}_maxp.npy"),
                p.max(-1).values.numpy().astype(np.float32))
        t_fwd = time.time() - t1

        meta.append(dict(gi=gi, kind="informative", prompt=prompt, plen=plen,
                         n_new=len(new_ids), total_len=L, onset=onset, period=period,
                         label="positive" if onset is not None else "negative",
                         new_ids=new_ids, t_gen=t_gen, t_fwd=t_fwd,
                         text=tok.decode(new_ids)))
        with open(meta_path, "w") as f:      # incremental: nothing lost on a crash
            json.dump(meta, f)
        print(f"[{gi:03d}] onset={onset} p={period} gen={t_gen:.1f}s fwd={t_fwd:.1f}s "
              f"| {tok.decode(new_ids[:30])!r}", flush=True)

    npos = sum(m["label"] == "positive" for m in meta)
    nlate = sum(m["label"] == "positive" and m["onset"] >= 32 for m in meta)
    print(f"\nround2: positives={npos} (onset>=32: {nlate}) negatives={len(meta)-npos}")


if __name__ == "__main__":
    sys.exit(main())
