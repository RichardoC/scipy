"""INVESTIGATION2 candidate-B probe: are Gated DeltaNet recurrent states low-rank?

Runs Qwen3.5-0.8B (stand-in) prefill in 64-token chunks, snapshots every
DeltaNet layer's recurrent state S (num_v_heads, 128, 128) at several token
counts, and reports singular-value spectra / truncation errors.

Output: results/inv2_state_rank.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
import numpy as np
import torch

torch.set_num_threads(4)

BASE = "/home/user/scipy/tools/fsrd_qwen"
MODEL = os.path.join(BASE, "models", "Qwen3.5-0.8B")
OUT = os.path.join(BASE, "results", "inv2_state_rank.json")
CHECKPOINTS = [int(x) for x in os.environ.get("INV2_CKPTS", "64,256,1024").split(",")]
RANKS = [4, 8, 16, 32, 64]
N_PROMPTS = int(os.environ.get("INV2_NPROMPTS", "6"))

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()

# prompts: reuse the wt2 prompt file, take long ones
with open(os.path.join(BASE, "prompts_wt2.txt")) as f:
    prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
if os.environ.get("INV2_LONG"):
    # concatenated streams so checkpoints reach 1024 tokens
    k = 8
    prompts = ["\n\n".join(prompts[i * k:(i + 1) * k]) for i in range(len(prompts) // k)]
    OUT = OUT.replace(".json", os.environ.get("INV2_SUFFIX", "_long") + ".json")

cfg = model.config
lin_layers = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
print(f"linear layers: {len(lin_layers)}", flush=True)

def get_recurrent_states(cache):
    """Return dict layer_idx -> np.ndarray (H, dk, dv) from the cache object."""
    out = {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise RuntimeError(f"cache type {type(cache)} has no .layers")
    for i, lyr in enumerate(layers):
        rs = getattr(lyr, "recurrent_states", None)
        if rs is None:
            continue
        t = rs[0] if isinstance(rs, (list, tuple, dict)) else rs
        if t is None:
            continue
        arr = t.detach().to(torch.float32).numpy()
        # shape (B, H, dk, dv)
        out[i] = arr[0]
    return out

results = {"checkpoints": CHECKPOINTS, "ranks": RANKS, "prompts": [], "per_layer": {}}
# accumulators: (ckpt, layer) -> list over prompts+heads of spectra stats
acc = {}

for pi, prompt in enumerate(prompts[:N_PROMPTS]):
    ids = tok(prompt, return_tensors="pt").input_ids[:, : max(CHECKPOINTS)]
    T = ids.shape[1]
    if T < max(CHECKPOINTS):
        # pad conceptually: just use what we have; skip too-short
        if T < 256:
            continue
    results["prompts"].append({"idx": pi, "tokens": int(T)})
    print(f"prompt {pi}: {T} tokens", flush=True)
    for ck in CHECKPOINTS:
        if ck > T:
            break
        # fresh full prefill per checkpoint: keeps the fast chunked kernel path
        # (continuing from a cached state falls into the ~0.3 s/token
        # sequential recurrence in the torch fallback)
        with torch.no_grad():
            out = model(input_ids=ids[:, :ck], use_cache=True)
        past = out.past_key_values
        states = get_recurrent_states(past)
        for li, S in states.items():  # S: (H, dk, dv)
            H = S.shape[0]
            for h in range(H):
                sv = np.linalg.svd(S[h], compute_uv=False)
                tot = float((sv ** 2).sum())
                if tot <= 0:
                    continue
                cum = np.cumsum(sv ** 2) / tot
                ent = {
                    "sv1_frac": float(sv[0] ** 2 / tot),
                    "r90": int(np.searchsorted(cum, 0.90) + 1),
                    "r99": int(np.searchsorted(cum, 0.99) + 1),
                }
                for r in RANKS:
                    # relative Frobenius error of best rank-r approx
                    err = float(np.sqrt(max(0.0, 1.0 - cum[min(r, len(cum)) - 1])))
                    ent[f"relF_r{r}"] = err
                acc.setdefault((ck, li), []).append(ent)
        del past, out

# aggregate: median over prompts x heads, per checkpoint per layer; also overall
def med(key, entries):
    return float(np.median([e[key] for e in entries]))

summary = {}
for (ck, li), entries in acc.items():
    summary.setdefault(str(ck), {})[str(li)] = {
        k: med(k, entries) for k in entries[0].keys()
    }
overall = {}
for ck in CHECKPOINTS:
    ents = [e for (c, l), es in acc.items() if c == ck for e in es]
    if ents:
        overall[str(ck)] = {k: med(k, ents) for k in ents[0].keys()}
results["per_layer"] = summary
results["overall_median"] = overall
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(overall, indent=1), flush=True)
print("DONE", flush=True)
