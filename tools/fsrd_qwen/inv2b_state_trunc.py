#!/usr/bin/env python
"""INVESTIGATION2.md section 5b (arm B): DeltaNet state truncation quality test.

Explicitly NOT an fSRD experiment -- the tool here is truncated SVD on the
Gated-DeltaNet recurrent states S (per layer, per head, 128x128 fp32) of the
Qwen3.5-0.8B stand-in. Two pre-registered parts:

Part 1 (at-rest / prefix-cache resume, section 5b.1):
    prefill 256 tokens; replace every DeltaNet S by its best rank-r
    approximation (r in {8, 16, 32}) ONCE; teacher-force the next 256 tokens
    of the same stream; measure mean KL(untouched || truncated) of the
    continuation next-token distributions and top-1 agreement, over 20
    fixed prompt streams.
    Gate: rank-16 mean KL <= 0.05 nats AND top-1 >= 95%.
    Kill: one-shot rank-32 KL > 0.2 nats kills the whole candidate.

Part 2 (in-loop periodic re-truncation, section 5b.2):
    process a 1024-token stream in 64-token chunks; after every chunk,
    re-truncate every S to rank r (r in {16, 32, 64}); measure teacher-forced
    perplexity delta vs an untouched run using the *identical* chunked
    protocol (so the delta isolates truncation, not the forward path).
    Gate: rank-32 costs <= 2% relative perplexity; > 5% at rank-64 kills
    in-loop compression.

Determinism: everything is teacher-forced (no sampling of any kind); all runs
consume the same token windows; the untouched and truncated runs differ only
in the state edit. model.eval(), fp32, fixed thread count.

Outputs: results/inv2b_state_trunc.json
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "models", "Qwen3.5-0.8B")
OUT = os.path.join(BASE, "results", "inv2b_state_trunc.json")
SCRATCH = os.environ.get(
    "INV2B_SCRATCH",
    "/tmp/claude-0/-home-user-scipy/8ec6b2c3-62da-55ca-a96d-bd9df70737b7/scratchpad",
)
N_STREAMS = int(os.environ.get("INV2B_NSTREAMS", "20"))
PREFILL, CONT = 256, 256          # part 1
LOOP_LEN, CHUNK = 1024, 64        # part 2
RANKS_REST = [8, 16, 32]
RANKS_LOOP = [16, 32, 64]
BATCH = N_STREAMS                 # all streams in one batch (no padding: equal lengths)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def log(*a):
    print(*a, flush=True)


def build_streams(tok, n, length):
    """n non-overlapping token windows of `length`, holdout text first then wt2."""
    text = (open(os.path.join(BASE, "prompts_wt2_holdout.txt")).read()
            + "\n\n" + open(os.path.join(BASE, "prompts_wt2.txt")).read())
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = n * length
    assert ids.numel() >= need, f"need {need} tokens, have {ids.numel()}"
    return ids[:need].reshape(n, length)


def truncate_states(past, r):
    """In-place best rank-r truncation of every DeltaNet recurrent state."""
    n_lin = 0
    for lyr in past.layers:
        rs = getattr(lyr, "recurrent_states", None)
        if not isinstance(rs, dict) or 0 not in rs:
            continue
        S = rs[0]                                # (B, H, dk, dv) fp32
        B, H, dk, dv = S.shape
        M = S.reshape(B * H, dk, dv)
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        s[:, r:] = 0.0
        rs[0].copy_(((U * s.unsqueeze(-2)) @ Vh).reshape(B, H, dk, dv))
        n_lin += 1
    return n_lin


@torch.no_grad()
def prefill(model, ids):
    out = model(input_ids=ids, use_cache=True)
    return out.past_key_values


@torch.no_grad()
def continue_chunks(model, past, ids_cont, p_memmap=None, p_argmax=None,
                    store=False):
    """Teacher-force ids_cont in CHUNK-sized chunks continuing from `past`.

    store=True: write log-probs (fp16) into p_memmap and argmax ids into
    p_argmax (the untouched reference run).
    store=False: compare against the reference; return (kl_sum, agree_sum, n).
    """
    B, T = ids_cont.shape
    kl_sum, agree_sum, n_pos = 0.0, 0, 0
    for c0 in range(0, T, CHUNK):
        chunk = ids_cont[:, c0:c0 + CHUNK]
        out = model(input_ids=chunk, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logq = torch.log_softmax(out.logits.float(), dim=-1)  # (B, chunk, V)
        del out
        if store:
            p_memmap[:, c0:c0 + CHUNK, :] = logq.to(torch.float16).numpy()
            p_argmax[:, c0:c0 + CHUNK] = logq.argmax(-1).numpy()
        else:
            for b in range(B):  # keep peak memory bounded
                logp = torch.from_numpy(
                    np.asarray(p_memmap[b, c0:c0 + CHUNK, :])).float()
                p = torch.exp(logp)
                kl = (p * (logp - logq[b])).sum(-1)          # (chunk,)
                kl_sum += float(kl.sum())
                agree_sum += int((logq[b].argmax(-1).numpy()
                                  == p_argmax[b, c0:c0 + CHUNK]).sum())
                n_pos += kl.numel()
        del logq
    return kl_sum, agree_sum, n_pos


@torch.no_grad()
def loop_nll(model, ids, rank=None):
    """Chunked teacher-forced mean NLL over a (B, LOOP_LEN) stream.

    rank=None: untouched (but same chunked protocol). Otherwise re-truncate
    every DeltaNet S to `rank` after each chunk.
    """
    B, T = ids.shape
    past = None
    nll_sum, n_pos = 0.0, 0
    prev_last_logits = None
    for c0 in range(0, T, CHUNK):
        chunk = ids[:, c0:c0 + CHUNK]
        out = model(input_ids=chunk, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits.float()
        del out
        logp = torch.log_softmax(logits, dim=-1)
        del logits
        # within-chunk: position i predicts chunk token i+1
        tgt = chunk[:, 1:]
        nll_sum += float(-torch.gather(
            logp[:, :-1, :], 2, tgt.unsqueeze(-1)).sum())
        n_pos += tgt.numel()
        # across-boundary: previous chunk's last logits predict this chunk's first token
        if prev_last_logits is not None:
            nll_sum += float(-torch.gather(
                prev_last_logits, 1, chunk[:, :1]).sum())
            n_pos += B
        prev_last_logits = logp[:, -1, :].clone()
        del logp
        if rank is not None and c0 + CHUNK < T:
            truncate_states(past, rank)
    return nll_sum / n_pos


def main():
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    vocab = model.config.text_config.vocab_size if hasattr(
        model.config, "text_config") else model.config.vocab_size
    log(f"model loaded, vocab={vocab}")

    results = {"config": {
        "n_streams": N_STREAMS, "prefill": PREFILL, "cont": CONT,
        "loop_len": LOOP_LEN, "chunk": CHUNK, "ranks_rest": RANKS_REST,
        "ranks_loop": RANKS_LOOP,
        "protocol": "teacher-forced, greedy-free (no sampling), fp32 CPU; "
                    "untouched and truncated runs consume identical tokens"}}

    streams = build_streams(tok, N_STREAMS, LOOP_LEN)
    log(f"streams: {tuple(streams.shape)}")

    # ---------------- Part 1: at-rest one-shot truncation -----------------
    ids_pre = streams[:, :PREFILL]
    ids_cont = streams[:, PREFILL:PREFILL + CONT]
    os.makedirs(SCRATCH, exist_ok=True)
    pm_path = os.path.join(SCRATCH, "inv2b_ref_logprobs.f16")
    p_memmap = np.memmap(pm_path, dtype=np.float16, mode="w+",
                         shape=(N_STREAMS, CONT, vocab))
    p_argmax = np.zeros((N_STREAMS, CONT), dtype=np.int64)

    log("part1: untouched reference run")
    t0 = time.time()
    past = prefill(model, ids_pre)
    continue_chunks(model, past, ids_cont, p_memmap, p_argmax, store=True)
    p_memmap.flush()
    log(f"  reference done in {time.time()-t0:.0f}s")

    results["at_rest"] = {}
    # r=None: determinism control -- fresh prefill, NO truncation, same compare
    # path; must give KL ~ 0 and top-1 ~ 1.0, proving the KL below measures
    # truncation only (binding rule: identical protocol/no sampling).
    for r in [None] + RANKS_REST:
        t0 = time.time()
        past = prefill(model, ids_pre)          # identical fresh prefill
        n_lin = truncate_states(past, r) if r is not None else 0
        kl_sum, agree_sum, n_pos = continue_chunks(
            model, past, ids_cont, p_memmap, p_argmax, store=False)
        ent = {"rank": r, "n_lin_layers": n_lin,
               "mean_kl_nats": kl_sum / n_pos,
               "top1_agree": agree_sum / n_pos, "n_positions": n_pos,
               "seconds": round(time.time() - t0, 1)}
        key = str(r) if r is not None else "untouched_control"
        results["at_rest"][key] = ent
        log(f"part1 rank={r}: KL={ent['mean_kl_nats']:.6f} nats, "
            f"top1={ent['top1_agree']:.4f} ({ent['seconds']}s)")

    del p_memmap
    os.remove(pm_path)

    # ---------------- Part 2: in-loop periodic re-truncation --------------
    log("part2: untouched chunked-NLL baseline")
    t0 = time.time()
    nll0 = loop_nll(model, streams, rank=None)
    ppl0 = float(np.exp(nll0))
    log(f"  untouched: nll={nll0:.4f} ppl={ppl0:.2f} ({time.time()-t0:.0f}s)")
    results["in_loop"] = {"untouched": {"nll": nll0, "ppl": ppl0}}
    for r in RANKS_LOOP:
        t0 = time.time()
        nll = loop_nll(model, streams, rank=r)
        ppl = float(np.exp(nll))
        ent = {"rank": r, "nll": nll, "ppl": ppl,
               "rel_ppl_increase": ppl / ppl0 - 1.0,
               "seconds": round(time.time() - t0, 1)}
        results["in_loop"][str(r)] = ent
        log(f"part2 rank={r}: ppl={ppl:.3f} (+{100*ent['rel_ppl_increase']:.2f}%) "
            f"({ent['seconds']}s)")

    # ---------------- gates ----------------
    g = {}
    r16 = results["at_rest"]["16"]
    g["at_rest_rank16_kl<=0.05"] = r16["mean_kl_nats"] <= 0.05
    g["at_rest_rank16_top1>=0.95"] = r16["top1_agree"] >= 0.95
    r32 = results["at_rest"]["32"]
    g["kill_rank32_kl>0.2"] = r32["mean_kl_nats"] > 0.2
    g["in_loop_rank32_ppl<=2pct"] = results["in_loop"]["32"]["rel_ppl_increase"] <= 0.02
    g["in_loop_rank64_ppl>5pct_dead"] = results["in_loop"]["64"]["rel_ppl_increase"] > 0.05
    results["gates"] = g
    results["wall_seconds"] = round(time.time() - t_start, 1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    log(json.dumps(g, indent=1))
    log(f"DONE in {results['wall_seconds']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
