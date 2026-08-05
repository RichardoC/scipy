#!/usr/bin/env python
"""INVESTIGATION2.md section 5 (arm A), Data step: fetch the 8 pre-registered
Qwen3.6-27B BF16 weight tensors plus their imatrix importance vectors.

Tensors (from the BF16 shard of unsloth/Qwen3.6-27B-MTP-GGUF, HTTP range
requests via gguf_fetch.py's machinery -- no new fetch tool):
  blk.{0,20,40}.ffn_down.weight   (np 5120 x 17408)
  blk.{0,20,40}.ffn_gate.weight   (np 17408 x 5120)
  blk.3.attn_q.weight             (np 12288 x 5120, full-attention layer)
  blk.1.attn_qkv.weight           (np 10240 x 5120, DeltaNet in-projection)

Each tensor is fetched, converted to fp16 and written to states/inv2a/ (kept
only until its arm-A evaluation completes -- see inv2a_quant.py); the raw
download buffer is freed immediately. Importance vectors
w_j = sqrt(in_sum2[j] / counts) are parsed from imatrix_unsloth.gguf_file and
saved to results/inv2a_imatrix.npz. Disk free space is logged before/after
every tensor (binding rule: never below 8 GB).
"""
import json
import os
import shutil
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from gguf_fetch import HF_URL, RangeClient, fetch_header, fetch_tensor  # noqa: E402

SHARD = "BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"
IMATRIX = "imatrix_unsloth.gguf_file"
REPO = "unsloth/Qwen3.6-27B-MTP-GGUF"
CACHE = os.path.join(BASE, ".gguf_cache")
OUTDIR = os.path.join(BASE, "states", "inv2a")
IM_OUT = os.path.join(BASE, "results", "inv2a_imatrix.npz")
LOG_OUT = os.path.join(BASE, "results", "inv2a_fetch_log.json")

TARGETS = [
    "blk.0.ffn_down.weight", "blk.20.ffn_down.weight", "blk.40.ffn_down.weight",
    "blk.0.ffn_gate.weight", "blk.20.ffn_gate.weight", "blk.40.ffn_gate.weight",
    "blk.3.attn_q.weight", "blk.1.attn_qkv.weight",
]
MIN_FREE_GB = 8.0


def free_gb(path="/"):
    return shutil.disk_usage(path).free / 2**30


def log(*a):
    print(*a, flush=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report = {"targets": TARGETS, "tensors": {}, "df_start_gb": free_gb()}
    log(f"disk free at start: {report['df_start_gb']:.1f} GB")

    # ---- imatrix importance vectors (13.6 MB file; header is cached) ----
    im_client = RangeClient(HF_URL.format(repo=REPO, path=IMATRIX), verbose=False)
    im_h = fetch_header(im_client, CACHE)
    by_name = {t.name: t for t in im_h.tensors}
    im = {}
    for name in TARGETS:
        s2, _, _ = fetch_tensor(im_client, by_name[name + ".in_sum2"])
        cnt, _, _ = fetch_tensor(im_client, by_name[name + ".counts"])
        w = np.sqrt(np.asarray(s2, dtype=np.float64) / float(np.asarray(cnt).ravel()[0]))
        im[name] = w
        log(f"imatrix {name}: n={w.size} counts={float(np.asarray(cnt).ravel()[0]):.0f} "
            f"w[min,med,max]=({w.min():.4g},{np.median(w):.4g},{w.max():.4g})")
    np.savez(IM_OUT, **{k: v for k, v in im.items()})
    log(f"wrote {IM_OUT} (imatrix bytes fetched: {im_client.bytes_downloaded:,})")

    # ---- the 8 BF16 tensors, one range request each ----
    client = RangeClient(HF_URL.format(repo=REPO, path=SHARD), verbose=False)
    h = fetch_header(client, CACHE)
    by_name = {t.name: t for t in h.tensors}
    for name in TARGETS:
        out = os.path.join(OUTDIR, name + ".fp16.npy")
        if os.path.exists(out):
            log(f"skip {name} (already on disk)")
            continue
        if free_gb() < MIN_FREE_GB:
            raise RuntimeError(f"free disk {free_gb():.1f} GB < {MIN_FREE_GB} GB floor")
        t = by_name[name]
        t0 = time.time()
        arr, got, dt = fetch_tensor(client, t, dtype="fp16")
        np.save(out, arr)
        rec = {"np_shape": list(arr.shape), "bytes": got, "fetch_s": round(dt, 1),
               "df_after_gb": round(free_gb(), 1)}
        report["tensors"][name] = rec
        log(f"fetched {name} {arr.shape} in {dt:.0f}s "
            f"({got/2**20:.0f} MiB, {got/dt/2**20:.1f} MiB/s); "
            f"disk free {rec['df_after_gb']:.1f} GB")
        del arr

    report["total_bytes"] = client.bytes_downloaded
    report["df_end_gb"] = round(free_gb(), 1)
    with open(LOG_OUT, "w") as f:
        json.dump(report, f, indent=1)
    log(f"DONE: {client.bytes_downloaded:,} B downloaded, "
        f"disk free {report['df_end_gb']} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
