"""INVESTIGATION2 feasibility probe: parse GGUF tensor index of the BF16 shard
of unsloth/Qwen3.6-27B-MTP-GGUF via HTTP range requests, and fetch one small
tensor to verify per-tensor range access. NumPy + urllib only. No multi-GB
downloads: grabs header bytes (up to ~32 MB) + one small tensor.

Output: results/inv2_gguf_index.json (tensor name/shape/type/offset for all
tensors) + stdout summary.
"""
import json
import os
import ssl
import struct
import sys
import urllib.request

import numpy as np

REPO = "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main"
SHARD1 = REPO + "/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"
OUT = "/home/user/scipy/tools/fsrd_qwen/results/inv2_gguf_index.json"

CA = "/root/.ccr/ca-bundle.crt"
ctx = ssl.create_default_context(cafile=CA if os.path.exists(CA) else None)


def fetch_range(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        code = r.getcode()
        data = r.read()
    return code, data


# GGUF v3 parsing
GGUF_TYPES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
              7: "bool", 8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64"}
SCALAR_FMT = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
              5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
              12: ("<d", 8)}


class Buf:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def read(self, n):
        if self.o + n > len(self.d):
            raise EOFError(f"need {self.o+n} bytes, have {len(self.d)}")
        b = self.d[self.o:self.o + n]
        self.o += n
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def s(self):
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def scalar(self, t):
        fmt, n = SCALAR_FMT[t]
        return struct.unpack(fmt, self.read(n))[0]

    def value(self, t, skip_big_arrays=True):
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u32()
            n = self.u64()
            if et == 8:
                vals = [self.s() for _ in range(n)]
                return f"<str array n={n}>" if n > 8 else vals
            fmt, sz = SCALAR_FMT[et]
            raw = self.read(sz * n)
            if n > 16:
                return f"<{GGUF_TYPES[et]} array n={n}>"
            return list(struct.unpack("<" + fmt[1] * n, raw))
        return self.scalar(t)


# GGML tensor type ids -> (name, block_elems, block_bytes)
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2),
        2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 8: ("Q8_0", 32, 34),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210)}


def parse_header(data):
    b = Buf(data)
    magic = b.read(4)
    assert magic == b"GGUF", magic
    version = b.u32()
    n_tensors = b.u64()
    n_kv = b.u64()
    kv = {}
    for _ in range(n_kv):
        k = b.s()
        t = b.u32()
        kv[k] = b.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = b.s()
        nd = b.u32()
        dims = [b.u64() for _ in range(nd)]
        ttype = b.u32()
        off = b.u64()
        tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
    align = kv.get("general.alignment", 32)
    data_start = (b.o + align - 1) // align * align
    return version, kv, tensors, data_start


def main():
    # 1) grab header progressively
    for size in (4 << 20, 16 << 20, 48 << 20):
        code, data = fetch_range(SHARD1, 0, size - 1)
        print(f"range fetch {size>>20} MiB -> HTTP {code}, got {len(data)} bytes", flush=True)
        try:
            version, kv, tensors, data_start = parse_header(data)
            break
        except EOFError as e:
            print(f"  header larger than {size>>20} MiB ({e}); retrying", flush=True)
    else:
        print("FAILED to parse header")
        return 1

    print(f"GGUF v{version}, {len(tensors)} tensors, data_start={data_start}")
    interesting = {k: v for k, v in kv.items() if not isinstance(v, str) or len(str(v)) < 200}
    for k in sorted(interesting):
        if k.startswith(("general.", "qwen", "split")):
            print(f"  {k} = {interesting[k]}")

    # tensor table with absolute offsets and sizes
    idx = []
    for t in tensors:
        tt = GGML.get(t["type"], (f"type{t['type']}", None, None))
        n_elem = 1
        for d in t["dims"]:
            n_elem *= d
        nbytes = None
        if tt[1]:
            nbytes = n_elem // tt[1] * tt[2]
        idx.append({"name": t["name"], "dims": t["dims"], "ggml_type": tt[0],
                    "abs_offset": data_start + t["offset"], "nbytes": nbytes})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"kv_keys": sorted(kv.keys()), "n_tensors": len(idx),
                   "data_start": data_start, "tensors": idx}, f)
    print(f"wrote {OUT}")

    # 3) show the tensors candidate C would target
    for pat in ("blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.attn_q.weight",
                "blk.1.linear_attn.in_proj_qkvz.weight", "token_embd.weight", "output.weight"):
        hits = [t for t in idx if t["name"] == pat]
        for h in hits:
            print(f"  target: {h['name']} dims={h['dims']} type={h['ggml_type']} "
                  f"bytes={h['nbytes']} off={h['abs_offset']}")

    # 4) fetch one small tensor fully + a slice of one big bf16 tensor
    small = next((t for t in idx if t["name"].endswith("attn_norm.weight")), None)
    if small and small["nbytes"] and small["nbytes"] < 1 << 20:
        code, data = fetch_range(SHARD1, small["abs_offset"], small["abs_offset"] + small["nbytes"] - 1)
        arr = np.frombuffer(data, dtype=np.uint16)
        f32 = arr.astype(np.uint32) << 16
        vals = f32.view(np.float32)
        print(f"  fetched {small['name']}: HTTP {code}, {len(data)} B, "
              f"mean={vals.mean():.4f} std={vals.std():.4f} min={vals.min():.3f} max={vals.max():.3f}")

    big = next((t for t in idx if t["name"] == "blk.0.ffn_down.weight" and t["ggml_type"] == "BF16"), None)
    if big is None:
        big = next((t for t in idx if "ffn" in t["name"] and t["ggml_type"] == "BF16"), None)
    if big:
        # fetch first 2 rows worth (dims[0] = ne0 = fastest axis)
        ne0 = big["dims"][0]
        nb = ne0 * 2 * 2
        code, data = fetch_range(SHARD1, big["abs_offset"], big["abs_offset"] + nb - 1)
        arr = np.frombuffer(data, dtype=np.uint16).astype(np.uint32) << 16
        vals = arr.view(np.float32)
        print(f"  fetched 2 rows of {big['name']} (ne0={ne0}): HTTP {code}, {len(data)} B, "
              f"absmax={np.abs(vals).max():.4f} std={vals.std():.5f}, finite={np.isfinite(vals).all()}")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
