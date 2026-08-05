#!/usr/bin/env python3
"""
gguf_fetch.py -- read a GGUF model's metadata and extract individual tensors
from a remote file (e.g. on HuggingFace) using HTTP range requests only.

Why this works
--------------
A GGUF file is:

    [magic u32 "GGUF"][version u32][tensor_count u64][kv_count u64]
    [kv_count  x  KV entries        ]   <- key/value metadata (arch, hparams, vocab...)
    [tensor_count x tensor-info recs ]   <- name, n_dims, dims[], ggml_type, offset
    [padding to general.alignment    ]
    [tensor data blob                ]   <- tensor i lives at data_start + info.offset

Every tensor's absolute byte range is therefore known after reading only the
header (a few MB), so a single `Range:` request per tensor suffices.  The size
in bytes is n_elements * type_size // block_size for the tensor's ggml type.

HuggingFace `resolve/main/...` URLs answer 302 with a *signed* CDN URL.  The
signature covers the path only, not the Range header, so we resolve once and
then reuse the CDN URL for every subsequent ranged GET (redirect-following also
works but costs an extra round trip and HF re-signs each time).  The CDN
returns 206 + Content-Range for single ranges; multi-range (`bytes=a-b,c-d`)
is *not* supported and returns 416, so ranges are fetched one at a time.

Usage
-----
  # list tensors (downloads only the header)
  python gguf_fetch.py --file Qwen3.6-27B-Q4_K_M.gguf --list
  python gguf_fetch.py --file Qwen3.6-27B-Q4_K_M.gguf --kv          # metadata
  python gguf_fetch.py --file Qwen3.6-27B-Q4_K_M.gguf --summary     # grouped inventory

  # extract tensors matching a regex, dequantized, to .npy
  python gguf_fetch.py --file Qwen3.6-27B-Q4_K_M.gguf \
      --tensor '^blk\\.0\\.(attn_q|attn_kv)\\.weight$' --out states/ --stats

Options of note:
  --repo         HF repo id (default unsloth/Qwen3.6-27B-MTP-GGUF)
  --dtype        fp32 (default) or fp16 output
  --raw          skip dequantization, save the raw quantized bytes
  --stats        print min/max/mean/std, zero fraction and SVD spectrum
  --cache-dir    where fetched headers are cached (default .gguf_cache/)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np
import requests

from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize

GGUF_MAGIC_LE = b"GGUF"
DEFAULT_REPO = "unsloth/Qwen3.6-27B-MTP-GGUF"
HF_URL = "https://huggingface.co/{repo}/resolve/main/{path}"


# --------------------------------------------------------------------------- #
# HTTP range client
# --------------------------------------------------------------------------- #
class NeedMoreBytes(Exception):
    """Raised by the parser when the fetched prefix is too short."""

    def __init__(self, want: int):
        super().__init__(f"need at least {want} bytes")
        self.want = want


class RangeClient:
    """Minimal HTTP byte-range reader that keeps a resolved CDN URL."""

    def __init__(self, url: str, verbose: bool = True):
        self.origin_url = url
        self.verbose = verbose
        self.session = requests.Session()
        self.bytes_downloaded = 0
        self.requests_made = 0
        self._cdn_url: str | None = None
        self.size: int | None = None
        self.etag: str | None = None
        self._resolve()

    def _resolve(self) -> None:
        # allow_redirects=True: requests re-sends our headers on the 302 hop.
        r = self.session.head(self.origin_url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self._cdn_url = r.url
        self.size = int(r.headers["Content-Length"])
        self.etag = r.headers.get("ETag", "").strip('"')
        if self.verbose:
            print(f"[http] {os.path.basename(self.origin_url)}: {self.size:,} bytes "
                  f"(accept-ranges={r.headers.get('Accept-Ranges')})", file=sys.stderr)

    def get_range(self, start: int, length: int) -> bytes:
        """Fetch exactly `length` bytes starting at `start`. Verifies the reply."""
        if length <= 0:
            return b""
        end = start + length - 1
        hdrs = {"Range": f"bytes={start}-{end}"}
        t0 = time.time()
        r = self.session.get(self._cdn_url, headers=hdrs, timeout=600, stream=True)
        if r.status_code == 403 and self._cdn_url != self.origin_url:
            # signed CDN URL expired -> re-resolve once and retry
            self._resolve()
            r = self.session.get(self._cdn_url, headers=hdrs, timeout=600, stream=True)
        if r.status_code != 206:
            raise RuntimeError(f"expected 206 for Range {hdrs['Range']}, got {r.status_code}")
        buf = r.content
        if len(buf) != length:
            raise RuntimeError(f"range {start}+{length} returned {len(buf)} bytes")
        self.bytes_downloaded += len(buf)
        self.requests_made += 1
        if self.verbose:
            dt = time.time() - t0
            print(f"[http] 206 {len(buf):,} B @ {start:,} in {dt:.2f}s "
                  f"({len(buf)/max(dt,1e-9)/1e6:.1f} MB/s)", file=sys.stderr)
        return buf


# --------------------------------------------------------------------------- #
# GGUF header parser that fails loudly on a truncated prefix
# --------------------------------------------------------------------------- #
_SCALAR = {
    GGUFValueType.UINT8: ("<B", 1), GGUFValueType.INT8: ("<b", 1),
    GGUFValueType.UINT16: ("<H", 2), GGUFValueType.INT16: ("<h", 2),
    GGUFValueType.UINT32: ("<I", 4), GGUFValueType.INT32: ("<i", 4),
    GGUFValueType.FLOAT32: ("<f", 4), GGUFValueType.UINT64: ("<Q", 8),
    GGUFValueType.INT64: ("<q", 8), GGUFValueType.FLOAT64: ("<d", 8),
    GGUFValueType.BOOL: ("<?", 1),
}


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]          # GGUF order (fastest-varying first)
    ggml_type: GGMLQuantizationType
    rel_offset: int                 # offset within the tensor-data region
    n_elements: int
    n_bytes: int
    abs_offset: int = 0             # filled in once data_start is known

    @property
    def np_shape(self) -> tuple[int, ...]:
        return tuple(reversed(self.shape))


@dataclass
class GGUFHeader:
    version: int
    tensor_count: int
    kv_count: int
    kv: "OrderedDict[str, Any]" = dc_field(default_factory=OrderedDict)
    kv_types: dict = dc_field(default_factory=dict)
    tensors: list = dc_field(default_factory=list)
    header_bytes: int = 0           # bytes up to end of tensor-info block
    alignment: int = 32
    data_start: int = 0


class _Cursor:
    def __init__(self, buf: bytes):
        self.b = buf
        self.o = 0

    def need(self, n: int) -> None:
        if self.o + n > len(self.b):
            raise NeedMoreBytes(self.o + n)

    def raw(self, n: int) -> bytes:
        self.need(n)
        out = self.b[self.o:self.o + n]
        self.o += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > 1 << 30:
            raise RuntimeError(f"absurd string length {n} at {self.o}")
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, vtype: int) -> Any:
        gt = GGUFValueType(vtype)
        if gt == GGUFValueType.STRING:
            return self.string()
        if gt in _SCALAR:
            fmt, sz = _SCALAR[gt]
            return struct.unpack(fmt, self.raw(sz))[0]
        if gt == GGUFValueType.ARRAY:
            itype = self.u32()
            alen = self.u64()
            igt = GGUFValueType(itype)
            if igt in _SCALAR:                      # bulk-decode numeric arrays
                fmt, sz = _SCALAR[igt]
                blob = self.raw(sz * alen)
                return list(np.frombuffer(blob, dtype=np.dtype(fmt[1:])))
            return [self.value(itype) for _ in range(alen)]
        raise RuntimeError(f"unhandled GGUF value type {vtype}")


def parse_header(buf: bytes) -> GGUFHeader:
    """Parse a GGUF header from a (possibly truncated) file prefix.

    Raises NeedMoreBytes(want) if `buf` is too short; the caller should refetch
    at least `want` bytes and try again.
    """
    c = _Cursor(buf)
    if c.raw(4) != GGUF_MAGIC_LE:
        raise RuntimeError("not a GGUF file (bad magic)")
    version = c.u32()
    if version not in (2, 3):
        raise RuntimeError(f"unsupported GGUF version {version}")
    tensor_count = c.u64()
    kv_count = c.u64()

    h = GGUFHeader(version=version, tensor_count=tensor_count, kv_count=kv_count)
    for _ in range(kv_count):
        key = c.string()
        vtype = c.u32()
        h.kv_types[key] = GGUFValueType(vtype).name
        h.kv[key] = c.value(vtype)

    for _ in range(tensor_count):
        name = c.string()
        n_dims = c.u32()
        dims = tuple(c.u64() for _ in range(n_dims))
        ggml_type = GGMLQuantizationType(c.u32())
        rel = c.u64()
        n_elem = int(np.prod(dims)) if dims else 0
        blk, tsz = GGML_QUANT_SIZES[ggml_type]
        h.tensors.append(TensorInfo(name, dims, ggml_type, rel, n_elem,
                                    n_elem * tsz // blk))

    h.header_bytes = c.o
    h.alignment = int(h.kv.get("general.alignment", 32))
    pad = c.o % h.alignment
    h.data_start = c.o + (h.alignment - pad if pad else 0)
    for t in h.tensors:
        t.abs_offset = h.data_start + t.rel_offset
    return h


def fetch_header(client: RangeClient, cache_dir: str, first: int = 1 << 20,
                 max_probe: int = 1 << 28) -> GGUFHeader:
    """Progressively range-fetch the file prefix until the header parses.

    Caches the exact header prefix on disk keyed by the file's ETag so repeat
    runs cost zero bytes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = (client.etag or os.path.basename(client.origin_url)).replace("/", "_")
    cache = os.path.join(cache_dir, tag + ".header")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            buf = f.read()
        try:
            h = parse_header(buf)
            print(f"[hdr] cache hit {cache} ({len(buf):,} B)", file=sys.stderr)
            return h
        except (NeedMoreBytes, RuntimeError):
            buf = b""
    else:
        buf = b""

    want = first
    while True:
        if want > max_probe:
            raise RuntimeError(f"header still not parsed after {want:,} bytes")
        if len(buf) < want:
            buf += client.get_range(len(buf), want - len(buf))
        try:
            h = parse_header(buf)
        except NeedMoreBytes as e:
            want = max(e.want, len(buf) * 2)
            continue
        with open(cache, "wb") as f:
            f.write(buf[:h.header_bytes])
        print(f"[hdr] parsed: header_bytes={h.header_bytes:,} "
              f"(fetched {len(buf):,} B)", file=sys.stderr)
        return h


# --------------------------------------------------------------------------- #
# tensor extraction
# --------------------------------------------------------------------------- #
def fetch_tensor(client: RangeClient, t: TensorInfo, raw: bool = False,
                 dtype: str = "fp32") -> tuple[np.ndarray, int, float]:
    """Range-fetch and (unless raw) dequantize one tensor.

    Returns (array, bytes_downloaded, seconds).
    """
    t0 = time.time()
    b0 = client.bytes_downloaded
    blob = client.get_range(t.abs_offset, t.n_bytes)
    got = client.bytes_downloaded - b0
    if raw:
        return np.frombuffer(blob, dtype=np.uint8).copy(), got, time.time() - t0

    qt = t.ggml_type
    if qt == GGMLQuantizationType.F32:
        arr = np.frombuffer(blob, dtype=np.float32).reshape(t.np_shape)
    elif qt == GGMLQuantizationType.F16:
        arr = np.frombuffer(blob, dtype=np.float16).reshape(t.np_shape).astype(np.float32)
    elif qt in (GGMLQuantizationType.I8, GGMLQuantizationType.I16,
                GGMLQuantizationType.I32, GGMLQuantizationType.I64):
        npt = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}[int(qt.name[1:])]
        arr = np.frombuffer(blob, dtype=npt).reshape(t.np_shape)
    else:
        # gguf.quants.dequantize wants the *byte* shape: rows x row_bytes
        blk, tsz = GGML_QUANT_SIZES[qt]
        row_bytes = t.shape[0] // blk * tsz
        byte_shape = t.np_shape[:-1] + (row_bytes,)
        q = np.frombuffer(blob, dtype=np.uint8).reshape(byte_shape)
        arr = dequantize(q, qt).reshape(t.np_shape)
    arr = np.ascontiguousarray(arr, dtype=np.float16 if dtype == "fp16" else np.float32)
    return arr, got, time.time() - t0


def tensor_stats(a: np.ndarray, svd_k: int = 20) -> dict:
    x = a.astype(np.float64)
    out = {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean()), "std": float(x.std()),
        "zero_frac": float((a == 0).mean()),
    }
    if a.ndim == 2 and min(a.shape) > 1:
        sv = np.linalg.svd(x, compute_uv=False)
        e = sv ** 2
        cum = np.cumsum(e) / e.sum()
        out["sv_top"] = [float(v) for v in sv[:svd_k]]
        out["sv_n"] = int(sv.size)
        out["sv_max"] = float(sv[0])
        out["sv_min"] = float(sv[-1])
        for frac in (0.5, 0.9, 0.99):
            out[f"rank_{int(frac*100)}"] = int(np.searchsorted(cum, frac) + 1)
        out["rank_frac_90"] = out["rank_90"] / sv.size
        out["rank_frac_99"] = out["rank_99"] / sv.size
        out["stable_rank"] = float(e.sum() / e[0])
    return out


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
_LAYER_RE = re.compile(r"\.\d+\.")


def collapse(name: str) -> str:
    return _LAYER_RE.sub(".N.", name)


def print_summary(h: GGUFHeader) -> None:
    groups: dict[str, dict] = OrderedDict()
    for t in h.tensors:
        k = (collapse(t.name), t.shape, t.ggml_type.name)
        g = groups.setdefault(k, {"count": 0, "bytes": 0, "layers": []})
        g["count"] += 1
        g["bytes"] += t.n_bytes
        m = re.search(r"\.(\d+)\.", t.name)
        if m:
            g["layers"].append(int(m.group(1)))
    total = sum(t.n_bytes for t in h.tensors)
    print(f"{'pattern':46s} {'shape':22s} {'type':7s} {'n':>4s} "
          f"{'bytes':>15s} {'%':>6s}  layers")
    for (name, shape, ty), g in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"]):
        ls = sorted(g["layers"])
        lstr = ""
        if ls:
            lstr = f"{ls[0]}..{ls[-1]}" if len(ls) > 3 else ",".join(map(str, ls))
        print(f"{name:46s} {str(shape):22s} {ty:7s} {g['count']:4d} "
              f"{g['bytes']:15,d} {100*g['bytes']/total:6.2f}  {lstr}")
    print(f"\ntotal tensor bytes: {total:,}  tensors: {len(h.tensors)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="path within the HF repo")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--url", help="full URL, overrides --repo/--file for fetching")
    ap.add_argument("--tensor", help="tensor name or regex to extract")
    ap.add_argument("--out", help="output dir (or .npy path for a single tensor)")
    ap.add_argument("--list", action="store_true", help="list all tensors")
    ap.add_argument("--kv", action="store_true", help="dump KV metadata")
    ap.add_argument("--kv-grep", help="only KV keys matching this regex")
    ap.add_argument("--summary", action="store_true", help="grouped tensor inventory")
    ap.add_argument("--stats", action="store_true", help="stats + SVD per tensor")
    ap.add_argument("--raw", action="store_true", help="do not dequantize")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--json", help="write a JSON report here")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), ".gguf_cache"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    url = a.url or HF_URL.format(repo=a.repo, path=a.file)
    client = RangeClient(url, verbose=not a.quiet)
    h = fetch_header(client, a.cache_dir)

    report: dict[str, Any] = {
        "file": a.file, "file_size": client.size, "etag": client.etag,
        "gguf_version": h.version, "tensor_count": h.tensor_count,
        "kv_count": h.kv_count, "alignment": h.alignment,
        "header_bytes": h.header_bytes, "data_start": h.data_start,
    }
    print(f"gguf v{h.version}  tensors={h.tensor_count}  kv={h.kv_count}  "
          f"alignment={h.alignment}  header_bytes={h.header_bytes:,}  "
          f"data_start={h.data_start:,}  file_size={client.size:,}")

    if a.kv or a.kv_grep:
        rx = re.compile(a.kv_grep, re.I) if a.kv_grep else None
        for k, v in h.kv.items():
            if rx and not rx.search(k):
                continue
            if isinstance(v, list):
                s = f"[{len(v)} items] {v[:6]}..." if len(v) > 6 else str(v)
            else:
                s = str(v)
            if len(s) > 300:
                s = s[:300] + "..."
            print(f"  {k:52s} {h.kv_types[k]:10s} {s}")

    if a.list:
        for t in h.tensors:
            print(f"  {t.name:44s} {str(t.shape):22s} {t.ggml_type.name:8s} "
                  f"off={t.abs_offset:,} bytes={t.n_bytes:,}")
    if a.summary:
        print_summary(h)

    if a.tensor:
        rx = re.compile(a.tensor)
        matches = [t for t in h.tensors if rx.search(t.name)]
        if not matches:
            print(f"no tensor matches {a.tensor!r}", file=sys.stderr)
            return 2
        outdir = a.out or "."
        if outdir.endswith(".npy"):
            os.makedirs(os.path.dirname(outdir) or ".", exist_ok=True)
        else:
            os.makedirs(outdir, exist_ok=True)
        results = []
        for t in matches:
            arr, got, dt = fetch_tensor(client, t, raw=a.raw, dtype=a.dtype)
            path = outdir if outdir.endswith(".npy") else os.path.join(
                outdir, t.name.replace("/", "_") + f".{t.ggml_type.name}.npy")
            np.save(path, arr)
            rec = {
                "name": t.name, "ggml_type": t.ggml_type.name,
                "gguf_shape": list(t.shape), "np_shape": list(arr.shape),
                "quant_bytes": t.n_bytes, "downloaded_bytes": got,
                "frac_of_file": got / client.size, "seconds": dt,
                "abs_offset": t.abs_offset, "out": path,
            }
            if a.stats:
                rec["stats"] = tensor_stats(arr)
            results.append(rec)
            print(json.dumps(rec, indent=2))
        report["extracted"] = results

    report["bytes_downloaded"] = client.bytes_downloaded
    report["http_requests"] = client.requests_made
    print(f"\n[total] downloaded {client.bytes_downloaded:,} B of "
          f"{client.size:,} ({100*client.bytes_downloaded/client.size:.6f}%) "
          f"in {client.requests_made} range requests")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
