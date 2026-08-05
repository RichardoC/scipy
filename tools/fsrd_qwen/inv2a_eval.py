#!/usr/bin/env python
"""INVESTIGATION2.md section 5 (arm A), pipeline steps 3-4 + baselines B2 and
gates: per-region bit allocation at exactly-equal total bits, honest itemised
bit ledgers, and the pre-registered PASS/PARTIAL/FAIL evaluation.

Inputs: states/inv2a/<t>.tables.npz (inv2a_prep.py: exact per-(row, 32-block)
weighted SSE at every level b in {2,3,4,5,6,8} on the sorted tensor) and
results/inv2a_fsrd.json (inv2a_fsrd.py: leaf boxes + BIC region counts).

Budget: B0's total = (4 + 0.5) bits/weight (payload + Q_K scale overhead).
Every adaptive method must fit payload + scales + superblock references +
row/column permutations + region/band tables inside that budget.

Allocator (identical for B2 and fSRD, per the spec): exact per-region SSE
tables + Lagrangian bisection on lambda, followed by a greedy upgrade polish
of leftover budget. For B2 the band boundaries (K <= 8 contiguous bands of
sorted columns) are jointly optimised with the bit levels by a DP at each
lambda -- the full-strength version of "exhaustive best split + water
filling". Because the SSE tables come from actually quantizing at each level,
"predicted" MSE == measured MSE and the final numbers are exact.

Region semantics for fSRD: leaf bounding boxes (which overlap by one
row/column at fuzzy splits) are painted in (level desc, area asc) order onto
the grid; metadata charged as an ordered box list. Charged per region:
payload 32*cells*b, scales 12 bits per 32-block, fp16 superblock refs
32 bits per ceil(row-width/8) per row, box table (4*16+3 bits per region),
plus both permutations.

Output: results/inv2a_eval.json + printed per-tensor ledgers.
"""
import json
import math
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "states", "inv2a")
FSRD_IN = os.path.join(BASE, "results", "inv2a_fsrd.json")
PREP_IN = os.path.join(BASE, "results", "inv2a_prep.json")
OUT = os.path.join(BASE, "results", "inv2a_eval.json")

LEVELS = [2, 3, 4, 5, 6, 8]
BLK, SUP = 32, 8
BOX_BITS = 4 * 16 + 3          # bbox coords (4 x u16) + bit-level code
BAND_BITS = 10 + 3             # band start block (u10) + bit-level code

TARGETS = [
    "blk.0.ffn_down.weight", "blk.20.ffn_down.weight", "blk.40.ffn_down.weight",
    "blk.0.ffn_gate.weight", "blk.20.ffn_gate.weight", "blk.40.ffn_gate.weight",
    "blk.3.attn_q.weight", "blk.1.attn_qkv.weight",
]


def log(*a):
    print(*a, flush=True)


def perm_bits(n):
    return n * math.ceil(math.log2(n))


# --------------------------------------------------------------------------
# generic region allocator: exact SSE + Lagrangian bisection + greedy polish
# --------------------------------------------------------------------------
def allocate_regions(sse, payload_bits, fixed_bits, budget):
    """Choose a level per region minimising total SSE s.t. bits <= budget.

    sse: (n_reg, n_lev) exact SSE per region per level.
    payload_bits: (n_reg, n_lev) payload bits per region per level.
    fixed_bits: scalar level-independent bits (scales, supers, metadata, perms).
    Returns (choice indices, total_sse, total_bits) or None if infeasible.
    """
    n_reg, n_lev = sse.shape
    if fixed_bits + payload_bits[:, 0].sum() > budget:
        return None

    if n_reg <= 8:
        # exact: enumerate all level assignments (6^8 = 1.7M max), so the
        # fSRD number cannot be blamed on allocator suboptimality.
        import itertools
        combos = np.array(list(itertools.product(range(n_lev), repeat=n_reg)),
                          dtype=np.int8)
        tot_sse = np.zeros(len(combos))
        tot_pay = np.zeros(len(combos))
        for i in range(n_reg):
            tot_sse += sse[i, combos[:, i]]
            tot_pay += payload_bits[i, combos[:, i]]
        ok = fixed_bits + tot_pay <= budget
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)[np.argmin(tot_sse[ok])]
        ch = combos[idx].astype(int)
        return ch, float(tot_sse[idx]), float(fixed_bits + tot_pay[idx])

    def solve(lam):
        cost = sse + lam * payload_bits
        ch = cost.argmin(axis=1)
        idx = np.arange(n_reg)
        return ch, float(sse[idx, ch].sum()), float(payload_bits[idx, ch].sum())

    lo, hi = 0.0, 1.0
    while True:                      # find hi with feasible solution
        ch, s, p = solve(hi)
        if fixed_bits + p <= budget:
            break
        hi *= 4.0
        if hi > 1e18:
            return None
    best = None
    for _ in range(80):              # bisect lambda
        mid = 0.5 * (lo + hi)
        ch, s, p = solve(mid)
        if fixed_bits + p <= budget:
            hi = mid
            if best is None or s < best[1]:
                best = (ch.copy(), s, p)
        else:
            lo = mid
    ch, s, p = best
    # greedy polish: spend leftover budget on the best SSE-per-bit upgrades
    improved = True
    while improved:
        improved = False
        slack = budget - fixed_bits - p
        gain, pick = 0.0, None
        for i in range(n_reg):
            for l in range(ch[i] + 1, len(LEVELS)):
                db = payload_bits[i, l] - payload_bits[i, ch[i]]
                if db <= slack:
                    g = (sse[i, ch[i]] - sse[i, l]) / max(db, 1.0)
                    if g > gain:
                        gain, pick = g, (i, l)
        if pick:
            i, l = pick
            p += payload_bits[i, l] - payload_bits[i, ch[i]]
            s += sse[i, l] - sse[i, ch[i]]
            ch[i] = l
            improved = True
    return ch, s, fixed_bits + p


# --------------------------------------------------------------------------
# B2: joint (<=8 contiguous column bands) x (bit level) DP under the budget
# --------------------------------------------------------------------------
def b2_bands(tables, n_rows, budget, fixed_bits, max_bands=8):
    """tables: (n_lev, R, nb). Returns dict with bands, sse, bits."""
    n_lev, R, nb = tables.shape
    col = tables.sum(axis=1)                        # (n_lev, nb)
    csum = np.concatenate([np.zeros((n_lev, 1)), col.cumsum(axis=1)], axis=1)
    ii, jj = np.tril_indices(nb + 1, k=-1)          # j < i: band [j, i)
    # band SSE per level for every (j, i) pair
    band_sse = csum[:, ii] - csum[:, jj]            # (n_lev, n_pairs)
    width = (ii - jj)
    pay = (32.0 * R * width)[None, :] * np.array(LEVELS, float)[:, None]
    scales = 12.0 * R * width
    supers = 32.0 * R * np.ceil(width / SUP)
    band_fixed = scales + supers + BAND_BITS

    def solve(lam):
        # dense band cost matrix incl. fixed parts (lambda applies to ALL bits)
        dense = np.full((nb + 1, nb + 1), np.inf)
        dense[jj, ii] = (band_sse + lam * (pay + band_fixed)).min(axis=0)
        levm = np.full((nb + 1, nb + 1), -1)
        levm[jj, ii] = (band_sse + lam * (pay + band_fixed)).argmin(axis=0)
        D = np.full((max_bands + 1, nb + 1), np.inf)
        D[0, 0] = 0.0
        back = np.zeros((max_bands + 1, nb + 1), dtype=int)
        for k in range(1, max_bands + 1):
            tot = D[k - 1][:, None] + dense                     # (nb+1, nb+1)
            D[k] = tot.min(axis=0)
            back[k] = tot.argmin(axis=0)
        kbest = int(D[:, nb].argmin())
        # reconstruct
        bands = []
        pos, k = nb, kbest
        while k > 0:
            prev = int(back[k, pos])
            bands.append((prev, pos, int(levm[prev, pos])))
            pos, k = prev, k - 1
        bands.reverse()
        sse_tot = sum(float(csum[l, i] - csum[l, j]) for j, i, l in bands)
        bits_tot = sum(32.0 * R * (i - j) * LEVELS[l] + 12.0 * R * (i - j)
                       + 32.0 * R * math.ceil((i - j) / SUP) + BAND_BITS
                       for j, i, l in bands)
        return bands, sse_tot, bits_tot

    lo, hi = 0.0, 1.0
    while True:
        bands, s, p = solve(hi)
        if fixed_bits + p <= budget:
            break
        hi *= 4.0
        if hi > 1e18:
            return None
    best = None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        bands, s, p = solve(mid)
        if fixed_bits + p <= budget:
            hi = mid
            if best is None or s < best[1]:
                best = (bands, s, p)
        else:
            lo = mid
    bands, s, p = best
    return bands, s, fixed_bits + p


# --------------------------------------------------------------------------
# fSRD: paint leaf boxes to a hard partition, then allocate
# --------------------------------------------------------------------------
def paint_regions(boxes, R, C):
    """Ordered-box painter -> assignment map + per-region cell/super counts."""
    order = sorted(range(len(boxes)),
                   key=lambda i: (-boxes[i][4],
                                  (boxes[i][1] - boxes[i][0])
                                  * (boxes[i][3] - boxes[i][2])))
    assign = np.full((R, C), -1, dtype=np.int16)
    for i in order:
        r0, r1, c0, c1, _ = boxes[i]
        sl = assign[r0:r1, c0:c1]
        sl[sl == -1] = i
    if (assign == -1).any():
        # fallback: nearest box centre (should not happen; leaves cover the grid)
        ys, xs = np.nonzero(assign == -1)
        for y, x in zip(ys, xs):
            d = [((y - (b[0] + b[1]) / 2) ** 2 + (x - (b[2] + b[3]) / 2) ** 2)
                 for b in boxes]
            assign[y, x] = int(np.argmin(d))
    return assign


def fsrd_alloc(tables, boxes, budget, fixed_meta_bits):
    n_lev, R, nb = tables.shape
    assign = paint_regions(boxes, R, nb)
    n_reg = len(boxes)
    flat = assign.ravel().astype(np.int64)
    sse = np.stack([np.bincount(flat, weights=tables[l].ravel(),
                                minlength=n_reg) for l in range(n_lev)], axis=1)
    cells = np.bincount(flat, minlength=n_reg).astype(np.float64)
    # per-region fp16 superblock references: sum over rows of ceil(width/8)
    supers = np.zeros(n_reg)
    for i in range(n_reg):
        wpr = (assign == i).sum(axis=1)          # width per row
        supers[i] = np.ceil(wpr / SUP).sum()
    payload = 32.0 * cells[:, None] * np.array(LEVELS, float)[None, :]
    fixed = (12.0 * cells.sum()                  # 6+6-bit scale/min per block
             + 32.0 * supers.sum()               # fp16 refs
             + n_reg * BOX_BITS + fixed_meta_bits)
    res = allocate_regions(sse, payload, fixed, budget)
    if res is None:
        return None
    ch, s, bits = res
    ledger = {
        "n_regions": n_reg,
        "payload_bits": float(payload[np.arange(n_reg), ch].sum()),
        "scale_bits": float(12.0 * cells.sum()),
        "super_bits": float(32.0 * supers.sum()),
        "region_table_bits": n_reg * BOX_BITS,
        "perm_bits": fixed_meta_bits,
        "total_bits": float(bits),
        "levels": [LEVELS[c] for c in ch],
        "region_cells": cells.tolist(),
    }
    return s, bits, ledger


def main():
    fsrd_res = json.load(open(FSRD_IN))
    prep = json.load(open(PREP_IN))
    out = {"tensors": {}}
    for name in TARGETS:
        npz = np.load(os.path.join(DIR, name + ".tables.npz"))
        tables = npz["tables"].astype(np.float64)      # (n_lev, R, nb)
        denom = float(npz["denom"])
        n_lev, R, nb = tables.shape
        C = nb * BLK
        n = R * C
        budget = 4.5 * n                                # B0 total bits
        rp_bits, cp_bits = perm_bits(R), perm_bits(C)

        ent = {"shape": [R, C], "budget_bits": budget,
               "budget_bpw": 4.5,
               "b0": {"rel_rmse": prep[name]["b0_rel_rmse_weighted"],
                      "rel_frob_unweighted": prep[name]["b0_rel_frob_unweighted"],
                      "total_bits": budget,
                      "ledger": {"payload_bits": 4 * n,
                                 "scale_bits": int(0.375 * n),
                                 "super_bits": int(0.125 * n)}},
               "b1": {"rel_rmse": prep[name]["b1_rel_rmse_weighted"],
                      "total_bits": budget + cp_bits,
                      "note": "diagnostic only; exceeds the budget by the "
                              "column-perm bits (row sort is error-neutral "
                              "for a within-row block codec and need not be "
                              "stored)"}}

        # ---- B2 ----
        bands, s2, bits2 = b2_bands(tables, R, budget,
                                    fixed_bits=cp_bits + 4)
        ent["b2"] = {
            "rel_rmse": float(np.sqrt(s2 / denom)),
            "total_bits": float(bits2), "bpw": float(bits2 / n),
            "bands": [[int(j), int(i), LEVELS[l]] for j, i, l in bands],
            "ledger": {"col_perm_bits": cp_bits, "band_count_bits": 4,
                       "band_table_bits": len(bands) * BAND_BITS},
        }

        # ---- fSRD: all 8 sweep configs ----
        ent["fsrd"] = {}
        best = None
        for cfg, rec in fsrd_res[name].items():
            r = fsrd_alloc(tables, rec["boxes_asc"], budget,
                           fixed_meta_bits=rp_bits + cp_bits)
            if r is None:
                ent["fsrd"][cfg] = {"infeasible": True}
                continue
            s, bits, ledger = r
            e = {"rel_rmse": float(np.sqrt(s / denom)),
                 "total_bits": float(bits), "bpw": float(bits / n),
                 "n_regions": rec["n_regions"], "ledger": ledger}
            ent["fsrd"][cfg] = e
            if best is None or e["rel_rmse"] < best[1]["rel_rmse"]:
                best = (cfg, e)
        ent["fsrd_best_cfg"] = best[0]
        ent["fsrd_best"] = best[1]

        # ---- margins ----
        f = best[1]["rel_rmse"]
        ent["margin_vs_b0"] = 1.0 - f / ent["b0"]["rel_rmse"]
        ent["margin_vs_b2"] = 1.0 - f / ent["b2"]["rel_rmse"]
        ent["b2_margin_vs_b0"] = 1.0 - ent["b2"]["rel_rmse"] / ent["b0"]["rel_rmse"]
        out["tensors"][name] = ent
        log(f"{name}: B0={ent['b0']['rel_rmse']:.5f} B1={ent['b1']['rel_rmse']:.5f} "
            f"B2={ent['b2']['rel_rmse']:.5f} "
            f"fSRD[{best[0]}, {best[1]['n_regions']}reg]={f:.5f} "
            f"| vsB0 {100*ent['margin_vs_b0']:+.1f}% vsB2 {100*ent['margin_vs_b2']:+.1f}% "
            f"(B2 vs B0 {100*ent['b2_margin_vs_b0']:+.1f}%)")

    # ---- gates ----
    ts = out["tensors"]
    wins = sum(1 for t in ts.values()
               if t["margin_vs_b0"] >= 0.10 and t["margin_vs_b2"] >= 0.05)
    partial_wins = sum(1 for t in ts.values() if t["margin_vs_b0"] >= 0.10)
    one_region = sum(1 for t in ts.values()
                     if t["fsrd_best"]["n_regions"] == 1)
    one_region_all_cfg = sum(
        1 for name in TARGETS
        if all(v.get("n_regions", 1) == 1
               for v in json.load(open(FSRD_IN))[name].values()))
    gates = {
        "pass_wins_over_b0_and_b2": wins,
        "b0_margin>=10pct_count": partial_wins,
        "bic_1region_bestcfg_count": one_region,
        "bic_1region_allcfg_count": one_region_all_cfg,
        "median_margin_vs_b0": float(np.median([t["margin_vs_b0"] for t in ts.values()])),
        "median_margin_vs_b2": float(np.median([t["margin_vs_b2"] for t in ts.values()])),
        "median_b2_margin_vs_b0": float(np.median([t["b2_margin_vs_b0"] for t in ts.values()])),
    }
    if one_region_all_cfg >= 4:
        verdict = "AUTO-FAIL (BIC returned 1 region on >=4 of 8 tensors)"
    elif wins >= 6:
        verdict = "PASS"
    elif partial_wins >= 6:
        verdict = "PARTIAL: adaptive allocation works; fSRD is ceremony"
    else:
        verdict = "FAIL"
    gates["verdict"] = verdict
    out["gates"] = gates
    log(json.dumps(gates, indent=1))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
