"""Positive control for pilot_ckpt_compress: run the SAME pipeline on a
synthetic trajectory of the same shape (350 stored checkpoints x P params)
built from 3 genuinely piecewise-linear regimes with KNOWN switch points.

Each regime rotates/decays a 12-dim latent inside its OWN 12-dim subspace of
a 36-dim ambient latent (subspaces disjoint), so the global rank is 36 while
each regime is rank 12: exactly the structure fSRD claims to exploit.
Regime switches at stored positions 111 and 236 (fit-grid 55.5 and 118.0 —
generic locations, not on fSRD's deterministic split lattice).

If the protocol cannot detect this planted structure, a null on the real
data means nothing. dloss does not apply (synthetic params are not a
network); the control is judged on region recovery + frobenius metrics.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fsrd_standalone as fs
from pilot_ckpt_eval import (build_basis, project_holdout, fsrd_fit_leaves,
                             blend_eval, fsrd_ledger, split_kinds,
                             temporal_boundaries, svd_holdout, useg_holdout,
                             sub_holdout, arm_frob, DATADIR, RCONDS, N_STORE,
                             N_FIT, N_HO)

P = 470_000                      # same order as the real run (~4.7e5)
K_LAT = 12                       # latent dims per regime
SWITCHES = [111, 236]            # stored positions (fit-grid 55.5, 118.0)
SEED = 777


def rot_block(theta, radius):
    c, s = np.cos(theta), np.sin(theta)
    return radius * np.array([[c, -s], [s, c]])


def make_regime_op(rng, base_theta):
    """Rotation angles are per STORED step; the fit grid sees every 2nd stored
    step, so the largest angle must satisfy 2*theta_max << pi (Nyquist on the
    fit grid), or no method -- and no fractional-power interpolation -- is
    well-posed. Max here: 2 * 0.11 * 4.5 ~ 0.99 rad/fit-step."""
    blocks = []
    for i in range(K_LAT // 2):
        theta = base_theta * (1.0 + 0.7 * i) + rng.uniform(-0.005, 0.005)
        radius = rng.uniform(0.9975, 1.002)
        blocks.append(rot_block(theta, radius))
    A = np.zeros((K_LAT, K_LAT))
    for i, b in enumerate(blocks):
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = b
    return A


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATADIR, exist_ok=True)

    # disjoint 12-dim subspaces of a 36-dim ambient latent, mapped to P dims
    Wfull = np.linalg.qr(rng.standard_normal((P, 3 * K_LAT)))[0]  # (P, 36)
    mu_s = 0.05 * rng.standard_normal(P)

    ops = [make_regime_op(rng, bt) for bt in (0.06, 0.11, 0.04)]
    bounds = [0] + SWITCHES + [N_STORE]

    Z = np.zeros((N_STORE, 3 * K_LAT))
    z = np.zeros(3 * K_LAT)
    z[:K_LAT] = rng.standard_normal(K_LAT)
    z /= np.linalg.norm(z) / 3.0
    for reg in range(3):
        lo, hi = bounds[reg], bounds[reg + 1]
        sl = slice(reg * K_LAT, (reg + 1) * K_LAT)
        if reg > 0:
            # hand over the state into the new regime's subspace (continuous
            # energy, new dynamics): copy the previous regime's latent
            prev = slice((reg - 1) * K_LAT, reg * K_LAT)
            z[sl] = z[prev]
            z[prev] = 0.0
        for t in range(lo, hi):
            Z[t] = z
            z[sl] = ops[reg] @ z[sl]

    mm_path = os.path.join(DATADIR, "ckpts_control.f32")
    mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(N_STORE, P))
    CH = 100_000
    for c0 in range(0, P, CH):
        mm[:, c0:c0 + CH] = (Z @ Wfull[c0:c0 + CH].T
                             + mu_s[c0:c0 + CH]).astype(np.float32)
    mm.flush()

    out = dict(P=P, switches_stored=SWITCHES,
               switches_fit_units=[s / 2 for s in SWITCHES],
               naive_bytes=int(349 * P * 4))

    basis = build_basis(mm, P, "control")
    ho = project_holdout(mm, P, basis)
    C = basis["C"]
    out["stageA"] = dict(K=basis["K"], orth_pre=basis["orth_pre"],
                         orth_post=basis["orth_post"],
                         svd_trunc_resid=basis["svd_resid"])
    print("control stage A", out["stageA"], flush=True)

    xs_ho = np.arange(N_HO) + 0.5
    per_rc = {}
    for rc in RCONDS:
        leaves, u1, t = fsrd_fit_leaves(C, rc)
        lib = fs._reconstruct(leaves, u1, np.linspace(0.0, 1.0, t), 1.0,
                              t, t, C.shape[0], True)
        mine = blend_eval(leaves, u1, t, np.arange(N_FIT, dtype=float))
        bmax = float(np.max(np.abs(mine - lib)))
        assert bmax <= 1e-9 * max(float(np.max(np.abs(lib))), 1.0)
        led = fsrd_ledger(leaves, P)
        n_reg = sum(1 for lf in leaves if lf.model is not None)
        n_t, n_r = split_kinds(leaves)
        B = led["bytes"]
        m = led["n_segments"]
        R_svd = max(0, (B - P * 4) // (P * 4 + N_FIT * 4))
        r_useg = max(0, (B - P * 4 - 4 * max(m - 1, 0)) //
                     (m * P * 4 + N_FIT * 4))
        n_keep = max(1, B // (P * 4))
        chat = dict(fsrd=blend_eval(leaves, u1, t, xs_ho),
                    svd=svd_holdout(C, int(R_svd)),
                    useg=useg_holdout(C, m, int(r_useg))[0],
                    sub=sub_holdout(C, int(n_keep))[0])
        gleaves, gu1, gt = fsrd_fit_leaves(C, rc, max_depth=0)
        chat["gdmd"] = blend_eval(gleaves, gu1, gt, xs_ho)
        summ = {}
        for arm in chat:
            fr = arm_frob(chat[arm], ho)
            summ[arm] = dict(
                frob_rel_c_mean=float(np.mean(fr["frob_rel_c"])),
                frob_rel_c_median=float(np.median(fr["frob_rel_c"])))
        per_rc[str(rc)] = dict(
            n_regions=n_reg, ranks=led["ranks"],
            bytes=B, ratio_vs_naive=out["naive_bytes"] / B,
            splits_temporal=n_t, splits_row=n_r,
            boundaries_fit_units=temporal_boundaries(leaves),
            blend_verify_maxabsdiff=bmax,
            R_svd=int(R_svd), r_useg=int(r_useg), n_keep=int(n_keep),
            frob_rel_c=summ)
        print("control", rc, per_rc[str(rc)], flush=True)
        out["per_rcond"] = per_rc
        with open(os.path.join(HERE, "pilot_ckpt_control.json"), "w") as f:
            json.dump(out, f, indent=1)

    # cleanup control artifacts
    del mm
    os.remove(mm_path)
    os.remove(basis["upath"])
    print("control done", flush=True)


if __name__ == "__main__":
    main()
