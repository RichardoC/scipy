"""pilot_ckpt_compress: basis, fSRD fits, baselines, byte ledger, dloss.

Implements the frozen pre-registration in pilot_ckpt_compress.md §0-§1.
Usage: python pilot_ckpt_eval.py SEED
Writes pilot_ckpt_results_seed{SEED}.json incrementally after every stage.

Budgets: the four byte budgets are fSRD's own achieved byte totals at
rcond in {1e-1, 1e-2, 1e-3, 1e-4} (fSRD's only rank/size knob), max_depth=3.
This grid was fixed before any fit was run (the pre-registration references
"the 4 byte budgets tested" without naming the generator; see "Notes on the
design" in §4 of the .md).
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fsrd_standalone as fs
from pilot_train_run import TinyLM, CORPUS, BLOCK

DATADIR = os.path.join(HERE, "pilot_ckpt_data")
RCONDS = [1e-1, 1e-2, 1e-3, 1e-4]      # budget generator (fixed a priori)
MAX_DEPTH = 3
THETA = 1.5
SMOOTH = 0.05
DT = 1.0
N_STORE = 350
N_FIT = 175
N_HO = 174
J_EVAL = [9, 32, 55, 78, 101, 124, 147, 170]   # held-out ordinals; step=4j+2
CHUNK = 100_000                          # row chunk for P-dim linear algebra
SIG_CUT = 1e-7                           # keep sigma/sigma0 > 1e-7 (spec 0.2)


# --------------------------------------------------------------- fSRD driver
def fsrd_fit_leaves(a, rcond, max_depth=MAX_DEPTH):
    """Replicates fsrd_standalone.fsrd() but returns the internal leaves
    (full split paths are needed for fractional-index evaluation and for the
    storage ledger). Deterministic: identical to the public call."""
    work = np.asarray(a, dtype=np.float64)
    m, t = work.shape
    u1 = np.linspace(0.0, 1.0, m) if m > 1 else np.zeros(1)
    u2 = np.linspace(0.0, 1.0, t) if t > 1 else np.zeros(1)
    leaves = fs._forward_pass(work, u1, u2, DT, rcond, 1e-4, SMOOTH, THETA,
                              max_depth, False)          # oblique=False
    if len(leaves) > 1:                                   # prune=True
        leaves = fs._prune(work, leaves, DT, THETA, rcond, 1e-4)
    return leaves, u1, t


def blend_eval(leaves, u1, t, xs):
    """Evaluate the fitted regions' Vandermonde at (possibly fractional)
    column positions xs with fsrd's own fuzzy membership blend.

    At integer xs this reproduces fs._reconstruct (verified in stage B):
    a leaf covers columns t0..t1-1, its Vandermonde argument at column c is
    c - t0, membership is the product of the leaf's path sigmoids, and cells
    are renormalised by the summed membership (weight > _EPS guard).
    For fractional x the leaf covers t0 <= x < t1 and the same expressions
    are evaluated at the continuous position: strict interpolation.

    Even with oblique=False *splits*, a leaf can be non-rectangular
    (node.oblique=True): the intersection of fuzzy row- and column-split
    memberships cuts corners off the bbox above the eta cutoff. The library
    then fits on the topologically transformed block and reconstructs through
    _inverse_topological_transform (per-row linear interpolation of the dense
    Vandermonde row onto the row's in-region span, zeros outside the span,
    while the membership weight still accumulates over the whole bbox). That
    path is replicated here exactly, extended to fractional columns by
    evaluating the same per-row linear interpolant at the fractional span
    coordinate."""
    xs = np.asarray(xs, dtype=np.float64)
    m = u1.size
    accum = np.zeros((m, xs.size), dtype=complex)
    weight = np.zeros((m, xs.size))
    u2x = xs / (t - 1)
    for leaf in leaves:
        if leaf.model is None:
            continue
        m0, m1, t0, t1 = leaf.bbox
        sel = np.flatnonzero((xs >= t0) & (xs < t1))
        if sel.size == 0:
            continue
        omega, phi, b = leaf.model
        loglam = omega * DT
        if leaf.oblique:
            # library-exact: dense Vandermonde on the transformed integer
            # grid, then per-row linear interpolation at the (fractional)
            # span coordinate; zeros outside the row's in-region span.
            q = t1 - t0
            grid = fs._dmd_reconstruct(leaf.model, q, DT)  # (f, q) complex
            xp = np.linspace(0.0, 1.0, q)
            f_rows = m1 - m0
            dense = np.zeros((f_rows, sel.size), dtype=complex)
            u_rel = xs[sel] - t0
            for jr, (a0, b0) in enumerate(leaf.spans):
                if a0 < 0:
                    continue
                lj = b0 - a0 + 1
                row = grid[jr]
                if lj == 1:
                    hit = u_rel == a0
                    if hit.any():
                        dense[jr, hit] = row.mean()
                    continue
                cq = (u_rel - a0) / (b0 - a0)
                ins = (cq >= 0.0) & (cq <= 1.0)
                if ins.any():
                    dense[jr, ins] = (np.interp(cq[ins], xp, row.real)
                                      + 1j * np.interp(cq[ins], xp, row.imag))
        else:
            with np.errstate(divide='ignore', invalid='ignore',
                             over='ignore'):
                logb = np.log(b.astype(complex))
                expo = (logb[:, None]
                        + loglam[:, None] * (xs[sel] - t0)[None, :])
                expo = np.minimum(expo.real, fs._LOG_CLIP) + 1j * expo.imag
                term = np.exp(expo)
            dense = phi @ term                            # (m1-m0, nsel)
        memb = np.ones((m1 - m0, sel.size))
        u1r = u1[m0:m1]
        u2r = u2x[sel]
        for v, tau, side, _sid in leaf.path:
            om = fs._sigmoid(u1r, u2r, v, tau)
            memb = memb * (om if side == 0 else (1.0 - om))
        sub_a = accum[m0:m1]
        sub_w = weight[m0:m1]
        sub_a[:, sel] += memb * dense
        sub_w[:, sel] += memb
    weight = np.where(weight > fs._EPS, weight, 1.0)
    return (accum / weight).real


def fsrd_ledger(leaves, P):
    """Achieved bytes per spec 0.3. Realified complex pairs: a conjugate pair
    = 2 real P-vectors; a real mode = 1; an unpaired complex mode = 2
    (reported). + 4*r floats eig+amp, bbox 4*int32, path 5*4 B/level, mu P*4."""
    total = P * 4                                        # mu
    n_unpaired = 0
    seg_intervals = set()
    ranks = []
    for leaf in leaves:
        if leaf.model is None:
            continue
        omega, phi, b = leaf.model
        r = omega.size
        lam = np.exp(omega * DT)
        used = np.zeros(r, bool)
        charge = 0
        for i in range(r):
            if used[i]:
                continue
            if abs(lam[i].imag) <= 1e-10 * max(1.0, abs(lam[i])):
                charge += 1
                used[i] = True
                continue
            part = None
            for j in range(i + 1, r):
                if not used[j] and np.isclose(lam[j], np.conj(lam[i]),
                                              rtol=1e-8, atol=1e-12):
                    part = j
                    break
            if part is not None:
                used[i] = used[part] = True
                charge += 2
            else:
                used[i] = True
                charge += 2                              # unpaired: double
                n_unpaired += 1
        total += charge * P * 4                          # realified modes
        total += 4 * r * 4                               # eigenvalues + amps
        total += 16                                      # bounding box
        total += 20 * len(leaf.path)                     # split path
        seg_intervals.add((leaf.bbox[2], leaf.bbox[3]))
        ranks.append(int(r))
    return dict(bytes=int(total), n_unpaired=int(n_unpaired),
                n_segments=len(seg_intervals), ranks=ranks)


def split_kinds(leaves):
    """(n_temporal, n_row) over the unique splits present in the tree."""
    seen = {}
    for leaf in leaves:
        for v, tau, side, sid in leaf.path:
            seen[sid] = v
    n_t = sum(1 for v in seen.values() if v[1] == 0.0)
    n_r = sum(1 for v in seen.values() if v[2] == 0.0)
    return n_t, n_r


def temporal_boundaries(leaves):
    """Interior column boundaries of the region column-intervals (fit units)."""
    ivs = sorted({(lf.bbox[2], lf.bbox[3]) for lf in leaves
                  if lf.model is not None})
    starts = sorted({iv[0] for iv in ivs if iv[0] > 0})
    return starts


# ------------------------------------------------------------------- basis
def build_basis(mm, P, tag, outdir=DATADIR):
    """mu, exactly-orthonormal U (fp64 memmap on disk), C = U^T A_c, checks.
    Touches ONLY even-indexed stored columns (assert: fit set = even)."""
    even = np.arange(0, N_STORE, 2)
    assert even.size == N_FIT and np.all(even % 2 == 0)
    Afit = np.asarray(mm[even], dtype=np.float32)         # (175, P)
    mu = Afit.mean(axis=0, dtype=np.float64)              # (P,)

    # Gram matrix in fp64, chunked over parameters
    G = np.zeros((N_FIT, N_FIT))
    normAc2 = 0.0
    for c0 in range(0, P, CHUNK):
        ch = Afit[:, c0:c0 + CHUNK].astype(np.float64) - mu[c0:c0 + CHUNK]
        G += ch @ ch.T
        normAc2 += float(np.sum(ch * ch))
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0.0, None)
    V = V[:, order]
    sig = np.sqrt(w)
    K = int(np.count_nonzero(sig > SIG_CUT * sig[0]))
    sig = sig[:K]
    Vk = V[:, :K]

    upath = os.path.join(outdir, f"U_{tag}.f64")
    U = np.memmap(upath, dtype=np.float64, mode="w+", shape=(P, K))
    for c0 in range(0, P, CHUNK):
        ch = Afit[:, c0:c0 + CHUNK].astype(np.float64) - mu[c0:c0 + CHUNK]
        U[c0:c0 + CHUNK] = ch.T @ (Vk / sig)
    # one Cholesky re-orthonormalisation, then exact C = U^T A_c
    W = np.zeros((K, K))
    for c0 in range(0, P, CHUNK):
        u = U[c0:c0 + CHUNK]
        W += u.T @ u
    orth_pre = float(np.linalg.norm(W - np.eye(K)))
    Linv = np.linalg.inv(np.linalg.cholesky(W))
    for c0 in range(0, P, CHUNK):
        U[c0:c0 + CHUNK] = U[c0:c0 + CHUNK] @ Linv.T
    W2 = np.zeros((K, K))
    C = np.zeros((K, N_FIT))
    resid2 = 0.0
    for c0 in range(0, P, CHUNK):
        u = U[c0:c0 + CHUNK]
        W2 += u.T @ u
        ch = Afit[:, c0:c0 + CHUNK].astype(np.float64) - mu[c0:c0 + CHUNK]
        C += u.T @ ch.T
    for c0 in range(0, P, CHUNK):
        ch = Afit[:, c0:c0 + CHUNK].astype(np.float64) - mu[c0:c0 + CHUNK]
        resid2 += float(np.sum((ch.T - U[c0:c0 + CHUNK] @ C) ** 2))
    orth_post = float(np.linalg.norm(W2 - np.eye(K)))
    svd_resid = float(np.sqrt(resid2 / normAc2))
    U.flush()
    return dict(mu=mu, upath=upath, K=K, C=C, sig=sig,
                orth_pre=orth_pre, orth_post=orth_post,
                svd_resid=svd_resid, normAc=float(np.sqrt(normAc2)))


def project_holdout(mm, P, basis):
    """c_true, out-of-span residual rho2, norms — for the 174 held-out cols."""
    odd = np.arange(1, N_STORE, 2)[:N_HO]                 # drop last (step 698)
    assert odd.size == N_HO and np.all(odd % 2 == 1)
    K = basis["K"]
    U = np.memmap(basis["upath"], dtype=np.float64, mode="r", shape=(P, K))
    mu = basis["mu"]
    Aho = np.asarray(mm[odd], dtype=np.float32)           # (174, P)
    Cho = np.zeros((K, N_HO))
    n2 = np.zeros(N_HO)                                   # ||x||^2
    nc2 = np.zeros(N_HO)                                  # ||x - mu||^2
    for c0 in range(0, P, CHUNK):
        ch = Aho[:, c0:c0 + CHUNK].astype(np.float64)
        n2 += np.sum(ch * ch, axis=1)
        ch -= mu[c0:c0 + CHUNK]
        nc2 += np.sum(ch * ch, axis=1)
        Cho += U[c0:c0 + CHUNK].T @ ch.T
    rho2 = np.clip(nc2 - np.sum(Cho * Cho, axis=0), 0.0, None)
    return dict(Cho=Cho, rho2=rho2, norm_x=np.sqrt(n2), norm_xc=np.sqrt(nc2))


# ------------------------------------------------------------- comparators
def svd_holdout(C, R):
    """Global rank-R SVD arm: held-out midpoint of stored coefficients."""
    K = C.shape[0]
    Ct = np.zeros_like(C)
    if R > 0:
        Ct[:R] = C[:R]          # C rows are already the SVD coefficients
    return 0.5 * (Ct[:, :-1] + Ct[:, 1:])                 # (K, 174)


def useg_holdout(C, m, r):
    """Uniform-boundary per-segment SVD (segment SVD in coefficient space is
    exactly the parameter-space segment SVD because U is orthonormal)."""
    K = C.shape[0]
    bounds = np.round(np.linspace(0, N_FIT, m + 1)).astype(int)
    recon = np.zeros_like(C)
    if r > 0:
        for s in range(m):
            s0, s1 = bounds[s], bounds[s + 1]
            blk = C[:, s0:s1]
            u, sv, vh = np.linalg.svd(blk, full_matrices=False)
            rr = min(r, sv.size)
            recon[:, s0:s1] = (u[:, :rr] * sv[:rr]) @ vh[:rr]
    return 0.5 * (recon[:, :-1] + recon[:, 1:]), bounds


def sub_holdout(C, n_keep):
    """Uniform checkpoint subsampling + linear interpolation (in coefficient
    space; the neglected fit-residual interpolation term is bounded by the
    reported SVD truncation residual, ~1e-7 relative)."""
    n_keep = max(1, min(n_keep, N_FIT))
    kept = np.unique(np.round(np.linspace(0, N_FIT - 1, n_keep)).astype(int))
    xs = np.arange(N_HO) + 0.5
    out = np.empty((C.shape[0], N_HO))
    for row in range(C.shape[0]):
        out[row] = np.interp(xs, kept.astype(float), C[row, kept])
    return out, kept


def arm_frob(Chat, ho):
    err2 = np.sum((Chat - ho["Cho"]) ** 2, axis=0) + ho["rho2"]
    err = np.sqrt(err2)
    return dict(frob_rel=(err / ho["norm_x"]).tolist(),
                frob_rel_c=(err / ho["norm_xc"]).tolist())


# ------------------------------------------------------------------ dloss
class ValLoss:
    """Fixed deterministic 3072-token validation loss (spec 0.4)."""

    def __init__(self):
        text = open(CORPUS, "r", encoding="utf-8").read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        val = data[-20000:]
        xs, ys = [], []
        for k in range(64):                       # 64 windows x 48 = 3072
            xs.append(val[k * BLOCK:(k + 1) * BLOCK])
            ys.append(val[k * BLOCK + 1:(k + 1) * BLOCK + 1])
        self.vx = torch.stack(xs)
        self.vy = torch.stack(ys)
        self.model = TinyLM(len(chars))
        self.n_par = sum(p.numel() for p in self.model.parameters())

    def __call__(self, vec):
        v = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
        torch.nn.utils.vector_to_parameters(v, self.model.parameters())
        with torch.no_grad():
            return float(self.model(self.vx, self.vy))


def dloss_for_arm(Chat, basis, P, vl, true_losses):
    """Chat: (K, 174) coefficient predictions; evaluate at J_EVAL."""
    K = basis["K"]
    U = np.memmap(basis["upath"], dtype=np.float64, mode="r", shape=(P, K))
    out = []
    for j in J_EVAL:
        xhat = basis["mu"] + U @ Chat[:, j]
        out.append(vl(xhat) - true_losses[j])
    return out


# ------------------------------------------------------------------- main
def save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=1)


def run_seed(seed):
    meta = json.load(open(os.path.join(DATADIR, f"ckpts_seed{seed}_meta.json")))
    P = meta["n_par"]
    mm = np.memmap(os.path.join(DATADIR, f"ckpts_seed{seed}.f32"),
                   dtype=np.float32, mode="r", shape=(N_STORE, P))
    rpath = os.path.join(HERE, f"pilot_ckpt_results_seed{seed}.json")
    results = dict(seed=seed, P=P, n_fit=N_FIT, n_ho=N_HO,
                   rconds=RCONDS, j_eval=J_EVAL,
                   naive_bytes=int(349 * P * 4),
                   naive_bytes_note="349 = 175 fit + 174 held-out columns "
                                    "represented")

    # ---- stage A: basis --------------------------------------------------
    t0 = time.time()
    basis = build_basis(mm, P, f"seed{seed}")
    ho = project_holdout(mm, P, basis)
    results["stageA"] = dict(K=basis["K"],
                             orth_pre=basis["orth_pre"],
                             orth_post=basis["orth_post"],
                             svd_trunc_resid=basis["svd_resid"],
                             sig_top5=basis["sig"][:5].tolist(),
                             sig_ratio_last=float(basis["sig"][-1] /
                                                  basis["sig"][0]),
                             wall=round(time.time() - t0, 1))
    save(results, rpath)
    print("stage A done", results["stageA"], flush=True)

    # ---- stage B: fSRD fits + blend verification -------------------------
    C = basis["C"]
    fits = {}
    stageB = {}
    for rc in RCONDS:
        t0 = time.time()
        leaves, u1, t = fsrd_fit_leaves(C, rc)
        # res.reconstruction of the model actually evaluated: the library's
        # own blend (fs._reconstruct) applied to these exact fitted regions.
        lib = fs._reconstruct(leaves, u1, np.linspace(0.0, 1.0, t), DT,
                              t, t, C.shape[0], True)
        mine = blend_eval(leaves, u1, t, np.arange(N_FIT, dtype=float))
        blend_maxdiff = float(np.max(np.abs(mine - lib)))
        scale = float(np.max(np.abs(lib)))
        assert blend_maxdiff <= 1e-9 * max(scale, 1.0), \
            f"blend re-implementation mismatch: {blend_maxdiff} vs scale {scale}"
        # diagnostic: an independent public fsrd() call (a separately fitted
        # tree can differ on near-tie splits; not a blend property)
        res_pub = fs.fsrd(C, dt=DT, max_depth=MAX_DEPTH, theta=THETA,
                          rcond=rc, smoothness=SMOOTH, oblique=False,
                          prune=True)
        pub_diff = float(np.max(np.abs(res_pub.reconstruction - lib)))
        led = fsrd_ledger(leaves, P)
        n_reg = sum(1 for lf in leaves if lf.model is not None)
        n_t, n_r = split_kinds(leaves)
        inwin = float(np.linalg.norm(lib - C) / np.linalg.norm(C))
        # gdmd at the same rcond
        gleaves, gu1, gt = fsrd_fit_leaves(C, rc, max_depth=0)
        gled = fsrd_ledger(gleaves, P)
        fits[rc] = dict(leaves=leaves, u1=u1, t=t, led=led,
                        gleaves=gleaves, gled=gled)
        stageB[str(rc)] = dict(
            n_regions=n_reg, n_regions_public_call=res_pub.n_regions,
            public_refit_recon_maxdiff=pub_diff,
            n_segments=led["n_segments"],
            ranks=led["ranks"], bytes=led["bytes"],
            ratio_vs_naive=results["naive_bytes"] / led["bytes"],
            n_unpaired_complex=led["n_unpaired"],
            splits_temporal=n_t, splits_row=n_r,
            boundaries_fit_units=temporal_boundaries(leaves),
            blend_verify_maxabsdiff=blend_maxdiff,
            recon_scale=scale, inwindow_rel_err=inwin,
            gdmd_rank=gled["ranks"], gdmd_bytes=gled["bytes"],
            wall=round(time.time() - t0, 1))
        print("stage B", rc, stageB[str(rc)], flush=True)
    results["stageB"] = stageB
    save(results, rpath)

    # ---- stage C: held-out predictions + frobenius metrics ---------------
    xs_ho = np.arange(N_HO) + 0.5
    stageC = {}
    preds = {}
    for rc in RCONDS:
        f = fits[rc]
        B = f["led"]["bytes"]
        m = f["led"]["n_segments"]
        # comparator sizing: largest that fits inside B (floor), spec 0.3
        R_svd = max(0, (B - P * 4) // (P * 4 + N_FIT * 4))
        r_useg = max(0, (B - P * 4 - 4 * max(m - 1, 0)) //
                     (m * P * 4 + N_FIT * 4))
        n_keep = max(1, B // (P * 4))

        chat_fsrd = blend_eval(f["leaves"], f["u1"], f["t"], xs_ho)
        chat_gdmd = blend_eval(f["gleaves"], f["u1"], f["t"], xs_ho)
        chat_svd = svd_holdout(C, int(R_svd))
        chat_useg, useg_bounds = useg_holdout(C, m, int(r_useg))
        chat_sub, kept = sub_holdout(C, int(n_keep))
        preds[rc] = dict(fsrd=chat_fsrd, svd=chat_svd, useg=chat_useg,
                         sub=chat_sub, gdmd=chat_gdmd)

        bytes_svd = int(P * 4 * R_svd + R_svd * N_FIT * 4 + P * 4)
        bytes_useg = int(P * 4 + m * r_useg * P * 4 + r_useg * N_FIT * 4
                         + 4 * max(m - 1, 0))
        bytes_sub = int(len(kept) * P * 4)
        stageC[str(rc)] = dict(
            budget_bytes=int(B), R_svd=int(R_svd), r_useg=int(r_useg),
            m_useg=int(m), n_keep=int(len(kept)),
            bytes=dict(fsrd=int(B), svd=bytes_svd, useg=bytes_useg,
                       sub=bytes_sub, gdmd=int(f["gled"]["bytes"])),
            frob={arm: arm_frob(preds[rc][arm], ho)
                  for arm in ("fsrd", "svd", "useg", "sub", "gdmd")})
        for arm in ("fsrd", "svd", "useg", "sub", "gdmd"):
            fr = stageC[str(rc)]["frob"][arm]
            stageC[str(rc)]["frob_summary_" + arm] = dict(
                frob_rel_mean=float(np.mean(fr["frob_rel"])),
                frob_rel_median=float(np.median(fr["frob_rel"])),
                frob_rel_c_mean=float(np.mean(fr["frob_rel_c"])),
                frob_rel_c_median=float(np.median(fr["frob_rel_c"])))
        print("stage C", rc,
              {k: v for k, v in stageC[str(rc)].items() if k != "frob"},
              flush=True)
    results["stageC"] = stageC
    save(results, rpath)

    # ---- stage D: dloss (primary metric) ----------------------------------
    vl = ValLoss()
    assert vl.n_par == P
    odd = np.arange(1, N_STORE, 2)[:N_HO]
    true_losses = {}
    for j in J_EVAL:
        true_losses[j] = vl(np.asarray(mm[odd[j]], dtype=np.float64))
    results["true_val_losses"] = {str(4 * j + 2): true_losses[j]
                                  for j in J_EVAL}
    stageD = {}
    for rc in RCONDS:
        t0 = time.time()
        entry = {}
        for arm in ("fsrd", "svd", "useg", "sub", "gdmd"):
            dl = dloss_for_arm(preds[rc][arm], basis, P, vl, true_losses)
            entry[arm] = dict(dloss=dl, mean=float(np.mean(dl)),
                              median=float(np.median(dl)))
        stageD[str(rc)] = entry
        print("stage D", rc,
              {a: round(entry[a]["mean"], 5) for a in entry},
              f"wall={time.time()-t0:.0f}s", flush=True)
        results["stageD"] = stageD
        save(results, rpath)

    # sanity: mu-only reconstruction dloss, for context (not a gate)
    mu_dl = [vl(basis["mu"]) - true_losses[j] for j in J_EVAL]
    results["mu_only_dloss_mean"] = float(np.mean(mu_dl))
    save(results, rpath)
    print("done seed", seed, flush=True)
    return results


if __name__ == "__main__":
    run_seed(int(sys.argv[1]))
