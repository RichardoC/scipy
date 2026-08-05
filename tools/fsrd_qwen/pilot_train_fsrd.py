"""PILOT analysis: fSRD on training-dynamics snapshot matrices.

T3 degeneracy check, T1 forecasting (vs persistence / linear / global DMD),
T2 temporal-boundary phase detection (vs loss & grad-norm changepoints).

All thresholds pre-registered in pilot_train_dynamics.md before this ran.
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsrd_standalone import fsrd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "pilot_train_data")
SEEDS = [0, 1, 2, 3, 4]
FIT_POINTS = [150, 250, 350, 450, 550]
HORIZONS = [10, 40]
KNOWN_EVENTS = [40, 350]
TOL = 25
NOVELTY = 40


def load(seed):
    d = np.load(os.path.join(DATA, f"run_seed{seed}.npz"))
    return {k: d[k] for k in d.files}


# ------------------------------------------------------------ forecasting
def lin_extrap(a, k, h, w=5):
    """Least-squares line through last w columns, extrapolated h steps."""
    t = np.arange(w, dtype=float)
    X = np.vstack([np.ones(w), t]).T
    Y = a[:, k - w:k]                       # (M, w)
    coef, *_ = np.linalg.lstsq(X, Y.T, rcond=None)   # (2, M)
    return coef[0] + coef[1] * (w - 1 + h)


def fsrd_forecast(a, k, h, depth):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fsrd(a[:, :k], dt=1.0, max_depth=depth, oblique=False, forecast=h)
    rec = res.reconstruction
    pred = np.real(rec[:, k + h - 1])
    inwin = np.linalg.norm(np.real(rec[:, :k]) - a[:, :k]) / np.linalg.norm(a[:, :k])
    return pred, res, inwin


def temporal_boundaries(res, n_cols):
    """Column positions of pure-time splits, in step units."""
    bnds, n_time, n_split = set(), 0, 0
    for r in res.regions:
        v = r.split_vector
        if v is None:
            continue
        n_split += 1
        if abs(v[1]) < 1e-12 and abs(v[2]) > 1e-12:
            n_time += 1
            u = -v[0] / v[2]                 # normalised column position in [0,1]
            bnds.add(int(round(u * (n_cols - 1))))
    return sorted(bnds), n_time, n_split


# ------------------------------------------------------------ changepoints
def binseg(y, n_cp, min_size=15):
    """Binary segmentation with piecewise-constant L2 cost."""
    y = np.asarray(y, float)

    def cost(i, j):
        seg = y[i:j]
        return float(((seg - seg.mean()) ** 2).sum()) if j - i > 0 else 0.0

    bounds = [0, len(y)]
    for _ in range(n_cp):
        best = None
        for bi in range(len(bounds) - 1):
            i, j = bounds[bi], bounds[bi + 1]
            base = cost(i, j)
            for s in range(i + min_size, j - min_size + 1):
                gain = base - cost(i, s) - cost(s, j)
                if best is None or gain > best[0]:
                    best = (gain, s)
        if best is None or best[0] <= 0:
            break
        bounds = sorted(bounds + [best[1]])
    return [b for b in bounds if b not in (0, len(y))]


def nearest(x, pts):
    return min((abs(x - p) for p in pts), default=10 ** 9)


def main():
    out = {"pre_registered": True, "seeds": SEEDS, "fit_points": FIT_POINTS,
           "horizons": HORIZONS}
    runs = {s: load(s) for s in SEEDS}
    out["run_meta"] = {str(s): dict(n_par=int(runs[s]["n_par"]),
                                    wall=float(runs[s]["wall"]),
                                    loss_start=float(runs[s]["loss"][:5].mean()),
                                    loss_end=float(runs[s]["loss"][-20:].mean()),
                                    val=runs[s]["val"].tolist())
                       for s in SEEDS}

    # ---------------- T1 + T3 --------------------------------------------
    rows = []
    t0 = time.time()
    for s in SEEDS:
        a = runs[s]["a"]
        for k in FIT_POINTS:
            for h in HORIZONS:
                truth = a[:, k + h - 1]
                pers = a[:, k - 1]
                lin = lin_extrap(a, k, h)
                rec = {}
                for tag, depth in (("fsrd", 3), ("gdmd", 0)):
                    p, res, inwin = fsrd_forecast(a, k, h, depth)
                    bnds, nt, ns = temporal_boundaries(res, k)
                    rec[tag] = dict(err=float(np.linalg.norm(p - truth)),
                                    n_regions=int(res.n_regions),
                                    inwin=float(inwin), n_time=nt, n_split=ns,
                                    bnds=bnds, bic=float(res.bic))
                den = np.linalg.norm(truth)
                rows.append(dict(seed=s, k=k, h=h,
                                 norm_truth=float(den),
                                 err_pers=float(np.linalg.norm(pers - truth)),
                                 err_lin=float(np.linalg.norm(lin - truth)),
                                 err_fsrd=rec["fsrd"]["err"],
                                 err_gdmd=rec["gdmd"]["err"],
                                 fsrd=rec["fsrd"], gdmd=rec["gdmd"]))
                print(f"seed{s} k={k} h={h}: pers={rows[-1]['err_pers']:.4g} "
                      f"lin={rows[-1]['err_lin']:.4g} "
                      f"gdmd={rows[-1]['err_gdmd']:.4g} "
                      f"fsrd={rows[-1]['err_fsrd']:.4g} "
                      f"nreg={rec['fsrd']['n_regions']} "
                      f"ntime={rec['fsrd']['n_time']}/{rec['fsrd']['n_split']} "
                      f"inwin={rec['fsrd']['inwin']:.3g}", flush=True)
    out["t1_rows"] = rows
    out["t1_wall"] = round(time.time() - t0, 1)

    def med(key, h):
        return float(np.median([r[key] for r in rows if r["h"] == h]))

    out["t1_summary"] = {}
    for h in HORIZONS:
        sel = [r for r in rows if r["h"] == h]
        ratio = float(np.median([r["err_fsrd"] / r["err_gdmd"] for r in sel]))
        out["t1_summary"][str(h)] = dict(
            med_err_pers=med("err_pers", h), med_err_lin=med("err_lin", h),
            med_err_gdmd=med("err_gdmd", h), med_err_fsrd=med("err_fsrd", h),
            median_ratio_fsrd_over_gdmd=ratio,
            skill_vs_pers_fsrd=float(np.median(
                [1 - r["err_fsrd"] / r["err_pers"] for r in sel])),
            skill_vs_pers_gdmd=float(np.median(
                [1 - r["err_gdmd"] / r["err_pers"] for r in sel])),
            skill_vs_pers_lin=float(np.median(
                [1 - r["err_lin"] / r["err_pers"] for r in sel])),
            n_fsrd_best=int(sum(1 for r in sel if r["err_fsrd"] ==
                                min(r["err_pers"], r["err_lin"],
                                    r["err_gdmd"], r["err_fsrd"]))),
            n=len(sel),
            gate_pass=bool(ratio <= 0.80 and
                           med("err_fsrd", h) < med("err_pers", h) and
                           med("err_fsrd", h) < med("err_lin", h)),
            verdict_regions_add_nothing=bool(0.80 < ratio < 1.25),
        )

    # T3 degeneracy
    nreg = [r["fsrd"]["n_regions"] for r in rows]
    ntime = [r["fsrd"]["n_time"] for r in rows]
    nsplit = [r["fsrd"]["n_split"] for r in rows]
    out["t3"] = dict(n_fits=len(nreg),
                     frac_single_region=float(np.mean([n == 1 for n in nreg])),
                     region_counts=sorted(set(nreg)),
                     mean_n_regions=float(np.mean(nreg)),
                     total_splits=int(sum(nsplit)),
                     total_time_splits=int(sum(ntime)),
                     frac_time_splits=(float(sum(ntime) / sum(nsplit))
                                       if sum(nsplit) else None),
                     hard_autofail=bool(np.mean([n == 1 for n in nreg]) > 0.5),
                     soft_autofail=bool(sum(ntime) == 0))

    # ---------------- T2 full-window fits --------------------------------
    t2 = {}
    for s in SEEDS:
        a = runs[s]["a"]
        T = a.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fsrd(a, dt=1.0, max_depth=3, oblique=False)
        bnds, nt, ns = temporal_boundaries(res, T)
        n_cp = max(1, len(bnds))
        loss_cp = binseg(runs[s]["loss"], n_cp)
        gn_cp = binseg(runs[s]["gnorm"], n_cp)
        # also on smoothed loss (the visible 'knee' structure)
        sm = np.convolve(runs[s]["loss"], np.ones(11) / 11, mode="same")
        loss_cp_sm = binseg(sm, n_cp)
        t2[str(s)] = dict(n_regions=int(res.n_regions), boundaries=bnds,
                          n_time_splits=nt, n_splits=ns,
                          inwin=float(np.linalg.norm(
                              np.real(res.reconstruction) - a) /
                              np.linalg.norm(a)),
                          loss_cp=loss_cp, loss_cp_smooth=loss_cp_sm,
                          gnorm_cp=gn_cp)
        print(f"[T2] seed{s}: nreg={res.n_regions} time-bnds={bnds} "
              f"loss_cp={loss_cp} loss_cp_sm={loss_cp_sm} gn_cp={gn_cp} "
              f"inwin={t2[str(s)]['inwin']:.3g}", flush=True)
    out["t2_per_seed"] = t2

    # consistency: cluster boundaries across seeds within +-TOL
    allb = sorted((b, s) for s in SEEDS for b in t2[str(s)]["boundaries"])
    clusters = []
    for b, s in allb:
        placed = False
        for c in clusters:
            if abs(b - np.mean(c["vals"])) <= TOL:
                c["vals"].append(b); c["seeds"].add(s); placed = True; break
        if not placed:
            clusters.append(dict(vals=[b], seeds={s}))
    cl = []
    for c in clusters:
        loc = float(np.mean(c["vals"]))
        d_loss = min(nearest(loc, t2[str(s)]["loss_cp"]) for s in SEEDS)
        d_loss_sm = min(nearest(loc, t2[str(s)]["loss_cp_smooth"]) for s in SEEDS)
        d_gn = min(nearest(loc, t2[str(s)]["gnorm_cp"]) for s in SEEDS)
        cl.append(dict(loc=loc, n_seeds=len(c["seeds"]),
                       seeds=sorted(c["seeds"]), vals=sorted(c["vals"]),
                       dist_nearest_loss_cp=d_loss,
                       dist_nearest_loss_cp_smooth=d_loss_sm,
                       dist_nearest_gnorm_cp=d_gn,
                       dist_nearest_known_event=nearest(loc, KNOWN_EVENTS)))
    cl.sort(key=lambda c: -c["n_seeds"])
    out["t2_clusters"] = cl
    consistent = [c for c in cl if c["n_seeds"] >= 4]
    novel = [c for c in consistent
             if min(c["dist_nearest_loss_cp"], c["dist_nearest_loss_cp_smooth"],
                    c["dist_nearest_gnorm_cp"]) > NOVELTY]
    grounded = [c for c in consistent if c["dist_nearest_known_event"] <= TOL]
    out["t2_summary"] = dict(
        n_clusters=len(cl), n_consistent_4of5=len(consistent),
        consistent_locs=[c["loc"] for c in consistent],
        n_novel=len(novel), novel_locs=[c["loc"] for c in novel],
        n_grounded=len(grounded), grounded_locs=[c["loc"] for c in grounded],
        gate_a_consistency=bool(len(consistent) >= 1),
        gate_b_novelty=bool(len(novel) >= 1),
        gate_c_grounded=bool(len(grounded) >= 1),
        gate_pass=bool(len(consistent) >= 1 and len(novel) >= 1),
    )

    out["verdict"] = dict(
        t3_autofail=out["t3"]["hard_autofail"] or out["t3"]["soft_autofail"],
        t1_pass=out["t1_summary"]["10"]["gate_pass"],
        t2_pass=out["t2_summary"]["gate_pass"],
    )
    out["verdict"]["GO"] = bool(
        not out["verdict"]["t3_autofail"] and
        (out["verdict"]["t1_pass"] or out["verdict"]["t2_pass"]))

    with open(os.path.join(HERE, "pilot_train_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in
                      ("t1_summary", "t3", "t2_summary", "verdict")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
