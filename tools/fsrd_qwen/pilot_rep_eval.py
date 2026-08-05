#!/usr/bin/env python
"""PILOT step 3: score detectors against independent ground truth, per pre-registration."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_raw():
    files = sorted(glob.glob(os.path.join(HERE, "pilot_rep_raw_*.json")))
    if not files:
        raise SystemExit("no pilot_rep_raw_*.json shards found")
    merged, cost = None, {}
    for f in files:
        d = json.load(open(f))
        if merged is None:
            merged = dict(W=d["W"], K=d["K"], layers=d["layers"], recs=[], cost={})
        merged["recs"].extend(d["recs"])
        for k, v in d["cost"].items():
            c = cost.setdefault(k, [0.0, 0])
            c[0] += v["median_ms"] * v["n"]
            c[1] += v["n"]
    merged["cost"] = {k: dict(median_ms=v[0] / max(1, v[1]), n=v[1])
                      for k, v in cost.items()}
    merged["recs"].sort(key=lambda r: r["gi"])
    merged["shards"] = [os.path.basename(f) for f in files]
    return merged


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    rp = r[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    d = load_raw()
    W, recs = d["W"], d["recs"]
    layers = [str(l) for l in d["layers"]]
    report = {"W": W, "K": d["K"], "cost_ms": d["cost"], "layers": {}}

    npos = sum(r["label"] == "positive" for r in recs)
    report["n_generations"] = len(recs)
    report["n_positive"] = npos
    report["n_negative"] = len(recs) - npos
    report["n_positive_onset0"] = sum(r["label"] == "positive" and r["onset"] == 0
                                      for r in recs)
    report["n_positive_usable_for_lead"] = sum(
        r["label"] == "positive" and r["onset"] >= W for r in recs)
    report["shards"] = d["shards"]
    report["onsets"] = {r["gi"]: r["onset"] for r in recs if r["label"] == "positive"}
    report["periods"] = {r["gi"]: r["period"] for r in recs if r["label"] == "positive"}

    for lay in layers:
        # assemble per-detector score series on the common window index range
        dets = {}
        def add(name, fn):
            dets[name] = {r["gi"]: fn(r) for r in recs}

        add("fsrd", lambda r: [r["layers"][lay]["fsrd"].get(str(t), np.nan)
                               for t in range(r["T"])])
        add("global_dmd", lambda r: [r["layers"][lay]["gdmd"].get(str(t), np.nan)
                                     for t in range(r["T"])])
        add("cos_raw", lambda r: r["layers"][lay]["cos_raw"])
        add("cos_std", lambda r: r["layers"][lay]["cos_std"])
        add("ngram", lambda r: r["ngram"])
        add("neg_entropy", lambda r: [-x for x in r["entropy"]])
        add("maxprob", lambda r: r["maxprob"])

        out = {}
        for name, series in dets.items():
            pos, neg, pos_late = [], [], []
            for r in recs:
                s = np.asarray(series[r["gi"]], float)
                idx = np.arange(W - 1, r["T"])
                v = s[idx]
                ok = np.isfinite(v)
                if r["label"] == "negative":
                    neg.extend(v[ok].tolist())
                else:
                    m = ok & (idx >= r["onset"])
                    pos.extend(v[m].tolist())
                    if r["onset"] >= W:
                        pos_late.extend(v[m].tolist())
            A = auc(pos, neg)
            A_late = auc(pos_late, neg)
            thr = float(np.percentile(neg, 95)) if neg else float("nan")
            leads, alarms = [], {}
            for r in recs:
                if r["label"] != "positive":
                    continue
                s = np.asarray(series[r["gi"]], float)
                idx = np.arange(W - 1, r["T"])
                v = s[idx]
                hit = np.where(np.isfinite(v) & (v > thr))[0]
                if hit.size:
                    ta = int(idx[hit[0]])
                    leads.append(r["onset"] - ta)
                    alarms[r["gi"]] = ta
                else:
                    leads.append(None)
                    alarms[r["gi"]] = None
            fired = [l for l in leads if l is not None]
            # (b) LEAD TIME: only positives with a genuine pre-onset window (onset >= W).
            # onset==0 samples are excluded entirely -- no pre-onset window exists there.
            late = [r for r in recs if r["label"] == "positive" and r["onset"] >= W]
            leads_late, det_late = [], 0
            for r in late:
                ta = alarms.get(r["gi"])
                if ta is not None:
                    leads_late.append(r["onset"] - ta)
                    det_late += 1
            # FPR of the calibrated threshold on negatives, per generation
            gen_fp = 0
            for r in recs:
                if r["label"] != "negative":
                    continue
                s = np.asarray(series[r["gi"]], float)
                v = s[W - 1:r["T"]]
                if np.nansum(v > thr) > 0:
                    gen_fp += 1
            out[name] = dict(
                auc=A, auc_usable_pos_only=A_late, thr=thr,
                n_pos_win=len(pos), n_pos_win_late=len(pos_late), n_neg_win=len(neg),
                detect_rate=len(fired) / max(1, len(leads)),
                median_lead=float(np.median(fired)) if fired else None,
                mean_lead=float(np.mean(fired)) if fired else None,
                leads=leads,
                n_late_onset=len(late),
                detect_rate_late=det_late / max(1, len(late)),
                leads_late=leads_late,
                mean_lead_late=float(np.mean(leads_late)) if leads_late else None,
                median_lead_late=float(np.median(leads_late)) if leads_late else None,
                gen_level_fp_rate=gen_fp / max(1, report["n_negative"]),
            )

        # degeneracy check + region boundary alignment
        nreg_all, nreg1 = 0, 0
        for r in recs:
            for v in r["layers"][lay]["nreg"].values():
                nreg_all += 1
                nreg1 += (v == 1)
        bhit, btot, bdens = 0, 0, []
        for r in recs:
            if r["label"] != "positive":
                continue
            on = r["onset"]
            for tstr, cuts in r["layers"][lay]["bounds"].items():
                t = int(tstr)
                lo = t - W + 1
                if not (lo <= on <= t):
                    continue
                btot += 1
                bdens.append(len(cuts))
                if any(abs(c - on) <= 4 for c in cuts):
                    bhit += 1
        chance = min(1.0, (9.0 / W) * float(np.mean(bdens))) if bdens else float("nan")
        report["layers"][lay] = dict(
            detectors=out,
            frac_windows_1region=nreg1 / max(1, nreg_all),
            n_windows=nreg_all,
            boundary_at_onset_rate=bhit / btot if btot else None,
            boundary_chance_rate=chance,
            mean_boundaries_per_window=float(np.mean(bdens)) if bdens else None,
        )

    json.dump(report, open(os.path.join(HERE, "pilot_repetition.json"), "w"), indent=1)

    # console summary
    print(f"generations={report['n_generations']} positives={report['n_positive']} "
          f"(onset==0: {report['n_positive_onset0']}, usable for lead (onset>={W}): "
          f"{report['n_positive_usable_for_lead']}) negatives={report['n_negative']}")
    print("onsets:", report["onsets"])
    print("cost (ms/window):", {k: round(v["median_ms"], 1)
                                for k, v in d["cost"].items()})
    for lay, L in report["layers"].items():
        print(f"\n=== layer {lay}  (1-region windows: {L['frac_windows_1region']:.1%} "
              f"of {L['n_windows']}) ===")
        print(f"{'detector':13s} {'AUC_a':>6s} {'AUCusb':>7s} {'det%_b':>7s} "
              f"{'medLead_b':>10s} {'meanLead_b':>11s} {'genFPR':>7s}")
        for k, v in L["detectors"].items():
            print(f"{k:13s} {v['auc']:6.3f} {v['auc_usable_pos_only']:7.3f} "
                  f"{v['detect_rate_late']*100:6.0f}% "
                  f"{str(v['median_lead_late']):>10s} "
                  f"{str(round(v['mean_lead_late'],1) if v['mean_lead_late'] is not None else None):>11s} "
                  f"{v['gen_level_fp_rate']:7.2f}")
        print(f"region boundary within +-4 of onset: {L['boundary_at_onset_rate']} "
              f"(chance {L['boundary_chance_rate']}, "
              f"{L['mean_boundaries_per_window']} cuts/window)")


if __name__ == "__main__":
    sys.exit(main())
