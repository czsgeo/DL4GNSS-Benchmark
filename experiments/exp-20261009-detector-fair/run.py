#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp-20261009-detector-fair -- review M3.

v1.0 compared a 10-SE classical detector against a 1.5-sigma learned peak
picker and reported a single F1 per amplitude; its large-amplitude verdict was
then decided by a post-hoc seed exclusion. This experiment removes both
problems:

  * both detectors are reduced to a decision FIELD (classical: the two-sample
    step statistic on the robust residual; learned: the network output) and read
    out with the same threshold + min-separation rule;
  * sweeping the threshold traces a full precision-recall curve per method, so we
    can report average precision and compare at matched recall as well as at each
    method's own F1-optimal threshold;
  * the learned threshold is chosen on a held-out validation fold and the cost of
    that selection step is reported as part of the method;
  * the detector's training-seed distribution is estimated over 20 seeds.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench  # noqa: E402
import dl  # noqa: E402
import rev  # noqa: E402

DT = bench.DT


def load_stations(cfg):
    npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
    split = json.load(open(os.path.join(ROOT, cfg["split_file"]), encoding="utf-8"))
    want = set(split[cfg["use_split"]])
    sid = sorted(str(s) for s in npz["sites"] if str(s) in want)
    k = int(cfg["max_stations"])
    if len(sid) > k:
        idx = np.linspace(0, len(sid) - 1, k).round().astype(int)
        sid = [sid[i] for i in sorted(set(idx))]
    return npz, sid


def f1_at(P, R, F, ths, mode="f1opt"):
    """Return (threshold, precision, recall, f1) at an operating point."""
    P, R, F = np.asarray(P), np.asarray(R), np.asarray(F)
    if mode == "f1opt":
        j = int(np.argmax(F))
    else:
        j = int(np.argmin(np.abs(R - float(mode))))
    return float(ths[j]), float(P[j]), float(R[j]), float(F[j])


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    out_dir = os.environ.get("PAPER_OUT", HERE)
    import torch
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ndir = os.path.join(ROOT, cfg["nets_dir"])
    nets = rev.load_nets(ndir, "unet_detect_in_s*.pt", device)
    seeds = sorted(s for s, _ in nets.values())
    print(f"device {device} detect nets {len(nets)}", flush=True)

    npz, sids = load_stations(cfg)
    noise = cfg["noise"]
    base_seed = int(cfg["seed"])
    ths = np.asarray(cfg["thresholds"], float)
    tol_years = cfg["detect_tol"] = cfg["tol_days"] / 365.25
    msep = cfg["min_sep_days"]

    # per (A) accumulators: list over realisations of per-method curve + point
    #   cl_curves[ A ] : list of (P,R,F) tuples for the classical detector
    #   dl_curves[ A ][ seed ] : list of (P,R,F)
    cl_curves, dl_curves = {}, {}
    realisations = []   # (A, true_epochs, cl_point..., ) for val-selection

    for si, sid in enumerate(sids):
        t = np.asarray(npz[f"t__{sid}"], float)
        enu = np.asarray(npz[f"enu__{sid}"], float)
        if enu.ndim == 1:
            enu = enu[:, None]
        for comp in cfg["components"]:
            if comp >= enu.shape[1]:
                continue
            p = bench.signal_params_from_real(t, enu[:, comp], seed=base_seed + si)
            nsp = bench.make_noise_spec(noise["sigma_wn_mm"], noise["sigma_fn_mm"],
                                        noise["sigma_rw_mm"], len(t), DT)
            for A in cfg["amplitudes_mm"]:
                for k in range(cfg["n_realisations"]):
                    rng = np.random.default_rng(base_seed + 3 * si + comp + int(A) + 1000 * k)
                    epc = bench.random_epochs(t, cfg["n_offsets"],
                                              min_sep_years=cfg["min_sep_years"], rng=rng)
                    p2 = dict(p)
                    p2["steps"] = [(e, float(rng.choice([-1, 1]) * A)) for e in epc]
                    S2 = bench.clean_signal(t, p2)
                    y2 = S2 + bench.inject(len(t), nsp, DT, rng)
                    clf = rev.classical_step_field(t, y2)
                    cP, cR, cF = rev.pr_curve([clf], epc, t, ths, tol_years, msep)
                    cl_curves.setdefault(A, []).append((cP, cR, cF))
                    dlf = {}
                    for stem, (_s, net) in nets.items():
                        dlf[stem] = dl.detect_field(net, y2, device=device)
                    dl_curve = {}
                    for stem, fld in dlf.items():
                        dP, dR, dF = rev.pr_curve([fld], epc, t, ths, tol_years, msep)
                        dl_curves.setdefault(A, {}).setdefault(stem, []).append((dP, dR, dF))
                        dl_curve[stem] = (dP.tolist(), dR.tolist(), dF.tolist())
                    # Store the per-realisation PR *curves*, not the raw fields,
                    # for the validation-selection sweep. `pr_curve` is
                    # deterministic, so a curve (3 x len(ths) floats) is all the
                    # sweep needs; recomputing it inside each of the
                    # val_splits x seeds loops would repeat the same work ~20x.
                    realisations.append({"A": int(A), "true": list(epc),
                                         "cl_curve": (cP.tolist(), cR.tolist(), cF.tolist()),
                                         "dl_curve": dl_curve})
        print(f"  [{si+1}/{len(sids)}] {sid}", flush=True)

    def avg_points(curves):
        P = np.mean([c[0] for c in curves], axis=0)
        R = np.mean([c[1] for c in curves], axis=0)
        F = np.mean([c[2] for c in curves], axis=0)
        return P, R, F

    results = {"exp_id": cfg["exp_id"], "n_stations": len(sids),
               "detect_seeds": seeds, "thresholds": list(ths),
               "detect": {}}

    for A in sorted(cl_curves):
        P, R, F = avg_points(cl_curves[A])
        cl = {"ap": rev.average_precision(P, R),
              "pr_curve": {"precision": P.tolist(), "recall": R.tolist(), "f1": F.tolist()}}
        cl["f1_opt"] = dict(zip(("threshold", "precision", "recall", "f1"),
                                f1_at(P, R, F, ths, "f1opt")))
        for tr in cfg["matched_recall_targets"]:
            cl[f"at_recall_{tr}"] = dict(zip(("threshold", "precision", "recall", "f1"),
                                             f1_at(P, R, F, ths, tr)))
        # learned: per-seed F1-opt, then across-seed mean of the curves
        per_seed = {}
        for stem in dl_curves[A]:
            P2, R2, F2 = avg_points(dl_curves[A][stem])
            per_seed[stem] = {"ap": rev.average_precision(P2, R2),
                              "f1_opt": dict(zip(("threshold", "precision", "recall", "f1"),
                                                 f1_at(P2, R2, F2, ths, "f1opt")))}
        # across-seed: average the per-seed F1-optimal F1 (the "as tuned" figure)
        f1s = np.array([per_seed[s]["f1_opt"]["f1"] for s in per_seed])
        aps = np.array([per_seed[s]["ap"] for s in per_seed])
        dl_all = {}
        for stem in dl_curves[A]:
            P2, R2, F2 = avg_points(dl_curves[A][stem])
            dl_all.setdefault("P", []).append(P2)
            dl_all.setdefault("R", []).append(R2)
            dl_all.setdefault("F", []).append(F2)
        Pm = np.mean(dl_all["P"], axis=0); Rm = np.mean(dl_all["R"], axis=0)
        Fm = np.mean(dl_all["F"], axis=0)
        dl_summary = {"ap_seed_mean": float(np.mean(aps)), "ap_seed_std": float(np.std(aps, ddof=1)),
              "ap_of_seed_mean_curve": rev.average_precision(Pm, Rm),
              "f1_opt_per_seed_mean": float(np.mean(f1s)),
              "f1_opt_per_seed_std": float(np.std(f1s, ddof=1)),
              "f1_opt_per_seed_min": float(np.min(f1s)),
              "f1_opt_per_seed_max": float(np.max(f1s)),
              "n_seeds": len(f1s),
              "seed_mean_pr_curve": {"precision": Pm.tolist(), "recall": Rm.tolist(),
                                     "f1": Fm.tolist()},
              "per_seed": per_seed}
        results["detect"][str(int(A))] = {"classical": cl, "learned": dl_summary}

    # ---------------- validation-threshold selection (cost of model selection)
    # For each (A) we simulate: split realisations into val/test. Choose the
    # threshold maximising F1 on val (per seed), report the test F1 at that
    # threshold. Compared against the oracle F1-optimal and the fixed v1.0 1.5.
    rng = np.random.default_rng(int(cfg["seed"]) + 99)
    valres = {}
    by_A = {}
    for r in realisations:
        by_A.setdefault(r["A"], []).append(r)

    def _avg(rs, key, idx, method=None):
        """Mean P/R/F curve over the realisations indexed by `idx`."""
        P = np.zeros(len(ths)); R = np.zeros(len(ths)); F = np.zeros(len(ths))
        for i in idx:
            c = rs[i][key] if method is None else rs[i][key][method]
            P += c[0]; R += c[1]; F += c[2]
        return P / len(idx), R / len(idx), F / len(idx)

    for A, rs in by_A.items():
        n = len(rs)
        if n < 4:
            continue
        nval = max(2, n // 2)
        # The learned detector selects its own threshold per training seed on the
        # validation split; the classical detector is a single curve. They are
        # reported separately -- pooling them (as an earlier draft did) would
        # conflate a 20-seed ensemble with a single method in one mean.
        sel = {"learned": {"valsel": [], "oracle": []},
               "classical": {"valsel": [], "oracle": []}}
        for split in range(int(cfg["val_splits"])):
            perm = rng.permutation(n)
            vidx, tidx = perm[:nval], perm[nval:]
            for stem in rs[0]["dl_curve"]:
                _, _, Fv = _avg(rs, "dl_curve", vidx, stem)
                th_star = float(ths[int(np.argmax(Fv))])
                _, _, Ft = _avg(rs, "dl_curve", tidx, stem)
                j = int(np.argmin(np.abs(ths - th_star)))
                sel["learned"]["valsel"].append(float(Ft[j]))
                sel["learned"]["oracle"].append(float(np.max(Ft)))
            _, _, Fv = _avg(rs, "cl_curve", vidx)
            th_star = float(ths[int(np.argmax(Fv))])
            _, _, Ft = _avg(rs, "cl_curve", tidx)
            j = int(np.argmin(np.abs(ths - th_star)))
            sel["classical"]["valsel"].append(float(Ft[j]))
            sel["classical"]["oracle"].append(float(np.max(Ft)))
        valres[str(int(A))] = {
            meth: {
                "validator_selected_f1_mean": float(np.mean(v["valsel"])),
                "validator_selected_f1_std": float(np.std(v["valsel"], ddof=1)),
                "oracle_f1_mean": float(np.mean(v["oracle"])),
                "n_draws": len(v["valsel"]),
            }
            for meth, v in sel.items()
        }
    results["validation_selection"] = valres

    json.dump(results, open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8"),
              indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8") as fh:
        fh.write("amplitude,method,metric,value\n")
        for A, d in results["detect"].items():
            fh.write(f"{A},classical,ap,{d['classical']['ap']}\n")
            fh.write(f"{A},classical,f1_opt,{d['classical']['f1_opt']['f1']}\n")
            fh.write(f"{A},classical,f1_opt_threshold,{d['classical']['f1_opt']['threshold']}\n")
            fh.write(f"{A},learned,ap_seed_mean,{d['learned']['ap_seed_mean']}\n")
            fh.write(f"{A},learned,ap_seed_std,{d['learned']['ap_seed_std']}\n")
            fh.write(f"{A},learned,f1_opt_per_seed_mean,{d['learned']['f1_opt_per_seed_mean']}\n")
            fh.write(f"{A},learned,f1_opt_per_seed_std,{d['learned']['f1_opt_per_seed_std']}\n")
        for A, d in results["validation_selection"].items():
            for meth in ("learned", "classical"):
                fh.write(f"{A},{meth},valselected_f1,"
                         f"{d[meth]['validator_selected_f1_mean']}\n")
                fh.write(f"{A},{meth},oracle_f1,{d[meth]['oracle_f1_mean']}\n")
    print(json.dumps({A: {"cl_ap": round(d["classical"]["ap"], 3),
                          "cl_f1": round(d["classical"]["f1_opt"]["f1"], 3),
                          "dl_ap": round(d["learned"]["ap_seed_mean"], 3),
                          "dl_f1_mean": round(d["learned"]["f1_opt_per_seed_mean"], 3),
                          "dl_f1_std": round(d["learned"]["f1_opt_per_seed_std"], 3)}
                      for A, d in results["detect"].items()}, indent=2))
    print("val:", json.dumps(results["validation_selection"], indent=2))


if __name__ == "__main__":
    main()
