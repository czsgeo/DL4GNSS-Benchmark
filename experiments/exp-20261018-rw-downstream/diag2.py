import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench, gnss_lib as gl
cfg = json.load(open(os.path.join(HERE, "config.json")))
npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
for sid, comp, seed_off in (("0LEK", 0, 1), ("HOLP", 1, 1)):
    t = np.asarray(npz[f"t__{sid}"], float); enu = np.asarray(npz[f"enu__{sid}"], float)
    print(sid, "nan t", np.isnan(t).sum(), "nan y", np.isnan(enu[:, comp]).sum(),
          "inf", np.isinf(enu[:, comp]).sum())
    split = json.load(open(os.path.join(ROOT, cfg["split_file"])))
    sids = sorted(str(s) for s in npz["sites"] if str(s) in set(split[cfg["use_split"]]))
    k = int(cfg["max_stations"])
    idx = np.linspace(0, len(sids) - 1, k).round().astype(int)
    sids = [sids[i] for i in sorted(set(idx))]
    si = sids.index(sid); base = int(cfg["seed"])
    p = bench.signal_params_from_real(t, enu[:, comp], seed=base + si)
    S = bench.clean_signal(t, p)
    nspec = bench.make_noise_spec(1.5, 3.0, 0.0, len(t), bench.DT)
    rng = np.random.default_rng(base + 1000 * si + 10 * comp + seed_off)
    y = S + bench.inject(len(t), nspec, bench.DT, rng)
    print("  nan y after inject", np.isnan(y).sum(), "max|y|", np.nanmax(np.abs(y)))
    X, names = gl.build_design(t, orders=(1, 2), poly=2)
    print("  X nan", np.isnan(X).sum(), "Xmax", np.abs(X).max(), "cond", np.linalg.cond(X))
    beta, resid, w, out = gl.robust_fit(t, y, X)
    print("  beta", beta, "n_out", out.sum(), "n_w0", (w == 0).sum(),
          "resid_max", np.nanmax(np.abs(resid)))
    yhat = X @ beta
    print("  yhat range", np.nanmin(yhat), np.nanmax(yhat))
