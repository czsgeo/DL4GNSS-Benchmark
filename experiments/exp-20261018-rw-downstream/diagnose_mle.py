"""Diagnose the MLE clean-arm velocity outlier in exp-20261018-rw-downstream."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench, classical as cl, gnss_lib as gl

DT = bench.DT
cfg = json.load(open(os.path.join(HERE, "config.json")))
npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
split = json.load(open(os.path.join(ROOT, cfg["split_file"])))
want = set(split[cfg["use_split"]])
sids = sorted(str(s) for s in npz["sites"] if str(s) in want)
k = int(cfg["max_stations"])
idx = np.linspace(0, len(sids) - 1, k).round().astype(int)
sids = [sids[i] for i in sorted(set(idx))]
base = int(cfg["seed"])
rows = []
for si, sid in enumerate(sids):
    t = np.asarray(npz[f"t__{sid}"], float)
    enu = np.asarray(npz[f"enu__{sid}"], float)
    X, names = gl.build_design(t, orders=(1, 2), poly=2)
    vc = names.index("trend")
    for comp in cfg["components"]:
        p = bench.signal_params_from_real(t, enu[:, comp], seed=base + si)
        S = bench.clean_signal(t, p)
        tv = float(p["trend"])
        for kk in range(cfg["n_seeds"]):
            rng = np.random.default_rng(base + 1000 * si + 10 * comp + kk)
            nspec = bench.make_noise_spec(1.5, 3.0, 0.0, len(t), DT)
            y = S + bench.inject(len(t), nspec, DT, rng)
            yhat = cl.mle_signal(t, y)
            beta, resid, *_ = gl.robust_fit(t, yhat, X)
            verr = float(beta[vc]) - tv
            rows.append(dict(sid=sid, comp=comp, seed=kk, verr=verr,
                             span=float(t[-1] - t[0]), n=len(t),
                             vhat=float(beta[vc]), vtrue=tv,
                             resid_max=float(np.max(np.abs(resid)))))
rows.sort(key=lambda r: -abs(r["verr"]))
for r in rows[:8]:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
print("bias", np.mean([r["verr"] for r in rows]),
      "rmse", np.sqrt(np.mean([r["verr"] ** 2 for r in rows])))
