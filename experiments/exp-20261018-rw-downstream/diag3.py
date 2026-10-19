import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bench, classical as cl, gnss_lib as gl
cfg = json.load(open(os.path.join(HERE, "config.json")))
npz = np.load(os.path.join(ROOT, cfg["dataset"]), allow_pickle=True)
split = json.load(open(os.path.join(ROOT, cfg["split_file"])))
sids = sorted(str(s) for s in npz["sites"] if str(s) in set(split[cfg["use_split"]]))
k = int(cfg["max_stations"]); idx = np.linspace(0, len(sids) - 1, k).round().astype(int)
sids = [sids[i] for i in sorted(set(idx))]
base = int(cfg["seed"])
for sid, comp, so in (("0LEK", 0, 1), ("HOLP", 1, 1)):
    si = sids.index(sid)
    t = np.asarray(npz[f"t__{sid}"], float); enu = np.asarray(npz[f"enu__{sid}"], float)
    p = bench.signal_params_from_real(t, enu[:, comp], seed=base + si)
    S = bench.clean_signal(t, p)
    nspec = bench.make_noise_spec(1.5, 3.0, 0.0, len(t), bench.DT)
    rng = np.random.default_rng(base + 1000 * si + 10 * comp + so)
    y = S + bench.inject(len(t), nspec, bench.DT, rng)
    yhat = cl.mle_signal(t, y)
    X, names = gl.build_design(t, orders=(1, 2), poly=2)
    print(sid, "mle_signal nan", np.isnan(yhat).sum(), "range", yhat.min(), yhat.max())
    print("  direct lstsq beta", np.linalg.lstsq(X, yhat, rcond=None)[0])
    # manual IRLS
    w = np.ones_like(yhat); beta = np.zeros(X.shape[1])
    for it in range(6):
        try:
            beta2, *_ = np.linalg.lstsq(X * w[:, None], yhat * w, rcond=None)
        except np.linalg.LinAlgError as e:
            print("  LinAlgError at", it, e); break
        r = yhat - X @ beta2
        s = 1.4826 * np.median(np.abs(r - np.median(r))); s = s if s > 0 else (np.std(r) or 1.0)
        rr = np.abs(r) / s; out = rr > 4.0
        w = np.where(out, 0.0, np.where(rr <= 1.5, 1.0, 1.5 / np.maximum(rr, 1e-9)))
        print("  it", it, "s", s, "nout", out.sum(), "trend", beta2[names.index("trend")])
