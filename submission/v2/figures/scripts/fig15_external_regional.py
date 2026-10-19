"""Fig. 15 -- external (second-centre) and regional transfer."""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

PATH = os.path.join(
    fs.WS, "experiments/exp-20261018-external-eval/results.csv")
rows = list(csv.DictReader(open(PATH)))

CLASS_DEN = [("mle", "trajectory MLE"), ("wavelet", "wavelet")]
CLASS_IMP = [("linear", "linear"), ("mle", "trajectory MLE")]

def series_mean(task, methods, key):
    """Average over realisations and, for the learned arm, over seeds.

    Returns {method: {group: [value per component series]}} -- one value per
    (site, component), so the intervals describe station-to-station spread
    rather than the much narrower seed-to-seed spread.
    """
    buckets = {}
    for r in rows:
        if r["task"] != task or r["method"] not in methods:
            continue
        k = (r["site"], r["component"], r["method"])
        buckets.setdefault(k, []).append(float(r["value"]))
    per_series = {}
    for (site, comp, method), vals in buckets.items():
        per_series.setdefault((site, comp), {})[method] = float(np.mean(vals))
    out = {m: {} for m in methods}
    if any(m.startswith("unet_") for m in methods):
        out["unet"] = {}
    for (site, comp), d in per_series.items():
        seed_vals = [v for m, v in d.items() if m.startswith("unet_")]
        if seed_vals:
            d["unet"] = float(np.mean(seed_vals))
        g = next(r[key] for r in rows
                 if r["site"] == site and r["component"] == comp)
        for m, v in d.items():
            if m in out:
                out[m].setdefault(g, []).append(v)
    return out


def boot(v, n=4000, seed=11):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    means = v[idx].mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def learned(task):
    return sorted({r["method"] for r in rows
                   if r["task"] == task and r["method"].startswith("unet_")})


den_methods = [m for m, _ in CLASS_DEN] + learned("denoise")
imp_methods = [m for m, _ in CLASS_IMP] + learned("impute30")

fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.4),
                         gridspec_kw=dict(wspace=0.34, hspace=0.55,
                                          left=0.085, right=0.985, top=0.95,
                                          bottom=0.08))

# ---------------------------------------------------- (a)(b) by centre
for ax, task, methods, classes, ylab, title, better in (
        (axes[0, 0], "denoise", den_methods, CLASS_DEN,
         "injected-error reduction", "(a) denoising by analysis centre", +1),
        (axes[0, 1], "impute30", imp_methods, CLASS_IMP,
         "30-day gap RMSE (mm)", "(b) gap filling by analysis centre", -1)):
    dat = series_mean(task, methods, "source")
    groups = ["ngl", "jpl"]
    x = np.arange(len(groups))
    width = 0.26
    shades = [fs.BLUE, fs.SKY]
    labels = [(m, lab, shades[i]) for i, (m, lab) in enumerate(classes)] + [
        ("unet", "U-Net", fs.ORANGE)]
    offs = np.linspace(-width, width, len(labels))
    for (m, lab, color), off in zip(labels, offs):
        vals = [np.mean(dat[m][g]) if dat[m].get(g) else np.nan for g in groups]
        lo = [boot(dat[m][g])[1] if dat[m].get(g) else np.nan for g in groups]
        hi = [boot(dat[m][g])[2] if dat[m].get(g) else np.nan for g in groups]
        ax.bar(x + off, vals, width * 0.92, color=color, label=lab, alpha=0.9)
        ax.errorbar(x + off, vals, yerr=[np.array(vals) - np.array(lo),
                                         np.array(hi) - np.array(vals)],
                    fmt="none", ecolor="#333333", lw=0.8, capsize=2,
                    elinewidth=0.7)
    ax.set_xticks(x, ["NGL", "JPL"])
    ax.set_ylabel(ylab)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.legend(fontsize=8, loc="upper center", ncol=3, handlelength=1.1,
              borderpad=0.2, columnspacing=1.0, frameon=False)

# ---------------------------------------------------- (c)(d) by region
for ax, task, methods, classes, ylab, title in (
        (axes[1, 0], "denoise", den_methods, CLASS_DEN,
         "injected-error reduction", "(c) denoising by region"),
        (axes[1, 1], "impute30", imp_methods, CLASS_IMP,
         "30-day gap RMSE (mm)", "(d) gap filling by region")):
    dat = series_mean(task, methods, "region")
    groups = [g for g in ("Europe", "North America", "Asia", "Pacific", "other")
              if dat["mle"].get(g) and len(dat["mle"][g]) >= 6]
    order = sorted(groups, key=lambda g: np.mean(dat["unet"][g]))
    y = np.arange(len(order))
    shades = [fs.BLUE, fs.SKY]
    for i, (m, lab) in enumerate(classes):
        v = [np.mean(dat[m][g]) for g in order]
        ax.scatter(v, y, s=15, color=shades[i], zorder=3, label=lab)
    v = [np.mean(dat["unet"][g]) for g in order]
    ax.scatter(v, y, s=15, color=fs.ORANGE, zorder=3, label="U-Net")
    ax.legend(fontsize=8, loc="upper left", handlelength=1.1, borderpad=0.2,
              labelspacing=0.25, frameon=False)
    for i, g in enumerate(order):
        ax.plot([np.mean(dat[classes[0][0]][g]), np.mean(dat["unet"][g])],
                [i, i], color="#bbbbbb", lw=1.2, zorder=1)
    ax.set_yticks(y, [f"{g}\n(n={len(dat['unet'][g])})" for g in order],
                  fontsize=8)
    ax.set_xlabel(ylab)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.axvline(0, color="#888888", lw=0.6, ls=":")

for a in axes.ravel():
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fs.save_fig(fig, "fig15_external_regional")
for task in ("denoise", "impute30"):
    dat = series_mean(task, den_methods if task == "denoise" else imp_methods,
                      "source")
    print(task)
    for g in ("ngl", "jpl", "Europe", "North America", "Asia", "Pacific", "other"):
        if dat["mle"].get(g):
            print(f"  {g:14s} n={len(dat['mle'][g]):3d} "
                  f"mle={np.mean(dat['mle'][g]):.3f} "
                  f"unet={np.mean(dat['unet'][g]):.3f}")
