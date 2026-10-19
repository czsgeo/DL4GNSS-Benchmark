"""Fig. 8 -- nonparametric truth families and who survives the misspecification."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

sys.path.insert(0, os.path.join(fs.WS, "experiments", "common"))
import bench  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261018-nonparam-signals/results_perwin.json")))
regs = d["regimes"]

SID, COMP, SEED = "J010", 0, 20261011
npz = np.load(os.path.join(fs.WS,
                           "experiments/exp-20261008-data-prep/data/series.npz"),
              allow_pickle=True)
t = np.asarray(npz[f"t__{SID}"], float)
y = np.asarray(npz[f"enu__{SID}"], float)[:, COMP]
p_in = bench.signal_params_from_real(t, y, seed=SEED)
p_oof = bench.signal_params_out_of_family(t, y, seed=SEED)
S_in = bench.clean_signal(t, p_in)
S_oof = bench.clean_signal(t, p_oof)
lin = p_in["trend"] * (t - t[0])
S_in, S_oof = S_in - lin, S_oof - lin
win = (t >= 1996.0) & (t <= 2014.0)

fig = plt.figure(figsize=(7.1, 3.15))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.36,
                      left=0.078, right=0.99, top=0.88, bottom=0.24)

# ------------------------------------------------------------ (a) truth families
ax = fig.add_subplot(gs[0, 0])
ax.plot(t[win], S_in[win], color=fs.BLUE, lw=1.0, label="in-family truth")
ax.plot(t[win], S_oof[win], color=fs.ORANGE, lw=0.8,
        label="out-of-family truth")
for t0, amp, tau in p_oof["transients"]:
    ax.axvline(t0, color=fs.ORANGE, lw=0.6, ls=":")
for t0, amp, tau in p_oof.get("logdecay", []):
    ax.annotate("log relaxation", xy=(t0 + 3.0, S_oof[win].max() * 0.75),
                fontsize=8, color="#444444",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"))
t0, amp, tau = p_oof["transients"][0]
ax.annotate("transient\n$\\tau=%.1f$ y" % tau,
            xy=(t0, S_oof[win].max() * 0.72),
            xytext=(t[win][0] + 0.3, S_oof[win].max() * 0.86),
            fontsize=8, color="#444444", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"))
ax.set_xlabel("year")
ax.set_ylabel(f"{SID} {['E', 'N', 'U'][COMP]} signal (mm, detrended)")
ax.set_title("(a) seasonal amplitude is not constant", fontsize=8.5, loc="left")
ax.legend(fontsize=8, loc="upper right", handlelength=1.2,
          labelspacing=0.25, borderpad=0.2)

# ------------------------------------------------------- (b,c) reductions
METHODS = [("mle", "MLE"), ("rts", "RTS"), ("ssa", "SSA"),
           ("wavelet", "wavelet"), ("median", "median")]
COLORS = [fs.BLUE, "#4393C3", "#92C5DE", "#6BAED6", "#B8D7EA"]


def unet(reg):
    v = [x["reduction_mean"] for m, x in regs[reg].items() if "__" in m]
    return np.asarray(v)


handles = []
def panel(ax, pair, title, ylim):
    pairs = [(pair[0], "in"), (pair[1], "out")]
    centers = np.arange(len(pairs)) * 1.55
    wbar = 0.15
    offs = np.linspace(-2.5, 2.5, 6) * wbar
    handles.clear()
    for gi, (reg, _lab) in enumerate(pairs):
        r = regs[reg]
        for j, (m, lab) in enumerate(METHODS):
            h = ax.bar(centers[gi] + offs[j], r[m]["reduction_mean"], width=wbar,
                       color=COLORS[j], label=lab if gi == 0 else None,
                       edgecolor="white", linewidth=0.3)
            if gi == 0:
                handles.append(h)
        u = unet(reg)
        x = centers[gi] + offs[5]
        h = ax.bar(x, float(np.mean(u)), width=wbar, color=fs.ORANGE,
                   edgecolor="white", linewidth=0.3,
                   label="U-Net" if gi == 0 else None)
        if gi == 0:
            handles.append(h)
        ss = np.linspace(-1, 1, len(u)) * 0.036
        ax.scatter(np.full(len(u), x) + ss, u, s=3.0, color="#7F2A00",
                   zorder=4, linewidths=0)
        ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(centers, ["in-family", "out-of-family"], fontsize=8)
    ax.set_ylabel(r"RMSE reduction")
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=8.5, loc="left")


panel(fig.add_subplot(gs[0, 1]), ("hybrid_in", "hybrid_oof"),
      "(b) real-station truth", (-0.42, 1.02))
panel(fig.add_subplot(gs[0, 2]), ("synth_in", "synth_oof"),
      "(c) synthetic truth", (-1.35, 0.82))
fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=6,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.005),
           handlelength=1.0, columnspacing=0.9, handletextpad=0.3)

for a in fig.axes:
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fs.save_fig(fig, "fig08_truth_families")
for reg in ("hybrid_in", "hybrid_oof", "synth_in", "synth_oof"):
    r = regs[reg]
    u = unet(reg)
    print(reg, {m: round(r[m]["reduction_mean"], 3) for m in
                ("mle", "rts", "ssa", "wavelet", "median")},
          "unet %.3f+-%.3f" % (np.mean(u), np.std(u, ddof=1)), "n_seeds", len(u))
