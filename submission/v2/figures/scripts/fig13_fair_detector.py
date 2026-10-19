"""Fig. 13 -- detector comparison under validation-based threshold selection."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

d = json.load(open(os.path.join(
    fs.WS, "experiments/exp-20261009-detector-fair/results.json")))
amps = [2, 4, 8, 16, 25]

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55),
                         gridspec_kw=dict(width_ratios=[1.15, 1, 1], wspace=0.36,
                                          left=0.075, right=0.99, top=0.86, bottom=0.18))
ax = axes[0]
cl_f1 = [d["detect"][str(a)]["classical"]["f1_opt"]["f1"] for a in amps]
ln = [d["detect"][str(a)]["learned"] for a in amps]
ln_f1 = [x["f1_opt_per_seed_mean"] for x in ln]
ln_f1_sd = [x["f1_opt_per_seed_std"] for x in ln]
vcl = [d["validation_selection"][str(a)]["classical"]["validator_selected_f1_mean"] for a in amps]
vcl_sd = [d["validation_selection"][str(a)]["classical"]["validator_selected_f1_std"] for a in amps]
vln = [d["validation_selection"][str(a)]["learned"]["validator_selected_f1_mean"] for a in amps]
vln_sd = [d["validation_selection"][str(a)]["learned"]["validator_selected_f1_std"] for a in amps]

x = np.arange(len(amps))
ax.errorbar(x - 0.06, ln_f1, yerr=ln_f1_sd, fmt="o-", color=fs.ORANGE, lw=1.3,
            ms=3.5, capsize=2, elinewidth=0.7,
            label="learned, best threshold (20 seeds)")
ax.plot(x - 0.06, vln, "s--", color=fs.ORANGE, ms=3.2, lw=1.0, alpha=0.75,
        label="learned, validation-selected")
ax.plot(x + 0.06, cl_f1, "o-", color=fs.BLUE, lw=1.3, ms=3.5,
        label="classical, best threshold")
ax.errorbar(x + 0.06, vcl, yerr=vcl_sd, fmt="s--", color=fs.BLUE, ms=3.2,
            capsize=2, elinewidth=0.7, lw=1.0, alpha=0.75,
            label="classical, validation-selected")
ax.set_xticks(x, [str(a) for a in amps])
ax.set_xlabel("injected offset amplitude (mm)")
ax.set_ylabel(r"$F_1$")
ax.set_ylim(-0.02, 1.04)
ax.set_title("(a) selection does not close the gap", fontsize=8.5, loc="left")
ax.legend(fontsize=8, loc="upper left", handlelength=1.4, labelspacing=0.3)

for k, amp in enumerate([8, 16]):
    a = axes[k + 1]
    cc = d["detect"][str(amp)]["classical"]["pr_curve"]
    ll = d["detect"][str(amp)]["learned"]["seed_mean_pr_curve"]
    a.plot(cc["recall"], cc["precision"], color=fs.BLUE, lw=1.4,
           label="classical")
    a.plot(ll["recall"], ll["precision"], color=fs.ORANGE, lw=1.4,
           label="learned (seed mean)")
    a.set_xlim(0, 1.02)
    a.set_ylim(0, 1.02)
    a.set_xlabel("recall")
    if k == 0:
        a.set_ylabel("precision")
    a.set_title(f"({'bc'[k]}) {amp} mm", fontsize=8.5, loc="left")
    a.text(0.04, 0.08, f"AP {d['detect'][str(amp)]['classical']['ap']:.2f}"
           f" vs {d['detect'][str(amp)]['learned']['ap_seed_mean']:.2f}",
           fontsize=8, color="#333333")

axes[1].legend(fontsize=8, loc="upper right", handlelength=1.5)
for a in axes:
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fs.save_fig(fig, "fig13_fair_detector")
