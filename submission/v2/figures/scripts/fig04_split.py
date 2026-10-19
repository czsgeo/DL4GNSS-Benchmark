"""Fig. 4 -- station-level train/validation/test split: geography, counts, record length."""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

import figstyle as fs

WS = fs.WS
df = pd.read_csv(os.path.join(WS, "experiments/exp-20261008-data-prep/data/stations.csv"))
split = json.load(open(os.path.join(WS, "experiments/exp-20261008-data-prep/data/split.json")))

df["split"] = "unassigned"
for k, ids in split.items():
    df.loc[df["sid"].isin(ids), "split"] = k

colors = {"train": "#8C8C8C", "val": fs.SKY, "test": fs.ORANGE}
labels = {"train": "train", "val": "validation", "test": "test"}

fig = plt.figure(figsize=(7.1, 2.85))
gs = fig.add_gridspec(1, 3, width_ratios=[1.95, 0.92, 0.92], wspace=0.42,
                      left=0.03, right=0.99, top=0.90, bottom=0.16)
axm = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
axh = fig.add_subplot(gs[0, 2])

# --- basemap
with open(os.path.join(os.path.dirname(fs.OUT), "figures/data/ne_110m_land.geojson")) as fh:
    gj = json.load(fh)
for feat in gj["features"]:
    geom = feat["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        ring = np.asarray(poly[0])
        axm.add_patch(Polygon(ring, closed=True, facecolor="#EEEEEE",
                              edgecolor="#CCCCCC", linewidth=0.3, zorder=0))

for k in ["train", "val", "test"]:
    sub = df[df["split"] == k].drop_duplicates("site")
    lo = ((sub["lon"].values + 180.0) % 360.0) - 180.0
    axm.scatter(lo, sub["lat"].values, s=7, c=colors[k], edgecolors="white",
                linewidths=0.25, label=f"{labels[k]} ({len(sub)})", zorder=3)

axm.set_xlim(-180, 180)
axm.set_ylim(-90, 90)
axm.set_aspect("equal")
axm.set_xticks([-180, -120, -60, 0, 60, 120, 180])
axm.set_yticks([-60, -30, 0, 30, 60])
axm.tick_params(length=2, pad=1)
axm.set_title("(a) stations by split", fontsize=8.5, loc="left")
axm.legend(loc="lower left", fontsize=8, markerscale=1.4, handletextpad=0.2,
           borderpad=0.2, labelspacing=0.25, ncol=3, bbox_to_anchor=(0.0, -0.02))
for s in ["top", "right"]:
    axm.spines[s].set_visible(False)

# --- counts
order = ["train", "val", "test"]
n_st = [int((df["split"] == k).sum()) for k in order]
n_site = [int(df.loc[df["split"] == k, "site"].nunique()) for k in order]
n_ep = [int(df.loc[df["split"] == k, "n_epochs"].sum()) for k in order]
x = np.arange(3)
bars = axb.bar(x, n_st, color=[colors[k] for k in order], width=0.62)
for i, (b, e) in enumerate(zip(bars, n_ep)):
    axb.text(i, b.get_height() + 4, f"{n_site[i]} st\n{n_st[i]} ser",
             ha="center", va="bottom", fontsize=8, linespacing=0.95)
    axb.text(i, b.get_height() / 2, f"{e/1e6:.2f}M", ha="center", va="center",
             fontsize=8, color="white" if i != 1 else "#333333")
axb.set_xticks(x, [labels[k] for k in order], fontsize=8)
axb.set_ylabel("series", fontsize=8)
axb.set_ylim(0, 228)
axb.set_title("(b) split sizes", fontsize=8.5, loc="left")
axb.text(0.97, 0.86, "0 sites\nshared", transform=axb.transAxes,
         fontsize=8, va="top", ha="right", color="#333333")

# --- record length
data = [df.loc[df["split"] == k, "span_years"].values for k in order]
bp = axh.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                 medianprops=dict(color="#222222", linewidth=1.0))
for patch, k in zip(bp["boxes"], order):
    patch.set_facecolor(colors[k]); patch.set_edgecolor("#555555"); patch.set_linewidth(0.6)
axh.set_xticks([1, 2, 3], [labels[k] for k in order], fontsize=8)
axh.set_ylabel("record length (years)", fontsize=8)
axh.set_title("(c) comparable spans", fontsize=8.5, loc="left")
axh.set_ylim(0, 32)
axh.text(0.03, 0.06, "median 30 y", transform=axh.transAxes, fontsize=8,
         color="#333333")

for a in [axb, axh]:
    a.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)

fs.save_fig(fig, "fig04_split")
print("series", n_st, "sites", n_site, "epochs", n_ep)
print("spans", {k: (float(np.median(v)), float(np.min(v)), float(np.max(v))) for k, v in zip(order, data)})
