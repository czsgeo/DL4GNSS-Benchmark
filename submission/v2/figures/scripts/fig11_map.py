"""Fig. 11 -- station coverage, folds and the five evaluation regions."""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

WS = fs.WS
stations = os.path.join(WS, "experiments/exp-20261008-data-prep/data/stations.csv")
holdings = os.path.join(WS, "experiments/exp-20261008-data-prep/data/ngl_holdings.txt")
split = json.load(open(os.path.join(
    WS, "experiments/exp-20261008-data-prep/data/split.json"), encoding="utf-8"))
npz = np.load(os.path.join(
    WS, "experiments/exp-20261008-data-prep/data/series.npz"), allow_pickle=True)
src = {str(s): str(q) for s, q in zip(npz["sites"], npz["source"])}

lat, lon = {}, {}
for row in csv.DictReader(open(stations)):
    try:
        la, lo = float(row["lat"]), float(row["lon"])
    except (TypeError, ValueError):
        continue
    if np.isfinite(la) and np.isfinite(lo):
        lat[row["sid"]], lon[row["sid"]] = la, lo
for line in open(holdings):
    p = line.split()
    if len(p) < 3:
        continue
    try:
        la, lo = float(p[1]), float(p[2])
    except ValueError:
        continue
    lat.setdefault(p[0], la)
    lon.setdefault(p[0], lo)

fold_of = {}
for fold in ("train", "val", "test"):
    for s in split[fold]:
        fold_of[str(s)] = fold


def xy(s):
    code = s.split("@")[0]
    la = lat.get(s, lat.get(code))
    lo = lon.get(s, lon.get(code))
    if la is None or lo is None:
        return None
    return ((lo + 180.0) % 360.0) - 180.0, la


fig, ax = plt.subplots(figsize=(7.1, 3.15))
fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.14)

# region boxes (descriptive, not administrative boundaries)
BOXES = [
    ("Europe", (-25, 34, 45, 72)),
    ("North America", (-170, 10, -50, 75)),
    ("Asia", (45, -12, 180, 78)),
    ("Pacific", (150, -60, 180, 20)),
    ("Pacific", (-180, -60, -150, 20)),
]
for name, (x0, y0, x1, y1) in BOXES:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#B0B0B0", lw=0.5, ls=(0, (2, 2))))
ax.text(-11, 76, "Europe", fontsize=8, color="#555555", ha="center")
ax.text(-110, 78, "North America", fontsize=8, color="#555555", ha="center")
ax.text(112, 82, "Asia", fontsize=8, color="#555555", ha="center")
ax.text(168, 24, "Pacific", fontsize=8, color="#555555", ha="center", rotation=90)
ax.text(-40, -48, "other / S. America, Africa, polar",
        fontsize=8, color="#555555", ha="center")

FOLD = {"train": ("#C8C8C8", 0.55, "train (193 sites)"),
        "val": ("#92C5DE", 0.85, "validation (64 sites)"),
        "test": (fs.BLUE, 1.0, "test (64 sites)")}
for fold in ("train", "val", "test"):
    xs, ys, ms = [], [], []
    for s, f in fold_of.items():
        if f != fold:
            continue
        p = xy(s)
        if p is None:
            continue
        xs.append(p[0])
        ys.append(p[1])
        ms.append(3.2 if src[s] == "ngl" else 6.0)
    color, alpha, label = FOLD[fold]
    ax.scatter(xs, ys, s=ms, c=color, alpha=alpha, linewidths=0,
               label=label, zorder=3 if fold != "test" else 4)

# JPL triangles, drawn on top of the circles; only the held-out ones are orange
for fold in ("train", "val", "test"):
    xs, ys = [], []
    for s, f in fold_of.items():
        if f == fold and src[s] == "jpl":
            p = xy(s)
            if p is not None:
                xs.append(p[0])
                ys.append(p[1])
    edge = fs.ORANGE if fold == "test" else FOLD[fold][0]
    ax.scatter(xs, ys, s=13, marker="^", facecolors="none",
               edgecolors=edge, linewidths=0.8,
               label="JPL series (35 total, 6 held out)" if fold == "test" else None,
               zorder=5)

ax.set_xlim(-180, 180)
ax.set_ylim(-80, 88)
ax.set_xticks(np.arange(-180, 181, 60),
              ["180$\\degree$", "120$\\degree$W", "60$\\degree$W", "0$\\degree$",
               "60$\\degree$E", "120$\\degree$E", "180$\\degree$"])
ax.set_yticks(np.arange(-60, 81, 30))
ax.tick_params(labelsize=8)
ax.grid(color="#E8E8E8", lw=0.4)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_title("Held-out sites cover all five regions and both analysis centres",
             fontsize=9.0, loc="left")
ax.legend(fontsize=8, loc="lower left", ncol=2, handletextpad=0.3,
          columnspacing=0.9, borderpad=0.3, labelspacing=0.25,
          frameon=False)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=2, pad=1)
fs.save_fig(fig, "fig11_map")
print("saved fig11")
