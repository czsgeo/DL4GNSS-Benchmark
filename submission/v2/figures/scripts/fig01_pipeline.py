"""Fig. 1 -- benchmark pipeline overview."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as fs

fig, ax = plt.subplots(figsize=(7.1, 4.15))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, title, lines, edge=fs.GREY):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=1.0, edgecolor=edge, facecolor="none"))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top",
            fontsize=9, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x + 2.2, y + h - 5.8 - 3.7 * i, line, ha="left", va="top",
                fontsize=8.0, color="#222222")


# row 1
box(0.8, 36, 29, 23, "Daily GNSS series",
    ["333 stations, E/N/U", "NGL + JPL", "1994-2024", "station-level split"])
box(33.0, 36, 30, 23, "Controlled injection",
    ["parametric families", "nonparametric truth:",
     "seasonal AM, transient", "WN + FN (+ RW)"])
box(66.5, 36, 32.5, 23, "Three benchmark tasks",
    ["T1 denoise: RMSE", "T2 gaps: 5/30/120 d",
     "T3 offsets: 2-25 mm", "shared field readout"])

for x0, x1 in [(30.0, 32.6), (63.2, 65.9)]:
    ax.add_patch(FancyArrowPatch((x0, 47.5), (x1, 47.5), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.0, color=fs.GREY))

# row 2: candidates
ax.add_patch(FancyBboxPatch((0.8, 4), 63.7, 27,
                            boxstyle="round,pad=0.25,rounding_size=1.2",
                            linewidth=1.0, edgecolor=fs.GREY, facecolor="none"))
ax.text(32.65, 29.4, "Classical and learned candidates", ha="center", va="top",
        fontsize=9, fontweight="bold")
ax.text(3.2, 24.4, "Classical", fontsize=8.5, fontweight="bold", color=fs.BLUE)
ax.text(3.2, 20.8, "MLE/GLS, SSA, wavelet,\nRTS, moving median,\ntwo-sample step statistic",
        fontsize=8.0, va="top", color="#222222", linespacing=1.45)
ax.text(34.5, 24.4, "Learned", fontsize=8.5, fontweight="bold", color=fs.ORANGE)
ax.text(34.5, 20.8, "U-Net 213 k, TCN 228 k,\nBiGRU 183 k;\n10-20 training seeds",
        fontsize=8.0, va="top", color="#222222", linespacing=1.45)

# task -> candidates (elbow routed above the evaluation box)
ax.plot([75.0, 75.0, 64.9], [35.4, 33.4, 33.4], color=fs.GREY, linewidth=1.0,
        solid_capstyle="butt")
ax.annotate("", xy=(64.9, 20.0), xytext=(64.9, 33.4),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=9,
                            linewidth=1.0, color=fs.GREY))

# evaluation
box(66.5, 4, 32.5, 27, "Held-out evaluation",
    ["threshold on validation", "P/R/F1 and delay",
     "velocity bias, sigma,", "coverage", "NGL catalog + JPL"])

ax.add_patch(FancyArrowPatch((65.0, 17.5), (66.1, 17.5), arrowstyle="-|>",
                             mutation_scale=9, linewidth=1.0, color=fs.GREY))

fs.save_fig(fig, "fig01_pipeline")
