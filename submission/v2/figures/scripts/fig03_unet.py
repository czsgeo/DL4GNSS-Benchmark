"""Fig. 3 -- one-dimensional U-Net used for denoising and gap filling."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import figstyle as fs

fig, ax = plt.subplots(figsize=(7.1, 3.15))
ax.set_xlim(0, 141)
ax.set_ylim(0, 17)
ax.axis("off")

BW, BH = 14.5, 2.35


def blk(xc, yc, text, color, edge, w=BW):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - BH / 2), w, BH,
                                boxstyle="round,pad=0.12,rounding_size=0.5",
                                linewidth=1.0, edgecolor=edge, facecolor=color))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=8)


enc = [(10, 13.0, "in 1×1024\nout 12×1024"),
       (28, 10.0, "in 12×512\nout 24×512"),
       (46, 7.0, "in 24×256\nout 48×256")]
bot = (64, 4.0, "in 48×128\nout 96×128")
dec = [(82, 7.0, "96×256 ⊕ 48\n→ 48×256"),
       (100, 10.0, "48×512 ⊕ 24\n→ 24×512")]

for x, y, txt in enc:
    blk(x, y, txt, "#E8F1F8", fs.BLUE)
blk(*bot, color="#E8F1F8", edge=fs.BLUE)
for x, y, txt in dec:
    blk(x, y, txt, "#FBE9DE", fs.ORANGE)
blk(116, 13.0, "24×1024 ⊕ 12\n→ 12×1024", "#FBE9DE", fs.ORANGE, w=17.0)
blk(132, 13.0, "1×1024\noutput", "white", fs.GREY, w=8.5)
ax.add_patch(FancyArrowPatch((124.8, 13.0), (127.5, 13.0), arrowstyle="-|>",
                             mutation_scale=8, linewidth=0.9, color=fs.GREY))

# down / up arrows
for (x0, y0), (x1, y1) in [((10, 11.8), (28, 11.2)), ((28, 8.8), (46, 8.2)),
                           ((46, 5.8), (64, 5.2))]:
    ax.add_patch(FancyArrowPatch((x0 + 3.6, y0), (x1 - 3.6, y1), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.9, color=fs.GREY))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.45, "↓2", ha="center", fontsize=8,
            color=fs.GREY)
for (x0, y0), (x1, y1) in [((66.5, 5.2), (80, 5.8)), ((84, 8.2), (98, 8.8)),
                           ((102, 11.2), (114, 11.8))]:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=0.9, color=fs.GREY))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.45, "↑2", ha="center", fontsize=8,
            color=fs.GREY)

# skip connections
for xa, xb, y, lbl in [(17.3, 107.3, 13.0, "skip"),
                       (35.3, 92.8, 10.0, "skip"),
                       (53.3, 74.8, 7.0, "skip")]:
    ax.add_patch(FancyArrowPatch((xa, y), (xb, y),
                                 connectionstyle="arc3,rad=-0.08",
                                 arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.8, color=fs.GREEN, linestyle=(0, (2.5, 1.5))))

ax.text(10, 16.0, "encoder", ha="center", fontsize=8, color=fs.BLUE, fontweight="bold")
ax.text(100, 16.0, "decoder", ha="center", fontsize=8, color=fs.ORANGE, fontweight="bold")
ax.text(64, 1.25, "bottleneck", ha="center", fontsize=8, color=fs.BLUE)
ax.text(68, 15.0, "213 385 parameters;  kernel 7, ReLU, max-pool 2, nearest up-sampling",
        ha="center", fontsize=8, color="#333333")
ax.text(72, 8.45, "skip connections", ha="center", fontsize=8, color=fs.GREEN)

fs.save_fig(fig, "fig03_unet")
