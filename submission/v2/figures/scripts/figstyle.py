"""Shared style for the DL4GNSS-Benchmark v2.0 figures.

All figures are vector PDF + PNG, 9 pt sans-serif, blue = classical,
orange = learned, no top/right spines.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
WS = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

sys.path.insert(0, os.path.join(WS, "..", "paper-tools"))
from figure_style import apply_style, save, C, GREY, LIGHT  # noqa: E402

apply_style(lang="en", font_size=9)

import matplotlib.pyplot as _plt  # noqa: E402
_plt.rcParams["savefig.dpi"] = 240  # preview PNG; PDF stays vector

BLUE = C[0]
ORANGE = C[1]
GREEN = C[2]
PURPLE = C[3]
YELLOW = C[4]
SKY = C[5]

CLASSICAL = BLUE
LEARNED = ORANGE


def save_fig(fig, name):
    return save(fig, name, formats=("pdf", "png"), out_dir=OUT)
