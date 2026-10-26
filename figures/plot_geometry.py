"""Own and time-induced neighborhoods in Figure 4."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter

import os
from hc.figure_data import geometry_data

PAPER = Path(os.environ["HC_FIGURE_WORK"])
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
PREVIEW = PAPER / "previews"
BLACK = "#000000"
COORDS = [0, 2, 5]
SEED, INDEX = 321023, 256


def load(case, value):
    return geometry_data(SOURCE, case, value)


def cloud(ax, points, color, title, own):
    z = points[:, COORDS]
    z = z - z.mean(0)
    if own:
        ax.scatter(
            *z.T,
            s=4.5,
            facecolors="none",
            edgecolors=color,
            marker="o",
            depthshade=False,
            linewidths=0.35,
            alpha=0.95,
        )
    else:
        ax.scatter(
            *z.T,
            s=4.5,
            c=color,
            marker="x",
            depthshade=False,
            linewidths=0.35,
            alpha=0.95,
        )

    limit, ticks = (0.065, [-0.05, 0, 0.05]) if own else (0.70, [-0.5, 0, 0.5])
    assert np.max(np.abs(z)) < limit
    for setter in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        setter(-limit, limit)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_box_aspect((1, 1, 1), zoom=1.10)
    ax.view_init(24, -58)
    ax.tick_params(labelsize=4.7, pad=-4, length=1, width=0.4)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.line.set_linewidth(0.45)
        axis._axinfo["grid"]["linewidth"] = 0.35
        axis._axinfo["grid"]["color"] = (0.84, 0.84, 0.84, 1)
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_title(title, color=color, fontsize=7.3, pad=-2)


def panel(case, values, name):
    fig = plt.figure(figsize=(3.4, 1.38))
    records = []

    for index, value in enumerate(values):
        own, induced, info = load(case, value)
        shift = 0.49 * index
        positions = [[shift + 0.02, 0.08, 0.20, 0.56], [shift + 0.26, 0.08, 0.20, 0.56]]
        specs = [
            (induced[0], BLACK, r"$R_{X\mid Y}^{40}(t)$", False),
            (own[1], BLACK, r"$R_Y^{40}(t)$", True),
        ]
        for position, spec in zip(positions, specs):
            cloud(fig.add_axes(position, projection="3d"), *spec)
        center = shift + 0.24
        fig.add_artist(
            FancyArrowPatch(
                (center + 0.06, 0.84),
                (center - 0.06, 0.84),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=7,
                lw=0.6,
                color=".2",
            )
        )
        fig.text(center, 0.92, r"$M^t_{Y\to X}$", ha="center", fontsize=7.2)
        info["asset"] = "figures/" + name + ".pdf"
        records.append(info)
    for ext in ["pdf", "png"]:
        target = PAPER / "figures" if ext == "pdf" else PREVIEW
        fig.savefig(target / (name + "." + ext), dpi=240, facecolor="white")
    plt.close(fig)
    return records


def main():
    PREVIEW.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
        }
    )
    records = panel("symmetry", [1.0, 0.03], "Fig4_geometry_symmetry")
    records += panel("fast", [1.0, 4.0], "Fig4_geometry_fast_slow")
    assert len({r["physical_sample"] for r in records}) == 1
    return records
