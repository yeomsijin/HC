"""Compact vector panels for the validated fast-slow window diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from embedding_window_style import REFERENCE_GUIDES, REFERENCE_LINEWIDTH

import os

PAPER = Path(os.environ["HC_FIGURE_WORK"])
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
PREVIEW = PAPER / "previews"
COLORS = ["#A0A0A0", "#5C5C5C", "#111111"]
REFERENCE_WINDOWS = [guide["window"] for guide in REFERENCE_GUIDES]
CMAP = LinearSegmentedColormap.from_list("dimension_gray", COLORS)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2, width=0.45, pad=2, labelsize=6)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#383633")


def save(fig, name):
    fig.savefig(PAPER / "figures" / f"{name}.pdf", facecolor="white")
    fig.savefig(PREVIEW / f"{name}.png", facecolor="white", dpi=300)
    plt.close(fig)


def main():
    frame = pd.read_csv(SOURCE / "fig5_window.csv")
    assert len(frame) == 120 and len(frame[["m", "tau"]].drop_duplicates()) == 120
    PREVIEW.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
        }
    )
    fig = plt.figure(figsize=(3.4, 1.22))
    for h, left in [(0, 0.135), (1, 0.565)]:
        ax = fig.add_axes([left, 0.29, 0.34, 0.65])
        for m, g in frame.groupby("m"):
            ax.plot(
                g["T"], g[f"h{h}"], color=CMAP((m - 3) / 9), lw=0.65, marker=".", ms=1.3
            )
        ax.axhline(0, color="#C9C4BE", lw=0.45, zorder=0)
        for guide in REFERENCE_GUIDES:
            window, linestyle = guide["window"], guide["linestyle"]

            heights = []
            for _, group in frame.groupby("m"):
                group = group.sort_values("T")
                if group["T"].min() <= window <= group["T"].max():
                    heights.append(np.interp(window, group["T"], group[f"h{h}"]))
            ax.plot(
                [window, window],
                [-1, min(heights)],
                color=guide["color"],
                lw=REFERENCE_LINEWIDTH,
                ls=linestyle,
                zorder=1,
                solid_capstyle="butt",
                dash_capstyle="butt",
            )
        ax.set(xlim=(0, 2.7), ylim=(-1, 1), xticks=[0.2, 0.8, 2], yticks=[-1, 0, 1])
        ax.set_xticklabels(["0.2", "0.8", "2"])
        for tick, guide in zip(ax.get_xticklabels()[:2], REFERENCE_GUIDES):
            tick.set_color(guide["color"])
        ax.set_xlabel(r"$T$", fontsize=7, labelpad=1)
        if h == 0:
            ax.set_ylabel(r"$S_{\mathrm{HC}}$", fontsize=7, labelpad=3)
        else:
            ax.set_yticklabels([])
        style(ax)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(3, 12), cmap=CMAP),
        cax=fig.add_axes([0.94, 0.38, 0.017, 0.50]),
        ticks=[3, 6, 12],
    )
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.35)
    cb.ax.set_title(r"$m$", fontsize=6, pad=3)
    save(fig, "Fig5_window_scores")

    return


if __name__ == "__main__":
    main()
