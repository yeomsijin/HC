from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

PAPER = Path(os.environ["HC_FIGURE_WORK"])
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
PREVIEW = PAPER / "previews"
COLORS = ["#A34B5D", "#C78C98", "#254F69", "#7D756F"]
KEYS = ["h0", "h1", "tc", "d_ccm"]
LABELS = [
    r"$S_{\mathrm{HC}}$ ($H_0$)",
    r"$S_{\mathrm{HC}}$ ($H_1$)",
    r"$S_{\mathrm{TC}}$",
    r"$S_{\mathrm{CCM}}$",
]
LINESTYLES = ["-", (0, (4, 2)), (0, (4, 1.7, 1, 1.7)), (0, (1, 2))]
MARKERS = ["o", "D", "s", "^"]
MARKER_SIZES = [1.85, 2.0, 1.65, 1.9]


def save(fig, name):
    fig.savefig(PAPER / "figures" / (name + ".pdf"), facecolor="white")
    fig.savefig(PREVIEW / (name + ".png"), dpi=300, facecolor="white")
    plt.close(fig)


def combined(data):
    settings = [
        ("g", "a", r"Coupling strength $g$"),
        ("a", "b", r"Symmetry-breaking coefficient $a$"),
        ("q", "c", r"Speed asymmetry parameter $q$"),
    ]
    for index, (df, (parameter, panel, xlabel)) in enumerate(
        zip((data[0], data[2], data[1]), settings)
    ):
        fig, ax = plt.subplots(figsize=(3.4, 1.20))
        fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.32)
        means = df.groupby(parameter)[KEYS].agg(lambda x: np.mean(x.to_numpy(float)))
        means.to_csv(PAPER / f"figure_{parameter}_means.csv")
        for j, key in enumerate(KEYS):
            ax.plot(
                means.index,
                means[key],
                color=COLORS[j],
                ls=LINESTYLES[j],
                lw=[0.95, 0.9, 0.9, 0.8][j],
                marker=MARKERS[j],
                ms=MARKER_SIZES[j],
                markerfacecolor="white" if j in [1, 3] else COLORS[j],
                markeredgecolor=COLORS[j],
                markeredgewidth=0.45,
                solid_capstyle="round",
                dash_capstyle="round",
                label=LABELS[j],
            )
        ax.axhline(0, color="#C9C4BE", lw=0.45, zorder=0)
        ax.set_xlabel(xlabel, fontsize=7, labelpad=4)
        ax.set_ylim(-0.65, 1.0)
        ax.set_yticks([-0.5, 0, 0.5, 1.0])
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#383633")
        ax.tick_params(direction="out", length=2.3, width=0.5, pad=2, labelsize=6.5)
        ax.set_xticks([[2, 10, 20, 30], [0, 0.5, 1.0], [1, 2, 3, 4]][index])
        if index == 1:
            ax.set_ylabel("Directional score", fontsize=7, labelpad=6)
        handles, labels = ax.get_legend_handles_labels()
        save(fig, "Fig3_unified_" + panel)
    fig = plt.figure(figsize=(3.4, 0.22))
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.0),
        fontsize=7,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=0.8,
        borderaxespad=0,
    )
    save(fig, "Fig3_unified_legend")


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
        }
    )
    combined(
        [
            pd.read_csv(SOURCE / f"fig3_{name}.csv")
            for name in ["coupling", "fast_slow", "symmetry"]
        ]
    )
