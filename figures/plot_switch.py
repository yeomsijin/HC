"""Directional scores and local expansion in Figure 6."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

PAPER = Path(os.environ["HC_FIGURE_WORK"])
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
PREVIEW = PAPER / "previews"
DATA = SOURCE / "fig6_windows.csv"
METHODS = ["h0", "h1", "tc", "d_ccm"]
LABELS = [
    r"$S_{\mathrm{HC}}$ ($H_0$)",
    r"$S_{\mathrm{HC}}$ ($H_1$)",
    r"$S_{\mathrm{TC}}$",
    r"$S_{\mathrm{CCM}}$",
]
COLORS = ["#A34B5D", "#C78C98", "#254F69", "#7D756F"]
EXPANSION_COLORS = ["#202020", "#696969"]

LINESTYLES = ["-", (0, (4, 2)), (0, (4, 1.7, 1, 1.7)), (0, (1, 2))]
MARKERS = ["o", "D", "s", "^"]
MARKER_SIZES = [1.85, 2.0, 1.65, 1.9]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(ax, frame, key, color, label, ls):
    g = frame.groupby("time")[key]
    assert g.size().eq(6).all()
    mean = g.agg(lambda x: np.mean(x.to_numpy(float)))
    kwargs = dict(color=color, label=label, ls=ls, lw=0.95)
    if key in METHODS:
        j = METHODS.index(key)
        kwargs.update(
            lw=[0.95, 0.9, 0.9, 0.8][j],
            marker=MARKERS[j],
            ms=MARKER_SIZES[j],
            markevery=(j, 4),
            markerfacecolor="white" if j in [1, 3] else COLORS[j],
            markeredgecolor=COLORS[j],
            markeredgewidth=0.45,
            solid_capstyle="round",
            dash_capstyle="round",
        )
    (line,) = ax.plot(mean.index, mean.values, **kwargs)
    return line


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    assert len(frame) == 576
    assert frame.groupby("seed").size().eq(96).all()
    assert np.isfinite(
        frame[METHODS + ["mean_log_y_to_x", "mean_log_x_to_y"]].to_numpy()
    ).all()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure_height = 2.12
    fig = plt.figure(figsize=(3.4, figure_height))

    score = fig.add_axes([0.16, 1.1255 / figure_height, 0.81, 0.8925 / figure_height])
    logs = fig.add_axes(
        [0.16, 0.2075 / figure_height, 0.81, 0.714 / figure_height], sharex=score
    )
    for ax in [score, logs]:
        for t in [327.68, 655.36]:
            ax.axvspan(
                t - 81.92, t + 81.92, color="#DEDEDE", alpha=0.5, lw=0, zorder=-2
            )
            ax.axvline(t, color="#858585", ls="--", lw=0.5, zorder=-1)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_linewidth(0.55)
        ax.tick_params(direction="out", length=2.5, width=0.55, pad=2)
        ax.set_xlim(0, 983.04)
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.annotate(
            r"$t$",
            xy=(1, 0),
            xycoords="axes fraction",
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7,
            annotation_clip=False,
        )
    handles = [
        curve(score, frame, key, color, label, ls)
        for key, color, label, ls in zip(METHODS, COLORS, LABELS, LINESTYLES)
    ]
    score.axhline(0, color="#AAA", lw=0.5, zorder=0)
    score.set(ylim=(-1, 1), yticks=[-1, -0.5, 0, 0.5, 1], ylabel="Directional score")
    score.tick_params(labelbottom=False)

    score.legend(
        handles,
        LABELS,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        handlelength=1.8,
        handletextpad=0.35,
        columnspacing=0.55,
        borderaxespad=0,
        fontsize=6.5,
        labelcolor="black",
    )

    curve(logs, frame, "mean_log_y_to_x", EXPANSION_COLORS[0], r"$R_Y\to R_{X|Y}$", "-")
    curve(
        logs, frame, "mean_log_x_to_y", EXPANSION_COLORS[1], r"$R_X\to R_{Y|X}$", "--"
    )
    logs.set(ylabel="Mean log expansion", yticks=[10, 20, 30])

    for ax in (score, logs):
        ax.yaxis.set_label_coords(-0.145, 0.5)
    logs.set_xticks([0, 327.68, 655.36], labels=["0", r"$t_1$", r"$t_2$"])
    logs.legend(
        frameon=False,
        loc="upper left",
        borderaxespad=0.3,
        handlelength=1.8,
        labelcolor="linecolor",
    )

    for letter, left, right, label in zip(
        "ABC",
        [0, 327.68, 655.36],
        [327.68, 655.36, 983.04],
        [r"$X\to Y$", r"$Y\to X$", r"$X\to Y$"],
    ):
        center = (left + right) / 2
        logs.text(
            center,
            0,
            letter,
            transform=logs.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.3,
            color=".25",
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
            zorder=5,
        )
        logs.text(
            center,
            -0.18,
            label,
            transform=logs.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.5,
        )
    fig.savefig(PAPER / "figures/Fig6_direction_scores.pdf", facecolor="white")
    fig.savefig(PREVIEW / "Fig6_direction_scores.png", dpi=350, facecolor="white")
    plt.close(fig)
    return


if __name__ == "__main__":
    main()
