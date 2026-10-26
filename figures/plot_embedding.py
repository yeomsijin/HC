"""Single-column vector heatmaps for the frozen three-case embedding sweep."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from embedding_window_style import REFERENCE_GUIDES, REFERENCE_LINEWIDTH

import os

PAPER = Path(os.environ["HC_FIGURE_WORK"])
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
PREVIEW = PAPER / "previews"
CASES = ["synchronized", "symmetry", "fast_slow"]
KEYS = ["h0", "h1", "tc", "d_ccm"]
REFERENCE_WINDOWS = [guide["window"] for guide in REFERENCE_GUIDES]
SAMPLING_INTERVAL = 0.02
PALETTE = ["#287E68", "#87B8A5", "#FFFFFF", "#D7837C", "#AB2E3C"]
CMAP = LinearSegmentedColormap.from_list("teal_white_crimson", PALETTE, N=257)
CMAP.set_bad("#D4D1CD")


def save(fig, name):
    fig.savefig(PAPER / "figures" / f"{name}.pdf", facecolor="white")
    fig.savefig(PREVIEW / f"{name}.png", facecolor="white", dpi=300)
    plt.close(fig)


def main():
    frame = pd.read_csv(SOURCE / "fig5_mean_scores.csv")
    assert len(frame) in (360, 396) and not frame.duplicated(["case", "m", "tau"]).any()
    displayed = frame[frame.m >= 3].copy()
    assert len(displayed) == 360
    PREVIEW.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.4,
        }
    )
    for case in CASES:
        fig = plt.figure(figsize=(3.4, 0.94))
        group = displayed[displayed.case == case]
        for j, key in enumerate(KEYS):
            ax = fig.add_axes([0.10 + 0.23 * j, 0.28, 0.19, 0.65])
            matrix = (
                group.pivot(index="m", columns="tau", values=key)
                .reindex(index=range(3, 13), columns=range(1, 13))
                .to_numpy(float)
            )
            finite = matrix[np.isfinite(matrix)]
            assert np.all(np.abs(finite) <= 1 + 1e-12)
            ax.imshow(
                np.ma.masked_invalid(matrix),
                origin="lower",
                interpolation="nearest",
                extent=[0.5, 12.5, 2.5, 12.5],
                aspect="auto",
                cmap=CMAP,
                vmin=-1,
                vmax=1,
            )
            if case == "fast_slow" and key in ["h0", "h1"]:
                m_curve = np.linspace(3, 12, 600)
                for guide in REFERENCE_GUIDES:
                    window, linestyle = guide["window"], guide["linestyle"]
                    tau_curve = window / (SAMPLING_INTERVAL * (m_curve - 1))
                    inside = (tau_curve >= 1) & (tau_curve <= 12)
                    ax.plot(
                        tau_curve[inside],
                        m_curve[inside],
                        color=guide["color"],
                        lw=REFERENCE_LINEWIDTH,
                        ls=linestyle,
                        zorder=3,
                        solid_capstyle="butt",
                        dash_capstyle="butt",
                    )
            ax.set_xticks([1, 4, 8, 12])
            ax.set_yticks([3, 6, 12])
            ax.set_xlabel(r"$\tau$", fontsize=6, labelpad=0)
            ax.tick_params(direction="out", labelsize=5, length=1.5, width=0.4, pad=1.1)
            if j == 0:
                ax.set_ylabel(r"$m$", fontsize=6, labelpad=1)
            else:
                ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#6F6B67")
        save(fig, "Fig5_embedding_" + case)
    missing = not np.isfinite(displayed[KEYS].to_numpy(float)).all()
    fig = plt.figure(figsize=(3.4, 0.37))
    ax = fig.add_axes(
        [0.18 if missing else 0.20, 0.54, 0.58 if missing else 0.60, 0.20]
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(-1, 1), cmap=CMAP),
        cax=ax,
        orientation="horizontal",
        ticks=[-1, 0, 1],
    )
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
    cb.outline.set_linewidth(0.35)
    fig.text(
        0.47 if missing else 0.50, 0.07, "Directional score", ha="center", fontsize=6
    )
    if missing:
        fig.add_artist(
            Rectangle(
                (0.83, 0.51),
                0.035,
                0.26,
                transform=fig.transFigure,
                fc="#D4D1CD",
                ec="#6F6B67",
                lw=0.35,
            )
        )
        fig.text(0.88, 0.58, "undefined", fontsize=5.5, va="center")
    save(fig, "Fig5_embedding_colorbar")
    return


if __name__ == "__main__":
    main()
