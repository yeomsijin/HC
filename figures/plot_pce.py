from pathlib import Path
import json
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os
from tpce.entropy import induced_probability, entropy_excess

OUTPUT = Path(os.environ["HC_FIGURE_WORK"]) / "figures"
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
cloud = np.load(SOURCE / "fig2_point_clouds.npz")
pd = np.load(SOURCE / "fig2_diagrams.npz")
settings = json.loads((SOURCE / "fig2_original_settings.json").read_text())
keys = ["large loop", "reference"]
symbols = ["X", "Y"]
eta = settings["tau"]
records = []
probs = {}
for c, key in enumerate(keys):
    row = []
    for rev in [0, 1]:
        target_key = keys[rev]
        explanation_key = keys[rev] if c == 0 else keys[1 - rev]
        a, b = pd[target_key], pd[explanation_key]
        p, logq, bound = induced_probability(a, b, (0.5, 0.5), eta)
        q = np.exp(logq)
        ce = float(-p @ logq)
        ex = entropy_excess(a, b, (0.5, 0.5), eta)
        if c == 0:
            np.testing.assert_allclose(q, p, atol=1e-14)
            np.testing.assert_allclose(
                [ce, ex, bound], [-p @ np.log(p), 0, 0], atol=1e-13
            )
        np.testing.assert_allclose(q.sum() + bound, 1, atol=2e-14)
        row.append(
            dict(
                target=symbols[rev],
                explanation=symbols[rev if c == 0 else 1 - rev],
                target_cloud=target_key,
                explanation_cloud=explanation_key,
                cross_entropy=ce,
                excess=ex,
                boundary=bound,
            )
        )
        probs[f"{key}_{rev}"] = q
    records.append(row)
assert records[1][0]["excess"] < records[1][1]["excess"]
np.savez_compressed(OUTPUT / "induced_probabilities.npz", **probs)
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 5,
        "axes.linewidth": 0.25,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{fix-cm}",
    }
)
fig = plt.figure(figsize=(3.4, 2.35))
red = "#B43C45"
blue = "#2868A4"
cmap = LinearSegmentedColormap.from_list(
    "rose_wine", ["#eedcdd", "#d6a3a6", "#ac626a", "#6c2333"]
)
norm = Normalize(0, 0.8)
headings = [None, None, None, None]
for row, heading in enumerate(headings):
    group = 0 if row < 2 else 1
    upper = row in (0, 2)
    bottom = 0.61 if upper else 0.15
    if heading is not None:
        fig.text(
            0.025 + group * 0.51, 0.97 if upper else 0.48, heading, fontsize=6.0444
        )
    for c, key in enumerate(keys):
        upper = (c == 0) if row < 2 else (row == 2)
        bottom = 0.61 if upper else 0.15
        column = row if row < 2 else 2 + c
        left = 0.035 + column * 0.245
        ax = fig.add_axes([left, bottom, 0.195, 0.28])
        ax.set_box_aspect(1)
        sym = symbols[c]
        system_color = "black"
        if row == 0:
            ax.scatter(*cloud[key].T, s=0.70, color=system_color, linewidths=0)
            ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), xticks=[], yticks=[])
            ax.set_title("$" + sym + "$", fontsize=5.5, pad=2, color=system_color)
        else:
            a = pd[key] if row == 1 else pd[keys[row - 2]]
            if row == 1:
                ax.scatter(
                    *a.T,
                    s=5.2,
                    marker="D",
                    facecolors="none",
                    edgecolors=system_color,
                    linewidths=0.3,
                )
                pe = -np.sum(
                    (a[:, 1] / a[:, 1].sum()) * np.log(a[:, 1] / a[:, 1].sum())
                )
                ax.set_title(
                    "$D_{" + sym + "}$", fontsize=5.5, pad=2, color=system_color
                )
            else:
                rev = row - 2
                record = records[c][rev]
                im = ax.scatter(
                    *a.T,
                    c=probs[f"{key}_{rev}"],
                    cmap=cmap,
                    norm=norm,
                    s=6.5,
                    marker="D",
                    edgecolors="#888888",
                    linewidths=0.18,
                )
                label = (
                    (r"p_X" if c == 0 else r"p_X^Y")
                    if row == 2
                    else (r"p_Y" if c == 0 else r"p_Y^X")
                )
                ax.set_title("$" + label + "$", fontsize=5.5, pad=2, color="black")
                box = ax.get_position()
                bar_bottom = box.y0 - 0.035
                bar = fig.add_axes([box.x0, bar_bottom, box.width, 0.008])
                retained = 1 - record["boundary"]
                bar.barh(0, retained, color="#eedcdd", height=1)
                bar.barh(
                    0, record["boundary"], left=retained, color="#6c2333", height=1
                )
                bar.set(xlim=(0, 1), ylim=(-0.5, 0.5), xticks=[], yticks=[])
                for spine in bar.spines.values():
                    spine.set_linewidth(0.15)
                fig.text(
                    box.x0 + box.width / 2,
                    bar_bottom - 0.044,
                    "$"
                    + label
                    + r"(\partial)=$"
                    + ("0" if record["boundary"] == 0 else f"{record['boundary']:.1f}"),
                    ha="center",
                    fontsize=4.5,
                )
            ax.set(
                xlim=(-0.06, 1.28),
                ylim=(-0.045, 1.28),
                xticks=[0, 1.2],
                yticks=[0, 1.2],
            )
            ax.tick_params(labelsize=2.8, length=1, pad=0.3)
            if c != 0:
                ax.set_yticklabels([])
            ax.grid(False)
cax = fig.add_axes([1.01, 0.15, 0.018, 0.74])
cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, 0.4, 0.8])
cb.ax.tick_params(labelsize=4, length=1, pad=1)
cb.set_label("Probability mass", fontsize=4.5, labelpad=2)
cb.outline.set_linewidth(0.2)
cb.solids.set_rasterized(False)
fig.savefig(
    OUTPUT / "PCE_two_loops_compact.pdf",
    backend="pgf",
    bbox_inches="tight",
    pad_inches=0.025,
)
fig.savefig(OUTPUT / "PCE_two_loops_compact.pgf", bbox_inches="tight", pad_inches=0.025)

shutil.copy2(OUTPUT / "PCE_two_loops_compact.pdf", OUTPUT / "Fig2_PCE.pdf")
plt.close(fig)
