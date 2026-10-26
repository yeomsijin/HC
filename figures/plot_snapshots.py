"""Neighborhoods and persistence diagrams in Figure 6."""

from pathlib import Path
import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

import os
from hc.figure_data import snapshot_data

PAPER = Path(os.environ["HC_FIGURE_WORK"])
SOURCE = Path(os.environ["HC_FIGURE_SOURCE"])
PREVIEW = PAPER / "previews"
ROLE_COLORS = dict(response="#252525", driver="#777777")
WINDOWS = [16, 48, 80]
QUERY = 64


def load():
    return snapshot_data(SOURCE)


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    snapshots, evidence, singular = load()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6,
            "axes.linewidth": 0.45,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    unit = 0.01
    pooled = np.vstack([x for s in snapshots for x in s["display"]]) / unit
    spans = np.ptp(pooled, axis=0)
    lower = np.floor((pooled.min(0) - 0.04 * spans) * 2) / 2
    upper = np.ceil((pooled.max(0) + 0.04 * spans) * 2) / 2
    h1max = max(ds[1][:, 0].max(initial=0) for s in snapshots for ds in s["pds"]) / unit
    pdymax = (
        max(
            a.max(initial=0) if a.ndim == 1 else a[:, 1].max(initial=0)
            for s in snapshots
            for ds in s["pds"]
            for a in ds
        )
        / unit
    )

    pdxlim = float(np.ceil(h1max * 1.08))
    pdylim = float(np.ceil(pdymax * 1.06))
    max_persistence = max(
        bar["persistence"]
        for snapshot in snapshots
        for bars in snapshot["all_cycles"]
        for bar in bars
    )
    alpha_min, alpha_max = 0.045, 0.34
    outputs = []
    for i, snapshot in enumerate(snapshots):
        response = 1 if i in (0, 2) else 0
        system_colors = [
            ROLE_COLORS["response" if c == response else "driver"] for c in (0, 1)
        ]
        evidence[i]["role_colors"] = dict(
            response="XY"[response],
            driver="XY"[1 - response],
            X=system_colors[0],
            Y=system_colors[1],
        )

        fig = plt.figure(figsize=(3.4, 0.82))
        cloud_axes = []
        for c in (0, 1):
            ax = fig.add_axes(
                [0.0075 + c * 0.25, 0.24, 0.225, 0.70],
                projection="3d",
                computed_zorder=False,
            )
            points = snapshot["clouds"][c]
            xyz = snapshot["display"][c] / unit
            tree = np.array(minimum_spanning_tree(cdist(points, points)).nonzero()).T
            for u, v in tree:
                ax.plot(
                    *xyz[[u, v]].T,
                    color=system_colors[c],
                    alpha=0.35,
                    lw=0.24,
                    zorder=1,
                )

            cycle_edges = sorted(
                {tuple(e) for bar in snapshot["all_cycles"][c] for e in bar["edges"]}
            )
            for u, v in cycle_edges:
                ax.plot(
                    *xyz[[u, v]].T,
                    color=system_colors[c],
                    alpha=0.85,
                    lw=0.40,
                    zorder=2,
                )

            face_persistence = {}
            for bar in snapshot["all_cycles"][c]:
                for face in bar["faces"]:
                    key = tuple(face)
                    face_persistence[key] = max(
                        face_persistence.get(key, 0.0), bar["persistence"]
                    )
            if face_persistence:
                faces = sorted(face_persistence)
                opacities = np.array(
                    [
                        alpha_min
                        + (alpha_max - alpha_min)
                        * face_persistence[f]
                        / max_persistence
                        for f in faces
                    ]
                )
                colors = np.tile(
                    matplotlib.colors.to_rgba(system_colors[c]), (len(faces), 1)
                )
                colors[:, 3] = opacities
                ax.add_collection3d(
                    Poly3DCollection(
                        [xyz[list(f)] for f in faces],
                        facecolors=colors,
                        edgecolors="none",
                        zorder=1.8,
                    )
                )
                evidence[i].setdefault("filling_display", []).append(
                    dict(
                        system="XY"[c],
                        cycles=len(snapshot["all_cycles"][c]),
                        unique_triangles=len(faces),
                        alpha_range=[float(opacities.min()), float(opacities.max())],
                    )
                )
            cycle = snapshot["cycles"][c]
            if cycle:
                for u, v in cycle["edges"]:
                    ax.plot(*xyz[[u, v]].T, color=system_colors[c], lw=0.85, zorder=3)
            ax.scatter(
                *xyz.T,
                color=system_colors[c],
                marker=["o", "^"][c],
                s=3.5,
                linewidths=0,
                depthshade=False,
                zorder=4,
            )

            display_lower, display_upper = lower, upper
            evidence[i].setdefault("display_limits", []).append(
                dict(
                    system="XY"[c],
                    lower=display_lower.tolist(),
                    upper=display_upper.tolist(),
                )
            )
            for j, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
                setter(display_lower[j], display_upper[j])
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
                axis.line.set_linewidth(0.35)
                axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1)
                axis._axinfo["grid"]["linewidth"] = 0.3
                from matplotlib.ticker import MaxNLocator

                axis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            ax.tick_params(labelsize=4, pad=-5, length=1, width=0.35)
            ax.set_box_aspect(upper - lower, zoom=1.18)
            ax.view_init(24, -54)
            cloud_axes.append(ax)

        pair_gap = 0.03
        fig.canvas.draw()

        def box_extent(ax):
            corners = np.array(list(itertools.product(*zip(lower, upper))))
            px, py, _ = proj3d.proj_transform(*corners.T, ax.get_proj())
            xy = ax.transData.transform(np.column_stack([px, py]))
            return xy[:, 0].min() / fig.bbox.width, xy[:, 0].max() / fig.bbox.width

        first_extent, second_extent = [box_extent(ax) for ax in cloud_axes]
        shift = first_extent[1] + pair_gap - second_extent[0]
        pos = cloud_axes[1].get_position()
        cloud_axes[1].set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
        fig.canvas.draw()
        cloud_extents = [box_extent(ax) for ax in cloud_axes]
        cloud_centers = [sum(v) / 2 for v in cloud_extents]
        evidence[i]["column_layout"] = dict(
            pair_gap_fraction=pair_gap,
            cloud_box_extents=cloud_extents,
            heading_centers=cloud_centers + [0.64, 0.84],
            pd_box_extents=[[0.555, 0.725], [0.755, 0.925]],
        )

        for c in (0, 1):
            pdax = fig.add_axes([0.555 + c * 0.20, 0.235, 0.17, 0.17 * 3.4 / 0.82])
            pdax.set_box_aspect(1)
            h0, h1 = snapshot["pds"][c]
            pdax.scatter(
                np.zeros(len(h0)),
                h0 / unit,
                s=7,
                marker="o",
                facecolors="none",
                edgecolors=system_colors[c],
                linewidths=0.45,
                zorder=3 + c,
            )
            pdax.scatter(
                h1[:, 0] / unit,
                h1[:, 1] / unit,
                s=8.7,
                marker="D",
                facecolors="none",
                edgecolors=system_colors[c],
                linewidths=0.5,
                zorder=3 + c,
            )
            pdax.set(
                xlim=(-0.05 * pdxlim, pdxlim),
                ylim=(-0.04 * pdylim, pdylim),
                xticks=[0, 4, 8],
                yticks=[0, 5, 10],
            )
            pdax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)
            if c:
                pdax.set_yticklabels([])
            else:
                pdax.set_ylabel("Persistence", fontsize=5, labelpad=0.6)
            pdax.set_xlabel("Birth", fontsize=5, labelpad=0.3)
        name = f"Fig6_snapshot_{'ABC'[i]}"
        fig.savefig(PAPER / "figures" / f"{name}.pdf", facecolor="white")
        fig.savefig(PREVIEW / f"{name}.png", dpi=400, facecolor="white")
        outputs.append(f"figures/{name}.pdf")
        plt.close(fig)
    return


if __name__ == "__main__":
    main()
