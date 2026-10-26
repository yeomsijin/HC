"""Fixed snapshots and matched point clouds used in Figures 4 and 6."""

from pathlib import Path
import json
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from tpce.geometry import embed
from .experiments import stationary_record, score_record, switching_record


def geometry_data(source, case, value):
    source = Path(source)
    reference = source / "fig4_geometry.npz"
    if reference.exists():
        index = {
            ("symmetry", 1.0): 0,
            ("symmetry", 0.03): 1,
            ("fast", 1.0): 2,
            ("fast", 4.0): 3,
        }[(case, value)]
        data = np.load(reference)
        own = [data[f"r{index}_own{c}"] for c in (0, 1)]
        induced = [data[f"r{index}_induced{c}"] for c in (0, 1)]
        physical = int(data[f"r{index}_physical"])
    else:
        spec = (
            dict(g=24.0, q=1.0, a=value)
            if case == "symmetry"
            else dict(g=2.0, q=value, a=1.0)
        )
        quantiles = stationary_record(source, 321023, **spec)
        folder = source / "geometry" / f"{case}{value:g}"
        score_record(folder, quantiles)
        data = np.load(folder / "scores.npz")
        spaces = embed(quantiles, 6, 4)
        neighbors = [data["neighbors_x"][256], data["neighbors_y"][256]]
        own = [spaces[c][neighbors[c]] for c in (0, 1)]
        induced = [spaces[c][neighbors[1 - c]] for c in (0, 1)]
        physical = int(data["physical_sample"][256])
    return (
        own,
        induced,
        dict(
            case=case,
            parameter=value,
            seed=321023,
            evaluation_index=256,
            physical_sample=physical,
            time=physical * 0.02,
        ),
    )


def persistent_cycles(points):
    """Reduced triangle columns over F2, with explicit bounding 2-chains."""
    dist = cdist(points, points)
    n = len(points)
    edges = sorted(itertools.combinations(range(n), 2), key=lambda e: (dist[e], e))
    edge_index = {e: i for i, e in enumerate(edges)}
    triangles = sorted(
        itertools.combinations(range(n), 3),
        key=lambda t: (max(dist[u, v] for u, v in itertools.combinations(t, 2)), t),
    )
    basis, bars = {}, []
    for index, triangle in enumerate(triangles):
        filling = 1 << index
        column = sum(1 << edge_index[e] for e in itertools.combinations(triangle, 2))
        while column:
            pivot = column.bit_length() - 1
            if pivot not in basis:
                break
            old_column, old_filling = basis[pivot]
            column ^= old_column
            filling ^= old_filling
        if not column:
            continue
        basis[pivot] = column, filling
        birth = float(dist[edges[pivot]])
        death = float(max(dist[u, v] for u, v in itertools.combinations(triangle, 2)))
        if death - birth <= 1e-9:
            continue
        cycle = [edges[k] for k in range(len(edges)) if (column >> k) & 1]
        faces = [triangles[k] for k in range(len(triangles)) if (filling >> k) & 1]
        boundary = 0
        for face in faces:
            for e in itertools.combinations(face, 2):
                boundary ^= 1 << edge_index[e]
        assert boundary == column
        degree = np.zeros(n, int)
        for u, v in cycle:
            degree[u] += 1
            degree[v] += 1
        assert np.all(degree % 2 == 0)
        bars.append(
            dict(birth=birth, persistence=death - birth, edges=cycle, faces=faces)
        )
    return bars


def snapshot_data(source):
    source = Path(source)
    cache = source / "fig6_snapshots.npz"
    if cache.exists():
        arrays = np.load(cache)
        records = json.loads((source / "fig6_cycles.json").read_text())
        clouds = [[arrays[f"{label}_c{c}_cloud"] for c in (0, 1)] for label in "ABC"]
        basis = arrays["common_projection"]
    else:
        quantiles = switching_record(source, 481025)
        records, clouds = [], []
        for label, window in zip("ABC", [16, 48, 80]):
            center = 4096 + window * 512
            window_data = quantiles[center - 4096 : center + 4096]
            folder = source / "switching" / "s481025" / f"w{window:03d}"
            score_record(folder, window_data, queries=128)
            scores = np.load(folder / "scores.npz")
            spaces = embed(window_data, 6, 4)
            current = [
                spaces[c][scores["neighbors_x" if c == 0 else "neighbors_y"][64]]
                for c in (0, 1)
            ]
            clouds.append(current)
            all_cycles = [persistent_cycles(cloud) for cloud in current]
            pds = []
            for c in (0, 1):
                ds = []
                for h in (0, 1):
                    offsets = scores[f"pd_h{h}_c{c}_offsets"]
                    ds.append(scores[f"pd_h{h}_c{c}"][offsets[64] : offsets[65]])
                pds.append(ds)
            records.append(
                dict(
                    label=label,
                    all_displayed_cycles=all_cycles,
                    pds=pds,
                    highlighted_cycles=[
                        max(bars, key=lambda b: b["persistence"]) if bars else None
                        for bars in all_cycles
                    ],
                )
            )
        centered = [cloud - cloud.mean(0) for pair in clouds for cloud in pair]
        _, _, vh = np.linalg.svd(np.vstack(centered), full_matrices=False)
        basis = vh[:3].T
        for j in range(3):
            if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
                basis[:, j] *= -1
    centered = [cloud - cloud.mean(0) for pair in clouds for cloud in pair]
    singular = np.linalg.svd(np.vstack(centered), compute_uv=False)
    snapshots = []
    for i, (label, pair, record) in enumerate(zip("ABC", clouds, records)):
        pds = (
            [[arrays[f"{label}_c{c}_h{h}"] for h in (0, 1)] for c in (0, 1)]
            if cache.exists()
            else record["pds"]
        )
        snapshots.append(
            dict(
                clouds=pair,
                pds=pds,
                cycles=record["highlighted_cycles"],
                all_cycles=record["all_displayed_cycles"],
                display=[centered[2 * i + c] @ basis for c in (0, 1)],
            )
        )
    return snapshots, records, singular
