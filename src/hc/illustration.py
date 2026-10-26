"""Regenerate the two-loop clouds, diagrams and probability masses in Fig2."""

from pathlib import Path
import json
import numpy as np
from tpce.persistence import diagrams
from tpce.entropy import explanation_discrepancy, induced_probability, entropy_excess
from .two_loop_geometry import make_all_clouds, CASE_ORDER
from .io import ROOT, save_arrays, save_json


def run(output):
    output = Path(output)
    tuning = json.loads((ROOT / "configs/two_loops.json").read_text())
    clouds = make_all_clouds(tuning, seed=7)
    pds = {}
    for name, cloud in clouds.items():
        _, pd = diagrams(cloud, scale=1.0, h1_min_persistence=1e-3)
        pds[name] = pd[np.argsort(-pd[:, 1])]
    residuals = np.concatenate(
        [
            np.abs(explanation_discrepancy(pds["reference"], pds[name], (0.5, 0.5)))
            for name in CASE_ORDER[1:]
        ]
    )
    eta = float(np.median(residuals[residuals > 1e-12]))
    save_arrays(output / "fig2_point_clouds.npz", **clouds)
    save_arrays(output / "fig2_diagrams.npz", **pds)
    save_json(
        output / "fig2_original_settings.json", dict(sigma_b=0.5, sigma_l=0.5, tau=eta)
    )
    metrics = []
    for target, explanation in [
        ("large loop", "large loop"),
        ("reference", "reference"),
        ("large loop", "reference"),
        ("reference", "large loop"),
    ]:
        p, logq, boundary = induced_probability(
            pds[target], pds[explanation], (0.5, 0.5), eta
        )
        metrics.append(
            dict(
                target=target,
                explanation=explanation,
                cross_entropy=float(-p @ logq),
                excess=float(
                    entropy_excess(pds[target], pds[explanation], (0.5, 0.5), eta)
                ),
                unexplained_mass=boundary,
            )
        )
    save_json(output / "fig2_metrics.json", metrics)
    print("Two-loop illustration regenerated", flush=True)
