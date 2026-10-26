"""Own-neighborhood HC, main-text TC and scalar simplex CCM."""

import numpy as np
from ripser import ripser
from scipy.spatial import cKDTree
from tpce.geometry import embed, neighbor_indices
from tpce.h0_float64 import batch_lifetimes
from tpce.own_own import own_own_costs, H1_METHOD
from tpce.tc import logs_from_neighbors, METHOD as TC_METHOD


def direction(values):
    values = np.asarray(values, dtype=float)
    den = values.sum(axis=0)
    return np.divide(
        values[1] - values[0],
        den,
        out=np.full_like(den, np.nan),
        where=np.isfinite(den) & (den > 0),
    )


def correlation(x, y):
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ccm(spaces, anchors, m, span):
    predictions, neighbors, skills = [], [], []
    half = int(np.ceil(span / 2))
    count = min(len(spaces[0]), m + 1 + 2 * half + 1)
    for source in (1, 0):
        candidates = cKDTree(spaces[source]).query(spaces[source][anchors], k=count)[1]
        ids = np.array(
            [
                row[np.abs(row - anchor) > half][: m + 1]
                for anchor, row in zip(anchors, candidates)
            ]
        )
        if ids.shape != (len(anchors), m + 1):
            raise ValueError("Insufficient eligible CCM neighbors")
        distances = np.linalg.norm(
            spaces[source][ids] - spaces[source][anchors, None], axis=2
        )
        weights = np.empty_like(distances)
        nonzero = distances[:, 0] > 0
        weights[nonzero] = np.exp(-distances[nonzero] / distances[nonzero, :1])
        weights[~nonzero] = (distances[~nonzero] == 0).astype(float)
        weights /= weights.sum(axis=1, keepdims=True)
        predicted = np.sum(weights * spaces[1 - source][ids, 0], axis=1)
        predictions.append(predicted)
        neighbors.append(ids)
        skills.append(correlation(predicted, spaces[1 - source][anchors, 0]))
    return np.array(skills), np.array(predictions), np.array(neighbors)


def pack(diagrams):
    return np.concatenate(diagrams), np.r_[
        0, np.cumsum([len(d) for d in diagrams])
    ].astype(np.int32)


def evaluate(quantiles, m=6, tau=4, k=40, queries=512, anchors=None):
    spaces = embed(quantiles, m, tau)
    span = (m - 1) * tau
    if anchors is None:
        anchors = np.unique(
            np.linspace(span + 1, len(spaces[0]) - span - 2, queries, dtype=int)
        )
    anchors = np.asarray(anchors, dtype=int)
    neighbors = [neighbor_indices(s, anchors, k, span) for s in spaces]
    logs, dimensions = logs_from_neighbors(spaces, neighbors)
    h0 = [list(batch_lifetimes(s[ids])) for s, ids in zip(spaces, neighbors)]
    h1 = [[], []]
    for c in (0, 1):
        for ids in neighbors[c]:
            pd = ripser(spaces[c][ids], maxdim=1, coeff=2)["dgms"][1].astype(float)
            if not np.isfinite(pd).all():
                raise ValueError("Nonfinite H1 persistence")
            pd[:, 1] -= pd[:, 0]
            h1[c].append(pd)
    costs = np.full((2, 2, len(anchors)), np.nan)
    for h, diagrams in enumerate((h0, h1)):
        for j in range(len(anchors)):
            try:
                costs[h, :, j] = own_own_costs(diagrams[0][j], diagrams[1][j])[0]
            except ValueError as error:
                if "Unidentifiable H0 bandwidth" not in str(error):
                    raise
    rho, predictions, ccm_neighbors = ccm(spaces, anchors, m, span)
    arrays = dict(
        anchors=anchors.astype(np.int32),
        physical_sample=anchors + span,
        neighbors_x=neighbors[0],
        neighbors_y=neighbors[1],
        logs=logs,
        dimensions=dimensions,
        costs=costs,
        tc_local=direction(logs),
        hc0_local=direction(costs[0]),
        hc1_local=direction(costs[1]),
        ccm_rho=rho,
        ccm_predictions=predictions,
        ccm_neighbors=ccm_neighbors,
        ccm_truth=np.array([s[anchors, 0] for s in spaces]),
        tc_method=TC_METHOD,
        h1_method=H1_METHOD,
    )
    for h, diagrams in enumerate((h0, h1)):
        for c in (0, 1):
            arrays[f"pd_h{h}_c{c}"], arrays[f"pd_h{h}_c{c}_offsets"] = pack(diagrams[c])
    for name, ids in [("own_rms", neighbors), ("induced_rms", neighbors[::-1])]:
        clouds = [space[nn] for space, nn in zip(spaces, ids)]
        arrays[name] = np.array(
            [
                np.sqrt(
                    np.mean(np.sum((v - v.mean(1, keepdims=True)) ** 2, axis=2), axis=1)
                )
                for v in clouds
            ]
        )
    scores = dict(
        h0=float(direction(costs[0].sum(1))),
        h1=float(direction(costs[1].sum(1))),
        tc=float(direction(logs.sum(1))),
        d_ccm=float((rho[0] - rho[1]) / 2),
        rho_x_from_y=float(rho[0]),
        rho_y_from_x=float(rho[1]),
        mean_log_y_to_x=float(logs[0].mean()),
        mean_log_x_to_y=float(logs[1].mean()),
    )
    return arrays, scores
