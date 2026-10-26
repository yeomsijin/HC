"""Paper experiments, run serially with resumable records."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from .dynamics import integrate_stationary, integrate_switch
from .methods import evaluate
from tpce.geometry import kde_quantile
from .io import configuration, save_arrays, save_json


def initial_state(seed):
    state = np.random.default_rng(seed).uniform(-5, 5, 9)
    state[2::3] = 25
    return state


def stationary_record(root, seed, g, q, a):
    cfg = configuration()["stationary"]
    folder = Path(root) / "recordings" / f"g{g:g}_q{q:g}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "state.npz"
    if not state_path.exists():
        initial = initial_state(seed)
        post = integrate_stationary(
            initial.copy(),
            np.tile([q, g], (round(cfg["burn"] / cfg["dt"]) + 1, 1)),
            h=cfg["step"],
        )[-1]
        state = integrate_stationary(
            post.copy(), np.tile([q, g], (cfg["samples"] + 1, 1)), h=cfg["step"]
        )
        save_arrays(state_path, state=state, initial=initial, post_burn_initial=post)
    data_path = folder / f"observations_a{a:.17g}.npz"
    if not data_path.exists():
        state = np.load(state_path)["state"]
        observations = np.column_stack([state[:, 0], state[:, 5] + a * state[:, 3]])
        save_arrays(
            data_path, observations=observations, quantiles=kde_quantile(observations)
        )
    return np.load(data_path)["quantiles"]


def score_record(folder, quantiles, **settings):
    folder = Path(folder)
    if not (folder / "scores.json").exists():
        arrays, scores = evaluate(quantiles, **settings)
        save_arrays(folder / "scores.npz", **arrays)
        save_json(folder / "scores.json", scores)
    with (folder / "scores.json").open() as f:
        scores = json.load(f)
    return {
        key: (float("nan") if value is None else value) for key, value in scores.items()
    }


def stationary(root, case=None, seed=None, limit=None):
    root = Path(root)
    config = configuration()["stationary"]
    definitions = [
        ("coupling", "g", config["g"], dict(q=1.0, a=1.0)),
        ("symmetry", "a", config["a"], dict(g=24.0, q=1.0)),
        ("fast_slow", "q", config["q"], dict(g=2.0, a=1.0)),
    ]
    for name, parameter, values, fixed in definitions:
        if case and name != case:
            continue
        rows = []
        jobs = [
            (value, s)
            for value in values
            for s in ([seed] if seed is not None else config["seeds"])
        ]
        for value, s in jobs[:limit]:
            spec = dict(fixed, **{parameter: value})
            quantiles = stationary_record(root, s, **spec)
            folder = root / "stationary" / name / f"{parameter}{value:.17g}_s{s}"
            scores = score_record(
                folder,
                quantiles,
                **{key: config[key] for key in ("m", "tau", "k", "queries")},
            )
            rows.append(dict(**{parameter: value}, seed=s, **scores))
            pd.DataFrame(rows).to_csv(root / f"fig3_{name}.csv", index=False)
            print(
                name,
                value,
                s,
                {k: round(scores[k], 6) for k in ("h0", "h1", "tc", "d_ccm")},
                flush=True,
            )


def embedding(root, case=None, seed=None, limit=None):
    root = Path(root)
    config = configuration()
    definitions = config["embedding"]
    jobs = [
        (c, s, m, tau)
        for c in definitions["cases"]
        if not case or c == case
        for s in ([seed] if seed is not None else config["stationary"]["seeds"])
        for m in definitions["m"]
        for tau in definitions["tau"]
    ]
    rows = []
    for c, s, m, tau in jobs[:limit]:
        quantiles = stationary_record(root, s, **definitions["cases"][c])
        arrays_anchors = np.array(definitions["physical_sample"]) - (m - 1) * tau
        scores = score_record(
            root / "embedding" / c / f"s{s}_m{m}_t{tau}",
            quantiles,
            m=m,
            tau=tau,
            anchors=arrays_anchors,
            k=config["stationary"]["k"],
        )
        rows.append(dict(case=c, seed=s, m=m, tau=tau, **scores))
        pd.DataFrame(rows).to_csv(root / "fig5_scores.csv", index=False)
        print(
            c,
            s,
            m,
            tau,
            {k: round(scores[k], 6) for k in ("h0", "h1", "tc", "d_ccm")},
            flush=True,
        )
    summarize_embedding(root, rows)


def summarize_embedding(root, rows):
    frame = pd.DataFrame(rows)
    keys = ["h0", "h1", "tc", "d_ccm"]
    means = frame.groupby(["case", "m", "tau"])[keys].agg(
        lambda x: np.mean(x.to_numpy(float))
    )
    means.reset_index().to_csv(root / "fig5_mean_scores.csv", index=False)
    fast = means.reset_index().query("case == 'fast_slow'").copy()
    fast["T"] = (fast.m - 1) * fast.tau * 0.02
    fast.to_csv(root / "fig5_window.csv", index=False)


def switching_record(root, seed):
    cfg = configuration()["switching"]
    path = Path(root) / "switching" / f"s{seed}" / "recording.npz"
    if not path.exists():
        initial = initial_state(seed)[:6]
        q, g, block, step = cfg["q"], cfg["g"], cfg["phase_samples"], cfg["step"]
        post = integrate_switch(
            initial, q, g, 0.0, round(cfg["burn"] / cfg["dt"]), step=step
        )[-1]
        first = integrate_switch(post, q, g, 0.0, cfg["prefix"] + block, step=step)
        middle = integrate_switch(first[-1], q, 0.0, g, block, step=step)
        last = integrate_switch(middle[-1], q, g, 0.0, block + cfg["suffix"], step=step)
        state = np.concatenate([first, middle, last])
        observations = np.column_stack([state[:, 0], state[:, 5] + state[:, 3]])
        save_arrays(
            path,
            state=state,
            observations=observations,
            quantiles=kde_quantile(observations),
            initial=initial,
            post_burn_initial=post,
        )
    return np.load(path)["quantiles"]


def switching(root, seed=None, limit=None):
    root = Path(root)
    config = configuration()["switching"]
    rows = []
    for s in [seed] if seed is not None else config["seeds"]:
        quantiles = switching_record(root, s)
        for window in list(range(3 * config["phase_samples"] // config["stride"]))[
            :limit
        ]:
            center = config["prefix"] + window * config["stride"]
            left, right = center - config["window"] // 2, center + config["window"] // 2
            scores = score_record(
                root / "switching" / f"s{s}" / f"w{window:03d}",
                quantiles[left:right],
                **{key: config[key] for key in ("m", "tau", "k", "queries")},
            )
            phase = (center - config["prefix"]) // config["phase_samples"]
            changes = [config["prefix"] + j * config["phase_samples"] for j in (1, 2)]
            pure = not any(left < point < right for point in changes)
            rows.append(
                dict(
                    seed=s,
                    window=window,
                    time=window * config["stride"] * config["dt"],
                    phase=phase,
                    true_sign=1 if phase in (0, 2) else -1,
                    pure_window=pure,
                    **scores,
                )
            )
            pd.DataFrame(rows).to_csv(root / "fig6_windows.csv", index=False)
            print(
                "switching",
                s,
                window,
                {k: round(scores[k], 6) for k in ("h0", "h1", "tc", "d_ccm")},
                flush=True,
            )
