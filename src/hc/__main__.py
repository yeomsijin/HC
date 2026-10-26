import os
import argparse
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[name] = "1"


def main():
    from .io import ROOT, configuration, save_json

    parser = argparse.ArgumentParser(
        prog="hc", description="Homological Causality: paper reproduction"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stationary", "embedding", "switching"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, default=Path("results"))
        p.add_argument("--seed", type=int)
        p.add_argument("--limit", type=int)
        if name != "switching":
            p.add_argument(
                "--case",
                choices=["coupling", "symmetry", "fast_slow"]
                if name == "stationary"
                else ["synchronized", "symmetry", "fast_slow"],
            )
    p = sub.add_parser("illustration")
    p.add_argument("--output", type=Path, default=Path("results"))
    p = sub.add_parser("figures")
    p.add_argument("--source", type=Path, default=ROOT / "reference")
    p.add_argument("--output", type=Path, default=Path("results/figures"))
    p.add_argument(
        "--figure", type=int, nargs="+", choices=range(1, 7), default=list(range(1, 7))
    )
    p.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if hasattr(os, "setpriority"):
        os.setpriority(os.PRIO_PROCESS, 0, 19)
    if args.command == "figures":
        from .figures import build

        build(args.source, args.output, args.figure, not args.assets_only)
    elif args.command == "illustration":
        from .illustration import run

        run(args.output)
    else:
        import fcntl
        import json
        from . import experiments

        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".run.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol = configuration()
            path = args.output / "paper.json"
            if path.exists() and json.loads(path.read_text()) != protocol:
                raise ValueError("Output belongs to a different protocol")
            save_json(path, protocol)
            kwargs = dict(seed=args.seed, limit=args.limit)
            if hasattr(args, "case"):
                kwargs["case"] = args.case
            getattr(experiments, args.command)(args.output, **kwargs)


if __name__ == "__main__":
    main()
