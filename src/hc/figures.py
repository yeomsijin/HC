"""Build the paper figures from reference or recalculated results."""

from pathlib import Path
import importlib
import os
import shutil
import subprocess
import sys
from .io import ROOT


def build(source, output, numbers=(1, 2, 3, 4, 5, 6), assemble=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    work = output / "work"
    assets = output / "assets"
    work.mkdir(parents=True, exist_ok=True)
    (work / "figures").mkdir(exist_ok=True)
    assets.mkdir(exist_ok=True)
    os.environ["HC_FIGURE_SOURCE"] = str(source)
    os.environ["HC_FIGURE_WORK"] = str(work)
    sys.path.insert(0, str(ROOT / "figures"))
    pdflatex = shutil.which("pdflatex")
    if assemble and not pdflatex:
        raise RuntimeError(
            "pdflatex is required to assemble figures; use --assets-only for Python panels"
        )
    for number in numbers:
        if number == 1:
            if not pdflatex:
                raise RuntimeError("Fig1 needs pdflatex")
            shutil.copy2(ROOT / "figures/filtration.tex", work / "filtration.tex")
            compile_tex(work, "filtration.tex", pdflatex)
            shutil.copy2(work / "filtration.pdf", work / "figures/Fig1_filtration.pdf")
        for module in {
            1: [],
            2: ["plot_pce"],
            3: ["plot_stationary"],
            4: ["plot_geometry"],
            5: ["plot_embedding", "plot_windows"],
            6: ["plot_switch", "plot_snapshots"],
        }[number]:
            sys.modules.pop(module, None)
            obj = importlib.import_module(module)
            if hasattr(obj, "main"):
                obj.main()
        for path in (work / "figures").glob("*.pdf"):
            shutil.copy2(path, assets / path.name)
        if assemble:
            shutil.copy2(ROOT / f"figures/fig{number}.tex", output / f"Fig{number}.tex")
            compile_tex(output, f"Fig{number}.tex", pdflatex)
        print(f"Figure {number} built", flush=True)


def compile_tex(folder, filename, executable):
    log = Path(folder) / (Path(filename).stem + ".build.log")
    with log.open("w") as f:
        status = subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error", filename],
            cwd=folder,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if status.returncode:
        raise RuntimeError(f"LaTeX failed; see {log}")
