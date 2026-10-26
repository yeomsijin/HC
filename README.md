# HC

Reproduction code for *Homological Causality in Dynamical Systems*.

HC compares the persistent homology of two systems' own neighborhoods. This
repository contains the Lorenz experiments, HC/TC/CCM comparisons, figure sources
and reference results used in the paper. PCE and TC are imported from
[PCE](https://github.com/yeomsijin/PCE), pinned to commit
`c8e1ce36fa531500437b664036ac395fa09b6cb8`.

## Install

Python 3.9–3.12 and Git are required. Run from this repository:

```sh
git clone https://github.com/yeomsijin/HC.git
cd HC
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Numerical runs use one process and one numerical-library thread. Completed
recordings and score files are reused when a run is restarted.

## Rebuild the figures

The `reference/` directory contains the per-seed scores, selected point clouds,
diagrams and cycle fillings underlying the paper figures. No simulation is needed
for this route:

```sh
python -m hc figures
```

The six assembled vector PDFs are written to `results/figures/Fig1.pdf` through
`Fig6.pdf`. Assembly uses `pdflatex` with the `standalone`, `newtx`, `pgf` and
`preview` packages. Panel headings are native LaTeX. No manuscript is modified.

For Python-only panels, without assembling the LaTeX headings:

```sh
python -m hc figures --figure 3 4 5 6 --assets-only
```

| Figure | Contents | Reference inputs |
| --- | --- | --- |
| 1 | Filtration, barcodes and persistence diagram | `figures/filtration.tex` |
| 2 | Two-loop PCE example | `fig2_*` |
| 3 | Coupling, observation symmetry and speed asymmetry | `fig3_*.csv` |
| 4 | Own and time-induced neighborhoods | `fig4_geometry.npz` |
| 5 | Embedding grid and window-size comparison | `fig5_*.csv` |
| 6 | Coupling-direction reversals and snapshots | `fig6_*` |

## Recalculate the experiments

```sh
python -m hc illustration
python -m hc stationary
python -m hc embedding
python -m hc switching
python -m hc figures --source results --output results/recalculated_figures
```

These commands generate the observations, apply KDE-CDF preprocessing, select
neighbors, compute persistence diagrams, evaluate the methods and save CSV/NPZ
results. Full sweeps take substantially longer than rebuilding the figures.

For a short run on the actual paper grid:

```sh
python -m hc stationary --case coupling --seed 321023 --limit 1 --output results/check
python -m hc switching --seed 481025 --limit 1 --output results/check
```

`--limit` restricts completed parameter settings or windows; it does not shorten
a recording, change the neighbor library or reduce the evaluation times. Partial
runs are not substitutes for the full paper results. Use the same command without
`--limit` to continue. `--seed` selects an initial condition; it does not create a
new parameter grid.

## Settings

The complete configuration is in `configs/paper.json`.

- Stationary recordings: 32,768 observations, sampling interval 0.02, RK4 step
  0.002, transient 240, seeds 321023–321025. Observations are `X=x1` and
  `Y=y3+a*y1`. The coupling is outside the reciprocal response clock.
- Figure 3: `m=6`, `tau=4`, `k=40`, 512 evaluation times. Coupling uses
  `g=2,...,30`; symmetry uses `a=j/30`, `j=0,...,30`; speed asymmetry uses
  `q=1,...,4` in steps of 0.1. Fixed controls are `(q,a)=(1,1)`, `(g,q)=(24,1)`
  and `(g,a)=(2,1)`, respectively.
- Figure 5: all `m=3,...,12`, `tau=1,...,12`, `k=40`. The three cases use
  `(g,q,a)=(24,1,1)`, `(24,1,0.03)` and `(2,4,1)`. All embeddings share 512
  interior physical sample indices, referring to the last delay coordinate.
- Figure 6: `g=2`, `q=2`, `a=1`, RK4 step 0.001, six seeds 481025–481030.
  Continuous forward/reverse/forward phases last 327.68 time units each. Centered
  windows contain 8,192 observations, move by 512 observations and use 128 common
  evaluation times. Preprocessing uses the whole recording; this is an offline
  comparison. The state is never reset at a reversal.

HC and TC sum directional costs or log expansions before forming a score.
Figures average the resulting per-seed scores. Undefined values remain undefined;
means do not silently exclude them. Positive scores favor `X -> Y`.

HC uses own–own H0/H1 PCE with pairwise bandwidth and response scale 5. Figure 2
instead retains its illustrative fixed widths `(0.5, 0.5)` and calibrated response
scale. H0 uses float64 MST lifetimes; H1 uses Ripser birth/lifetime coordinates.
TC uses `tc_main_text_source_neighbors_v1`. CCM uses `m+1` simplex neighbors and
exponential distance weights, with score `(rho_XfromY-rho_YfromX)/2`.

Temporal exclusion matches the frozen implementation: HC/TC call PCE's
`neighbor_indices` with full window `(m-1)*tau`; CCM excludes
`abs(index-anchor) <= ceil((m-1)*tau/2)`. At odd spans these conventions can differ
at one boundary. No estimator definition is changed to make them identical.

## Verification

The tests compare neighbors, directional costs, mapping expansions and CCM
predictions against saved evaluations from the stationary and switching
experiments. They also check input exchange and undefined-score handling.
`PROVENANCE.json` records the upstream commit and input checksums.

Long chaotic trajectories need not be bitwise identical across platforms. Use the
bundled reference inputs to regenerate the published numerical figures exactly.
