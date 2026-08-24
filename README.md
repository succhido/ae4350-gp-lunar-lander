# GP-evolved symbolic controller for LunarLanderContinuous-v3

Code and saved results for the AE4350 Bio-Inspired Intelligence assignment
report by Leonardo Sigolo (TU Delft, 2026). Genetic Programming evolves a pair
of expression trees, one per engine command, whose fitness is the mean episode
return over a fixed set of training scenarios. Every number, table, and figure
in the report can be regenerated from this repository.

## Setup

Python 3.11 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
```

## Layout

| Path | Contents |
|---|---|
| `rollout.py` | Episode rollout, fitness, outcome classification (land/crash/timeout), fixed TRAIN/HELDOUT seed protocol, linear baseline |
| `engine/` | The GP core: tree representation, ramped half-and-half init, tournament selection, role-matched crossover, the four mutations, the evolve loop |
| `experiments/` | `run_single.py` (one configured run) and `run_sweep.py` (parallel grid of runs) |
| `analysis/` | Figures/tables generator, exact-tree re-extraction, convergence plots, human-render viewer |
| `results/` | The complete result set of all 42 runs reported: per-generation history, champion trees (exact full-precision forms included), train and held-out returns |
| `watch_lander.py` | Smoke test: opens a render window with a random pilot |

## Reproducing the report

Everything is deterministic: a run is fully determined by its configuration and
GP seed, so any result file can be reproduced bit-for-bit by re-running its
config (the `config` block is stored inside every JSON).

Regenerate every figure and table from the saved results (about two minutes;
outputs land in `figures/`):

```
python analysis/make_report_figs.py
```

Watch an evolved champion fly (opens a window):

```
python analysis/watch_tree.py --lam 0 --gp-seed 5 --seeds 100 101 102
```

Re-run a single evolution (the λ=0.2, seed 3 cell shown; roughly two hours):

```
python experiments/run_single.py --seed 3 --parsimony 0.2 --heldout-every 1
```

Quick smoke test of the whole loop (small population, five generations):

```
python engine/gp.py
```

The full grid behind the report (six parsimony coefficients × five GP seeds at
50 generations, plus 150-generation convergence runs and the 5-training-seed
comparison batch) is what `results/` already contains; re-running it end to end
costs roughly 100 CPU-hours via `experiments/run_sweep.py`.

## Result files

`results/single_pop200_gen<G>_seed<S>_ts<N>_lam<L>.json` holds one run:
config, per-generation history (best/mean fitness, mean size, and for the
monitored runs the champion's held-out return and land/crash/timeout counts),
and the final champion. Champion trees are stored both as readable
3-significant-figure strings and as exact full-precision dictionaries; analysis
always uses the exact forms. `results/exact_*.json` are full-precision
re-extractions for the six runs recorded before exact serialisation was added.
