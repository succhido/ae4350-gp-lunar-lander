"""Convergence plot: best/mean fitness and mean tree size vs. generation.

This is the "learning effect" figure the assignment rubric explicitly asks
for (AE4350 lecture-1 slide 17: "Show the 'learning effect'!"). Reads the
`history` list already logged by run_single.py / run_sweep.py in every
single_<tag>.json -- no new experiments needed.

Run me:
    python analysis/plot_convergence.py
    python analysis/plot_convergence.py --tags pop200_gen50_seed1_ts20_lam0 pop200_gen50_seed1_ts20_lam0.2
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")

DEFAULT_TAGS = [
    "pop200_gen50_seed1_ts20_lam0",
    "pop200_gen50_seed1_ts20_lam0.2",
    "pop200_gen50_seed1_ts20_lam0.5",
]


def load_history(tag):
    path = os.path.join(RESULTS_DIR, f"single_{tag}.json")
    with open(path) as fh:
        data = json.load(fh)
    return data["config"]["parsimony"], data["history"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "convergence.png"))
    args = ap.parse_args()

    fig, (ax_fit, ax_size) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    for tag in args.tags:
        try:
            parsimony, history = load_history(tag)
        except FileNotFoundError:
            print(f"skipping {tag}: no result file yet")
            continue

        gens = [h["gen"] for h in history]
        best = [h["best"] for h in history]
        mean = [h["mean"] for h in history]
        mean_size = [h["mean_size"] for h in history]

        label = f"lambda={parsimony:g}"
        line, = ax_fit.plot(gens, best, label=f"{label} (best)")
        ax_fit.plot(gens, mean, "--", color=line.get_color(), alpha=0.5,
                    label=f"{label} (mean)")
        ax_size.plot(gens, mean_size, color=line.get_color())

    ax_fit.set_ylabel("fitness (lower = better)")
    ax_fit.set_title("GP convergence: fitness vs. generation")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(alpha=0.3)

    ax_size.set_xlabel("generation")
    ax_size.set_ylabel("mean tree size (nodes)")
    ax_size.set_title("bloat: mean population tree size vs. generation")
    ax_size.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
