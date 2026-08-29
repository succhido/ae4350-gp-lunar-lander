"""Every chapter-4 figure and generated table, straight from the saved runs.

One script, rerunnable, so a figure can never silently disagree with the
JSONs it was drawn from. Outputs land in figures/:

  fig-convergence-gen150.pdf  held-out return + landing counts vs generation
                              for the six 150-generation runs. The "was 50
                              generations enough?" figure (it was not), and
                              the hovering-self-cure figure in one.
  fig-lambda-sweep.pdf        held-out return and champion size vs lambda,
                              per-seed points + 5-seed means (gen-50 grid).
  fig-frontier.pdf            champion size vs held-out return, every ts20
                              run. The size/performance trade-off at a glance.
  fig-traj-grid.pdf           lander x-y flight paths on all 10 held-out
                              seeds for the linear baseline and three evolved
                              champions, coloured by outcome.
  tab-lambda-sweep.tex        generated tabular ROWS for the lambda table
                              (report supplies the surrounding table env).
  tab-seedcount.tex           generated rows for the 5-vs-20 training-seed
                              comparison.

Trees are loaded via tree_io.load_controller, which prefers exact
tree_main_dict payloads (new runs) or exact_<tag>.json re-extractions (old
runs) over the lossy 3-sig-fig strings -- returns plotted here are computed
from the same Node objects the run itself scored.

Run me (from the repository root):
    python analysis/make_report_figs.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from rollout import ENV_ID, HELDOUT_SEEDS, linear_main, linear_lat   # noqa: E402
from analysis.tree_io import load_controller                         # noqa: E402

RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIG_DIR = os.path.join(CODE_DIR, "figures")

SEEDS = (1, 2, 3, 4, 5)
LAMBDAS = (0, 0.1, 0.2, 0.3, 0.5, 1)

# One colour per lambda, used consistently across every figure so the reader
# can track a coefficient between plots without re-reading legends.
LAM_COLOR = {0: "tab:blue", 0.1: "tab:cyan", 0.2: "tab:orange",
             0.3: "tab:olive", 0.5: "tab:purple", 1: "tab:red"}
OUTCOME_COLOR = {"land": "#2a9d3a", "crash": "#d62728", "timeout": "#e8a13a"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "figure.constrained_layout.use": True,
})


def save(fig, name):
    """PDF for the report, PNG preview beside it for quick eyeballing."""
    out = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def gtag(seed, lam, gens=50, ts=20):
    return f"pop{200}_gen{gens}_seed{seed}_ts{ts}_lam{lam:g}"


def load(tag):
    with open(os.path.join(RESULTS_DIR, f"single_{tag}.json")) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Figure 1: the 150-generation convergence runs
# ---------------------------------------------------------------------------

def fig_convergence():
    """Columns = lambda in {0, 0.2, 1}; top row held-out return, bottom row
    held-out landing count (with timeouts dashed). Both GP seeds per panel.

    The three claims this figure carries:
      - held-out return is still climbing at gen 50 (vertical marker), so the
        main grid stopped early;
      - the lambda ranking at gen 50 does not survive to gen 150;
      - seed 3 / lambda 0's rising return before ~gen 90 is hovering, not
        landing -- the timeout curve is doing the scoring, then flips.
    """
    lams = (0, 0.2, 1)
    seed_style = {3: ("tab:blue", "seed 3"), 5: ("tab:red", "seed 5")}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.0), sharex=True,
                             sharey="row")

    for col, lam in enumerate(lams):
        ax_ret, ax_out = axes[0, col], axes[1, col]
        for seed, (color, label) in seed_style.items():
            hist = load(gtag(seed, lam, gens=150))["history"]
            gens = [h["gen"] for h in hist]
            ax_ret.plot(gens, [h["heldout_return"] for h in hist],
                        color=color, lw=1.2, label=f"{label} held-out")
            # lw/alpha chosen so the train curve reads at print size while
            # staying visually secondary -- 0.7/0.35 was illegible on paper.
            ax_ret.plot(gens, [h["train_return"] for h in hist],
                        color=color, lw=1.0, alpha=0.55,
                        label=f"{label} train")
            ax_out.plot(gens, [h["heldout_land"] for h in hist],
                        color=color, lw=1.2, label=f"{label} landings")
            ax_out.plot(gens, [h["heldout_timeout"] for h in hist],
                        color=color, lw=0.8, ls="--", alpha=0.6,
                        label=f"{label} timeouts")
        for ax in (ax_ret, ax_out):
            ax.axvline(49, color="grey", lw=0.8, ls=":")
        ax_ret.set_title(f"$\\lambda = {lam:g}$")
        ax_out.set_xlabel("generation")
        ax_out.set_ylim(-0.5, 10.5)

    axes[0, 0].set_ylabel("mean return")
    axes[1, 0].set_ylabel("episodes / 10")
    # Legends go where the data is not: the lambda=0.2 return panel is empty
    # below ~100 after gen 70, the lambda=1 outcome panel above 6 landings.
    axes[0, 1].legend(*axes[0, 0].get_legend_handles_labels(),
                      loc="lower right")
    axes[1, 2].legend(*axes[1, 0].get_legend_handles_labels(),
                      loc="upper left")

    save(fig, "fig-convergence-gen150")


# ---------------------------------------------------------------------------
# Figure 2 + generated lambda table: the gen-50 grid
# ---------------------------------------------------------------------------

def sweep_stats():
    """{lambda: dict of per-seed lists} for the 6 x 5 gen-50 ts20 grid."""
    stats = {}
    for lam in LAMBDAS:
        rows = [load(gtag(s, lam))["best"] for s in SEEDS]
        stats[lam] = {
            "train": [r["train_return"] for r in rows],
            "heldout": [r["heldout_return"] for r in rows],
            "size": [r["size"] for r in rows],
        }
    return stats


def fig_lambda_sweep(stats):
    """Per-seed points (jittered) + 5-seed mean, on categorical lambda
    positions -- a linear lambda axis would pile 4 of 6 values into the
    left tenth of the plot."""
    xs = np.arange(len(LAMBDAS))
    fig, (ax_ret, ax_size) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    rng = np.random.default_rng(0)   # jitter only, cosmetic
    for i, lam in enumerate(LAMBDAS):
        jitter = rng.uniform(-0.10, 0.10, len(SEEDS))
        ax_ret.scatter(i + jitter, stats[lam]["heldout"], s=14,
                       color=LAM_COLOR[lam], alpha=0.8, zorder=3)
        ax_size.scatter(i + jitter, stats[lam]["size"], s=14,
                        color=LAM_COLOR[lam], alpha=0.8, zorder=3)

    ax_ret.plot(xs, [np.mean(stats[l]["heldout"]) for l in LAMBDAS],
                color="k", lw=1.2, marker="_", ms=14, label="held-out mean")
    ax_ret.plot(xs, [np.mean(stats[l]["train"]) for l in LAMBDAS],
                color="k", lw=0.8, ls="--", alpha=0.5, marker="_", ms=10,
                label="train mean")
    ax_size.plot(xs, [np.mean(stats[l]["size"]) for l in LAMBDAS],
                 color="k", lw=1.2, marker="_", ms=14, label="mean")

    for ax, ylab in ((ax_ret, "mean return"), (ax_size, "champion size (nodes)")):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
        ax.set_xlabel("$\\lambda$")
        ax.set_ylabel(ylab)
    ax_ret.legend(loc="lower left")

    save(fig, "fig-lambda-sweep")


def write_lambda_table(stats):
    """The complete tabular env (a bare row-fragment \\input inside tabular
    breaks TeX's alignment scanner: 'Misplaced \\noalign'). The report keeps
    the table env, caption and label, and \\inputs this whole tabular.

    Cells are shaded per column, darker teal = better, where better means a
    higher return but a smaller gap and a smaller size (the project's stated
    aim). Needs colortbl in the report preamble."""
    rows = []
    for lam in LAMBDAS:
        tr = np.mean(stats[lam]["train"])
        ho = np.mean(stats[lam]["heldout"])
        rows.append((lam, tr, ho, tr - ho, np.mean(stats[lam]["size"])))

    def shade(col, invert):
        vals = [r[col] for r in rows]
        lo, hi = min(vals), max(vals)
        pcts = {}
        for v in vals:
            score = (v - lo) / (hi - lo) if hi > lo else 0.0
            if invert:
                score = 1.0 - score
            pcts[v] = int(round(5 + 45 * score))
        return pcts

    # invert=True where smaller is better (gap and size).
    shades = [shade(1, False), shade(2, False), shade(3, True), shade(4, True)]

    lines = ["% GENERATED by analysis/make_report_figs.py -- do not edit by",
             "% hand, rerun the script. One row per lambda, means over GP",
             "% seeds 1-5 (gen-50, 20 training seeds).",
             "\\begin{tabular}{ccccc}",
             "    \\hline",
             "    $\\lambda$ & Train ($\\uparrow$) & Held-out ($\\uparrow$) & Gap ($\\downarrow$) & Size ($\\downarrow$) \\\\",
             "    \\hline"]
    for lam, tr, ho, gap, sz in rows:
        cells = [f"${lam:g}$"]
        for pcts, v in zip(shades, (tr, ho, gap, sz)):
            cells.append(f"\\cellcolor{{teal!{pcts[v]}}} ${v:.1f}$")
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\hline",
        "\\end{tabular}",
        "\\par\\smallskip",
        "{\\footnotesize Column headers mark which direction is better",
        "($\\uparrow$ higher, $\\downarrow$ lower). Shading follows the same",
        "direction within each column, from",
        "\\colorbox{teal!5}{\\strut worst} through",
        "\\colorbox{teal!28}{\\strut middle} to",
        "\\colorbox{teal!50}{\\strut best}: dark means a high return in the",
        "first two columns, but a low gap or size in the last two.}",
    ]
    out = os.path.join(FIG_DIR, "tab-lambda-sweep.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: size vs held-out return, every ts20 champion
# ---------------------------------------------------------------------------

def fig_frontier():
    """Lambda is grouped to three classes (none / moderate / maximum), not
    six hues: on a scatter every pair of colours is compared against every
    other, and no 6-colour palette passes an all-pairs colourblind check --
    while the chapter's narrative only ever distinguishes these three groups
    anyway. The three hues are a validated categorical triple."""
    groups = [
        ("$\\lambda = 0$", lambda lam: lam == 0, "#2a78d6"),
        ("$0 < \\lambda < 1$", lambda lam: 0 < lam < 1, "#eb6834"),
        ("$\\lambda = 1$", lambda lam: lam == 1, "#1baf7a"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for label, member, color in groups:
        xs, ys = [], []
        for lam in LAMBDAS:
            if not member(lam):
                continue
            for s in SEEDS:
                b = load(gtag(s, lam))["best"]
                xs.append(b["size"]), ys.append(b["heldout_return"])
        ax.scatter(xs, ys, s=24, color=color, label=label,
                   edgecolor="k", lw=0.3, alpha=0.9, zorder=3)
    # 150-generation champions as stars, same colour code.
    for label, member, color in groups:
        for lam in (0, 0.2, 1):
            if not member(lam):
                continue
            for s in (3, 5):
                b = load(gtag(s, lam, gens=150))["best"]
                ax.scatter(b["size"], b["heldout_return"], s=80, marker="*",
                           color=color, edgecolor="k", lw=0.4, zorder=4)
    ax.scatter([], [], s=80, marker="*", color="w", edgecolor="k", lw=0.6,
               label="150 generations")

    ax.set_xlabel("champion size (nodes, both trees)")
    ax.set_ylabel("held-out mean return")
    # No quadrant is actually empty (tried two), so the legend goes below
    # the axes -- constrained layout reserves the space for it.
    fig.legend(loc="outside lower center", ncol=4)

    save(fig, "fig-frontier")


# ---------------------------------------------------------------------------
# Figure 4: flight-path grid on the held-out seeds
# ---------------------------------------------------------------------------

def record_trajectory(f_main, f_lat, seed, env):
    """One episode -> (path Nx2 array of normalised (x, y), outcome).

    Same stepping loop and outcome classification as rollout_with_outcome
    (final reward is ASSIGNED +-100 on the terminating branches, so the
    classification is exact) -- duplicated here only to record positions,
    which the scoring path deliberately does not.
    """
    obs, _ = env.reset(seed=seed)
    path = [(obs[0], obs[1])]
    done, truncated, reward = False, False, 0.0
    while not done:
        action = [np.tanh(f_main(obs)), np.tanh(f_lat(obs))]
        obs, reward, terminated, truncated, _ = env.step(action)
        path.append((obs[0], obs[1]))
        done = terminated or truncated
    outcome = ("timeout" if truncated else
               "crash" if reward <= -99.0 else "land")
    return np.array(path), outcome


def fig_traj_grid():
    """One panel per controller, all 10 held-out episodes overlaid, coloured
    by how the episode ended. Obs coordinates are pad-relative, so the pad
    is at the origin by construction."""
    def gp(tag, title):
        f_main, f_lat, meta = load_controller(tag)
        if not meta["exact"]:
            raise RuntimeError(f"{tag}: only lossy trees available -- "
                               f"run extract_exact.py first")
        return (title, f_main, f_lat)

    controllers = [
        ("Linear baseline", linear_main, linear_lat),
        gp(gtag(3, 0, gens=150), "GP $\\lambda=0$, seed 3, gen 150"),
        gp(gtag(3, 0.2, gens=150), "GP $\\lambda=0.2$, seed 3, gen 150"),
        gp(gtag(4, 1), "GP $\\lambda=1$, seed 4, gen 50"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.1), sharex=True,
                             sharey=True)
    env = gym.make(ENV_ID)
    try:
        for ax, (title, f_main, f_lat) in zip(axes, controllers):
            counts = {"land": 0, "crash": 0, "timeout": 0}
            for seed in HELDOUT_SEEDS:
                path, outcome = record_trajectory(f_main, f_lat, seed, env)
                counts[outcome] += 1
                ax.plot(path[:, 0], path[:, 1], lw=0.7, alpha=0.8,
                        color=OUTCOME_COLOR[outcome], zorder=2)
                ax.plot(path[0, 0], path[0, 1], ".", ms=3, color="k",
                        zorder=3)
            # The landing pad: obs x,y are measured from its centre.
            ax.plot([-0.15, 0.15], [0, 0], color="k", lw=2.5,
                    solid_capstyle="butt", zorder=4)
            ax.set_title(f"{title}\n{counts['land']}/10 land, "
                         f"{counts['crash']} crash, "
                         f"{counts['timeout']} t/o", fontsize=7, pad=3)
            ax.set_xlabel("$x$")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-0.15, 1.55)
    finally:
        env.close()
    axes[0].set_ylabel("$y$")

    save(fig, "fig-traj-grid")


# ---------------------------------------------------------------------------
# Generated rows: the 5-vs-20 training-seed comparison
# ---------------------------------------------------------------------------

def write_seedcount_table():
    """The unconfounded before/after: GP seeds 0-4 at 5 training seeds vs
    GP seeds 1-5 at 20, both lambda=0, gen 50 -- the training-seed count is
    the only variable between the two batches."""
    old = [load(f"pop200_gen50_seed{s}")["best"] for s in range(5)]
    new = [load(gtag(s, 0))["best"] for s in SEEDS]

    lines = ["% GENERATED by analysis/make_report_figs.py -- do not edit by",
             "% hand, rerun the script. Means over 5 GP seeds each, lambda=0.",
             "\\begin{tabular}{cccc}",
             "    \\hline",
             "    Training seeds $N$ & Train & Held-out & Gap \\\\",
             "    \\hline"]
    for label, batch in (("$5$", old), ("$20$", new)):
        tr = np.mean([b["train_return"] for b in batch])
        ho = np.mean([b["heldout_return"] for b in batch])
        lines.append(f"    {label} & ${tr:.1f}$ & ${ho:.1f}$ & "
                     f"${tr - ho:.1f}$ \\\\")
    lines += ["    \\hline", "\\end{tabular}"]
    out = os.path.join(FIG_DIR, "tab-seedcount.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")
    for label, batch in (("5-seed", old), ("20-seed", new)):
        tr = np.mean([b["train_return"] for b in batch])
        ho = np.mean([b["heldout_return"] for b in batch])
        print(f"  {label}: train {tr:7.1f}   heldout {ho:7.1f}   "
              f"gap {tr - ho:7.1f}")


def print_key_numbers():
    """Every number chapter 4 quotes inline, printed from source so the
    prose can be checked against this output line by line."""
    from rollout import outcome_breakdown, mean_return, TRAIN_SEEDS
    env = gym.make(ENV_ID)
    try:
        tr = float(mean_return(linear_main, linear_lat, TRAIN_SEEDS, env))
        ho = outcome_breakdown(linear_main, linear_lat, HELDOUT_SEEDS, env)
    finally:
        env.close()
    print(f"linear baseline : train {tr:7.1f}   heldout {ho['mean_return']:7.1f}"
          f"   {ho['n_land']}/10 land {ho['n_crash']} crash {ho['n_timeout']} t/o")

    print("gen-150 champions (best.heldout_return / size):")
    for lam in (0, 0.2, 1):
        vals = []
        for s in (3, 5):
            b = load(gtag(s, lam, gens=150))["best"]
            vals.append(b["heldout_return"])
            print(f"  lam {lam:<4g} seed {s}: heldout {b['heldout_return']:7.1f}"
                  f"   size {b['size']:3d}")
        print(f"  lam {lam:<4g} 2-seed mean heldout: {np.mean(vals):7.1f}")

    h0 = load(gtag(3, 0, gens=150))["history"][0]
    print(f"gen-0 (seed 3, lam 0): best fitness {h0['best']:.1f}, "
          f"mean fitness {h0['mean']:.1f}  (fitness = -return at lam 0)")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    stats = sweep_stats()
    write_lambda_table(stats)
    write_seedcount_table()
    fig_lambda_sweep(stats)
    fig_frontier()
    fig_convergence()
    fig_traj_grid()      # these two fly episodes (~1 min together)
    print_key_numbers()
    print("done")
