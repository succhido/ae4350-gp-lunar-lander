"""Watch GP-evolved controller(s) fly LunarLander in human-render mode.

Select trees two ways:
  --lam / --gp-seed   numeric, direct: any (lambda, GP seed) cell from the
                       sweep, cartesian product if you pass several of each.
  --tag                nickname from TREES below, or a raw sweep tag.
--lam wins if both are given. Pass several --seeds to fly each selected tree
across multiple lander scenarios; pass several --lam/--gp-seed/--tag to watch
several trees back to back, one policy printout per tree.

One render window is opened ONCE and reused for every seed/tree in the run
(never recreated mid-run) -- recreating the window per seed is what caused
episodes after the first to silently stop showing.

Run me:
    python analysis/watch_tree.py
    python analysis/watch_tree.py --lam 0.5 --gp-seed 3 --seeds 100 101
    python analysis/watch_tree.py --lam 0 0.2 0.5 --gp-seed 1 --seeds 100
    python analysis/watch_tree.py --tag lam0.2_seed1 --seeds 100 101 5
"""
import argparse
import os
import sys

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rollout import rollout, ENV_ID              # noqa: E402
from analysis.tree_io import load_controller     # noqa: E402

# Nicknames for the trees worth re-watching without remembering raw sweep
# tags. --lam/--gp-seed below covers every other cell directly.
TREES = {
    "lam0.2_seed1": "pop200_gen50_seed1_ts20_lam0.2",   # train 129.8  heldout 134.5  size 53
    "lam0.2_seed3": "pop200_gen50_seed3_ts20_lam0.2",   # train 135.2  heldout 151.3  size 72
    "lam0_seed1":   "pop200_gen50_seed1_ts20_lam0",     # train  85.5  heldout  37.6  size 84
    "lam0.5_seed3": "pop200_gen50_seed3_ts20_lam0.5",   # train 123.5  heldout 107.6  size 61
    "lam0.5_seed5": "pop200_gen50_seed5_ts20_lam0.5",   # train  51.5  heldout  -33.9  size 25
}

DEFAULT_TAG = "lam0.2_seed3"
DEFAULT_SEEDS = [1, 2, 3, 100, 101, 102]  # a few TRAIN, a few HELDOUT


def resolve_tags(args):
    if args.lam:
        gp_seeds = args.gp_seed or [1]
        return [f"pop200_gen50_seed{s}_ts20_lam{lam:g}"
                for lam in args.lam for s in gp_seeds]
    tags = args.tag or [DEFAULT_TAG]
    return [TREES.get(t, t) for t in tags]


def watch_one(env, tag, seeds):
    f_main, f_lat, meta = load_controller(tag)

    print(f"\n{'=' * 60}")
    print(f"watching: {tag}")
    print(f"  lambda={meta['parsimony']:g}  size={meta['size']}  "
          f"train_return={meta['train_return']:.1f}  "
          f"heldout_return={meta['heldout_return']:.1f}")
    print(f"\n  main policy (throttle, pre-tanh):\n    {meta['tree_main']}")
    print(f"\n  lateral policy (pre-tanh):\n    {meta['tree_lat']}")
    if meta["exact"]:
        print("\n  (exact: loaded from results/exact_<tag>.json, full float64")
        print("   precision -- this flight reproduces train/heldout_return exactly.)")
    else:
        print("\n  (approximate: constants rounded to 3 sig figs for readability --")
        print("   this flight uses that same rounded reconstruction, so its")
        print("   return will be close to but not exactly train/heldout_return.")
        print("   run analysis/extract_exact.py for this tag to get exact constants.)")
    print(f"\n  seeds: {seeds}\n")

    for seed in seeds:
        regime = "TRAIN" if seed < 20 else "HELDOUT" if seed >= 100 else "?"
        r = rollout(f_main, f_lat, seed=seed, env=env)
        print(f"  seed {seed:3d} [{regime:7s}]: return = {r:8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=None,
                    help="parsimony value(s), e.g. --lam 0.2 0.5")
    ap.add_argument("--gp-seed", type=int, nargs="+", default=None,
                    help="GP seed(s) 1-5, paired with --lam (default: 1)")
    ap.add_argument("--tag", nargs="+", default=None,
                    help=f"nickname(s) ({list(TREES)}) or raw sweep tag(s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="lander scenario seeds to fly for EACH selected tree")
    ap.add_argument("--fps", type=int, default=50,
                    help="render speed; physics is unchanged, only the pygame "
                         "clock (50 = real time, 200 = 4x)")
    args = ap.parse_args()

    tags = resolve_tags(args)
    env = gym.make(ENV_ID, render_mode="human")
    # metadata is a class-level dict; copy before overriding so the change
    # cannot leak into other envs created in the same process.
    env.unwrapped.metadata = {**env.unwrapped.metadata, "render_fps": args.fps}
    try:
        for tag in tags:
            watch_one(env, tag, args.seeds)
    finally:
        env.close()


if __name__ == "__main__":
    main()
