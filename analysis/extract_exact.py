"""Re-run evolve() for specific sweep cells to recover FULL-PRECISION evolved
constants. single_<tag>.json only ever stored tree_main/tree_lat through
Node.__str__, which rounds every constant to 3 significant figures (built for
report readability, not exact serialization) -- so those strings were never
an exact record of what was actually evolved.

GP is deterministic: fixed seed + fixed config (n_pop, n_gen, k, max_depth,
parsimony, train_seeds) reproduces the exact same run bit-for-bit, so
re-running evolve() with the recorded config recovers the real Node objects
without touching the lossy string at all. This script does that, checks the
recomputed fitness/size/train/heldout return against the ones already on
record (if these don't match, the environment or engine is not actually
reproducible and nothing below can be trusted), and writes the exact tree
(dict form, for lossless reload, + a full-precision infix string for the
report) to results/exact_<tag>.json.

Costs ~15-20 min per tag -- this repeats the ENTIRE evolution, not just the
final generation. Pick tags deliberately, not all 25.

Run me:
    python analysis/extract_exact.py lam0.2_seed3
    python analysis/extract_exact.py --lam 0.2 0.5 --gp-seed 1 3
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.gp import evolve                                        # noqa: E402
from rollout import ENV_ID, TRAIN_SEEDS, HELDOUT_SEEDS, mean_return  # noqa: E402
from analysis.watch_tree import TREES                                # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")

TOL = 1e-6  # exact match expected -- this is bit-reproducibility, not noise


def resolve_tags(tags, lam, gp_seed):
    if lam:
        gp_seeds = gp_seed or [1]
        return [f"pop200_gen50_seed{s}_ts20_lam{l:g}" for l in lam for s in gp_seeds]
    return [TREES.get(t, t) for t in (tags or [])]


def extract_one(tag):
    src_path = os.path.join(RESULTS_DIR, f"single_{tag}.json")
    with open(src_path) as fh:
        recorded = json.load(fh)
    cfg = recorded["config"]
    train_seeds = TRAIN_SEEDS[:cfg["n_train_seeds"]]

    print(f"\n{'=' * 60}\nre-evolving {tag}  "
          f"(seed={cfg['seed']} parsimony={cfg['parsimony']} "
          f"n_pop={cfg['n_pop']} n_gen={cfg['n_gen']})")
    t0 = time.time()
    best, _ = evolve(n_pop=cfg["n_pop"], n_gen=cfg["n_gen"], k=cfg["k"],
                     max_depth=cfg["max_depth"], parsimony=cfg["parsimony"],
                     seed=cfg["seed"], seeds=train_seeds, log=False)
    elapsed = time.time() - t0

    f_main, f_lat = best.as_callables()
    env = gym.make(ENV_ID)
    try:
        train_ret = float(mean_return(f_main, f_lat, train_seeds, env))
        heldout_ret = float(mean_return(f_main, f_lat, HELDOUT_SEEDS, env))
    finally:
        env.close()

    rec_best = recorded["best"]
    mismatches = []
    if abs(train_ret - rec_best["train_return"]) > TOL:
        mismatches.append(f"train_return {train_ret} != {rec_best['train_return']}")
    if abs(heldout_ret - rec_best["heldout_return"]) > TOL:
        mismatches.append(f"heldout_return {heldout_ret} != {rec_best['heldout_return']}")
    if best.size() != rec_best["size"]:
        mismatches.append(f"size {best.size()} != {rec_best['size']}")
    if mismatches:
        print("  DETERMINISM CHECK FAILED:")
        for m in mismatches:
            print(f"    {m}")
        raise RuntimeError(f"{tag}: re-run did not reproduce the recorded run -- "
                           "do not trust the exact constants below")
    print(f"  determinism check OK  ({elapsed / 60:.1f} min, "
          f"train={train_ret:.2f} heldout={heldout_ret:.2f} size={best.size()})")

    out_path = os.path.join(RESULTS_DIR, f"exact_{tag}.json")
    with open(out_path, "w") as fh:
        json.dump({
            "tag": tag,
            "config": cfg,
            "fitness": float(best.fitness),
            "size": int(best.size()),
            "train_return": train_ret,
            "heldout_return": heldout_ret,
            "tree_main_dict": best.tree_main.to_dict(),
            "tree_lat_dict": best.tree_lat.to_dict(),
            "tree_main_full": best.tree_main.full_str(),
            "tree_lat_full": best.tree_lat.full_str(),
        }, fh, indent=2)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=None,
                    help=f"nickname(s) ({list(TREES)}) or raw sweep tag(s)")
    ap.add_argument("--lam", type=float, nargs="+", default=None)
    ap.add_argument("--gp-seed", type=int, nargs="+", default=None)
    args = ap.parse_args()

    tags = resolve_tags(args.tags, args.lam, args.gp_seed)
    if not tags:
        ap.error("pass tag(s) or --lam/--gp-seed")
    for tag in tags:
        extract_one(tag)


if __name__ == "__main__":
    main()
