"""One full-size GP run. The timing probe that sizes the whole sweep budget.

This is deliberately NOT a sweep. It answers two questions before 180 runs get
launched on faith:
  1. does the engine actually learn (does return climb toward a landing)?
  2. how long does one run cost?  (2) sets the budget for plan section 4.3.

Run me (background, unbuffered so the log updates live):
    python experiments/run_single.py

Writes results/single_<tag>.json -- per-generation history plus
the best controller, ready for notebooks/results.ipynb.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.gp import evolve                                    # noqa: E402
from rollout import (ENV_ID, TRAIN_SEEDS, HELDOUT_SEEDS,          # noqa: E402
                     mean_return, outcome_breakdown)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pop", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parsimony", type=float, default=0.0)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-train-seeds", type=int, default=len(TRAIN_SEEDS),
                    help="how many of TRAIN_SEEDS to evaluate on (default: all). "
                         "Pass 5 to reproduce the original 5-seed config.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--heldout-every", type=int, default=0,
                    help="evaluate the generation champion on HELDOUT_SEEDS "
                         "every N generations and record it in history "
                         "(0 = off, the pre-convergence-experiment behaviour). "
                         "Costs 10 episodes per sampled generation against "
                         "~4000 spent evolving it, so 1 is affordable.")
    args = ap.parse_args()

    # Slice, never reshuffle: --n-train-seeds 5 must give back exactly the old
    # (0,1,2,3,4) so the before/after comparison is against the same scenarios.
    train_seeds = TRAIN_SEEDS[:args.n_train_seeds]

    # parsimony and seed-count belong in the tag: without them the lambda sweep
    # writes every run to the same file and silently overwrites itself.
    tag = args.tag or (f"pop{args.n_pop}_gen{args.n_gen}_seed{args.seed}"
                       f"_ts{len(train_seeds)}_lam{args.parsimony:g}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"single_{tag}.json")

    print(f"config: n_pop={args.n_pop} n_gen={args.n_gen} k={args.k} "
          f"max_depth={args.max_depth} parsimony={args.parsimony} seed={args.seed}")
    print(f"train seeds: {train_seeds}")
    print(f"budget: {args.n_pop * args.n_gen:,} evaluations "
          f"x {len(train_seeds)} rollouts = "
          f"{args.n_pop * args.n_gen * len(train_seeds):,} episodes (upper bound;\n"
          f"        elites and unchanged clones are cached, so the true count is lower)")
    print(f"output: {out_path}\n", flush=True)

    # ---- held-out monitoring hook (convergence experiment) -------------------
    # Without this, a 150-generation run can only show that TRAINING fitness
    # kept falling. The question it is meant to answer -- does lambda=0.2 reach
    # a better controller, or merely reach it sooner? -- is about HELD-OUT
    # return over time, so that curve has to be recorded as the run goes.
    # The individual tracked is the best by PENALISED fitness. At lambda > 0
    # that is NOT the best by raw return -- but it is exactly what evolve()
    # hands back if the run stops at this generation, so it is the honest
    # subject for a "should I have stopped earlier?" curve. Say so in the
    # report; tracking the best-by-raw-return instead would flatter the
    # lambda > 0 runs by reporting a controller the run would not have chosen.
    def heldout_monitor(gen, champion, env):
        """(gen, best-by-fitness individual, the run's env) -> dict merged
        into this generation's history entry."""
        f_main, f_lat = champion.as_callables()
        # outcome_breakdown flies exactly the episodes mean_return would, so
        # the land/crash/timeout counts are free. They are what makes a rising
        # heldout_return curve readable: without them, "return went up" cannot
        # be told apart from "learned to hover past the 1000-step limit
        # instead of landing" -- a distinction GP seed 3 makes real.
        ho = outcome_breakdown(f_main, f_lat, HELDOUT_SEEDS, env)
        heldout_ret = ho["mean_return"]

        # train_return comes for FREE: evaluate_population sets
        #     fitness = -mean_return(train) + parsimony * size
        # so the raw return inverts exactly, with no extra episodes. Re-flying
        # the 20 training seeds every generation would cost ~3000 needless
        # episodes per run for a number already implied by fitness.
        # (history["best"] is fitness, which mixes in the size penalty and so
        # is NOT comparable to heldout_return -- without this the per-
        # generation generalisation GAP cannot be plotted at all.)
        train_ret = float(args.parsimony * champion.size() - champion.fitness)

        # Guard that inversion once, since it hardcodes evaluate_population's
        # formula from a different module. 20 episodes, first sampled
        # generation only. Also catches a PENALTY=1e4 champion, where the
        # stored fitness is a sentinel and the inversion is meaningless.
        if gen == 0:
            check = float(mean_return(f_main, f_lat, train_seeds, env))
            if abs(check - train_ret) > 1e-6:
                print(f"WARNING: train_return inversion disagrees with a real "
                      f"rollout ({train_ret:.4f} vs {check:.4f}) -- the "
                      f"fitness formula assumed here is wrong, treat every "
                      f"train_return in this run's history as unusable.",
                      flush=True)

        return {"heldout_return": heldout_ret, "train_return": train_ret,
                "heldout_land": ho["n_land"],
                "heldout_crash": ho["n_crash"],
                "heldout_timeout": ho["n_timeout"]}

    monitor = heldout_monitor if args.heldout_every > 0 else None

    # ---- crash insurance -----------------------------------------------------
    # history only reaches disk when the run ENDS. A power cut at generation 128
    # of 150 destroyed four runs outright, the logs keeping just the training
    # fitness that happened to be printed. Checkpoints make the worst case
    # "lose the last few generations" instead of "lose the run".
    def write_payload(path, best_ind, hist, elapsed_sec, train_r, heldout_r,
                      complete):
        payload = {
            "config": vars(args),
            "complete": complete,          # False => a checkpoint, run died
            "elapsed_sec": elapsed_sec,
            "history": hist,
            "best": {
                "fitness": float(best_ind.fitness),
                "size": int(best_ind.size()),
                "tree_main": str(best_ind.tree_main),
                "tree_lat": str(best_ind.tree_lat),
                "tree_main_dict": best_ind.tree_main.to_dict(),
                "tree_lat_dict": best_ind.tree_lat.to_dict(),
                "tree_main_full": best_ind.tree_main.full_str(),
                "tree_lat_full": best_ind.tree_lat.full_str(),
                "train_return": train_r,
                "heldout_return": heldout_r,
            },
        }
        # Write-then-rename: a crash DURING the write must not leave a
        # half-written file where a valid one used to be. os.replace is atomic
        # on Windows and POSIX alike.
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)

    ckpt_path = os.path.join(RESULTS_DIR, f"partial_{tag}.json")

    def checkpoint(gen, hist, best_so_far):
        # train_return/heldout_return are left None here: recomputing them
        # costs real episodes, and the last monitored history entry already
        # carries both. The final write fills them in properly.
        write_payload(ckpt_path, best_so_far, hist, time.time() - t0,
                      None, None, complete=False)

    t0 = time.time()
    best, history = evolve(n_pop=args.n_pop, n_gen=args.n_gen, k=args.k,
                           max_depth=args.max_depth, parsimony=args.parsimony,
                           seed=args.seed, seeds=train_seeds, log=True,
                           monitor=monitor,
                           monitor_every=max(1, args.heldout_every),
                           checkpoint=checkpoint, checkpoint_every=5)
    elapsed = time.time() - t0

    # Held-out check: the control analogue of a test set. A large train/heldout
    # gap means the policy memorised the five training scenarios.
    f_main, f_lat = best.as_callables()
    env = gym.make(ENV_ID)
    try:
        train_ret = float(mean_return(f_main, f_lat, train_seeds, env))
        heldout_ret = float(mean_return(f_main, f_lat, HELDOUT_SEEDS, env))
    finally:
        env.close()

    print(f"\nwall clock       : {elapsed/60:.1f} min ({elapsed/args.n_gen:.1f} s/generation)")
    print(f"best fitness     : {best.fitness:.2f}   size {best.size()} nodes")
    print(f"TRAIN   return   : {train_ret:8.2f}   (seeds {train_seeds})")
    print(f"HELDOUT return   : {heldout_ret:8.2f}   (seeds {HELDOUT_SEEDS})")
    print(f"\nbest controller:\n{best}")

    # What one run costs -> what the full section 4.3 sweep would cost.
    sweep_runs = 180
    print(f"\nsweep projection : {sweep_runs} runs x {elapsed/3600:.2f} h = "
          f"{sweep_runs * elapsed / 3600:.0f} h serially "
          f"({sweep_runs * elapsed / 3600 / os.cpu_count():.0f} h on "
          f"{os.cpu_count()} cores)")

    # Same writer as the checkpoints, so the two formats cannot drift apart.
    # On tree_main/tree_lat vs the *_dict/*_full fields, see write_payload:
    # the strings are 3 s.f. and lossy (~70% of trees), the dicts round-trip
    # exactly. fitness/size/returns were always exact -- computed on the real
    # Node objects, never via a string.
    write_payload(out_path, best, history, elapsed, train_ret, heldout_ret,
                  complete=True)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)      # the real file exists now; drop the stub
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
