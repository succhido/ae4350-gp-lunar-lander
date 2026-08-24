"""The evolutionary loop. Implements plan section 3.5.

This is the file that actually differs from the CS4205 symbolic-regression
version -- and only in evaluate_population(), where fitness comes from flying
the lander instead of from MSE against a data table.

Run me:  python engine/gp.py
"""

import os
import sys

import numpy as np
import gymnasium as gym

#  on the path, so `rollout` and `engine.*` both resolve whether
# this file is imported as engine.gp or run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.policy import random_population                # noqa: E402
from engine.selection import tournament_selection, elitism  # noqa: E402
from engine.operators import subtree_crossover, mutate      # noqa: E402
from rollout import ENV_ID, TRAIN_SEEDS, HELDOUT_SEEDS, fitness, mean_return  # noqa: E402


# Score given to a controller whose trees blow up numerically. Must be finite
# (nan/inf would poison min/mean) and worse than any real lander -- the worst
# genuine returns sit around -1000, so fitness tops out near +1000.
PENALTY = 1e4


def structure_key(individual):
    """Identity of an individual by STRUCTURE, not by object.

    Two individuals with the same pair of trees fly identically -- the rollout
    is deterministic given fixed seeds -- so they must score identically. This
    string is what lets the cache recognise that.
    """
    return f"{individual.tree_main}|{individual.tree_lat}"


def evaluate_population(population, seeds, env=None, parsimony=0.0, cache=None):
    """Set .fitness on every individual that doesn't already have one.

    Returns (n_flown, n_cached) so the caller can report how much work the
    cache saved.

    Three tiers of avoiding a rollout, cheapest first:
      1. .fitness already set -- elites carried over untouched.
      2. cache hit on the structure key -- a clone that survived crossover and
         mutation unchanged, or any individual whose exact trees were scored
         earlier in the run. copy() clears .fitness by design (so a mutated
         child can never inherit a stale score), which means roughly 8% of
         children arrive unscored despite being identical to a known parent.
         The cache recovers that without weakening the safe default.
      3. actually fly it.

    Only the RAW fitness is cached. The parsimony term is re-added per lookup
    so the same cache stays valid if lambda changes.

    Wraps the rollout so a tree that overflows to inf/nan gets a large finite
    penalty rather than killing generation 47 of 50.
    """
    n_flown = n_cached = 0

    for ind in population:
        if ind.fitness is not None:
            continue

        key = None if cache is None else structure_key(ind)
        if key is not None and key in cache:
            ind.fitness = cache[key] + parsimony * ind.size()
            n_cached += 1
            continue

        f_main, f_lat = ind.as_callables()
        try:
            raw = fitness(f_main, f_lat, seeds=seeds, env=env)
            if not np.isfinite(raw):
                raw = PENALTY
        except (OverflowError, FloatingPointError, ValueError):
            raw = PENALTY
        n_flown += 1

        if key is not None:
            cache[key] = raw
        ind.fitness = raw + parsimony * ind.size()

    return n_flown, n_cached


def evolve(n_pop=200, n_gen=50, k=3, p_crossover=0.9, p_mutation=0.15,
           max_depth=5, max_depth_cap=8, n_elites=1, parsimony=0.0,
           seed=0, seeds=None, use_cache=True, log=True,
           monitor=None, monitor_every=1,
           checkpoint=None, checkpoint_every=10):
    """Main loop (plan section 3.5). Returns (best_individual, history).

    max_depth     -- depth budget for INITIALISATION and subtree mutation
    max_depth_cap -- hard ceiling crossover children may not exceed. Kept
                     separate and looser: clamping offspring to the initial
                     depth forbids the population from ever growing more
                     expressive than generation 0.

    monitor       -- optional callable (gen, best_individual, env) -> dict.
                     Whatever it returns is merged into that generation's
                     history entry. Added for the convergence experiment: the
                     history below is ALL training-side, so a longer run can
                     only ever show that training fitness kept falling -- it
                     cannot show whether HELD-OUT return was still improving,
                     which is the thing actually worth knowing. Kept as a
                     callback rather than hardcoding held-out seeds here so
                     engine/ stays independent of rollout.py's seed protocol.
    monitor_every -- call it every N generations (1 = every generation).

    checkpoint    -- optional callable (gen, history, best_so_far) -> None,
                     called every checkpoint_every generations. Exists because
                     history is otherwise held in memory until the run ENDS:
                     a machine losing power at generation 128 of 150 destroyed
                     four runs and ~20 h of compute, logs preserving only the
                     training fitness that gets printed. Anything the caller
                     wants to survive a crash has to be written as it goes.
    checkpoint_every -- how often to call it. Keep well above 1; the point is
                     crash insurance, not a second results file per generation.

    history is what the report's learning curves are drawn from: per generation
    the best fitness, mean fitness and mean tree size.

    ONE env is made here and reused for the whole run -- gym.make() per rollout
    is a large avoidable cost over ~500k episodes.
    """
    seeds = TRAIN_SEEDS if seeds is None else seeds
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)
    history = []

    # structure -> raw fitness, for the whole run. Bounded by the number of
    # DISTINCT tree pairs ever produced, so at most n_pop * n_gen entries.
    cache = {} if use_cache else None

    try:
        population = random_population(rng, n_pop, max_depth=max_depth)
        evaluate_population(population, seeds, env, parsimony, cache)

        for gen in range(n_gen):
            next_pop = elitism(population, n_elites)

            while len(next_pop) < n_pop:
                pa = tournament_selection(rng, population, k)
                pb = tournament_selection(rng, population, k)

                if rng.random() < p_crossover:
                    ca, cb = subtree_crossover(rng, pa, pb, max_depth_cap)
                else:
                    ca, cb = pa.copy(), pb.copy()

                if rng.random() < p_mutation:
                    ca = mutate(rng, ca, max_depth=max_depth,
                                max_depth_cap=max_depth_cap)
                if rng.random() < p_mutation:
                    cb = mutate(rng, cb, max_depth=max_depth,
                                max_depth_cap=max_depth_cap)

                next_pop.extend([ca, cb])

            population = next_pop[:n_pop]     # trim the odd overshoot
            n_flown, n_cached = evaluate_population(population, seeds, env,
                                                    parsimony, cache)

            fits = np.array([ind.fitness for ind in population])
            sizes = np.array([ind.size() for ind in population])
            entry = {
                "gen": gen,
                "best": float(fits.min()),
                "mean": float(fits.mean()),
                "mean_size": float(sizes.mean()),
                "flown": n_flown,
                "cached": n_cached,
            }
            # Reuse the run's env: monitor rollouts are cheap (10 episodes)
            # next to a generation's ~4000, but gym.make() per call is not.
            if monitor is not None and gen % monitor_every == 0:
                champion = min(population, key=lambda ind: ind.fitness)
                entry.update(monitor(gen, champion, env))
            history.append(entry)
            if checkpoint is not None and (gen + 1) % checkpoint_every == 0:
                checkpoint(gen, history,
                           min(population, key=lambda ind: ind.fitness))
            if log:
                h = history[-1]
                print(f"gen {gen:3d}  best {h['best']:9.2f}  "
                      f"mean {h['mean']:9.2f}  mean size {h['mean_size']:6.1f}"
                      f"  flown {n_flown:4d}  cached {n_cached:4d}")
    finally:
        env.close()

    return min(population, key=lambda ind: ind.fitness), history


if __name__ == "__main__":
    # Smoke run: proves the loop turns over. Expect nonsense controllers at
    # this size -- this is testing plumbing, not performance.
    best, history = evolve(n_pop=30, n_gen=5, seed=0)

    print("\nbest individual (fitness = %.2f, size = %d):" % (best.fitness, best.size()))
    print(best)

    f_main, f_lat = best.as_callables()
    env = gym.make(ENV_ID)
    try:
        print(f"\n  TRAIN   mean return = {mean_return(f_main, f_lat, TRAIN_SEEDS, env):8.2f}")
        print(f"  HELDOUT mean return = {mean_return(f_main, f_lat, HELDOUT_SEEDS, env):8.2f}")
    finally:
        env.close()
