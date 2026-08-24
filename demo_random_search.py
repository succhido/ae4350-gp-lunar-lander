"""Demo: take the weights random_search_linear.py just found, score them
seed-by-seed, and WATCH the best and worst ones fly (human-render mode).

Why this is worth doing: random_search_linear.py reported a mean_return of
61.5 over TRAIN_SEEDS -- much weaker than the canonical linear_main/linear_lat
baseline in rollout.py (254.6). A single mean number hides that: this
controller might land great on some seeds and crash on others. Scanning more
seeds and watching the extremes makes that variance visible.

Run me:  python demo_random_search.py

Update W_MAIN / W_LAT below with your own random_search_linear.py output if
you re-run it (results change every run -- it's random search, no fixed seed
guarantee of reproducing the same optimum).
"""

from rollout import rollout
from random_search_linear import make_linear_controller

# Weights found by random_search_linear.py (1500 samples, this run).
W_MAIN = (-5.62, -5.854, -1.115)
W_LAT = (-4.735, -1.774, 2.238, 5.412)

SEEDS_TO_SCAN = range(20)  # wider net than TRAIN_SEEDS (0-4), to find clear cases


def main():
    f_main, f_lat = make_linear_controller(W_MAIN, W_LAT)

    print(f"scoring seeds {SEEDS_TO_SCAN.start}-{SEEDS_TO_SCAN.stop - 1} (no render)...\n")
    scored = []
    for seed in SEEDS_TO_SCAN:
        r = rollout(f_main, f_lat, seed=seed)
        scored.append((seed, r))
        print(f"  seed {seed:2d}: return = {r:8.1f}")

    scored.sort(key=lambda t: t[1])
    worst_seed, worst_r = scored[0]
    best_seed, best_r = scored[-1]

    input(f"\nworst: seed {worst_seed} (return {worst_r:.1f}) -- press Enter to watch it fail...")
    rollout(f_main, f_lat, seed=worst_seed, render=True)

    input(f"\nbest: seed {best_seed} (return {best_r:.1f}) -- press Enter to watch it land...")
    rollout(f_main, f_lat, seed=best_seed, render=True)


if __name__ == "__main__":
    main()
