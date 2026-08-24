"""Baseline: find the linear-controller weights by random search.

This IS the "how did you get those numbers" story behind linear_main /
linear_lat in rollout.py -- they were not designed, they were found by
exactly this script. Re-run it and you should reproduce (up to RNG) a
mean_return in the same ballpark (250+).

A linear controller is just 7 real numbers:
    f_main(obs) = w_main[0]*y  + w_main[1]*vy    + w_main[2]
    f_lat(obs)  = w_lat[0]*x   + w_lat[1]*vx
                + w_lat[2]*angle + w_lat[3]*angular_vel
So "search over controllers" here just means "search over 7 numbers,"
score each with the fitness() you already built, keep the best.

Run me:  python random_search_linear.py
"""

import numpy as np
import gymnasium as gym

from rollout import ENV_ID, TRAIN_SEEDS, mean_return

N_SAMPLES = 10             # how many random weight vectors to try
SAMPLE_RANGE = (-6.0, 6.0)   # uniform range each weight is drawn from
SEED = 1                     # RNG seed for the SEARCH (not the environment)


def make_linear_controller(w_main, w_lat):
    """TODO: given weight tuples w_main=(a, b, c) and w_lat=(a, b, c, d),
    return (f_main, f_lat) -- two functions with signature f(obs) -> float.

    obs = [x, y, vx, vy, angle, angular_vel, leg_left, leg_right]

    f_main(obs) = w_main[0]*obs[1] + w_main[1]*obs[3] + w_main[2]
    f_lat(obs)  = w_lat[0]*obs[0] + w_lat[1]*obs[2]
                + w_lat[2]*obs[4] + w_lat[3]*obs[5]

    Hint: define two small local functions (or lambdas) inside this
    function and return the pair -- they "close over" w_main/w_lat.
    """
    def f_main_local(obs):
        return w_main[0]*obs[1] + w_main[1]*obs[3] + w_main[2]

    def f_lat_local(obs):
        return w_lat[0]*obs[0] + w_lat[1]*obs[2] + w_lat[2]*obs[4] + w_lat[3]*obs[5]
    
    return (f_main_local, f_lat_local)


def random_search(n_samples=N_SAMPLES, seeds=TRAIN_SEEDS, env=None):
    """TODO: draw n_samples random (w_main, w_lat) pairs, score each with
    mean_return(), keep the best-scoring one.

    
    """
    rng = np.random.default_rng(SEED)
    best_score = -np.inf
    best_w_main = best_w_lat = None

    for i in range(n_samples):
        w_main = rng.uniform(*SAMPLE_RANGE, size=3)
        w_lat = rng.uniform(*SAMPLE_RANGE, size=4)
        f_main, f_lat = make_linear_controller(w_main, w_lat)
        score = mean_return(f_main, f_lat, seeds=seeds, env=env)
        if score > best_score:
            best_score, best_w_main, best_w_lat = score, w_main, w_lat
            print(f"  [{i:4d}] new best: {best_score:.1f}")
    
    return (best_w_main, best_w_lat, best_score)

    


if __name__ == "__main__":
    env = gym.make(ENV_ID)
    print(f"searching {N_SAMPLES} random linear controllers "
          f"over seeds {TRAIN_SEEDS}...\n")

    try:
        w_main, w_lat, score = random_search(env=env)
        print(f"\nbest mean return: {score:.1f}")
        print(f"W_MAIN = {tuple(round(float(w), 3) for w in w_main)}")
        print(f"W_LAT  = {tuple(round(float(w), 3) for w in w_lat)}")
    except NotImplementedError as e:
        print(f"not implemented yet ({e})")
    finally:
        env.close()
