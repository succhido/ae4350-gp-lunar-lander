"""Step 1 of the GP project: score ONE controller on ONE episode.

Nothing evolutionary lives here. This is the function the whole GP loop
will eventually call thousands of times, so it has to be right first.

A "controller" here is just two Python functions:
    f_main(obs) -> float     (raw, pre-tanh, main engine)
    f_lat(obs)  -> float     (raw, pre-tanh, lateral engines)
Later these get replaced by GP trees' __call__ methods. The rest of this
file will not need to change when that happens -- that is the point.

Run me:  python rollout.py
"""

import numpy as np
import gymnasium as gym

ENV_ID = "LunarLanderContinuous-v3"

# Fixed training seeds. These NEVER change once results are in the report:
# every candidate is judged on exactly these scenarios, so score differences
# come from the controller and not from luck.
#
# Widened 5 -> 20 after the 5-seed sweep (GP seeds 0-4) showed a systematic
# overfitting signature: mean TRAIN return +125.7 against mean HELDOUT -70.4.
# Five scenarios are few enough to memorise; twenty are not. Costs 4x the
# episodes per evaluation, which is the whole price of the fix.
# run_single.py --n-train-seeds 5 slices this back to the old (0,1,2,3,4)
# for the report's before/after comparison.
TRAIN_SEEDS = tuple(range(20))

# Held-out seeds -- the "test set". Only the final best controller touches these.
HELDOUT_SEEDS = (100, 101, 102, 103, 104, 105, 106, 107, 108, 109)


def rollout(f_main, f_lat, seed, env=None, render=False):
    """Fly one episode with this controller. Return the episode RETURN.

    Return = sum of rewards over the episode. Higher is better
    (random ~ -200, solved ~ +200). Sign flipping happens in fitness(),
    not here -- keep exactly one place in the codebase that negates.

    TODO 1: get an env. If `env` is None, make one
            (render_mode="human" if render else None).
    TODO 2: obs, info = env.reset(seed=seed)   <- the reproducibility hinge
    TODO 3: loop until terminated or truncated:
              - a0 = np.tanh(f_main(obs))      <- squash into [-1, 1]
              - a1 = np.tanh(f_lat(obs))
              - obs, reward, terminated, truncated, info = env.step([a0, a1])
              - accumulate reward
    TODO 4: return the accumulated total.

    Careful: env.step wants an array-like of 2 floats. And do NOT call
    env.close() here if the env was passed in -- the caller owns it.
    """
    return rollout_with_outcome(f_main, f_lat, seed, env, render)[0]


def rollout_with_outcome(f_main, f_lat, seed, env=None, render=False):
    """Fly one episode. Return (episode_return, outcome).

    outcome is one of:
        "land"    -- terminated with the +100 bonus (at rest on the pad)
        "crash"   -- terminated with the -100 penalty
        "timeout" -- truncated by the 1000-step TimeLimit, i.e. still flying

    Why this exists: return alone cannot tell a landing from a HOVER. A policy
    that flies to just above the pad and holds there until truncation banks the
    shaping reward and dodges the -100 for crashing, paying only 0.03/step of
    lateral fuel. Measured on the lambda=0 sweep, GP seed 3 scored 108.2 mean
    return while landing 0 times in 30 episodes -- above that lambda's 5-seed
    mean. So a rising return curve is ambiguous unless outcomes are recorded
    beside it.

    Classifying on the final reward is exact, not a heuristic: the env ASSIGNS
    (not adds) reward = -100 / +100 on the two terminating branches, so a
    terminated episode's last reward is exactly one of those two values.
    See gymnasium/envs/box2d/lunar_lander.py, the `terminated = True` branches.
    """
    made_one = False
    if env is None:
        env = gym.make(ENV_ID, render_mode="human" if render else None)
        made_one = True

    obs, _ = env.reset(seed=seed)
    done = False
    cum_reward = 0
    reward = 0.0
    truncated = False

    try:
        while not done:
            action = [np.tanh(f_main(obs)), np.tanh(f_lat(obs))]
            obs, reward, terminated, truncated, _ = env.step(action)
            cum_reward += reward
            done = terminated or truncated

    finally:
        if made_one:
            env.close()

    if truncated:
        outcome = "timeout"
    elif reward <= -99.0:
        outcome = "crash"
    else:
        outcome = "land"
    return cum_reward, outcome


def outcome_breakdown(f_main, f_lat, seeds=HELDOUT_SEEDS, env=None):
    """Fly `seeds` once each -> {"mean_return", "n_land", "n_crash",
    "n_timeout"}.

    Same episodes mean_return() would fly, so recording outcomes alongside a
    return costs nothing extra -- call this INSTEAD of mean_return, never as
    well as it.
    """
    rets, counts = [], {"land": 0, "crash": 0, "timeout": 0}
    for s in seeds:
        r, outcome = rollout_with_outcome(f_main, f_lat, s, env=env)
        rets.append(r)
        counts[outcome] += 1
    return {"mean_return": float(np.mean(rets)),
            "n_land": counts["land"],
            "n_crash": counts["crash"],
            "n_timeout": counts["timeout"]}


def mean_return(f_main, f_lat, seeds=TRAIN_SEEDS, env=None):
    """Average episode return over several fixed seeds.

    TODO: call rollout() once per seed, return the mean.
    One seed is not enough -- that is judging a model on a single data point.
    """
    return np.mean([rollout(f_main, f_lat, seed = s, env=env) for s in seeds])


def fitness(f_main, f_lat, seeds=TRAIN_SEEDS, env=None):
    """What GP MINIMISES. Lower is better, so we negate the return.

    The parsimony penalty (+ lambda * tree_size) gets added here later,
    once individuals are actually trees and have a size.
    """
    return -mean_return(f_main, f_lat, seeds=seeds, env=env)


# --------------------------------------------------------------------------
# Stand-in "controllers" -- hand-written now, GP-evolved later.
# obs = [x, y, vx, vy, angle, angular_vel, leg_left, leg_right]
# --------------------------------------------------------------------------

def random_main(obs):
    return np.random.uniform(-1, 1)


def random_lat(obs):
    return np.random.uniform(-1, 1)


# Weights found by random search over TRAIN_SEEDS (1500 samples, uniform).
# This is the "linear baseline" of the report: a = tanh(W @ obs + b).
W_MAIN = (-5.62, -5.854, -1.115)          # [y, vy, bias]
W_LAT = (-4.735, -1.774, 2.238, 5.412)      # [x, vx, angle, angular_vel]


def linear_main(obs):
    """Main engine: fire harder the faster you are falling / the lower you are.

    -1.59*y   : the lower you are, the less negative this is -> throttle up
    -2.72*vy  : vy < 0 means falling, so this term goes POSITIVE -> brake
    -0.09     : bias, keeps the engine off when already stable
    """
    y, vy = obs[1], obs[3]
    return W_MAIN[0] * y + W_MAIN[1] * vy + W_MAIN[2]


def linear_lat(obs):
    """Lateral engines: cancel tilt and spin, and steer back over the pad.

    2.70*angle + 5.77*angular_vel  is a PD pair -- the angular_vel term is the
    damping/derivative term, without which the lander oscillates.
    1.48*x - 4.42*vx               steers back toward x = 0 (the pad).
    """
    x, vx, angle, ang_vel = obs[0], obs[2], obs[4], obs[5]
    return (W_LAT[0] * x + W_LAT[1] * vx
            + W_LAT[2] * angle + W_LAT[3] * ang_vel)


if __name__ == "__main__":
    env = gym.make(ENV_ID)

    print(f"env: {ENV_ID}")
    print(f"train seeds: {TRAIN_SEEDS}\n")

    for name, fm, fl in [
        ("random", random_main, random_lat),
        ("linear", linear_main, linear_lat),
    ]:
        try:
            r = mean_return(fm, fl, env=env)
            print(f"{name:8s}  mean return = {r:8.1f}   fitness = {-r:8.1f}")
        except NotImplementedError as e:
            print(f"{name:8s}  not implemented yet ({e})")

    env.close()
