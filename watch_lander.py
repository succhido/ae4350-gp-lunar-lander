"""Watch the Lunar Lander with a RANDOM pilot.

Run this and a game window opens. The lander fires its engines at
random, so expect crashes -- the point is just to see the environment
working and understand the loop we'll later plug a GP controller into.

Close the window or wait for 3 episodes to end.
"""

import gymnasium as gym

# render_mode="human" means "open a window and draw the simulation".
env = gym.make("LunarLanderContinuous-v3", render_mode="human")

for episode in range(3):
    # reset() starts a new episode and returns the first observation:
    # 8 numbers = [x, y, x-velocity, y-velocity, angle, angular velocity,
    #              left leg touching?, right leg touching?]
    obs, info = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        # A random action: 2 numbers in [-1, 1] =
        # [main engine throttle, left/right side engine].
        # Later, our GP-evolved controller will compute this from `obs`.
        action = env.action_space.sample()

        # step() advances the physics by one tick (1/50th of a second).
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # terminated = landed or crashed; truncated = ran out of time.
        done = terminated or truncated

    print(f"Episode {episode + 1}: total reward = {total_reward:.1f}")

env.close()
print("Done! (A good controller scores 200+; random scores far below 0.)")
