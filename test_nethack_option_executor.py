import gymnasium as gym
import numpy as np
import nle

from nethack_option_executor import (
    DungeonMemory,
    option_explore,
    option_go_to_downstairs,
    option_descend,
)

ENV_ID = "NetHackStaircase-v0"
NUM_SEEDS = 20
MAX_EXPLORE_OPTIONS = 20

env = gym.make(ENV_ID)

successes = 0
stairs_discovered = 0
navigation_successes = 0
total_primitive_steps = []

print("=" * 65)
print("NETHACK HARD-CODED OPTION EXECUTOR TEST V2")
print("=" * 65)

for seed in range(NUM_SEEDS):
    obs, info = env.reset(seed=seed)
    memory = DungeonMemory()

    primitive_steps = 0
    discovered = False
    navigated = False
    success = False
    done = False

    for _ in range(MAX_EXPLORE_OPTIONS):
        memory.update(obs)

        if memory.downstairs_position() is not None:
            discovered = True
            break

        obs, reward, steps, explore_success, done = option_explore(
            env,
            obs,
            memory,
            max_steps=150,
            search_repetitions=5,
        )

        primitive_steps += steps

        if memory.downstairs_position() is not None:
            discovered = True
            break

        if done:
            break

    if discovered:
        stairs_discovered += 1

    if discovered and not done:
        obs, reward, steps, navigated, done = option_go_to_downstairs(
            env,
            obs,
            memory,
            max_steps=500,
        )

        primitive_steps += steps

        if navigated:
            navigation_successes += 1

    if navigated and not done:
        obs, reward, steps, success, done = option_descend(
            env,
            obs,
            memory,
        )

        primitive_steps += steps

        if success:
            successes += 1

    total_primitive_steps.append(primitive_steps)

    print(
        f"Seed {seed:02d}: "
        f"stairs={discovered} | "
        f"navigate={navigated} | "
        f"descend={success} | "
        f"steps={primitive_steps}"
    )

env.close()

print()
print("=" * 65)
print("OPTION EXECUTOR V2 SUMMARY")
print("=" * 65)
print(f"Stairs discovered: {stairs_discovered}/{NUM_SEEDS}")
print(f"Navigation successes: {navigation_successes}/{NUM_SEEDS}")
print(f"Successful descents: {successes}/{NUM_SEEDS}")
print("Mean primitive steps:", float(np.mean(total_primitive_steps)))