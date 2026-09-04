import statistics
import time

import gymnasium as gym

from nethack_option_executor import (
    DungeonMemory,
    option_explore,
    option_go_to_downstairs,
)


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "NetHackStaircase-v0"

NUM_EPISODES = 100

MAX_EXPLORE_OPTIONS = 5
EXPLORE_STEPS_PER_OPTION = 150
NAVIGATION_STEPS = 500


# ============================================================
# SINGLE EPISODE
# ============================================================

def run_episode(episode_id):
    env = gym.make(ENV_ID)

    try:
        obs, info = env.reset()

        memory = DungeonMemory()
        memory.update(obs)

        explore_options = 0
        explore_steps = 0

        found_stairs = False
        reached_stairs = False

        explore_terminated = False
        navigation_terminated = False

        zero_step_stop = False

        # ----------------------------------------------------
        # PHASE 1: EXPLORE
        # ----------------------------------------------------

        for _ in range(MAX_EXPLORE_OPTIONS):
            explore_options += 1

            (
                obs,
                reward,
                steps,
                success,
                done,
            ) = option_explore(
                env,
                obs,
                memory,
                max_steps=EXPLORE_STEPS_PER_OPTION,
            )

            explore_steps += steps

            if (
                success
                or memory.downstairs_position() is not None
            ):
                found_stairs = True
                break

            if done:
                explore_terminated = True
                break

            if steps == 0:
                zero_step_stop = True
                break

        # ----------------------------------------------------
        # PHASE 2: NAVIGATE
        # ----------------------------------------------------

        nav_steps = 0
        nav_reward = 0.0

        if found_stairs and not explore_terminated:

            (
                obs,
                nav_reward,
                nav_steps,
                nav_success,
                nav_done,
            ) = option_go_to_downstairs(
                env,
                obs,
                memory,
                max_steps=NAVIGATION_STEPS,
            )

            reached_stairs = nav_success
            navigation_terminated = nav_done

        total_steps = explore_steps + nav_steps

        return {
            "episode": episode_id,

            "found": found_stairs,
            "reached": reached_stairs,

            "explore_options": explore_options,
            "explore_steps": explore_steps,
            "nav_steps": nav_steps,
            "total_steps": total_steps,

            "explore_terminated": explore_terminated,
            "navigation_terminated": navigation_terminated,
            "zero_step_stop": zero_step_stop,

            "known": len(memory.known),
            "walkable": len(memory.walkable),

            "stairs": memory.downstairs_position(),
            "nav_reward": nav_reward,
        }

    finally:
        env.close()


# ============================================================
# HELPERS
# ============================================================

def percentage(count, total):
    if total == 0:
        return 0.0

    return 100.0 * count / total


def safe_mean(values):
    if not values:
        return 0.0

    return statistics.mean(values)


def safe_median(values):
    if not values:
        return 0.0

    return statistics.median(values)


# ============================================================
# BENCHMARK
# ============================================================

def main():

    print("=" * 100)
    print("NETHACK OPTION EXECUTOR BENCHMARK")
    print("=" * 100)

    print(
        f"Episodes: {NUM_EPISODES}"
    )

    print(
        f"Max explore options: "
        f"{MAX_EXPLORE_OPTIONS}"
    )

    print(
        f"Explore budget / option: "
        f"{EXPLORE_STEPS_PER_OPTION}"
    )

    print(
        f"Navigation budget: "
        f"{NAVIGATION_STEPS}"
    )

    print("=" * 100)

    results = []

    start_time = time.time()

    for episode in range(NUM_EPISODES):

        result = run_episode(
            episode
        )

        results.append(
            result
        )

        status = (
            "SUCCESS"
            if result["reached"]
            else (
                "FOUND_ONLY"
                if result["found"]
                else "FAIL"
            )
        )

        print(
            f"[{episode + 1:03d}/{NUM_EPISODES}] "
            f"{status:<10} "
            f"explore={result['explore_steps']:<4} "
            f"nav={result['nav_steps']:<4} "
            f"total={result['total_steps']:<4} "
            f"known={result['known']:<4} "
            f"stairs={result['stairs']}"
        )

    elapsed = time.time() - start_time

    # ========================================================
    # AGGREGATES
    # ========================================================

    total = len(
        results
    )

    found = [
        r
        for r in results
        if r["found"]
    ]

    reached = [
        r
        for r in results
        if r["reached"]
    ]

    failed_discovery = [
        r
        for r in results
        if not r["found"]
    ]

    found_but_not_reached = [
        r
        for r in results
        if r["found"]
        and not r["reached"]
    ]

    zero_step = [
        r
        for r in results
        if r["zero_step_stop"]
    ]

    explore_terminated = [
        r
        for r in results
        if r["explore_terminated"]
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"Downstairs discovery: "
        f"{len(found)}/{total} "
        f"({percentage(len(found), total):.1f}%)"
    )

    print(
        f"End-to-end success: "
        f"{len(reached)}/{total} "
        f"({percentage(len(reached), total):.1f}%)"
    )

    print(
        f"Navigation given discovery: "
        f"{len(reached)}/{len(found)} "
        f"({percentage(len(reached), len(found)):.1f}%)"
    )

    print()

    print(
        f"Found stairs but navigation failed: "
        f"{len(found_but_not_reached)}"
    )

    print(
        f"Exploration terminated before discovery: "
        f"{len(explore_terminated)}"
    )

    print(
        f"Zero-step exploration stops: "
        f"{len(zero_step)}"
    )

    print()

    # ========================================================
    # STEP STATISTICS
    # ========================================================

    discovery_steps = [
        r["explore_steps"]
        for r in found
    ]

    failure_steps = [
        r["explore_steps"]
        for r in failed_discovery
    ]

    navigation_steps = [
        r["nav_steps"]
        for r in reached
    ]

    successful_total_steps = [
        r["total_steps"]
        for r in reached
    ]

    print("DISCOVERY STEP STATISTICS")
    print("-" * 100)

    print(
        f"Mean explore steps when stairs found: "
        f"{safe_mean(discovery_steps):.1f}"
    )

    print(
        f"Median explore steps when stairs found: "
        f"{safe_median(discovery_steps):.1f}"
    )

    print(
        f"Mean explore steps on discovery failure: "
        f"{safe_mean(failure_steps):.1f}"
    )

    print(
        f"Median explore steps on discovery failure: "
        f"{safe_median(failure_steps):.1f}"
    )

    print()

    print("NAVIGATION STEP STATISTICS")
    print("-" * 100)

    print(
        f"Mean navigation steps on success: "
        f"{safe_mean(navigation_steps):.1f}"
    )

    print(
        f"Median navigation steps on success: "
        f"{safe_median(navigation_steps):.1f}"
    )

    print()

    print("END-TO-END STEP STATISTICS")
    print("-" * 100)

    print(
        f"Mean total steps on success: "
        f"{safe_mean(successful_total_steps):.1f}"
    )

    print(
        f"Median total steps on success: "
        f"{safe_median(successful_total_steps):.1f}"
    )

    print()

    print(
        f"Runtime: {elapsed:.1f} seconds"
    )

    if total:
        print(
            f"Mean runtime / episode: "
            f"{elapsed / total:.2f} seconds"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()