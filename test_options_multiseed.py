import gc
import csv

import jax
import numpy as np
import matplotlib.pyplot as plt
import navix as nx

from test_options import (
    go_to_key,
    pickup_key,
    go_to_door,
    open_door,
    go_to_goal,
)


def run_option_sequence(seed):
    """
    Run the complete hard-coded option sequence
    for one random DoorKey environment.
    """

    env = nx.make(
        "Navix-DoorKey-Random-8x8-v0"
    )

    rng = jax.random.PRNGKey(seed)
    timestep = env.reset(rng)

    total_primitive_steps = 0

    option_durations = {
        "go_to_key": 0,
        "pickup_key": 0,
        "go_to_door": 0,
        "open_door": 0,
        "go_to_goal": 0,
    }

    try:

        timestep, duration = go_to_key(
            env,
            timestep
        )

        option_durations["go_to_key"] = duration
        total_primitive_steps += duration

        timestep, duration = pickup_key(
            env,
            timestep
        )

        option_durations["pickup_key"] = duration
        total_primitive_steps += duration

        timestep, duration = go_to_door(
            env,
            timestep
        )

        option_durations["go_to_door"] = duration
        total_primitive_steps += duration

        timestep, duration = open_door(
            env,
            timestep
        )

        option_durations["open_door"] = duration
        total_primitive_steps += duration

        timestep, duration = go_to_goal(
            env,
            timestep
        )

        option_durations["go_to_goal"] = duration
        total_primitive_steps += duration

        success = (
            bool(timestep.is_done())
            and float(timestep.reward) > 0.0
        )

        return {
            "seed": seed,
            "success": success,
            "primitive_steps": total_primitive_steps,
            "option_durations": option_durations,
            "error": None,
        }

    except Exception as error:

        return {
            "seed": seed,
            "success": False,
            "primitive_steps": total_primitive_steps,
            "option_durations": option_durations,
            "error": str(error),
        }


# ============================================================
# Configuration
# ============================================================

NUM_SEEDS = 100

CSV_FILE = "option_multiseed_results.csv"

STEPS_PER_SEED_PLOT = (
    "option_steps_per_seed.png"
)

STEPS_DISTRIBUTION_PLOT = (
    "option_total_steps_distribution.png"
)

OPTION_DURATION_PLOT = (
    "option_duration_by_type.png"
)


# ============================================================
# Run experiment
# ============================================================

results = []

print(
    f"Testing complete option controller "
    f"across {NUM_SEEDS} seeds...\n"
)

for seed in range(NUM_SEEDS):

    result = run_option_sequence(seed)

    results.append(result)

    if result["success"]:

        print(
            f"Seed {seed:03d}: PASS "
            f"({result['primitive_steps']} primitive steps)"
        )

    else:

        print(
            f"Seed {seed:03d}: FAIL"
        )

        if result["error"] is not None:

            print(
                "    Error:",
                result["error"]
            )

    # Clear JAX caches periodically
    if (seed + 1) % 5 == 0:

        jax.clear_caches()
        gc.collect()


# ============================================================
# Split successful / failed runs
# ============================================================

successful_results = [
    result
    for result in results
    if result["success"]
]

failed_results = [
    result
    for result in results
    if not result["success"]
]


# ============================================================
# Summary
# ============================================================

print(
    "\n================================"
)

print(
    "MULTI-SEED OPTION TEST SUMMARY"
)

print(
    "================================"
)

print(
    "Total seeds:",
    NUM_SEEDS
)

print(
    "Successful:",
    len(successful_results)
)

print(
    "Failed:",
    len(failed_results)
)


success_rate = (
    len(successful_results)
    / NUM_SEEDS
)


print(
    "Success rate:",
    f"{success_rate * 100:.2f}%"
)


# ============================================================
# Primitive-step statistics
# ============================================================

if successful_results:

    primitive_steps = np.asarray(
        [
            result["primitive_steps"]
            for result in successful_results
        ]
    )

    print(
        "\nPrimitive step statistics"
    )

    print(
        "Mean:",
        primitive_steps.mean()
    )

    print(
        "Std:",
        primitive_steps.std()
    )

    print(
        "Min:",
        primitive_steps.min()
    )

    print(
        "Max:",
        primitive_steps.max()
    )

    print(
        "Mean temporal abstraction ratio:",
        primitive_steps.mean() / 5
    )


# ============================================================
# Individual option statistics
# ============================================================

option_names = [
    "go_to_key",
    "pickup_key",
    "go_to_door",
    "open_door",
    "go_to_goal",
]


if successful_results:

    print(
        "\nOption duration statistics"
    )

    for option_name in option_names:

        values = np.asarray(
            [
                result["option_durations"][option_name]
                for result in successful_results
            ]
        )

        print(
            f"{option_name}: "
            f"mean={values.mean():.2f}, "
            f"std={values.std():.2f}, "
            f"min={values.min()}, "
            f"max={values.max()}"
        )


# ============================================================
# Save CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "seed",
            "success",
            "primitive_steps",
            "go_to_key",
            "pickup_key",
            "go_to_door",
            "open_door",
            "go_to_goal",
            "error",
        ]
    )

    for result in results:

        writer.writerow(
            [
                result["seed"],
                result["success"],
                result["primitive_steps"],
                result["option_durations"]["go_to_key"],
                result["option_durations"]["pickup_key"],
                result["option_durations"]["go_to_door"],
                result["option_durations"]["open_door"],
                result["option_durations"]["go_to_goal"],
                result["error"],
            ]
        )


print(
    "\nSaved CSV:"
)

print(
    CSV_FILE
)


# ============================================================
# Plot 1:
# Primitive steps per seed
# ============================================================

if successful_results:

    successful_seeds = np.asarray(
        [
            result["seed"]
            for result in successful_results
        ]
    )

    successful_steps = np.asarray(
        [
            result["primitive_steps"]
            for result in successful_results
        ]
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        successful_seeds,
        successful_steps,
        marker="o",
        markersize=3,
        linewidth=1,
    )

    plt.axhline(
        successful_steps.mean(),
        linestyle="--",
        label=(
            f"Mean = "
            f"{successful_steps.mean():.2f}"
        ),
    )

    plt.xlabel(
        "Random seed"
    )

    plt.ylabel(
        "Primitive steps to solve"
    )

    plt.title(
        "Hard-Coded Options - DoorKey Random 8x8\n"
        "Primitive Steps Across Seeds"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        STEPS_PER_SEED_PLOT,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 2:
# Distribution of total primitive steps
# ============================================================

if successful_results:

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        primitive_steps,
        bins=12,
        edgecolor="black",
    )

    plt.axvline(
        primitive_steps.mean(),
        linestyle="--",
        label=(
            f"Mean = "
            f"{primitive_steps.mean():.2f}"
        ),
    )

    plt.xlabel(
        "Primitive steps to solve"
    )

    plt.ylabel(
        "Number of seeds"
    )

    plt.title(
        "Hard-Coded Options - DoorKey Random 8x8\n"
        "Distribution of Primitive Steps"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        STEPS_DISTRIBUTION_PLOT,
        dpi=200,
    )

    plt.close()


# ============================================================
# Plot 3:
# Option duration by type
# ============================================================

if successful_results:

    option_means = []
    option_stds = []

    for option_name in option_names:

        values = np.asarray(
            [
                result["option_durations"][option_name]
                for result in successful_results
            ]
        )

        option_means.append(
            values.mean()
        )

        option_stds.append(
            values.std()
        )


    display_names = [
        "Go to key",
        "Pick up key",
        "Go to door",
        "Open door",
        "Go to goal",
    ]


    x_positions = np.arange(
        len(display_names)
    )


    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x_positions,
        option_means,
        yerr=option_stds,
        capsize=5,
    )

    plt.xticks(
        x_positions,
        display_names,
        rotation=20,
    )

    plt.xlabel(
        "Option"
    )

    plt.ylabel(
        "Primitive actions per execution"
    )

    plt.title(
        "Hard-Coded Option Duration - DoorKey Random 8x8\n"
        "Mean ± Standard Deviation"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        OPTION_DURATION_PLOT,
        dpi=200,
    )

    plt.close()


# ============================================================
# Report generated files
# ============================================================

print(
    "\nGenerated plots:"
)

print(
    STEPS_PER_SEED_PLOT
)

print(
    STEPS_DISTRIBUTION_PLOT
)

print(
    OPTION_DURATION_PLOT
)


# ============================================================
# Failure report
# ============================================================

if failed_results:

    print(
        "\nFAILED SEEDS"
    )

    for result in failed_results:

        print(
            f"Seed {result['seed']}: "
            f"{result['error']}"
        )


# ============================================================
# Final result
# ============================================================

if len(failed_results) == 0:

    print(
        "\nAll 100 option-controller tests passed."
    )

else:

    print(
        f"\n{len(failed_results)} seed(s) failed."
    )