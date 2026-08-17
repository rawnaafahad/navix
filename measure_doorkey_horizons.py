import jax
import jax.numpy as jnp
import numpy as np

import navix as nx

from jax_option_executor import execute_option_jax


# ============================================================
# Configuration
# ============================================================

NUM_SEEDS = 100

ENV_IDS = [
    "Navix-DoorKey-Random-5x5-v0",
    "Navix-DoorKey-Random-6x6-v0",
    "Navix-DoorKey-Random-8x8-v0",
    "Navix-DoorKey-Random-16x16-v0",
]

OPTION_SEQUENCE = [
    0,  # GO_TO_KEY
    1,  # PICKUP_KEY
    2,  # GO_TO_DOOR
    3,  # OPEN_DOOR
    4,  # GO_TO_GOAL
]

OPTION_NAMES = [
    "GO_TO_KEY",
    "PICKUP_KEY",
    "GO_TO_DOOR",
    "OPEN_DOOR",
    "GO_TO_GOAL",
]


# ============================================================
# Evaluate one environment size
# ============================================================

def measure_environment(env_id):

    print()
    print("=" * 60)
    print("ENVIRONMENT:", env_id)
    print("=" * 60)

    try:
        env = nx.make(
            env_id,
            gamma=0.99,
        )

    except Exception as error:

        print("Could not create environment.")
        print("Error:", error)

        return None

    # --------------------------------------------------------
    # JIT option dispatcher for this environment
    # --------------------------------------------------------

    execute_option_fn = jax.jit(
        lambda timestep, option_id:
            execute_option_jax(
                env,
                timestep,
                option_id,
            )
    )

    successful_lengths = []
    successful_decisions = []

    option_durations = []

    failed_seeds = []

    # --------------------------------------------------------
    # Run expert option sequence
    # --------------------------------------------------------

    for seed in range(NUM_SEEDS):

        rng = jax.random.PRNGKey(
            seed
        )

        timestep = env.reset(
            rng
        )

        total_primitive_steps = 0
        total_option_decisions = 0

        seed_option_durations = []

        sequence_valid = True

        for option_id in OPTION_SEQUENCE:

            timestep, duration, option_reward, valid = (
                execute_option_fn(
                    timestep,
                    jnp.asarray(
                        option_id,
                        dtype=jnp.int32,
                    ),
                )
            )

            duration = int(
                duration
            )

            valid = bool(
                valid
            )

            total_primitive_steps += duration

            total_option_decisions += 1

            seed_option_durations.append(
                duration
            )

            if not valid:

                sequence_valid = False

                break

            if bool(
                timestep.is_done()
            ):

                break

        success = bool(
            timestep.is_done()
        ) and float(
            timestep.reward
        ) > 0.0

        if (
            success
            and sequence_valid
        ):

            successful_lengths.append(
                total_primitive_steps
            )

            successful_decisions.append(
                total_option_decisions
            )

            option_durations.append(
                seed_option_durations
            )

        else:

            failed_seeds.append(
                seed
            )

        # Print first few seeds for sanity checking
        if seed < 5:

            status = (
                "SUCCESS"
                if success
                else "FAIL"
            )

            print(
                f"Seed {seed:03d}: "
                f"{status} | "
                f"primitive steps="
                f"{total_primitive_steps} | "
                f"option decisions="
                f"{total_option_decisions}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    success_count = len(
        successful_lengths
    )

    success_rate = (
        success_count
        / NUM_SEEDS
    )

    print()
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)

    print(
        "Successful sequences:",
        f"{success_count}/{NUM_SEEDS}"
    )

    print(
        "Success rate:",
        f"{success_rate * 100:.1f}%"
    )

    if success_count == 0:

        print(
            "No successful trajectories."
        )

        print(
            "The current option executor probably "
            "does not generalise to this environment size."
        )

        return {
            "env_id": env_id,
            "success_rate": success_rate,
            "mean_steps": np.nan,
            "std_steps": np.nan,
            "min_steps": np.nan,
            "max_steps": np.nan,
        }

    successful_lengths = np.asarray(
        successful_lengths,
        dtype=np.float32,
    )

    successful_decisions = np.asarray(
        successful_decisions,
        dtype=np.float32,
    )

    print(
        "Mean primitive solution length:",
        f"{successful_lengths.mean():.2f}"
    )

    print(
        "Std primitive solution length:",
        f"{successful_lengths.std():.2f}"
    )

    print(
        "Minimum primitive solution length:",
        int(
            successful_lengths.min()
        )
    )

    print(
        "Maximum primitive solution length:",
        int(
            successful_lengths.max()
        )
    )

    print(
        "Mean option decisions:",
        f"{successful_decisions.mean():.2f}"
    )

    print(
        "Mean primitive steps per option:",
        f"{successful_lengths.mean() / successful_decisions.mean():.2f}"
    )

    if len(
        failed_seeds
    ) > 0:

        print(
            "Failed seeds:",
            failed_seeds[:20],
        )

        if len(
            failed_seeds
        ) > 20:

            print(
                "...",
                len(failed_seeds) - 20,
                "additional failures"
            )

    return {
        "env_id": env_id,
        "success_rate": success_rate,
        "mean_steps": float(
            successful_lengths.mean()
        ),
        "std_steps": float(
            successful_lengths.std()
        ),
        "min_steps": float(
            successful_lengths.min()
        ),
        "max_steps": float(
            successful_lengths.max()
        ),
    }


# ============================================================
# Main experiment
# ============================================================

results = []

print(
    "============================================"
)

print(
    "DOORKEY TEMPORAL-HORIZON MEASUREMENT"
)

print(
    "============================================"
)

print(
    "Seeds per environment:",
    NUM_SEEDS
)

print(
    "Candidate environments:"
)

for env_id in ENV_IDS:

    print(
        " -",
        env_id
    )


for env_id in ENV_IDS:

    result = measure_environment(
        env_id
    )

    if result is not None:

        results.append(
            result
        )


# ============================================================
# Final comparison
# ============================================================

print()
print()
print("=" * 75)
print("TEMPORAL-HORIZON COMPARISON")
print("=" * 75)

print(
    "Environment                         "
    "| Success | Mean steps | Std | Min | Max"
)

print(
    "-" * 75
)


for result in results:

    env_name = result[
        "env_id"
    ]

    success = result[
        "success_rate"
    ] * 100

    mean_steps = result[
        "mean_steps"
    ]

    std_steps = result[
        "std_steps"
    ]

    min_steps = result[
        "min_steps"
    ]

    max_steps = result[
        "max_steps"
    ]

    print(
        f"{env_name:<35} | "
        f"{success:>6.1f}% | "
        f"{mean_steps:>10.2f} | "
        f"{std_steps:>4.1f} | "
        f"{min_steps:>3.0f} | "
        f"{max_steps:>3.0f}"
    )


# ============================================================
# Save results
# ============================================================

np.savez(
    "doorkey_horizon_measurements.npz",

    env_ids=np.asarray(
        [
            result["env_id"]
            for result in results
        ]
    ),

    success_rates=np.asarray(
        [
            result["success_rate"]
            for result in results
        ]
    ),

    mean_steps=np.asarray(
        [
            result["mean_steps"]
            for result in results
        ]
    ),

    std_steps=np.asarray(
        [
            result["std_steps"]
            for result in results
        ]
    ),

    min_steps=np.asarray(
        [
            result["min_steps"]
            for result in results
        ]
    ),

    max_steps=np.asarray(
        [
            result["max_steps"]
            for result in results
        ]
    ),
)


print()
print(
    "Saved:"
)

print(
    "doorkey_horizon_measurements.npz"
)

print()
print(
    "Horizon measurement complete."
)