import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix import observations
from navix.environments.environment import Environment

from jax_option_executor import execute_option_jax


# ============================================================
# Configuration
# ============================================================

NUM_EVAL_EPISODES = 100
MAX_EVAL_PRIMITIVE_STEPS = 100
GAMMA = 0.99
NUM_OPTIONS = 5


# ============================================================
# Observation wrapper
# ============================================================

def FlattenObsWrapper(
    env: Environment,
):
    flatten_obs_fn = (
        lambda x:
            jnp.ravel(
                env.observation_fn(x)
            )
    )

    flatten_obs_shape = (
        int(
            np.prod(
                env.observation_space.shape
            )
        ),
    )

    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=(
            env.observation_space.replace(
                shape=flatten_obs_shape
            )
        ),
    )


# ============================================================
# Environment
# ============================================================

env_id = "Navix-DoorKey-Random-8x8-v0"

env = nx.make(
    env_id,
    observation_fn=observations.symbolic_first_person,
    gamma=GAMMA,
)

env = FlattenObsWrapper(
    env
)


# ============================================================
# JIT option executor
# ============================================================

execute_option_fn = jax.jit(
    lambda timestep, option_id:
        execute_option_jax(
            env,
            timestep,
            option_id,
            gamma=GAMMA,
        )
)


# ============================================================
# Evaluate one random-option episode
# ============================================================

def evaluate_episode(
    seed,
):
    """
    Evaluate a uniformly random high-level option policy.
    """

    reset_rng = jax.random.PRNGKey(
        seed
    )

    policy_rng = jax.random.PRNGKey(
        seed + 1_000_000
    )

    timestep = env.reset(
        reset_rng
    )

    primitive_steps = 0
    option_decisions = 0
    valid_options = 0
    invalid_options = 0

    option_counts = np.zeros(
        NUM_OPTIONS,
        dtype=np.int32,
    )

    while (
        not bool(
            timestep.is_done()
        )
        and primitive_steps
        < MAX_EVAL_PRIMITIVE_STEPS
    ):

        # ----------------------------------------------------
        # Uniform random option
        # ----------------------------------------------------

        policy_rng, option_rng = (
            jax.random.split(
                policy_rng
            )
        )

        option_id = jax.random.randint(
            option_rng,
            shape=(),
            minval=0,
            maxval=NUM_OPTIONS,
            dtype=jnp.int32,
        )

        # ----------------------------------------------------
        # Execute option
        # ----------------------------------------------------

        (
            timestep,
            duration,
            option_reward,
            valid,
        ) = execute_option_fn(
            timestep,
            option_id,
        )

        duration = int(
            duration
        )

        valid = bool(
            valid
        )

        primitive_steps += duration
        option_decisions += 1

        option_counts[
            int(option_id)
        ] += 1

        if valid:
            valid_options += 1
        else:
            invalid_options += 1

    success = (
        bool(
            timestep.is_done()
        )
        and float(
            timestep.reward
        ) > 0.0
    )

    return {
        "seed":
            seed,

        "success":
            success,

        "primitive_steps":
            primitive_steps,

        "option_decisions":
            option_decisions,

        "valid_options":
            valid_options,

        "invalid_options":
            invalid_options,

        "option_counts":
            option_counts,
    }


# ============================================================
# Run evaluation
# ============================================================

print(
    "Environment:",
    env_id
)

print(
    "Evaluation episodes:",
    NUM_EVAL_EPISODES
)

print(
    "Maximum primitive steps:",
    MAX_EVAL_PRIMITIVE_STEPS
)

print(
    "\n================================"
)

print(
    "RANDOM OPTION POLICY EVALUATION"
)

print(
    "================================\n"
)


results = []


for seed in range(
    NUM_EVAL_EPISODES
):

    result = evaluate_episode(
        seed
    )

    results.append(
        result
    )

    status = (
        "SUCCESS"
        if result["success"]
        else "FAIL"
    )

    print(
        f"Seed {seed:03d}: "
        f"{status} | "
        f"primitive steps="
        f"{result['primitive_steps']} | "
        f"option decisions="
        f"{result['option_decisions']}"
    )


# ============================================================
# Aggregate metrics
# ============================================================

successes = np.asarray(
    [
        result["success"]
        for result in results
    ],
    dtype=np.float32,
)


primitive_steps = np.asarray(
    [
        result["primitive_steps"]
        for result in results
    ],
    dtype=np.float32,
)


option_decisions = np.asarray(
    [
        result["option_decisions"]
        for result in results
    ],
    dtype=np.float32,
)


valid_options = np.asarray(
    [
        result["valid_options"]
        for result in results
    ],
    dtype=np.float32,
)


invalid_options = np.asarray(
    [
        result["invalid_options"]
        for result in results
    ],
    dtype=np.float32,
)


option_counts = np.stack(
    [
        result["option_counts"]
        for result in results
    ],
    axis=0,
)


success_rate = (
    successes.mean()
)


total_options = (
    valid_options
    + invalid_options
)


evaluation_valid_rate = np.divide(
    valid_options,
    total_options,
    out=np.zeros_like(
        valid_options
    ),
    where=(
        total_options > 0
    ),
)


total_option_counts = (
    option_counts.sum(
        axis=0
    )
)


option_selection_frequency = (
    total_option_counts
    / max(
        total_option_counts.sum(),
        1,
    )
)


successful_mask = (
    successes.astype(bool)
)


# ============================================================
# Summary
# ============================================================

print(
    "\n================================"
)

print(
    "RANDOM POLICY SUMMARY"
)

print(
    "================================"
)


print(
    "Successful episodes:",
    int(
        successes.sum()
    ),
    "/",
    NUM_EVAL_EPISODES,
)


print(
    "Success rate:",
    f"{success_rate * 100:.2f}%"
)


print(
    "Mean valid-option rate:",
    f"{evaluation_valid_rate.mean() * 100:.2f}%"
)


print(
    "Mean option decisions per episode:",
    option_decisions.mean()
)


print(
    "Mean primitive steps per episode:",
    primitive_steps.mean()
)


option_names = [
    "GO_TO_KEY",
    "PICKUP_KEY",
    "GO_TO_DOOR",
    "OPEN_DOOR",
    "GO_TO_GOAL",
]


print(
    "\nOption-selection frequencies:"
)


for option_id, option_name in enumerate(
    option_names
):

    print(
        f"{option_name}: "
        f"{option_selection_frequency[option_id] * 100:.2f}%"
    )


if successful_mask.any():

    print(
        "\nSuccessful episodes only:"
    )

    print(
        "Mean primitive steps:",
        primitive_steps[
            successful_mask
        ].mean()
    )

    print(
        "Min primitive steps:",
        primitive_steps[
            successful_mask
        ].min()
    )

    print(
        "Max primitive steps:",
        primitive_steps[
            successful_mask
        ].max()
    )

    print(
        "Mean option decisions:",
        option_decisions[
            successful_mask
        ].mean()
    )


# ============================================================
# Save
# ============================================================

np.savez(
    "random_option_evaluation.npz",

    successes=successes,

    primitive_steps=primitive_steps,

    option_decisions=option_decisions,

    valid_options=valid_options,

    invalid_options=invalid_options,

    evaluation_valid_rate=(
        evaluation_valid_rate
    ),

    option_counts=option_counts,

    option_selection_frequency=(
        option_selection_frequency
    ),
)


print(
    "\nSaved:"
)

print(
    "random_option_evaluation.npz"
)