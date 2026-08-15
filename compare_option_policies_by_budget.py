import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import navix as nx
from navix import observations
from navix.agents import PPOHparams, ActorCritic
from navix.environments.environment import Environment

from option_ppo import OptionPPO
from jax_option_executor import execute_option_jax


# ============================================================
# Configuration
# ============================================================

TRAIN_SEED = 0
TRAIN_BUDGET = 100_000

NUM_EVAL_EPISODES = 100

EVAL_BUDGETS = [
    15,
    20,
    25,
    30,
    40,
    50,
    75,
    100,
]

NUM_OPTIONS = 5
GAMMA = 0.99


# ============================================================
# Observation wrapper
# ============================================================

def FlattenObsWrapper(env: Environment):

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
# Option PPO
# ============================================================

config = PPOHparams().replace(
    budget=TRAIN_BUDGET,
    num_envs=16,
    num_steps=128,
    num_minibatches=8,
    num_epochs=1,
    lr=0.00025,
    anneal_lr=False,
)


agent = OptionPPO(
    hparams=config,
    network=ActorCritic(
        action_dim=NUM_OPTIONS,
    ),
    env=env,
)


# ============================================================
# Train once
# ============================================================

print(
    "Environment:",
    env_id
)

print(
    "Training primitive-frame budget:",
    TRAIN_BUDGET
)

print(
    "Evaluation episodes per budget:",
    NUM_EVAL_EPISODES
)

print(
    "Evaluation budgets:",
    EVAL_BUDGETS
)

print(
    "\nStarting 100k Option PPO training..."
)


rng = jax.random.PRNGKey(
    TRAIN_SEED
)

train_state, logs = agent.train(
    rng
)


print(
    "\nTraining complete."
)

print(
    "Final primitive frames:",
    train_state.frames
)

print(
    "Total option decisions:",
    train_state.option_decisions
)


# ============================================================
# Trained PPO option selection
# ============================================================

def select_trained_option(
    observation,
    rng,
):

    pi = agent.network.apply(
        train_state.params,
        observation,
        method="policy",
    )

    return pi.sample(
        seed=rng
    )


select_trained_option_jit = jax.jit(
    select_trained_option
)


# ============================================================
# Option execution
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
# Evaluate one episode
# ============================================================

def evaluate_episode(
    seed,
    primitive_budget,
    policy_type,
):

    reset_rng = jax.random.PRNGKey(
        seed
    )

    # Separate policy randomness from environment randomness.
    if policy_type == "trained":

        policy_rng = jax.random.PRNGKey(
            seed + 1_000_000
        )

    elif policy_type == "random":

        policy_rng = jax.random.PRNGKey(
            seed + 2_000_000
        )

    else:

        raise ValueError(
            f"Unknown policy type: {policy_type}"
        )


    timestep = env.reset(
        reset_rng
    )


    primitive_steps = 0

    option_decisions = 0

    valid_options = 0

    invalid_options = 0


    while (
        not bool(
            timestep.is_done()
        )
        and primitive_steps
        < primitive_budget
    ):

        policy_rng, option_rng = (
            jax.random.split(
                policy_rng
            )
        )


        # ----------------------------------------------------
        # Select option
        # ----------------------------------------------------

        if policy_type == "trained":

            option_id = (
                select_trained_option_jit(
                    timestep.observation,
                    option_rng,
                )
            )

        else:

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
            new_timestep,
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


        # ----------------------------------------------------
        # Do not count an option that would exceed the
        # evaluation primitive-step budget.
        #
        # The option itself has already been computed by JAX,
        # but for evaluation accounting we stop here.
        # ----------------------------------------------------

        if (
            primitive_steps
            + duration
            > primitive_budget
        ):

            break


        timestep = new_timestep

        primitive_steps += duration

        option_decisions += 1


        if bool(valid):

            valid_options += 1

        else:

            invalid_options += 1


    # ========================================================
    # True task success
    # ========================================================

    success = (
        bool(
            timestep.is_done()
        )
        and float(
            timestep.reward
        ) > 0.0
    )


    total_options = (
        valid_options
        + invalid_options
    )


    if total_options > 0:

        valid_rate = (
            valid_options
            / total_options
        )

    else:

        valid_rate = 0.0


    return {
        "success":
            success,

        "primitive_steps":
            primitive_steps,

        "option_decisions":
            option_decisions,

        "valid_rate":
            valid_rate,
    }


# ============================================================
# Evaluate one policy at one budget
# ============================================================

def evaluate_policy(
    policy_type,
    primitive_budget,
):

    results = []


    for seed in range(
        NUM_EVAL_EPISODES
    ):

        result = evaluate_episode(
            seed=seed,
            primitive_budget=primitive_budget,
            policy_type=policy_type,
        )

        results.append(
            result
        )


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


    valid_rates = np.asarray(
        [
            result["valid_rate"]
            for result in results
        ],
        dtype=np.float32,
    )


    successful_mask = (
        successes.astype(bool)
    )


    success_rate = (
        successes.mean()
    )


    if successful_mask.any():

        successful_mean_steps = (
            primitive_steps[
                successful_mask
            ].mean()
        )

        successful_mean_decisions = (
            option_decisions[
                successful_mask
            ].mean()
        )

    else:

        successful_mean_steps = np.nan

        successful_mean_decisions = np.nan


    return {
        "success_rate":
            success_rate,

        "mean_primitive_steps":
            primitive_steps.mean(),

        "mean_option_decisions":
            option_decisions.mean(),

        "mean_valid_rate":
            valid_rates.mean(),

        "successful_mean_steps":
            successful_mean_steps,

        "successful_mean_decisions":
            successful_mean_decisions,
    }


# ============================================================
# Run all budgets
# ============================================================

trained_results = []

random_results = []


print(
    "\n================================"
)

print(
    "BUDGET-SENSITIVITY EVALUATION"
)

print(
    "================================"
)


for budget in EVAL_BUDGETS:

    print(
        f"\nPrimitive-step budget: {budget}"
    )


    trained = evaluate_policy(
        policy_type="trained",
        primitive_budget=budget,
    )


    random = evaluate_policy(
        policy_type="random",
        primitive_budget=budget,
    )


    trained_results.append(
        trained
    )

    random_results.append(
        random
    )


    print(
        "Option PPO success:",
        f"{trained['success_rate'] * 100:.1f}%"
    )

    print(
        "Random success:",
        f"{random['success_rate'] * 100:.1f}%"
    )

    print(
        "Option PPO valid rate:",
        f"{trained['mean_valid_rate'] * 100:.1f}%"
    )

    print(
        "Random valid rate:",
        f"{random['mean_valid_rate'] * 100:.1f}%"
    )


# ============================================================
# Convert to arrays
# ============================================================

budgets = np.asarray(
    EVAL_BUDGETS
)


trained_success = np.asarray(
    [
        result["success_rate"]
        for result in trained_results
    ]
)


random_success = np.asarray(
    [
        result["success_rate"]
        for result in random_results
    ]
)


trained_valid = np.asarray(
    [
        result["mean_valid_rate"]
        for result in trained_results
    ]
)


random_valid = np.asarray(
    [
        result["mean_valid_rate"]
        for result in random_results
    ]
)


trained_decisions = np.asarray(
    [
        result["mean_option_decisions"]
        for result in trained_results
    ]
)


random_decisions = np.asarray(
    [
        result["mean_option_decisions"]
        for result in random_results
    ]
)


trained_success_steps = np.asarray(
    [
        result["successful_mean_steps"]
        for result in trained_results
    ]
)


random_success_steps = np.asarray(
    [
        result["successful_mean_steps"]
        for result in random_results
    ]
)


# ============================================================
# Save raw results
# ============================================================

np.savez(
    "option_policy_budget_comparison.npz",

    budgets=budgets,

    trained_success=trained_success,

    random_success=random_success,

    trained_valid=trained_valid,

    random_valid=random_valid,

    trained_decisions=trained_decisions,

    random_decisions=random_decisions,

    trained_success_steps=(
        trained_success_steps
    ),

    random_success_steps=(
        random_success_steps
    ),
)


# ============================================================
# Plot 1
# Success rate vs primitive budget
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    budgets,
    trained_success,
    marker="o",
    linewidth=2,
    label="Option PPO (100k)",
)


plt.plot(
    budgets,
    random_success,
    marker="o",
    linewidth=2,
    label="Random options",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Success rate"
)

plt.title(
    "DoorKey 8x8 - Success Under Fixed Interaction Budgets"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_policy_budget_success.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 2
# Valid option rate
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    budgets,
    trained_valid,
    marker="o",
    linewidth=2,
    label="Option PPO (100k)",
)


plt.plot(
    budgets,
    random_valid,
    marker="o",
    linewidth=2,
    label="Random options",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Valid-option rate"
)

plt.title(
    "DoorKey 8x8 - Valid Option Selection"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_policy_budget_validity.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 3
# Mean option decisions
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    budgets,
    trained_decisions,
    marker="o",
    linewidth=2,
    label="Option PPO (100k)",
)


plt.plot(
    budgets,
    random_decisions,
    marker="o",
    linewidth=2,
    label="Random options",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Mean option decisions per episode"
)

plt.title(
    "DoorKey 8x8 - High-Level Decision Efficiency"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_policy_budget_decisions.png",
    dpi=200,
)

plt.close()


# ============================================================
# Final table
# ============================================================

print(
    "\n============================================"
)

print(
    "FINAL BUDGET COMPARISON"
)

print(
    "============================================"
)

print(
    "Budget | PPO success | Random success"
)

print(
    "--------------------------------------------"
)


for i, budget in enumerate(
    budgets
):

    print(
        f"{budget:>6} | "
        f"{trained_success[i] * 100:>9.1f}% | "
        f"{random_success[i] * 100:>12.1f}%"
    )


print(
    "\nSaved:"
)

print(
    "option_policy_budget_comparison.npz"
)

print(
    "option_policy_budget_success.png"
)

print(
    "option_policy_budget_validity.png"
)

print(
    "option_policy_budget_decisions.png"
)