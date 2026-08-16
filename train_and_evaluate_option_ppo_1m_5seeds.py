import gc
import os
import pickle

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

TRAIN_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

TRAIN_BUDGET = 1_000_000

NUM_OPTIONS = 5

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

GAMMA = 0.99

# If a checkpoint already exists, load it instead of
# retraining that seed.
RESUME_FROM_CHECKPOINTS = True


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

env_id = (
    "Navix-DoorKey-Random-8x8-v0"
)

env = nx.make(
    env_id,
    observation_fn=(
        observations.symbolic_first_person
    ),
    gamma=GAMMA,
)

env = FlattenObsWrapper(
    env
)


# ============================================================
# PPO configuration
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


network = ActorCritic(
    action_dim=NUM_OPTIONS,
)


agent = OptionPPO(
    hparams=config,
    network=network,
    env=env,
)


# ============================================================
# Option executor
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
# Checkpoint utilities
# ============================================================

def checkpoint_path(
    seed,
):
    return (
        f"option_ppo_1m_seed{seed}.pkl"
    )


def save_checkpoint(
    seed,
    params,
    training_summary,
):
    checkpoint = {
        "params":
            params,

        "train_seed":
            seed,

        "train_budget":
            TRAIN_BUDGET,

        "env_id":
            env_id,

        "num_options":
            NUM_OPTIONS,

        "training_summary":
            training_summary,
    }

    path = checkpoint_path(
        seed
    )

    with open(
        path,
        "wb",
    ) as file:
        pickle.dump(
            checkpoint,
            file,
        )

    print(
        "Saved checkpoint:",
        path
    )

    return path


def load_checkpoint(
    path,
):
    with open(
        path,
        "rb",
    ) as file:
        checkpoint = (
            pickle.load(
                file
            )
        )

    assert (
        checkpoint["env_id"]
        == env_id
    )

    assert (
        checkpoint["num_options"]
        == NUM_OPTIONS
    )

    return checkpoint


# ============================================================
# Train one seed
# ============================================================

def train_seed(
    seed,
):
    canonical_path = (
        checkpoint_path(
            seed
        )
    )

    # --------------------------------------------------------
    # Resume from canonical checkpoint
    # --------------------------------------------------------

    if (
        RESUME_FROM_CHECKPOINTS
        and os.path.exists(
            canonical_path
        )
    ):
        print(
            f"\nSeed {seed}: "
            f"loading existing checkpoint..."
        )

        checkpoint = (
            load_checkpoint(
                canonical_path
            )
        )

        return (
            checkpoint["params"],
            checkpoint.get(
                "training_summary",
                {},
            ),
        )


    # --------------------------------------------------------
    # Special case:
    # reuse the 1M seed-0 checkpoint you already produced.
    # --------------------------------------------------------

    old_seed0_path = (
        "option_ppo_1m_checkpoint.pkl"
    )

    if (
        seed == 0
        and RESUME_FROM_CHECKPOINTS
        and os.path.exists(
            old_seed0_path
        )
    ):
        print(
            "\nSeed 0: reusing existing "
            "option_ppo_1m_checkpoint.pkl"
        )

        checkpoint = (
            load_checkpoint(
                old_seed0_path
            )
        )

        params = (
            checkpoint["params"]
        )

        summary = {
            "source":
                old_seed0_path,
        }

        save_checkpoint(
            seed=seed,
            params=params,
            training_summary=summary,
        )

        return (
            params,
            summary,
        )


    # --------------------------------------------------------
    # Train new model
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        f"TRAINING SEED {seed}"
    )

    print(
        "================================"
    )

    rng = jax.random.PRNGKey(
        seed
    )

    train_state, logs = (
        agent.train(
            rng
        )
    )


    active = np.asarray(
        logs[
            "iter/active"
        ]
    ).astype(
        bool
    )


    frames = np.asarray(
        logs[
            "iter/frames"
        ]
    )[active]


    valid_rate = np.asarray(
        logs[
            "option/valid_rate"
        ]
    )[active]


    mean_duration = np.asarray(
        logs[
            "option/mean_duration"
        ]
    )[active]


    option_frequencies = []

    for option_id in range(
        NUM_OPTIONS
    ):
        frequency = np.asarray(
            logs[
                f"option/frequency_{option_id}"
            ]
        )[active]

        option_frequencies.append(
            frequency
        )


    option_frequencies = (
        np.asarray(
            option_frequencies
        )
    )


    # --------------------------------------------------------
    # Save per-seed training metrics
    # --------------------------------------------------------

    np.savez(
        f"option_ppo_1m_seed{seed}_training.npz",

        frames=frames,

        valid_rate=valid_rate,

        mean_duration=(
            mean_duration
        ),

        option_frequencies=(
            option_frequencies
        ),
    )


    training_summary = {
        "final_frames":
            int(
                train_state.frames
            ),

        "option_decisions":
            int(
                train_state.option_decisions
            ),

        "optimiser_steps":
            int(
                train_state.step
            ),

        "active_updates":
            int(
                len(
                    frames
                )
            ),

        "final_valid_rate":
            float(
                valid_rate[-1]
            ),

        "final_mean_duration":
            float(
                mean_duration[-1]
            ),

        "final_option_frequencies":
            option_frequencies[
                :,
                -1,
            ].tolist(),
    }


    print(
        "\nTraining summary:"
    )

    print(
        "Final primitive frames:",
        training_summary[
            "final_frames"
        ]
    )

    print(
        "Final valid-option rate:",
        training_summary[
            "final_valid_rate"
        ]
    )

    print(
        "Final mean duration:",
        training_summary[
            "final_mean_duration"
        ]
    )


    save_checkpoint(
        seed=seed,
        params=train_state.params,
        training_summary=(
            training_summary
        ),
    )


    return (
        train_state.params,
        training_summary,
    )


# ============================================================
# Trained option selection
# ============================================================

def make_trained_policy(
    params,
):
    def select_option(
        observation,
        rng,
    ):
        pi = network.apply(
            params,
            observation,
            method="policy",
        )

        return pi.sample(
            seed=rng
        )

    return jax.jit(
        select_option
    )


# ============================================================
# Evaluate one episode
# ============================================================

def evaluate_episode(
    seed,
    primitive_budget,
    policy_type,
    trained_policy_fn=None,
):
    reset_rng = (
        jax.random.PRNGKey(
            seed
        )
    )


    # Same evaluation action-randomness stream for every
    # independently trained PPO policy.
    if policy_type == "trained":
        policy_rng = (
            jax.random.PRNGKey(
                seed + 1_000_000
            )
        )

    elif policy_type == "random":
        policy_rng = (
            jax.random.PRNGKey(
                seed + 2_000_000
            )
        )

    else:
        raise ValueError(
            f"Unknown policy type: "
            f"{policy_type}"
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
                trained_policy_fn(
                    timestep.observation,
                    option_rng,
                )
            )

        else:

            option_id = (
                jax.random.randint(
                    option_rng,
                    shape=(),
                    minval=0,
                    maxval=NUM_OPTIONS,
                    dtype=jnp.int32,
                )
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
        # Enforce exact primitive-step evaluation budget
        # ----------------------------------------------------

        if (
            primitive_steps
            + duration
            > primitive_budget
        ):
            break


        timestep = (
            new_timestep
        )

        primitive_steps += (
            duration
        )

        option_decisions += 1


        if bool(
            valid
        ):
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
# Evaluate one policy across all 100 environments
# ============================================================

def evaluate_policy(
    policy_type,
    primitive_budget,
    trained_policy_fn=None,
):
    results = []


    for eval_seed in range(
        NUM_EVAL_EPISODES
    ):
        result = evaluate_episode(
            seed=eval_seed,

            primitive_budget=(
                primitive_budget
            ),

            policy_type=(
                policy_type
            ),

            trained_policy_fn=(
                trained_policy_fn
            ),
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


    option_decisions = np.asarray(
        [
            result[
                "option_decisions"
            ]
            for result in results
        ],
        dtype=np.float32,
    )


    valid_rates = np.asarray(
        [
            result[
                "valid_rate"
            ]
            for result in results
        ],
        dtype=np.float32,
    )


    primitive_steps = np.asarray(
        [
            result[
                "primitive_steps"
            ]
            for result in results
        ],
        dtype=np.float32,
    )


    return {
        "success_rate":
            successes.mean(),

        "valid_rate":
            valid_rates.mean(),

        "mean_option_decisions":
            option_decisions.mean(),

        "mean_primitive_steps":
            primitive_steps.mean(),
    }


# ============================================================
# Random baseline
#
# Only needs to be calculated once because it does not depend
# on a PPO training seed.
# ============================================================

print(
    "================================"
)

print(
    "RANDOM OPTION BASELINE"
)

print(
    "================================"
)


random_success = []

random_valid = []

random_decisions = []


for budget in EVAL_BUDGETS:

    result = evaluate_policy(
        policy_type="random",
        primitive_budget=budget,
    )

    random_success.append(
        result[
            "success_rate"
        ]
    )

    random_valid.append(
        result[
            "valid_rate"
        ]
    )

    random_decisions.append(
        result[
            "mean_option_decisions"
        ]
    )


random_success = np.asarray(
    random_success
)

random_valid = np.asarray(
    random_valid
)

random_decisions = np.asarray(
    random_decisions
)


# ============================================================
# Train + evaluate all five PPO seeds
# ============================================================

all_success = []

all_valid = []

all_decisions = []

training_summaries = []


for seed in TRAIN_SEEDS:

    params, training_summary = (
        train_seed(
            seed
        )
    )

    training_summaries.append(
        training_summary
    )


    trained_policy_fn = (
        make_trained_policy(
            params
        )
    )


    seed_success = []

    seed_valid = []

    seed_decisions = []


    print(
        "\n================================"
    )

    print(
        f"EVALUATING PPO TRAINING SEED {seed}"
    )

    print(
        "================================"
    )


    for budget in EVAL_BUDGETS:

        result = evaluate_policy(
            policy_type="trained",

            primitive_budget=(
                budget
            ),

            trained_policy_fn=(
                trained_policy_fn
            ),
        )


        seed_success.append(
            result[
                "success_rate"
            ]
        )

        seed_valid.append(
            result[
                "valid_rate"
            ]
        )

        seed_decisions.append(
            result[
                "mean_option_decisions"
            ]
        )


        print(
            f"Budget {budget:>3}: "
            f"success="
            f"{result['success_rate'] * 100:>5.1f}% | "
            f"valid="
            f"{result['valid_rate'] * 100:>5.1f}% | "
            f"decisions="
            f"{result['mean_option_decisions']:.2f}"
        )


    all_success.append(
        seed_success
    )

    all_valid.append(
        seed_valid
    )

    all_decisions.append(
        seed_decisions
    )


    # Release compilation / Python caches between seeds.
    jax.clear_caches()

    gc.collect()


# ============================================================
# Convert to arrays
#
# shape:
#     [training_seed, evaluation_budget]
# ============================================================

all_success = np.asarray(
    all_success,
    dtype=np.float32,
)

all_valid = np.asarray(
    all_valid,
    dtype=np.float32,
)

all_decisions = np.asarray(
    all_decisions,
    dtype=np.float32,
)


budgets = np.asarray(
    EVAL_BUDGETS
)


# ============================================================
# Mean and SEM across independent training seeds
# ============================================================

mean_success = (
    all_success.mean(
        axis=0
    )
)

sem_success = (
    all_success.std(
        axis=0,
        ddof=1,
    )
    / np.sqrt(
        len(
            TRAIN_SEEDS
        )
    )
)


mean_valid = (
    all_valid.mean(
        axis=0
    )
)

sem_valid = (
    all_valid.std(
        axis=0,
        ddof=1,
    )
    / np.sqrt(
        len(
            TRAIN_SEEDS
        )
    )
)


mean_decisions = (
    all_decisions.mean(
        axis=0
    )
)

sem_decisions = (
    all_decisions.std(
        axis=0,
        ddof=1,
    )
    / np.sqrt(
        len(
            TRAIN_SEEDS
        )
    )
)


# ============================================================
# Save complete experiment
# ============================================================

np.savez(
    "option_ppo_1m_5seeds_evaluation.npz",

    training_seeds=np.asarray(
        TRAIN_SEEDS
    ),

    budgets=budgets,

    all_success=all_success,

    mean_success=mean_success,

    sem_success=sem_success,

    all_valid=all_valid,

    mean_valid=mean_valid,

    sem_valid=sem_valid,

    all_decisions=(
        all_decisions
    ),

    mean_decisions=(
        mean_decisions
    ),

    sem_decisions=(
        sem_decisions
    ),

    random_success=(
        random_success
    ),

    random_valid=(
        random_valid
    ),

    random_decisions=(
        random_decisions
    ),
)


# ============================================================
# Plot 1
# Mean success ± SEM
# ============================================================

plt.figure(
    figsize=(9, 6)
)


for seed_index, seed in enumerate(
    TRAIN_SEEDS
):
    plt.plot(
        budgets,
        all_success[
            seed_index
        ],
        alpha=0.25,
        linewidth=1,
    )


plt.plot(
    budgets,
    mean_success,
    marker="o",
    linewidth=2.5,
    label="Option PPO 1M mean",
)


plt.fill_between(
    budgets,

    np.clip(
        mean_success
        - sem_success,
        0,
        1,
    ),

    np.clip(
        mean_success
        + sem_success,
        0,
        1,
    ),

    alpha=0.2,

    label="±1 SEM",
)


plt.plot(
    budgets,
    random_success,
    marker="o",
    linewidth=2,
    linestyle="--",
    label="Random options",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Success rate"
)

plt.title(
    "DoorKey 8x8 - 1M Option PPO Across 5 Seeds"
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
    "option_ppo_1m_5seeds_success.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 2
# Valid-option rate ± SEM
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    budgets,
    mean_valid,
    marker="o",
    linewidth=2.5,
    label="Option PPO 1M mean",
)


plt.fill_between(
    budgets,

    np.clip(
        mean_valid
        - sem_valid,
        0,
        1,
    ),

    np.clip(
        mean_valid
        + sem_valid,
        0,
        1,
    ),

    alpha=0.2,

    label="±1 SEM",
)


plt.plot(
    budgets,
    random_valid,
    marker="o",
    linewidth=2,
    linestyle="--",
    label="Random options",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Valid-option rate"
)

plt.title(
    "DoorKey 8x8 - Option Validity Across 5 Seeds"
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
    "option_ppo_1m_5seeds_validity.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 3
# Mean option decisions ± SEM
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    budgets,
    mean_decisions,
    marker="o",
    linewidth=2.5,
    label="Option PPO 1M mean",
)


plt.fill_between(
    budgets,

    mean_decisions
    - sem_decisions,

    mean_decisions
    + sem_decisions,

    alpha=0.2,

    label="±1 SEM",
)


plt.plot(
    budgets,
    random_decisions,
    marker="o",
    linewidth=2,
    linestyle="--",
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
    "option_ppo_1m_5seeds_decisions.png",
    dpi=200,
)

plt.close()


# ============================================================
# Final report
# ============================================================

print(
    "\n============================================"
)

print(
    "1M OPTION PPO - FIVE-SEED SUMMARY"
)

print(
    "============================================"
)


print(
    "Budget | Mean success | SEM | Random"
)

print(
    "--------------------------------------------"
)


for i, budget in enumerate(
    budgets
):

    print(
        f"{budget:>6} | "
        f"{mean_success[i] * 100:>10.1f}% | "
        f"{sem_success[i] * 100:>4.1f}% | "
        f"{random_success[i] * 100:>6.1f}%"
    )


print(
    "\nPer-training-seed success rates:"
)


for seed_index, seed in enumerate(
    TRAIN_SEEDS
):

    values = " | ".join(
        [
            f"{value * 100:.0f}%"
            for value in all_success[
                seed_index
            ]
        ]
    )

    print(
        f"Seed {seed}: {values}"
    )


print(
    "\nGenerated plots:"
)

print(
    "option_ppo_1m_5seeds_success.png"
)

print(
    "option_ppo_1m_5seeds_validity.png"
)

print(
    "option_ppo_1m_5seeds_decisions.png"
)


print(
    "\nSaved raw results:"
)

print(
    "option_ppo_1m_5seeds_evaluation.npz"
)


print(
    "\nFive-seed Option PPO experiment complete."
)