import gc
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment


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

NUM_ACTIONS = len(
    env.action_set
)


print(
    "Environment:",
    env_id
)

print(
    "Observation shape:",
    env.observation_space.shape
)

print(
    "Primitive action count:",
    NUM_ACTIONS
)

print(
    "Training seeds:",
    TRAIN_SEEDS
)

print(
    "Primitive-frame budget per seed:",
    TRAIN_BUDGET
)


# ============================================================
# PPO configuration
#
# Match the Option PPO experiment where applicable.
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
    action_dim=NUM_ACTIONS,
)


agent = PPO(
    hparams=config,
    network=network,
    env=env,
)


# ============================================================
# Checkpoint utilities
# ============================================================

def checkpoint_path(
    seed,
):

    return (
        f"primitive_ppo_1m_seed{seed}.pkl"
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

        "num_actions":
            NUM_ACTIONS,

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
        checkpoint["num_actions"]
        == NUM_ACTIONS
    )


    return checkpoint


# ============================================================
# Train one primitive PPO seed
# ============================================================

def train_one_seed(
    seed,
):

    path = checkpoint_path(
        seed
    )


    # --------------------------------------------------------
    # Resume if already completed
    # --------------------------------------------------------

    if (
        RESUME_FROM_CHECKPOINTS
        and os.path.exists(
            path
        )
    ):

        print(
            f"\nSeed {seed}: "
            f"loading existing checkpoint..."
        )


        checkpoint = (
            load_checkpoint(
                path
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
    # Train
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        f"TRAINING PRIMITIVE PPO SEED {seed}"
    )

    print(
        "================================"
    )


    rng = jax.random.PRNGKey(
        seed
    )


    train_state, logs = agent.train(
        rng
    )


    # --------------------------------------------------------
    # Training diagnostics
    # --------------------------------------------------------

    frames = np.asarray(
        logs[
            "iter/frames"
        ]
    )


    # PPO's standard trainer does not have the OptionPPO
    # active/inactive mask, so keep the complete frame trace.
    training_summary = {
        "final_frames":
            int(
                train_state.frames
            ),

        "optimiser_steps":
            int(
                train_state.step
            ),

        "num_logged_updates":
            int(
                len(
                    frames
                )
            ),
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
        "Final optimiser step:",
        training_summary[
            "optimiser_steps"
        ]
    )


    # --------------------------------------------------------
    # Save lightweight training metrics
    # --------------------------------------------------------

    np.savez(
        f"primitive_ppo_1m_seed{seed}_training.npz",

        frames=frames,
    )


    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

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
# Build stochastic trained primitive policy
# ============================================================

def make_trained_policy(
    params,
):

    def select_action(
        observation,
        rng,
    ):

        pi = network.apply(
            params,
            observation,
            method="policy",
        )


        action = pi.sample(
            seed=rng
        )


        return action


    return jax.jit(
        select_action
    )


# ============================================================
# JIT primitive environment step
# ============================================================

primitive_step_fn = jax.jit(
    lambda timestep, action:
        env.step(
            timestep,
            action,
        )
)


# ============================================================
# Evaluate one complete episode
# ============================================================

def evaluate_episode(
    seed,
    primitive_budget,
    trained_policy_fn,
):

    reset_rng = (
        jax.random.PRNGKey(
            seed
        )
    )


    # Use the same policy-randomness convention used in
    # Option PPO evaluation.
    policy_rng = (
        jax.random.PRNGKey(
            seed + 1_000_000
        )
    )


    timestep = env.reset(
        reset_rng
    )


    primitive_steps = 0


    while (
        not bool(
            timestep.is_done()
        )
        and primitive_steps
        < primitive_budget
    ):

        policy_rng, action_rng = (
            jax.random.split(
                policy_rng
            )
        )


        # ----------------------------------------------------
        # Primitive PPO action
        # ----------------------------------------------------

        action = trained_policy_fn(
            timestep.observation,
            action_rng,
        )


        # ----------------------------------------------------
        # Exactly one primitive environment transition
        # ----------------------------------------------------

        timestep = primitive_step_fn(
            timestep,
            action,
        )


        primitive_steps += 1


    # --------------------------------------------------------
    # True episode success
    # --------------------------------------------------------

    success = (
        bool(
            timestep.is_done()
        )
        and float(
            timestep.reward
        ) > 0.0
    )


    return {
        "success":
            success,

        "primitive_steps":
            primitive_steps,
    }


# ============================================================
# Evaluate one trained primitive policy
# ============================================================

def evaluate_policy(
    primitive_budget,
    trained_policy_fn,
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


    primitive_steps = np.asarray(
        [
            result[
                "primitive_steps"
            ]
            for result in results
        ],
        dtype=np.float32,
    )


    successful_mask = (
        successes.astype(
            bool
        )
    )


    if successful_mask.any():

        successful_mean_steps = (
            primitive_steps[
                successful_mask
            ].mean()
        )

    else:

        successful_mean_steps = (
            np.nan
        )


    return {
        "success_rate":
            successes.mean(),

        "mean_primitive_steps":
            primitive_steps.mean(),

        "successful_mean_steps":
            successful_mean_steps,
    }


# ============================================================
# Train and evaluate all five independent training seeds
# ============================================================

all_success = []

all_mean_steps = []

all_successful_steps = []

training_summaries = []


for seed in TRAIN_SEEDS:

    params, training_summary = (
        train_one_seed(
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

    seed_mean_steps = []

    seed_successful_steps = []


    print(
        "\n================================"
    )

    print(
        f"EVALUATING PRIMITIVE PPO "
        f"TRAINING SEED {seed}"
    )

    print(
        "================================"
    )


    for budget in EVAL_BUDGETS:

        result = evaluate_policy(
            primitive_budget=budget,

            trained_policy_fn=(
                trained_policy_fn
            ),
        )


        seed_success.append(
            result[
                "success_rate"
            ]
        )


        seed_mean_steps.append(
            result[
                "mean_primitive_steps"
            ]
        )


        seed_successful_steps.append(
            result[
                "successful_mean_steps"
            ]
        )


        print(
            f"Budget {budget:>3}: "
            f"success="
            f"{result['success_rate'] * 100:>5.1f}% | "
            f"mean steps="
            f"{result['mean_primitive_steps']:.2f}"
        )


    all_success.append(
        seed_success
    )


    all_mean_steps.append(
        seed_mean_steps
    )


    all_successful_steps.append(
        seed_successful_steps
    )


    # --------------------------------------------------------
    # Free compilation caches between independent agents
    # --------------------------------------------------------

    jax.clear_caches()

    gc.collect()


# ============================================================
# Convert to arrays
#
# Shape:
#   [training_seed, evaluation_budget]
# ============================================================

all_success = np.asarray(
    all_success,
    dtype=np.float32,
)


all_mean_steps = np.asarray(
    all_mean_steps,
    dtype=np.float32,
)


all_successful_steps = np.asarray(
    all_successful_steps,
    dtype=np.float32,
)


budgets = np.asarray(
    EVAL_BUDGETS
)


# ============================================================
# Mean ± SEM across five independently trained PPO agents
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


mean_steps = (
    all_mean_steps.mean(
        axis=0
    )
)


sem_steps = (
    all_mean_steps.std(
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
# Save results
# ============================================================

np.savez(
    "primitive_ppo_1m_5seeds_evaluation.npz",

    training_seeds=np.asarray(
        TRAIN_SEEDS
    ),

    budgets=budgets,

    all_success=all_success,

    mean_success=mean_success,

    sem_success=sem_success,

    all_mean_steps=(
        all_mean_steps
    ),

    mean_steps=mean_steps,

    sem_steps=sem_steps,

    all_successful_steps=(
        all_successful_steps
    ),
)


# ============================================================
# Plot 1
# Primitive PPO success ± SEM
# ============================================================

plt.figure(
    figsize=(9, 6)
)


# Individual seed curves
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


# Mean
plt.plot(
    budgets,
    mean_success,
    marker="o",
    linewidth=2.5,
    label="Primitive PPO 1M mean",
)


# SEM
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


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Success rate"
)

plt.title(
    "DoorKey 8x8 - Primitive PPO 1M Across 5 Seeds"
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
    "primitive_ppo_1m_5seeds_success.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 2
# Primitive interactions consumed
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(
    budgets,
    mean_steps,
    marker="o",
    linewidth=2.5,
    label="Primitive PPO 1M mean",
)


plt.fill_between(
    budgets,

    mean_steps
    - sem_steps,

    mean_steps
    + sem_steps,

    alpha=0.2,
    label="±1 SEM",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Mean primitive steps per episode"
)

plt.title(
    "DoorKey 8x8 - Primitive PPO Interaction Usage"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "primitive_ppo_1m_5seeds_steps.png",
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
    "PRIMITIVE PPO 1M - FIVE-SEED SUMMARY"
)

print(
    "============================================"
)


print(
    "Budget | Mean success | SEM"
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
        f"{sem_success[i] * 100:>4.1f}%"
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
        f"Seed {seed}: "
        f"{values}"
    )


print(
    "\nGenerated plots:"
)

print(
    "primitive_ppo_1m_5seeds_success.png"
)

print(
    "primitive_ppo_1m_5seeds_steps.png"
)


print(
    "\nSaved raw results:"
)

print(
    "primitive_ppo_1m_5seeds_evaluation.npz"
)


print(
    "\nPrimitive PPO five-seed experiment complete."
)