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

from option_ppo import OptionPPO
from jax_option_executor import execute_option_jax


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_BUDGET = 1_000_000
TRAIN_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS = list(range(100))

NUM_OPTIONS = 5
MAX_EVAL_STEPS = 100

ENVIRONMENTS = {
    "5x5": {
        "env_id": "Navix-DoorKey-Random-5x5-v0",
        "mean_expert_horizon": 10.80,
    },
    "8x8": {
        "env_id": "Navix-DoorKey-Random-8x8-v0",
        "mean_expert_horizon": 18.89,
    },
    "16x16": {
        "env_id": "Navix-DoorKey-Random-16x16-v0",
        "mean_expert_horizon": 33.81,
    },
}


# ============================================================
# OBSERVATION WRAPPER
# ============================================================

def FlattenObsWrapper(env: Environment):

    flatten_obs_fn = lambda x: jnp.ravel(
        env.observation_fn(x)
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
        observation_space=env.observation_space.replace(
            shape=flatten_obs_shape
        ),
    )


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def create_env(env_id):

    env = nx.make(
        env_id,
        observation_fn=observations.symbolic_first_person,
        gamma=0.99,
    )

    return FlattenObsWrapper(env)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def get_params(state_or_checkpoint):
    """
    Return network parameters from either a live TrainState
    or a lightweight checkpoint dictionary.
    """

    if isinstance(state_or_checkpoint, dict):
        return state_or_checkpoint["params"]

    return state_or_checkpoint.params


def save_checkpoint(filename, train_state):
    """Save only serialisable numerical state."""

    checkpoint = {
        "params": jax.device_get(train_state.params),
        "frames": int(train_state.frames),
        "step": int(train_state.step),
    }

    with open(filename, "wb") as file:
        pickle.dump(checkpoint, file)


def load_checkpoint(filename):

    with open(filename, "rb") as file:
        return pickle.load(file)


# ============================================================
# OPTION PPO TRAINING
# ============================================================

def train_option_ppo(
    env,
    env_label,
    seed,
):

    # Reuse our existing 8x8 1M checkpoints
    # rather than training them again.
    if env_label == "8x8":

        existing_filename = (
            f"option_ppo_1m_seed{seed}.pkl"
        )

        if os.path.exists(
            existing_filename
        ):

            print(
                "Reusing existing checkpoint:",
                existing_filename,
            )

            return load_checkpoint(
                existing_filename
            )

    filename = (
        f"horizon_option_ppo_"
        f"{env_label}_1m_seed{seed}.pkl"
    )

    if os.path.exists(
        filename
    ):

        print(
            "Loading saved checkpoint:",
            filename,
        )

        return load_checkpoint(
            filename
        )

    print()
    print(
        "TRAINING OPTION PPO"
    )

    print(
        "Environment:",
        env_label,
    )

    print(
        "Seed:",
        seed,
    )

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
            action_dim=NUM_OPTIONS
        ),
        env=env,
    )

    rng = jax.random.PRNGKey(
        seed
    )

    train_state, logs = agent.train(
        rng
    )

    save_checkpoint(
        filename,
        train_state,
    )

    print(
        "Saved:",
        filename,
    )

    print(
        "Final primitive frames:",
        int(train_state.frames),
    )

    # Keep only the lightweight state required for evaluation.
    # Returning the full JAX/Navix TrainState keeps optimizer and
    # compiled-function-related objects alive and can exhaust LLVM
    # section memory when several agents are trained sequentially.
    checkpoint = {
        "params": jax.device_get(
            train_state.params
        ),
        "frames": int(
            train_state.frames
        ),
        "step": int(
            train_state.step
        ),
    }

    del logs
    del train_state
    del agent

    jax.clear_caches()
    gc.collect()

    return checkpoint


# ============================================================
# PRIMITIVE PPO TRAINING
# ============================================================

def train_primitive_ppo(
    env,
    env_label,
    seed,
):

    # Reuse existing 8x8 primitive checkpoints.
    if env_label == "8x8":

        existing_filename = (
            f"primitive_ppo_1m_seed{seed}.pkl"
        )

        if os.path.exists(
            existing_filename
        ):

            print(
                "Reusing existing checkpoint:",
                existing_filename,
            )

            return load_checkpoint(
                existing_filename
            )

    filename = (
        f"horizon_primitive_ppo_"
        f"{env_label}_1m_seed{seed}.pkl"
    )

    if os.path.exists(
        filename
    ):

        print(
            "Loading saved checkpoint:",
            filename,
        )

        return load_checkpoint(
            filename
        )

    print()
    print(
        "TRAINING PRIMITIVE PPO"
    )

    print(
        "Environment:",
        env_label,
    )

    print(
        "Seed:",
        seed,
    )

    config = PPOHparams().replace(
        budget=TRAIN_BUDGET,
        num_envs=16,
        num_steps=128,
        num_minibatches=8,
        num_epochs=1,
        lr=0.00025,
        anneal_lr=False,
    )

    agent = PPO(
        hparams=config,
        network=ActorCritic(
            action_dim=len(
                env.action_set
            )
        ),
        env=env,
    )

    rng = jax.random.PRNGKey(
        seed
    )

    train_state, logs = agent.train(
        rng
    )

    save_checkpoint(
        filename,
        train_state,
    )

    print(
        "Saved:",
        filename,
    )

    print(
        "Final primitive frames:",
        int(train_state.frames),
    )

    # Keep only the lightweight state required for evaluation.
    # Returning the full JAX/Navix TrainState keeps optimizer and
    # compiled-function-related objects alive and can exhaust LLVM
    # section memory when several agents are trained sequentially.
    checkpoint = {
        "params": jax.device_get(
            train_state.params
        ),
        "frames": int(
            train_state.frames
        ),
        "step": int(
            train_state.step
        ),
    }

    del logs
    del train_state
    del agent

    jax.clear_caches()
    gc.collect()

    return checkpoint


# ============================================================
# STOCHASTIC ACTION SAMPLING
# ============================================================

def sample_action(
    network,
    params,
    observation,
    rng,
):

    distribution = network.apply(
        params,
        observation,
        method=network.policy,
    )

    action = distribution.sample(
        seed=rng
    )

    return int(action)


# ============================================================
# EVALUATE OPTION PPO
# ============================================================

def evaluate_option_policy(
    env,
    train_state,
):

    network = ActorCritic(
        action_dim=NUM_OPTIONS
    )

    execute_option_fn = jax.jit(
        lambda timestep, option_id:
            execute_option_jax(
                env,
                timestep,
                option_id,
            )
    )

    successes = []
    primitive_steps_list = []
    option_decisions_list = []
    valid_rates = []

    for seed in EVAL_SEEDS:

        reset_rng = jax.random.PRNGKey(
            seed
        )

        policy_rng = jax.random.PRNGKey(
            100_000 + seed
        )

        timestep = env.reset(
            reset_rng
        )

        primitive_steps = 0
        option_decisions = 0
        valid_decisions = 0

        while (
            primitive_steps
            < MAX_EVAL_STEPS
            and not bool(
                timestep.is_done()
            )
        ):

            policy_rng, action_rng = (
                jax.random.split(
                    policy_rng
                )
            )

            option_id = sample_action(
                network,
                get_params(
                    train_state
                ),
                timestep.observation,
                action_rng,
            )

            (
                new_timestep,
                duration,
                option_reward,
                valid,
            ) = execute_option_fn(
                timestep,
                jnp.asarray(
                    option_id,
                    dtype=jnp.int32,
                ),
            )

            duration = int(
                duration
            )

            if (
                primitive_steps
                + duration
                > MAX_EVAL_STEPS
            ):
                break

            timestep = new_timestep

            primitive_steps += duration
            option_decisions += 1

            if bool(valid):

                valid_decisions += 1

        success = (
            bool(
                timestep.is_done()
            )
            and float(
                timestep.reward
            ) > 0.0
        )

        successes.append(
            float(success)
        )

        primitive_steps_list.append(
            primitive_steps
        )

        option_decisions_list.append(
            option_decisions
        )

        if option_decisions > 0:

            valid_rates.append(
                valid_decisions
                / option_decisions
            )

        else:

            valid_rates.append(
                0.0
            )

    return {
        "success": np.mean(
            successes
        ),
        "mean_steps": np.mean(
            primitive_steps_list
        ),
        "mean_decisions": np.mean(
            option_decisions_list
        ),
        "valid_rate": np.mean(
            valid_rates
        ),
    }


# ============================================================
# EVALUATE PRIMITIVE PPO
# ============================================================

def evaluate_primitive_policy(
    env,
    train_state,
):

    network = ActorCritic(
        action_dim=len(
            env.action_set
        )
    )

    successes = []
    primitive_steps_list = []

    for seed in EVAL_SEEDS:

        reset_rng = jax.random.PRNGKey(
            seed
        )

        policy_rng = jax.random.PRNGKey(
            200_000 + seed
        )

        timestep = env.reset(
            reset_rng
        )

        primitive_steps = 0

        while (
            primitive_steps
            < MAX_EVAL_STEPS
            and not bool(
                timestep.is_done()
            )
        ):

            policy_rng, action_rng = (
                jax.random.split(
                    policy_rng
                )
            )

            action = sample_action(
                network,
                get_params(
                    train_state
                ),
                timestep.observation,
                action_rng,
            )

            timestep = env.step(
                timestep,
                jnp.asarray(
                    action,
                    dtype=jnp.int32,
                ),
            )

            primitive_steps += 1

        success = (
            bool(
                timestep.is_done()
            )
            and float(
                timestep.reward
            ) > 0.0
        )

        successes.append(
            float(success)
        )

        primitive_steps_list.append(
            primitive_steps
        )

    return {
        "success": np.mean(
            successes
        ),
        "mean_steps": np.mean(
            primitive_steps_list
        ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print(
    "============================================"
)

print(
    "TEMPORAL-HORIZON STRESS TEST"
)

print(
    "============================================"
)

print(
    "Training budget:",
    TRAIN_BUDGET,
)

print(
    "Training seeds:",
    TRAIN_SEEDS,
)

print(
    "Evaluation episodes per policy:",
    len(EVAL_SEEDS),
)

print(
    "Maximum evaluation primitive steps:",
    MAX_EVAL_STEPS,
)

print(
    "Matched PPO settings:",
    "num_envs=16, num_steps=128, num_minibatches=8, "
    "num_epochs=1, lr=0.00025, anneal_lr=False",
)


option_success = []
option_sem = []

primitive_success = []
primitive_sem = []

option_validity = []
option_decisions = []

horizons = []


for env_label, env_info in (
    ENVIRONMENTS.items()
):

    env_id = env_info[
        "env_id"
    ]

    horizon = env_info[
        "mean_expert_horizon"
    ]

    horizons.append(
        horizon
    )

    print()
    print()
    print(
        "=" * 60
    )

    print(
        "ENVIRONMENT:",
        env_label,
    )

    print(
        "Environment ID:",
        env_id,
    )

    print(
        "Measured expert primitive horizon:",
        horizon,
    )

    print(
        "=" * 60
    )

    env = create_env(
        env_id
    )

    env_option_success = []
    env_primitive_success = []

    env_option_validity = []
    env_option_decisions = []


    # ========================================================
    # Train/evaluate each seed
    # ========================================================

    for seed in TRAIN_SEEDS:

        print()
        print(
            "--------------------------------"
        )

        print(
            "TRAINING SEED:",
            seed,
        )

        print(
            "--------------------------------"
        )

        # ----------------------------------------------------
        # Option PPO
        # ----------------------------------------------------

        option_state = train_option_ppo(
            env,
            env_label,
            seed,
        )

        option_result = (
            evaluate_option_policy(
                env,
                option_state,
            )
        )

        env_option_success.append(
            option_result[
                "success"
            ]
        )

        env_option_validity.append(
            option_result[
                "valid_rate"
            ]
        )

        env_option_decisions.append(
            option_result[
                "mean_decisions"
            ]
        )

        print(
            f"Option PPO: "
            f"success="
            f"{option_result['success'] * 100:.1f}% | "
            f"valid="
            f"{option_result['valid_rate'] * 100:.1f}% | "
            f"decisions="
            f"{option_result['mean_decisions']:.2f}"
        )

        # Release the option-policy object and compiled evaluation
        # executables before training the primitive policy.
        del option_state
        jax.clear_caches()
        gc.collect()

        # ----------------------------------------------------
        # Primitive PPO
        # ----------------------------------------------------

        primitive_state = (
            train_primitive_ppo(
                env,
                env_label,
                seed,
            )
        )

        primitive_result = (
            evaluate_primitive_policy(
                env,
                primitive_state,
            )
        )

        env_primitive_success.append(
            primitive_result[
                "success"
            ]
        )

        print(
            f"Primitive PPO: "
            f"success="
            f"{primitive_result['success'] * 100:.1f}% | "
            f"mean steps="
            f"{primitive_result['mean_steps']:.2f}"
        )

        del primitive_state
        jax.clear_caches()
        gc.collect()


    # ========================================================
    # Aggregate environment results
    # ========================================================

    env_option_success = np.asarray(
        env_option_success
    )

    env_primitive_success = np.asarray(
        env_primitive_success
    )

    env_option_validity = np.asarray(
        env_option_validity
    )

    env_option_decisions = np.asarray(
        env_option_decisions
    )

    option_success.append(
        env_option_success.mean()
    )

    option_sem.append(
        env_option_success.std(
            ddof=1
        ) / np.sqrt(
            len(TRAIN_SEEDS)
        )
    )

    primitive_success.append(
        env_primitive_success.mean()
    )

    primitive_sem.append(
        env_primitive_success.std(
            ddof=1
        ) / np.sqrt(
            len(TRAIN_SEEDS)
        )
    )

    option_validity.append(
        env_option_validity.mean()
    )

    option_decisions.append(
        env_option_decisions.mean()
    )

    print()
    print(
        "ENVIRONMENT SUMMARY:",
        env_label,
    )

    print(
        "Option PPO:",
        f"{option_success[-1] * 100:.1f}% "
        f"± {option_sem[-1] * 100:.1f}%"
    )

    print(
        "Primitive PPO:",
        f"{primitive_success[-1] * 100:.1f}% "
        f"± {primitive_sem[-1] * 100:.1f}%"
    )

    print(
        "Option validity:",
        f"{option_validity[-1] * 100:.1f}%"
    )

    print(
        "Mean high-level decisions:",
        f"{option_decisions[-1]:.2f}"
    )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

horizons = np.asarray(
    horizons,
    dtype=np.float32,
)

option_success = np.asarray(
    option_success,
)

option_sem = np.asarray(
    option_sem,
)

primitive_success = np.asarray(
    primitive_success,
)

primitive_sem = np.asarray(
    primitive_sem,
)

option_validity = np.asarray(
    option_validity,
)

option_decisions = np.asarray(
    option_decisions,
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

np.savez(
    "doorkey_temporal_horizon_1m_5seeds.npz",

    horizons=horizons,

    env_labels=np.asarray(
        list(
            ENVIRONMENTS.keys()
        )
    ),

    option_success=option_success,
    option_sem=option_sem,

    primitive_success=primitive_success,
    primitive_sem=primitive_sem,

    option_validity=option_validity,
    option_decisions=option_decisions,
)


print()
print(
    "Saved:"
)

print(
    "doorkey_temporal_horizon_1m_5seeds.npz"
)


# ============================================================
# GRAPH 1 — SUCCESS VS TEMPORAL HORIZON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.errorbar(
    horizons,
    option_success,
    yerr=option_sem,
    marker="o",
    linewidth=2,
    capsize=4,
    label="Option PPO",
)

plt.errorbar(
    horizons,
    primitive_success,
    yerr=primitive_sem,
    marker="o",
    linewidth=2,
    capsize=4,
    label="Primitive PPO",
)

plt.xlabel(
    "Mean expert primitive solution length"
)

plt.ylabel(
    "Success rate"
)

plt.title(
    "Temporal-Horizon Stress Test\n"
    "Option PPO vs Primitive PPO"
)

plt.ylim(
    -0.02,
    1.02,
)

plt.xticks(
    horizons,
    [
        "5×5\n10.8",
        "8×8\n18.9",
        "16×16\n33.8",
    ],
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "doorkey_temporal_horizon_success.png",
    dpi=200,
)

plt.close()


# ============================================================
# GRAPH 2 — OPTION VALIDITY VS HORIZON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    horizons,
    option_validity,
    marker="o",
    linewidth=2,
)

plt.xlabel(
    "Mean expert primitive solution length"
)

plt.ylabel(
    "Valid-option rate"
)

plt.title(
    "Option PPO Validity vs Temporal Horizon"
)

plt.ylim(
    0,
    1,
)

plt.xticks(
    horizons,
    [
        "5×5\n10.8",
        "8×8\n18.9",
        "16×16\n33.8",
    ],
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    "doorkey_temporal_horizon_validity.png",
    dpi=200,
)

plt.close()


# ============================================================
# GRAPH 3 — HIGH-LEVEL DECISIONS VS PRIMITIVE HORIZON
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    horizons,
    option_decisions,
    marker="o",
    linewidth=2,
)

plt.xlabel(
    "Mean expert primitive solution length"
)

plt.ylabel(
    "Mean high-level option decisions"
)

plt.title(
    "Effective Decision Horizon under Option Control"
)

plt.xticks(
    horizons,
    [
        "5×5\n10.8",
        "8×8\n18.9",
        "16×16\n33.8",
    ],
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    "doorkey_temporal_horizon_decisions.png",
    dpi=200,
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print(
    "=" * 75
)

print(
    "TEMPORAL-HORIZON STRESS TEST SUMMARY"
)

print(
    "=" * 75
)

print(
    "Environment | Horizon | Option PPO | Primitive PPO | Advantage"
)

print(
    "-" * 75
)


for index, env_label in enumerate(
    ENVIRONMENTS.keys()
):

    advantage = (
        option_success[index]
        - primitive_success[index]
    )

    print(
        f"{env_label:>11} | "
        f"{horizons[index]:>7.2f} | "
        f"{option_success[index] * 100:>7.1f}% | "
        f"{primitive_success[index] * 100:>12.1f}% | "
        f"{advantage * 100:>8.1f} pp"
    )


print()
print(
    "Generated plots:"
)

print(
    "doorkey_temporal_horizon_success.png"
)

print(
    "doorkey_temporal_horizon_validity.png"
)

print(
    "doorkey_temporal_horizon_decisions.png"
)

print()
print(
    "Temporal-horizon experiment complete."
)