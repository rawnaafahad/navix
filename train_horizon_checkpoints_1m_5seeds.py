import gc
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment

from option_ppo import OptionPPO


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_BUDGET = 1_000_000
TRAIN_SEEDS = [0, 1, 2, 3, 4]

NUM_OPTIONS = 5

ENVIRONMENTS = {
    "5x5": "Navix-DoorKey-Random-5x5-v0",
    "8x8": "Navix-DoorKey-Random-8x8-v0",
    "16x16": "Navix-DoorKey-Random-16x16-v0",
}


# ============================================================
# OBSERVATION WRAPPER
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
# ENVIRONMENT
# ============================================================

def create_env(env_id):

    env = nx.make(
        env_id,
        observation_fn=observations.symbolic_first_person,
        gamma=0.99,
    )

    return FlattenObsWrapper(
        env
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def option_checkpoint_path(
    env_label,
    seed,
):

    if env_label == "8x8":
        return (
            f"option_ppo_1m_seed{seed}.pkl"
        )

    return (
        f"horizon_option_ppo_"
        f"{env_label}_1m_seed{seed}.pkl"
    )


def primitive_checkpoint_path(
    env_label,
    seed,
):

    if env_label == "8x8":
        return (
            f"primitive_ppo_1m_seed{seed}.pkl"
        )

    return (
        f"horizon_primitive_ppo_"
        f"{env_label}_1m_seed{seed}.pkl"
    )


def save_lightweight_checkpoint(
    filename,
    train_state,
):

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

    with open(
        filename,
        "wb",
    ) as file:

        pickle.dump(
            checkpoint,
            file,
        )


# ============================================================
# TRAIN OPTION PPO
# ============================================================

def train_option_checkpoint(
    env,
    env_label,
    seed,
):

    filename = option_checkpoint_path(
        env_label,
        seed,
    )

    if os.path.exists(
        filename
    ):

        print(
            "Option PPO checkpoint exists:",
            filename,
        )

        return


    print()
    print(
        "TRAINING OPTION PPO"
    )

    print(
        "Environment:",
        env_label
    )

    print(
        "Seed:",
        seed
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


    save_lightweight_checkpoint(
        filename,
        train_state,
    )


    print(
        "Saved:",
        filename
    )

    print(
        "Final primitive frames:",
        int(
            train_state.frames
        )
    )


    del train_state
    del logs
    del agent

    jax.clear_caches()

    gc.collect()


# ============================================================
# TRAIN PRIMITIVE PPO
# ============================================================

def train_primitive_checkpoint(
    env,
    env_label,
    seed,
):

    filename = primitive_checkpoint_path(
        env_label,
        seed,
    )

    if os.path.exists(
        filename
    ):

        print(
            "Primitive PPO checkpoint exists:",
            filename,
        )

        return


    print()
    print(
        "TRAINING PRIMITIVE PPO"
    )

    print(
        "Environment:",
        env_label
    )

    print(
        "Seed:",
        seed
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


    save_lightweight_checkpoint(
        filename,
        train_state,
    )


    print(
        "Saved:",
        filename
    )

    print(
        "Final primitive frames:",
        int(
            train_state.frames
        )
    )


    del train_state
    del logs
    del agent

    jax.clear_caches()

    gc.collect()


# ============================================================
# MAIN
# ============================================================

print(
    "============================================"
)

print(
    "DOORKEY HORIZON CHECKPOINT TRAINING"
)

print(
    "============================================"
)

print(
    "Training budget:",
    TRAIN_BUDGET
)

print(
    "Training seeds:",
    TRAIN_SEEDS
)

print()


for env_label, env_id in (
    ENVIRONMENTS.items()
):

    print()
    print(
        "=" * 60
    )

    print(
        "ENVIRONMENT:",
        env_label
    )

    print(
        env_id
    )

    print(
        "=" * 60
    )


    env = create_env(
        env_id
    )


    for seed in TRAIN_SEEDS:

        print()
        print(
            "--------------------------------"
        )

        print(
            "SEED:",
            seed
        )

        print(
            "--------------------------------"
        )


        # ----------------------------------------------
        # OPTION PPO
        # ----------------------------------------------

        train_option_checkpoint(
            env,
            env_label,
            seed,
        )


        # Aggressive cleanup before switching algorithm.
        jax.clear_caches()

        gc.collect()


        # ----------------------------------------------
        # PRIMITIVE PPO
        # ----------------------------------------------

        train_primitive_checkpoint(
            env,
            env_label,
            seed,
        )


        # Cleanup before next seed.
        jax.clear_caches()

        gc.collect()


    del env

    jax.clear_caches()

    gc.collect()


print()
print(
    "============================================"
)

print(
    "CHECKPOINT TRAINING COMPLETE"
)

print(
    "============================================"
)

print(
    "All existing checkpoints were reused."
)

print(
    "All missing checkpoints were trained and saved."
)