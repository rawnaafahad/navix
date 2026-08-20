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
from doorkey_reward_ablation import shaped_doorkey_reward


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "Navix-DoorKey-Random-8x8-v0"

TRAIN_BUDGET = 1_000_000

TRAIN_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

NUM_OPTIONS = 5

GAMMA = 0.99


# ============================================================
# OBSERVATION WRAPPER
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
# CREATE SHAPED-REWARD ENVIRONMENT
# ============================================================

def create_shaped_env():

    env = nx.make(
        ENV_ID,
        observation_fn=(
            observations.symbolic_first_person
        ),
        gamma=GAMMA,
    )

    env = env.replace(
        reward_fn=shaped_doorkey_reward
    )

    env = FlattenObsWrapper(
        env
    )

    return env


# ============================================================
# CHECKPOINT PATHS
# ============================================================

def option_checkpoint_path(
    seed,
):
    return (
        f"reward_ablation_option_ppo_"
        f"shaped_1m_seed{seed}.pkl"
    )


def primitive_checkpoint_path(
    seed,
):
    return (
        f"reward_ablation_primitive_ppo_"
        f"shaped_1m_seed{seed}.pkl"
    )


# ============================================================
# SAVE LIGHTWEIGHT CHECKPOINT
# ============================================================

def save_lightweight_checkpoint(
    filename,
    train_state,
):

    checkpoint = {
        "params":
            jax.device_get(
                train_state.params
            ),

        "frames":
            int(
                train_state.frames
            ),

        "step":
            int(
                train_state.step
            ),

        "env_id":
            ENV_ID,

        "reward_mode":
            "shaped",

        "train_budget":
            TRAIN_BUDGET,
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
# TRAIN SHAPED OPTION PPO
# ============================================================

def train_option_checkpoint(
    env,
    seed,
):

    filename = option_checkpoint_path(
        seed
    )


    if os.path.exists(
        filename
    ):

        print(
            "Shaped Option PPO checkpoint exists:",
            filename,
        )

        return


    print()
    print(
        "TRAINING SHAPED OPTION PPO"
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
            action_dim=NUM_OPTIONS,
        ),
        env=env,
    )


    rng = jax.random.PRNGKey(
        seed
    )


    train_state, logs = (
        agent.train(
            rng
        )
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
# TRAIN SHAPED PRIMITIVE PPO
# ============================================================

def train_primitive_checkpoint(
    env,
    seed,
):

    filename = primitive_checkpoint_path(
        seed
    )


    if os.path.exists(
        filename
    ):

        print(
            "Shaped Primitive PPO checkpoint exists:",
            filename,
        )

        return


    print()
    print(
        "TRAINING SHAPED PRIMITIVE PPO"
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
            ),
        ),
        env=env,
    )


    rng = jax.random.PRNGKey(
        seed
    )


    train_state, logs = (
        agent.train(
            rng
        )
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
    "DOORKEY REWARD-ABLATION CHECKPOINT TRAINING"
)

print(
    "============================================"
)

print(
    "Environment:",
    ENV_ID
)

print(
    "Reward mode:",
    "SHAPED"
)

print(
    "Training budget:",
    TRAIN_BUDGET
)

print(
    "Training seeds:",
    TRAIN_SEEDS
)

print(
    "Shaping:"
)

print(
    "  key pickup  = +0.25"
)

print(
    "  door opened = +0.25"
)

print(
    "  goal reached = +0.50"
)

print()


env = create_shaped_env()


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


    # ========================================================
    # OPTION PPO
    # ========================================================

    train_option_checkpoint(
        env,
        seed,
    )


    jax.clear_caches()

    gc.collect()


    # ========================================================
    # PRIMITIVE PPO
    # ========================================================

    train_primitive_checkpoint(
        env,
        seed,
    )


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
    "SHAPED CHECKPOINT TRAINING COMPLETE"
)

print(
    "============================================"
)

print(
    "All existing shaped checkpoints were reused."
)

print(
    "All missing shaped checkpoints were trained and saved."
)

print()
print(
    "Sparse 1M checkpoints were not modified."
)