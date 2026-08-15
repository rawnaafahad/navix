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


# ============================================================
# Configuration
# ============================================================

TRAIN_SEED = 0
TRAIN_BUDGET = 1_000_000
NUM_OPTIONS = 5


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
    gamma=0.99,
)

env = FlattenObsWrapper(
    env
)


# ============================================================
# Option PPO configuration
#
# Identical to the successful 100k experiment except for
# the primitive-frame training budget.
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
# Experiment information
# ============================================================

print(
    "Environment:",
    env_id
)

print(
    "Observation shape:",
    env.observation_space.shape
)

print(
    "Number of high-level options:",
    agent.network.action_dim
)

print(
    "Primitive-frame budget:",
    config.budget
)

print(
    "Training seed:",
    TRAIN_SEED
)

print(
    "\nStarting 1M Option PPO training..."
)


# ============================================================
# Train
# ============================================================

rng = jax.random.PRNGKey(
    TRAIN_SEED
)

train_state, logs = agent.train(
    rng
)


print(
    "\nTraining completed."
)

print(
    "Final primitive frames:",
    train_state.frames
)

print(
    "Total option decisions:",
    train_state.option_decisions
)

print(
    "Final optimiser step:",
    train_state.step
)


# ============================================================
# Save trained model checkpoint
#
# This stores the trained network parameters so that future
# evaluation scripts can load the trained 1M policy without
# retraining it.
# ============================================================

checkpoint = {
    "params": train_state.params,
    "train_seed": TRAIN_SEED,
    "train_budget": TRAIN_BUDGET,
    "env_id": env_id,
    "num_options": NUM_OPTIONS,
}


with open(
    "option_ppo_1m_checkpoint.pkl",
    "wb",
) as file:

    pickle.dump(
        checkpoint,
        file,
    )


print(
    "\nSaved trained checkpoint:"
)

print(
    "option_ppo_1m_checkpoint.pkl"
)


# ============================================================
# Extract active PPO updates
# ============================================================

active = np.asarray(
    logs["iter/active"]
).astype(bool)


frames = np.asarray(
    logs["iter/frames"]
)[active]


valid_rate = np.asarray(
    logs["option/valid_rate"]
)[active]


mean_duration = np.asarray(
    logs["option/mean_duration"]
)[active]


print(
    "\nActive PPO updates:",
    len(frames)
)


# ============================================================
# Option-selection frequencies
# ============================================================

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


option_frequencies = np.asarray(
    option_frequencies
)


# ============================================================
# PPO losses
# ============================================================

total_loss = np.asarray(
    logs["loss/total_loss"]
)[active]


value_loss = np.asarray(
    logs["loss/value_loss"]
)[active]


actor_loss = np.asarray(
    logs["loss/actor_loss"]
)[active]


entropy = np.asarray(
    logs["loss/entropy"]
)[active]


# ============================================================
# Save training metrics
#
# True task success is deliberately NOT estimated from the
# training returns here. It will be measured separately using
# complete evaluation episodes.
# ============================================================

np.savez(
    "option_ppo_1m_results.npz",

    frames=frames,

    valid_rate=valid_rate,

    mean_duration=mean_duration,

    option_frequencies=(
        option_frequencies
    ),

    total_loss=total_loss,

    value_loss=value_loss,

    actor_loss=actor_loss,

    entropy=entropy,
)


print(
    "\nSaved raw training results:"
)

print(
    "option_ppo_1m_results.npz"
)


# ============================================================
# Plot 1
# Valid-option rate
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    frames,
    valid_rate,
)


plt.xlabel(
    "Primitive environment frames"
)

plt.ylabel(
    "Valid-option rate"
)

plt.title(
    "Option PPO 1M - Valid Option Selection"
)

plt.ylim(
    0,
    1
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_ppo_1m_valid_rate.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 2
# Mean option duration
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    frames,
    mean_duration,
)


plt.xlabel(
    "Primitive environment frames"
)

plt.ylabel(
    "Mean primitive steps per option"
)

plt.title(
    "Option PPO 1M - Temporal Abstraction"
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_ppo_1m_duration.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 3
# Option-selection frequencies
# ============================================================

option_names = [
    "GO_TO_KEY",
    "PICKUP_KEY",
    "GO_TO_DOOR",
    "OPEN_DOOR",
    "GO_TO_GOAL",
]


plt.figure(
    figsize=(9, 5)
)


for option_id in range(
    NUM_OPTIONS
):

    plt.plot(
        frames,
        option_frequencies[
            option_id
        ],
        label=option_names[
            option_id
        ],
    )


plt.xlabel(
    "Primitive environment frames"
)

plt.ylabel(
    "Selection frequency"
)

plt.title(
    "Option PPO 1M - Option Selection"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_ppo_1m_option_frequency.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 4
# PPO total loss
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    frames,
    total_loss,
)


plt.xlabel(
    "Primitive environment frames"
)

plt.ylabel(
    "Total PPO loss"
)

plt.title(
    "Option PPO 1M - Training Loss"
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    "option_ppo_1m_loss.png",
    dpi=200,
)

plt.close()


# ============================================================
# Final summary
# ============================================================

print(
    "\nGenerated plots:"
)

print(
    "option_ppo_1m_valid_rate.png"
)

print(
    "option_ppo_1m_duration.png"
)

print(
    "option_ppo_1m_option_frequency.png"
)

print(
    "option_ppo_1m_loss.png"
)


print(
    "\n================================"
)

print(
    "1M OPTION PPO TRAINING SUMMARY"
)

print(
    "================================"
)


print(
    "Final primitive frames:",
    int(
        train_state.frames
    )
)


print(
    "Total option decisions:",
    int(
        train_state.option_decisions
    )
)


print(
    "Final optimiser step:",
    int(
        train_state.step
    )
)


print(
    "Active PPO updates:",
    len(
        frames
    )
)


print(
    "Final valid-option rate:",
    float(
        valid_rate[-1]
    )
)


print(
    "Final mean option duration:",
    float(
        mean_duration[-1]
    )
)


print(
    "\nFinal option-selection frequencies:"
)


for option_id, option_name in enumerate(
    option_names
):

    print(
        f"{option_name}: "
        f"{option_frequencies[option_id, -1] * 100:.2f}%"
    )


print(
    "\nCheckpoint:"
)

print(
    "option_ppo_1m_checkpoint.pkl"
)


print(
    "\n1M Option PPO training complete."
)