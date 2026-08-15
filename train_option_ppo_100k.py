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
# ============================================================

config = PPOHparams().replace(
    budget=100_000,
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
        action_dim=5,
    ),
    env=env,
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
    "Number of high-level options:",
    agent.network.action_dim
)

print(
    "Primitive-frame budget:",
    config.budget
)

print(
    "\nStarting 100k Option PPO training..."
)


# ============================================================
# Train
# ============================================================

rng = jax.random.PRNGKey(
    0
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


# ============================================================
# Extract only active updates
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

done_mask = np.asarray(
    logs["done_mask"]
)[active]

returns = np.asarray(
    logs["returns"]
)[active]


print(
    "Active PPO updates:",
    len(frames)
)


# ============================================================
# Success rate
# ============================================================

success_rates = []

for update in range(
    len(frames)
):

    completed = (
        done_mask[update].astype(bool)
    )

    completed_returns = (
        returns[update][completed]
    )

    if len(completed_returns) > 0:

        success_rate = np.mean(
            completed_returns > 0.0
        )

    else:

        success_rate = np.nan

    success_rates.append(
        success_rate
    )


success_rates = np.asarray(
    success_rates
)


# ============================================================
# Smooth success rate
# ============================================================

window = 5

smoothed_success = []

for i in range(
    len(success_rates)
):

    start = max(
        0,
        i - window + 1
    )

    values = success_rates[
        start:i + 1
    ]

    values = values[
        ~np.isnan(values)
    ]

    if len(values) > 0:

        smoothed_success.append(
            values.mean()
        )

    else:

        smoothed_success.append(
            np.nan
        )


smoothed_success = np.asarray(
    smoothed_success
)


# ============================================================
# Option frequencies
# ============================================================

option_frequencies = []

for option_id in range(5):

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
# Save raw results
# ============================================================

np.savez(
    "option_ppo_100k_results.npz",
    frames=frames,
    success_rate=success_rates,
    smoothed_success=smoothed_success,
    valid_rate=valid_rate,
    mean_duration=mean_duration,
    option_frequencies=option_frequencies,
)


print(
    "\nSaved raw results:"
)

print(
    "option_ppo_100k_results.npz"
)


# ============================================================
# Plot 1:
# Success rate
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    frames,
    success_rates,
    alpha=0.4,
    label="Success rate",
)

plt.plot(
    frames,
    smoothed_success,
    linewidth=2,
    label="5-update moving average",
)

plt.xlabel(
    "Primitive environment frames"
)

plt.ylabel(
    "Success rate"
)

plt.title(
    "Option PPO - DoorKey 8x8"
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
    "option_ppo_100k_success_rate.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 2:
# Valid option rate
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
    "Valid option rate"
)

plt.title(
    "Option PPO - Valid Option Selection"
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
    "option_ppo_100k_valid_rate.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 3:
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
    "Option PPO - Temporal Abstraction"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "option_ppo_100k_duration.png",
    dpi=200,
)

plt.close()


# ============================================================
# Plot 4:
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

for option_id in range(5):

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
    "Option PPO - Option Selection"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "option_ppo_100k_option_frequency.png",
    dpi=200,
)

plt.close()


# ============================================================
# Final summary
# ============================================================

valid_success_values = (
    smoothed_success[
        ~np.isnan(
            smoothed_success
        )
    ]
)


print(
    "\nGenerated plots:"
)

print(
    "option_ppo_100k_success_rate.png"
)

print(
    "option_ppo_100k_valid_rate.png"
)

print(
    "option_ppo_100k_duration.png"
)

print(
    "option_ppo_100k_option_frequency.png"
)


if len(
    valid_success_values
) > 0:

    print(
        "\nFinal smoothed success rate:",
        valid_success_values[-1],
    )


print(
    "Final valid option rate:",
    valid_rate[-1],
)

print(
    "Final mean option duration:",
    mean_duration[-1],
)