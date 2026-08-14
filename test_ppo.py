import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import navix as nx
from navix import observations
from navix.agents import PPO, PPOHparams, ActorCritic
from navix.environments.environment import Environment


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)

    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(
            shape=flatten_obs_shape
        ),
    )


# -----------------------------
# 1. Create environment
# -----------------------------
env_id = "Navix-DoorKey-Random-8x8-v0"

env = nx.make(
    env_id,
    observation_fn=observations.symbolic_first_person,
    gamma=0.99,
)

env = FlattenObsWrapper(env)

print("Environment:", env_id)
print("Observation shape:", env.observation_space.shape)
print("Number of actions:", len(env.action_set))


# -----------------------------
# 2. PPO configuration
# -----------------------------
ppo_config = PPOHparams().replace(
    budget=10_000_000
)

print("Training budget per seed:", ppo_config.budget)


# -----------------------------
# 3. Create PPO agent
# -----------------------------
agent = PPO(
    hparams=ppo_config,
    network=ActorCritic(
        action_dim=len(env.action_set),
    ),
    env=env,
)


# -----------------------------
# 4. Create 5-seed experiment
# -----------------------------
seeds = (0, 1, 2, 3, 4)

experiment = nx.Experiment(
    name="primitive-ppo-baseline-5seeds",
    agent=agent,
    env=env,
    env_id=env_id,
    seeds=seeds,
)


# -----------------------------
# 5. Train
# -----------------------------
print("\nStarting 5-seed PPO training...")

train_state, logs = experiment.run(do_log=False)

print("\nTraining completed successfully!")
print("Final frames:", train_state.frames)


# -----------------------------
# 6. Extract logs
# -----------------------------
frames = np.asarray(logs["iter/frames"])
returns = np.asarray(logs["returns"])
done_mask = np.asarray(logs["done_mask"])

print("Frames shape:", frames.shape)
print("Returns shape:", returns.shape)
print("Done mask shape:", done_mask.shape)


# Expected shapes:
# frames:    (n_seeds, n_updates)
# returns:   (n_seeds, n_updates, num_steps, num_envs)
# done_mask: (n_seeds, n_updates, num_steps, num_envs)


# -----------------------------
# 7. Calculate success rate
#    separately for each seed
# -----------------------------
all_success_rates = []

for seed_idx in range(len(seeds)):

    seed_frames = frames[seed_idx]
    seed_returns = returns[seed_idx]
    seed_done_mask = done_mask[seed_idx]

    seed_success_rate = []

    for update in range(len(seed_frames)):

        update_returns = seed_returns[update]
        update_done = seed_done_mask[update]

        completed_returns = update_returns[
            update_done.astype(bool)
        ]

        if len(completed_returns) > 0:
            seed_success_rate.append(
                completed_returns.mean()
            )
        else:
            seed_success_rate.append(np.nan)

    all_success_rates.append(seed_success_rate)


all_success_rates = np.asarray(all_success_rates)

print(
    "Success-rate array shape:",
    all_success_rates.shape
)


# -----------------------------
# 8. Smooth each seed
# -----------------------------
window = 50

all_smoothed = []

for seed_idx in range(len(seeds)):

    success_rate = all_success_rates[seed_idx]

    smoothed = []

    for i in range(len(success_rate)):

        start = max(0, i - window + 1)

        values = success_rate[start:i + 1]
        values = values[~np.isnan(values)]

        if len(values) > 0:
            smoothed.append(values.mean())
        else:
            smoothed.append(np.nan)

    all_smoothed.append(smoothed)


all_smoothed = np.asarray(all_smoothed)


# -----------------------------
# 9. Calculate mean + SEM
# -----------------------------
mean_success = np.nanmean(
    all_smoothed,
    axis=0
)

std_success = np.nanstd(
    all_smoothed,
    axis=0,
    ddof=1
)

sem_success = (
    std_success / np.sqrt(len(seeds))
)

mean_frames = np.mean(
    frames,
    axis=0
)


# -----------------------------
# 10. Plot each individual seed
# -----------------------------
plt.figure(figsize=(9, 6))

for seed_idx in range(len(seeds)):

    plt.plot(
        frames[seed_idx],
        all_smoothed[seed_idx],
        alpha=0.25,
        linewidth=1,
        label=f"Seed {seeds[seed_idx]}"
    )


# -----------------------------
# 11. Plot mean
# -----------------------------
plt.plot(
    mean_frames,
    mean_success,
    linewidth=2.5,
    label="Mean success rate"
)


# -----------------------------
# 12. Add SEM uncertainty band
# -----------------------------
plt.fill_between(
    mean_frames,
    mean_success - sem_success,
    mean_success + sem_success,
    alpha=0.2,
    label="±1 SEM"
)


plt.xlabel(
    "Primitive environment frames"
)

plt.ylabel(
    "Success rate"
)

plt.title(
    "Primitive PPO - Navix DoorKey Random 8x8\n"
    "5 Seeds"
)

plt.ylim(
    -0.02,
    1.02
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# -----------------------------
# 13. Save graph
# -----------------------------
output_file = (
    "primitive_ppo_10m_5seeds_success_rate.png"
)

plt.savefig(
    output_file,
    dpi=200
)

print("\nSaved plot as:")
print(output_file)


# -----------------------------
# 14. Print final statistics
# -----------------------------
final_mean = mean_success[-1]
final_sem = sem_success[-1]

print(
    "\nFinal smoothed mean success rate:",
    final_mean
)

print(
    "Final SEM:",
    final_sem
)

print(
    "Final result:",
    f"{final_mean:.4f} ± {final_sem:.4f}"
)


plt.show()