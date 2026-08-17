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


TRAIN_SEEDS = [0, 1, 2, 3, 4]
TRAIN_BUDGETS = [100_000, 250_000, 500_000, 750_000, 1_000_000]
EVAL_BUDGETS = [15, 20, 25, 30, 40, 50, 75, 100]

NUM_EVAL_EPISODES = 100
NUM_OPTIONS = 5
GAMMA = 0.99
RESUME_FROM_CHECKPOINTS = True


def budget_label(budget):
    if budget == 1_000_000:
        return "1m"
    if budget % 1_000 == 0:
        return f"{budget // 1_000}k"
    return str(budget)


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)

    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(
            shape=flatten_obs_shape
        ),
    )


env_id = "Navix-DoorKey-Random-8x8-v0"

env = nx.make(
    env_id,
    observation_fn=observations.symbolic_first_person,
    gamma=GAMMA,
)

env = FlattenObsWrapper(env)

network = ActorCritic(action_dim=NUM_OPTIONS)

execute_option_fn = jax.jit(
    lambda timestep, option_id:
        execute_option_jax(
            env,
            timestep,
            option_id,
            gamma=GAMMA,
        )
)


def checkpoint_path(train_budget, seed):
    return f"option_ppo_{budget_label(train_budget)}_seed{seed}.pkl"


def training_metrics_path(train_budget, seed):
    return (
        f"option_ppo_{budget_label(train_budget)}_"
        f"seed{seed}_training.npz"
    )


def save_checkpoint(
    train_budget,
    seed,
    params,
    training_summary,
):
    checkpoint = {
        "params": params,
        "train_seed": seed,
        "train_budget": train_budget,
        "env_id": env_id,
        "num_options": NUM_OPTIONS,
        "training_summary": training_summary,
    }

    path = checkpoint_path(train_budget, seed)

    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)

    print("Saved checkpoint:", path)

    return path


def load_checkpoint(
    path,
    expected_budget,
    expected_seed,
):
    with open(path, "rb") as file:
        checkpoint = pickle.load(file)

    assert checkpoint["env_id"] == env_id
    assert checkpoint["num_options"] == NUM_OPTIONS
    assert checkpoint["train_budget"] == expected_budget
    assert checkpoint["train_seed"] == expected_seed

    return checkpoint


def create_agent(train_budget):
    config = PPOHparams().replace(
        budget=train_budget,
        num_envs=16,
        num_steps=128,
        num_minibatches=8,
        num_epochs=1,
        lr=0.00025,
        anneal_lr=False,
    )

    return OptionPPO(
        hparams=config,
        network=network,
        env=env,
    )


def get_trained_params(
    train_budget,
    seed,
):
    path = checkpoint_path(
        train_budget,
        seed,
    )

    if (
        RESUME_FROM_CHECKPOINTS
        and os.path.exists(path)
    ):
        print(
            f"\nTraining budget "
            f"{budget_label(train_budget)} "
            f"| Seed {seed}: loading existing checkpoint..."
        )

        checkpoint = load_checkpoint(
            path=path,
            expected_budget=train_budget,
            expected_seed=seed,
        )

        return (
            checkpoint["params"],
            checkpoint.get(
                "training_summary",
                {},
            ),
        )

    legacy_seed0_path = "option_ppo_1m_checkpoint.pkl"

    if (
        train_budget == 1_000_000
        and seed == 0
        and RESUME_FROM_CHECKPOINTS
        and os.path.exists(legacy_seed0_path)
    ):
        print(
            "\n1M | Seed 0: reusing legacy "
            "option_ppo_1m_checkpoint.pkl"
        )

        with open(
            legacy_seed0_path,
            "rb",
        ) as file:
            checkpoint = pickle.load(file)

        assert checkpoint["env_id"] == env_id
        assert checkpoint["num_options"] == NUM_OPTIONS
        assert checkpoint["train_budget"] == 1_000_000
        assert checkpoint["train_seed"] == 0

        summary = checkpoint.get(
            "training_summary",
            {
                "source": legacy_seed0_path,
            },
        )

        save_checkpoint(
            train_budget=train_budget,
            seed=seed,
            params=checkpoint["params"],
            training_summary=summary,
        )

        return (
            checkpoint["params"],
            summary,
        )

    print("\n========================================")
    print("TRAINING OPTION PPO")
    print("Training budget:", train_budget)
    print("Seed:", seed)
    print("========================================")

    agent = create_agent(train_budget)

    rng = jax.random.PRNGKey(seed)

    train_state, logs = agent.train(rng)

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

    option_frequencies = []

    for option_id in range(NUM_OPTIONS):
        frequency = np.asarray(
            logs[
                f"option/frequency_{option_id}"
            ]
        )[active]

        option_frequencies.append(frequency)

    option_frequencies = np.asarray(
        option_frequencies
    )

    np.savez(
        training_metrics_path(
            train_budget,
            seed,
        ),
        frames=frames,
        valid_rate=valid_rate,
        mean_duration=mean_duration,
        option_frequencies=option_frequencies,
    )

    training_summary = {
        "final_frames": int(train_state.frames),
        "option_decisions": int(
            train_state.option_decisions
        ),
        "optimiser_steps": int(
            train_state.step
        ),
        "active_updates": int(
            len(frames)
        ),
        "final_valid_rate": float(
            valid_rate[-1]
        ),
        "final_mean_duration": float(
            mean_duration[-1]
        ),
        "final_option_frequencies":
            option_frequencies[:, -1].tolist(),
    }

    print("\nTraining summary:")
    print(
        "Final primitive frames:",
        training_summary["final_frames"]
    )
    print(
        "Final valid-option rate:",
        training_summary["final_valid_rate"]
    )
    print(
        "Final mean option duration:",
        training_summary["final_mean_duration"]
    )

    save_checkpoint(
        train_budget=train_budget,
        seed=seed,
        params=train_state.params,
        training_summary=training_summary,
    )

    params = train_state.params

    del train_state
    del logs
    del agent

    jax.clear_caches()
    gc.collect()

    return (
        params,
        training_summary,
    )


def make_trained_policy(params):
    def select_option(
        observation,
        rng,
    ):
        pi = network.apply(
            params,
            observation,
            method="policy",
        )

        return pi.sample(seed=rng)

    return jax.jit(select_option)


def evaluate_episode(
    eval_seed,
    primitive_budget,
    trained_policy_fn,
):
    reset_rng = jax.random.PRNGKey(
        eval_seed
    )

    policy_rng = jax.random.PRNGKey(
        eval_seed + 1_000_000
    )

    timestep = env.reset(
        reset_rng
    )

    primitive_steps = 0
    option_decisions = 0
    valid_options = 0
    invalid_options = 0

    while (
        not bool(timestep.is_done())
        and primitive_steps < primitive_budget
    ):
        policy_rng, option_rng = (
            jax.random.split(policy_rng)
        )

        option_id = trained_policy_fn(
            timestep.observation,
            option_rng,
        )

        (
            new_timestep,
            duration,
            option_reward,
            valid,
        ) = execute_option_fn(
            timestep,
            option_id,
        )

        duration = int(duration)

        if (
            primitive_steps + duration
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

    success = (
        bool(timestep.is_done())
        and float(timestep.reward) > 0.0
    )

    total_options = (
        valid_options
        + invalid_options
    )

    valid_rate = (
        valid_options / total_options
        if total_options > 0
        else 0.0
    )

    return {
        "success": success,
        "primitive_steps": primitive_steps,
        "option_decisions": option_decisions,
        "valid_rate": valid_rate,
    }


def evaluate_policy(
    primitive_budget,
    trained_policy_fn,
):
    results = []

    for eval_seed in range(
        NUM_EVAL_EPISODES
    ):
        result = evaluate_episode(
            eval_seed=eval_seed,
            primitive_budget=primitive_budget,
            trained_policy_fn=trained_policy_fn,
        )

        results.append(result)

    successes = np.asarray(
        [
            result["success"]
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

    option_decisions = np.asarray(
        [
            result["option_decisions"]
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

    return {
        "success_rate": successes.mean(),
        "valid_rate": valid_rates.mean(),
        "mean_option_decisions":
            option_decisions.mean(),
        "mean_primitive_steps":
            primitive_steps.mean(),
    }


all_success = []
all_valid = []
all_decisions = []

print(
    "============================================"
)
print(
    "OPTION PPO SAMPLE-EFFICIENCY EXPERIMENT"
)
print(
    "============================================"
)
print(
    "Training budgets:",
    TRAIN_BUDGETS
)
print(
    "Training seeds:",
    TRAIN_SEEDS
)
print(
    "Evaluation budgets:",
    EVAL_BUDGETS
)
print(
    "Evaluation episodes per policy/budget:",
    NUM_EVAL_EPISODES
)


for train_budget in TRAIN_BUDGETS:
    budget_success = []
    budget_valid = []
    budget_decisions = []

    print(
        "\n############################################"
    )
    print(
        "TRAINING BUDGET:",
        train_budget
    )
    print(
        "############################################"
    )

    for seed in TRAIN_SEEDS:
        params, training_summary = (
            get_trained_params(
                train_budget=train_budget,
                seed=seed,
            )
        )

        trained_policy_fn = (
            make_trained_policy(params)
        )

        seed_success = []
        seed_valid = []
        seed_decisions = []

        print(
            "\n----------------------------------------"
        )
        print("EVALUATING")
        print(
            "Training budget:",
            train_budget
        )
        print(
            "Training seed:",
            seed
        )
        print(
            "----------------------------------------"
        )

        for eval_budget in EVAL_BUDGETS:
            result = evaluate_policy(
                primitive_budget=eval_budget,
                trained_policy_fn=(
                    trained_policy_fn
                ),
            )

            seed_success.append(
                result["success_rate"]
            )
            seed_valid.append(
                result["valid_rate"]
            )
            seed_decisions.append(
                result[
                    "mean_option_decisions"
                ]
            )

            print(
                f"Eval budget {eval_budget:>3}: "
                f"success="
                f"{result['success_rate'] * 100:>5.1f}% | "
                f"valid="
                f"{result['valid_rate'] * 100:>5.1f}% | "
                f"decisions="
                f"{result['mean_option_decisions']:.2f}"
            )

        budget_success.append(
            seed_success
        )
        budget_valid.append(
            seed_valid
        )
        budget_decisions.append(
            seed_decisions
        )

        del params
        del trained_policy_fn

        jax.clear_caches()
        gc.collect()

    all_success.append(
        budget_success
    )
    all_valid.append(
        budget_valid
    )
    all_decisions.append(
        budget_decisions
    )


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

train_budgets = np.asarray(
    TRAIN_BUDGETS,
    dtype=np.int64,
)

eval_budgets = np.asarray(
    EVAL_BUDGETS,
    dtype=np.int64,
)


mean_success = all_success.mean(
    axis=1
)

sem_success = (
    all_success.std(
        axis=1,
        ddof=1,
    )
    / np.sqrt(
        len(TRAIN_SEEDS)
    )
)

mean_valid = all_valid.mean(
    axis=1
)

sem_valid = (
    all_valid.std(
        axis=1,
        ddof=1,
    )
    / np.sqrt(
        len(TRAIN_SEEDS)
    )
)

mean_decisions = (
    all_decisions.mean(
        axis=1
    )
)

sem_decisions = (
    all_decisions.std(
        axis=1,
        ddof=1,
    )
    / np.sqrt(
        len(TRAIN_SEEDS)
    )
)


np.savez(
    "option_ppo_sample_efficiency_5seeds.npz",
    training_budgets=train_budgets,
    evaluation_budgets=eval_budgets,
    training_seeds=np.asarray(
        TRAIN_SEEDS
    ),
    all_success=all_success,
    mean_success=mean_success,
    sem_success=sem_success,
    all_valid=all_valid,
    mean_valid=mean_valid,
    sem_valid=sem_valid,
    all_decisions=all_decisions,
    mean_decisions=mean_decisions,
    sem_decisions=sem_decisions,
)


selected_eval_budgets = [
    20,
    25,
    30,
    40,
]

plt.figure(figsize=(9, 6))

for eval_budget in selected_eval_budgets:
    eval_index = EVAL_BUDGETS.index(
        eval_budget
    )

    means = mean_success[
        :,
        eval_index,
    ]

    sems = sem_success[
        :,
        eval_index,
    ]

    plt.plot(
        train_budgets,
        means,
        marker="o",
        linewidth=2.2,
        label=(
            f"{eval_budget}-step "
            f"evaluation budget"
        ),
    )

    plt.fill_between(
        train_budgets,
        np.clip(
            means - sems,
            0,
            1,
        ),
        np.clip(
            means + sems,
            0,
            1,
        ),
        alpha=0.12,
    )

plt.xlabel(
    "Primitive environment frames used for training"
)
plt.ylabel(
    "Success rate"
)
plt.title(
    "DoorKey 8x8 - Option PPO Sample Efficiency Across 5 Seeds"
)
plt.ylim(
    0,
    1.05
)
plt.xscale(
    "log"
)
plt.xticks(
    train_budgets,
    [
        "100k",
        "250k",
        "500k",
        "750k",
        "1M",
    ],
)
plt.legend()
plt.grid(
    True,
    alpha=0.3,
)
plt.tight_layout()

plt.savefig(
    "option_ppo_sample_efficiency_5seeds.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


plt.figure(figsize=(9, 6))

for train_index, train_budget in enumerate(
    TRAIN_BUDGETS
):
    plt.plot(
        eval_budgets,
        mean_success[
            train_index
        ],
        marker="o",
        linewidth=2,
        label=(
            f"{budget_label(train_budget)} "
            f"training"
        ),
    )

plt.xlabel(
    "Primitive-step evaluation budget"
)
plt.ylabel(
    "Success rate"
)
plt.title(
    "DoorKey 8x8 - Performance vs Training Budget"
)
plt.ylim(
    0,
    1.05
)
plt.legend()
plt.grid(
    True,
    alpha=0.3,
)
plt.tight_layout()

plt.savefig(
    "option_ppo_training_budget_sweep_5seeds.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


validity_eval_budget = 30
validity_index = EVAL_BUDGETS.index(
    validity_eval_budget
)

validity_mean = mean_valid[
    :,
    validity_index,
]

validity_sem = sem_valid[
    :,
    validity_index,
]

plt.figure(figsize=(9, 6))

plt.plot(
    train_budgets,
    validity_mean,
    marker="o",
    linewidth=2.5,
)

plt.fill_between(
    train_budgets,
    np.clip(
        validity_mean
        - validity_sem,
        0,
        1,
    ),
    np.clip(
        validity_mean
        + validity_sem,
        0,
        1,
    ),
    alpha=0.2,
)

plt.xlabel(
    "Primitive environment frames used for training"
)
plt.ylabel(
    "Valid-option rate"
)
plt.title(
    "DoorKey 8x8 - Option Validity vs Training Budget\n"
    "30-Step Evaluation Budget"
)
plt.ylim(
    0,
    1
)
plt.xscale(
    "log"
)
plt.xticks(
    train_budgets,
    [
        "100k",
        "250k",
        "500k",
        "750k",
        "1M",
    ],
)
plt.grid(
    True,
    alpha=0.3,
)
plt.tight_layout()

plt.savefig(
    "option_ppo_sample_efficiency_validity_5seeds.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


main_eval_budget = 30
main_eval_index = EVAL_BUDGETS.index(
    main_eval_budget
)

print(
    "\n============================================"
)
print(
    "OPTION PPO SAMPLE-EFFICIENCY SUMMARY"
)
print(
    "============================================"
)
print(
    f"Main evaluation budget: "
    f"{main_eval_budget} primitive steps"
)
print()
print(
    "Training budget | Mean success | SEM | Mean validity"
)
print(
    "-----------------------------------------------------"
)

for train_index, train_budget in enumerate(
    TRAIN_BUDGETS
):
    print(
        f"{budget_label(train_budget):>15} | "
        f"{mean_success[train_index, main_eval_index] * 100:>10.1f}% | "
        f"{sem_success[train_index, main_eval_index] * 100:>4.1f}% | "
        f"{mean_valid[train_index, main_eval_index] * 100:>11.1f}%"
    )

print(
    "\nPer-seed success at 30-step evaluation budget:"
)

for train_index, train_budget in enumerate(
    TRAIN_BUDGETS
):
    values = " | ".join(
        [
            f"{value * 100:.0f}%"
            for value in all_success[
                train_index,
                :,
                main_eval_index,
            ]
        ]
    )

    print(
        f"{budget_label(train_budget):>5}: "
        f"{values}"
    )

print(
    "\nGenerated plots:"
)
print(
    "option_ppo_sample_efficiency_5seeds.png"
)
print(
    "option_ppo_training_budget_sweep_5seeds.png"
)
print(
    "option_ppo_sample_efficiency_validity_5seeds.png"
)
print(
    "\nSaved raw results:"
)
print(
    "option_ppo_sample_efficiency_5seeds.npz"
)
print(
    "\nSample-efficiency experiment complete."
)