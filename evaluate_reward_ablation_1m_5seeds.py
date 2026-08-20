import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import navix as nx
from navix import observations
from navix.agents import ActorCritic
from navix.environments.environment import Environment

from jax_option_executor import execute_option_jax
from doorkey_reward_ablation import (
    sparse_doorkey_reward,
    shaped_doorkey_reward,
)


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "Navix-DoorKey-Random-8x8-v0"

TRAIN_SEEDS = [
    0,
    1,
    2,
    3,
    4,
]

EVAL_SEEDS = list(
    range(100)
)

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
# ENVIRONMENT CREATION
# ============================================================

def create_env(
    reward_mode,
):

    env = nx.make(
        ENV_ID,
        observation_fn=(
            observations.symbolic_first_person
        ),
        gamma=GAMMA,
    )

    if reward_mode == "sparse":

        env = env.replace(
            reward_fn=sparse_doorkey_reward
        )

    elif reward_mode == "shaped":

        env = env.replace(
            reward_fn=shaped_doorkey_reward
        )

    else:

        raise ValueError(
            f"Unknown reward mode: "
            f"{reward_mode}"
        )

    return FlattenObsWrapper(
        env
    )


# ============================================================
# CHECKPOINT PATHS
# ============================================================

def option_checkpoint_path(
    reward_mode,
    seed,
):

    if reward_mode == "sparse":

        return (
            f"option_ppo_1m_seed{seed}.pkl"
        )

    return (
        f"reward_ablation_option_ppo_"
        f"shaped_1m_seed{seed}.pkl"
    )


def primitive_checkpoint_path(
    reward_mode,
    seed,
):

    if reward_mode == "sparse":

        return (
            f"primitive_ppo_1m_seed{seed}.pkl"
        )

    return (
        f"reward_ablation_primitive_ppo_"
        f"shaped_1m_seed{seed}.pkl"
    )


# ============================================================
# LOAD PARAMS
# ============================================================

def load_params(
    filename,
):

    with open(
        filename,
        "rb",
    ) as file:

        checkpoint = (
            pickle.load(
                file
            )
        )

    if isinstance(
        checkpoint,
        dict
    ):

        return checkpoint[
            "params"
        ]

    return checkpoint.params


# ============================================================
# TRUE TASK SUCCESS
#
# Do not use reward > 0 in the shaped environment because
# pickup / door-opening rewards can be positive before the
# task is solved.
# ============================================================

def reached_goal(
    timestep,
):

    player = timestep.state.get_player(
        idx=0
    )

    goals = timestep.state.get_goals()

    goal_position = (
        goals.position[0]
    )

    return bool(
        jnp.all(
            player.position
            == goal_position
        )
    )


# ============================================================
# BUILD OPTION POLICY
# ============================================================

def make_option_policy(
    params,
):

    network = ActorCritic(
        action_dim=NUM_OPTIONS
    )

    def select_option(
        observation,
        rng,
    ):

        distribution = network.apply(
            params,
            observation,
            method=network.policy,
        )

        return distribution.sample(
            seed=rng
        )

    return jax.jit(
        select_option
    )


# ============================================================
# BUILD PRIMITIVE POLICY
# ============================================================

def make_primitive_policy(
    params,
    num_actions,
):

    network = ActorCritic(
        action_dim=num_actions
    )

    def select_action(
        observation,
        rng,
    ):

        distribution = network.apply(
            params,
            observation,
            method=network.policy,
        )

        return distribution.sample(
            seed=rng
        )

    return jax.jit(
        select_action
    )


# ============================================================
# EVALUATE OPTION POLICY
# ============================================================

def evaluate_option_policy(
    env,
    params,
    primitive_budget,
):

    select_option = (
        make_option_policy(
            params
        )
    )

    execute_option_fn = jax.jit(
        lambda timestep, option_id:
            execute_option_jax(
                env,
                timestep,
                option_id,
                gamma=GAMMA,
            )
    )

    successes = []

    valid_rates = []

    option_decisions_list = []

    primitive_steps_list = []


    for eval_seed in EVAL_SEEDS:

        reset_rng = (
            jax.random.PRNGKey(
                eval_seed
            )
        )

        policy_rng = (
            jax.random.PRNGKey(
                eval_seed
                + 1_000_000
            )
        )

        timestep = env.reset(
            reset_rng
        )

        primitive_steps = 0

        option_decisions = 0

        valid_decisions = 0


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

            option_id = (
                select_option(
                    timestep.observation,
                    option_rng,
                )
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

            # Keep exact primitive evaluation budget.
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

                valid_decisions += 1


        success = reached_goal(
            timestep
        )

        successes.append(
            float(
                success
            )
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
        "success_rate":
            float(
                np.mean(
                    successes
                )
            ),

        "valid_rate":
            float(
                np.mean(
                    valid_rates
                )
            ),

        "mean_option_decisions":
            float(
                np.mean(
                    option_decisions_list
                )
            ),

        "mean_primitive_steps":
            float(
                np.mean(
                    primitive_steps_list
                )
            ),
    }


# ============================================================
# EVALUATE PRIMITIVE POLICY
# ============================================================

def evaluate_primitive_policy(
    env,
    params,
    primitive_budget,
):

    select_action = (
        make_primitive_policy(
            params,
            len(
                env.action_set
            ),
        )
    )

    step_fn = jax.jit(
        lambda timestep, action:
            env.step(
                timestep,
                action,
            )
    )

    successes = []

    primitive_steps_list = []


    for eval_seed in EVAL_SEEDS:

        reset_rng = (
            jax.random.PRNGKey(
                eval_seed
            )
        )

        policy_rng = (
            jax.random.PRNGKey(
                eval_seed
                + 2_000_000
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

            action = (
                select_action(
                    timestep.observation,
                    action_rng,
                )
            )

            timestep = step_fn(
                timestep,
                jnp.asarray(
                    action,
                    dtype=jnp.int32,
                ),
            )

            primitive_steps += 1


        success = reached_goal(
            timestep
        )

        successes.append(
            float(
                success
            )
        )

        primitive_steps_list.append(
            primitive_steps
        )


    return {
        "success_rate":
            float(
                np.mean(
                    successes
                )
            ),

        "mean_primitive_steps":
            float(
                np.mean(
                    primitive_steps_list
                )
            ),
    }


# ============================================================
# CONDITIONS
# ============================================================

CONDITIONS = [
    (
        "option_sparse",
        "Option PPO - Sparse",
        "option",
        "sparse",
    ),
    (
        "option_shaped",
        "Option PPO - Shaped",
        "option",
        "shaped",
    ),
    (
        "primitive_sparse",
        "Primitive PPO - Sparse",
        "primitive",
        "sparse",
    ),
    (
        "primitive_shaped",
        "Primitive PPO - Shaped",
        "primitive",
        "shaped",
    ),
]


# ============================================================
# STORAGE
#
# shape for each condition:
#   [training_seed, evaluation_budget]
# ============================================================

condition_success = {}

condition_validity = {}

condition_decisions = {}


print(
    "============================================"
)

print(
    "DOORKEY REWARD-ABLATION EVALUATION"
)

print(
    "============================================"
)

print(
    "Environment:",
    ENV_ID
)

print(
    "Training seeds:",
    TRAIN_SEEDS
)

print(
    "Evaluation episodes per trained policy:",
    len(
        EVAL_SEEDS
    )
)

print(
    "Evaluation budgets:",
    EVAL_BUDGETS
)


# ============================================================
# LOOP OVER CONDITIONS
# ============================================================

for (
    condition_key,
    condition_label,
    policy_type,
    reward_mode,
) in CONDITIONS:

    print()
    print(
        "=" * 60
    )

    print(
        "CONDITION:",
        condition_label
    )

    print(
        "=" * 60
    )


    env = create_env(
        reward_mode
    )


    all_seed_success = []

    all_seed_validity = []

    all_seed_decisions = []


    for seed in TRAIN_SEEDS:

        print()
        print(
            "Training seed:",
            seed
        )


        if policy_type == "option":

            checkpoint_file = (
                option_checkpoint_path(
                    reward_mode,
                    seed,
                )
            )

        else:

            checkpoint_file = (
                primitive_checkpoint_path(
                    reward_mode,
                    seed,
                )
            )


        print(
            "Loading:",
            checkpoint_file
        )


        params = load_params(
            checkpoint_file
        )


        seed_success = []

        seed_validity = []

        seed_decisions = []


        for budget in EVAL_BUDGETS:

            if policy_type == "option":

                result = (
                    evaluate_option_policy(
                        env,
                        params,
                        primitive_budget=budget,
                    )
                )

                seed_validity.append(
                    result[
                        "valid_rate"
                    ]
                )

                seed_decisions.append(
                    result[
                        "mean_option_decisions"
                    ]
                )

            else:

                result = (
                    evaluate_primitive_policy(
                        env,
                        params,
                        primitive_budget=budget,
                    )
                )


            seed_success.append(
                result[
                    "success_rate"
                ]
            )


            print(
                f"Budget {budget:>3}: "
                f"success="
                f"{result['success_rate'] * 100:>5.1f}%"
            )


        all_seed_success.append(
            seed_success
        )


        if policy_type == "option":

            all_seed_validity.append(
                seed_validity
            )

            all_seed_decisions.append(
                seed_decisions
            )


        del params

        jax.clear_caches()


    condition_success[
        condition_key
    ] = np.asarray(
        all_seed_success,
        dtype=np.float32,
    )


    if policy_type == "option":

        condition_validity[
            condition_key
        ] = np.asarray(
            all_seed_validity,
            dtype=np.float32,
        )

        condition_decisions[
            condition_key
        ] = np.asarray(
            all_seed_decisions,
            dtype=np.float32,
        )


# ============================================================
# AGGREGATE
# ============================================================

mean_success = {}

sem_success = {}


for (
    condition_key,
    condition_label,
    policy_type,
    reward_mode,
) in CONDITIONS:

    values = (
        condition_success[
            condition_key
        ]
    )

    mean_success[
        condition_key
    ] = values.mean(
        axis=0
    )

    sem_success[
        condition_key
    ] = (
        values.std(
            axis=0,
            ddof=1,
        )
        / np.sqrt(
            len(
                TRAIN_SEEDS
            )
        )
    )


mean_validity = {}

sem_validity = {}

mean_decisions = {}

sem_decisions = {}


for condition_key in [
    "option_sparse",
    "option_shaped",
]:

    validity = (
        condition_validity[
            condition_key
        ]
    )

    decisions = (
        condition_decisions[
            condition_key
        ]
    )


    mean_validity[
        condition_key
    ] = validity.mean(
        axis=0
    )

    sem_validity[
        condition_key
    ] = (
        validity.std(
            axis=0,
            ddof=1,
        )
        / np.sqrt(
            len(
                TRAIN_SEEDS
            )
        )
    )


    mean_decisions[
        condition_key
    ] = decisions.mean(
        axis=0
    )

    sem_decisions[
        condition_key
    ] = (
        decisions.std(
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
# SAVE RESULTS
# ============================================================

np.savez(
    "doorkey_reward_ablation_1m_5seeds.npz",

    evaluation_budgets=np.asarray(
        EVAL_BUDGETS
    ),

    training_seeds=np.asarray(
        TRAIN_SEEDS
    ),

    option_sparse_success=(
        condition_success[
            "option_sparse"
        ]
    ),

    option_shaped_success=(
        condition_success[
            "option_shaped"
        ]
    ),

    primitive_sparse_success=(
        condition_success[
            "primitive_sparse"
        ]
    ),

    primitive_shaped_success=(
        condition_success[
            "primitive_shaped"
        ]
    ),

    option_sparse_mean=(
        mean_success[
            "option_sparse"
        ]
    ),

    option_shaped_mean=(
        mean_success[
            "option_shaped"
        ]
    ),

    primitive_sparse_mean=(
        mean_success[
            "primitive_sparse"
        ]
    ),

    primitive_shaped_mean=(
        mean_success[
            "primitive_shaped"
        ]
    ),

    option_sparse_sem=(
        sem_success[
            "option_sparse"
        ]
    ),

    option_shaped_sem=(
        sem_success[
            "option_shaped"
        ]
    ),

    primitive_sparse_sem=(
        sem_success[
            "primitive_sparse"
        ]
    ),

    primitive_shaped_sem=(
        sem_success[
            "primitive_shaped"
        ]
    ),

    option_sparse_validity=(
        mean_validity[
            "option_sparse"
        ]
    ),

    option_shaped_validity=(
        mean_validity[
            "option_shaped"
        ]
    ),

    option_sparse_decisions=(
        mean_decisions[
            "option_sparse"
        ]
    ),

    option_shaped_decisions=(
        mean_decisions[
            "option_shaped"
        ]
    ),
)


# ============================================================
# GRAPH 1
# SUCCESS — ALL FOUR CONDITIONS
# ============================================================

budgets = np.asarray(
    EVAL_BUDGETS
)


plt.figure(
    figsize=(10, 6)
)


for (
    condition_key,
    condition_label,
    policy_type,
    reward_mode,
) in CONDITIONS:

    means = (
        mean_success[
            condition_key
        ]
    )

    sems = (
        sem_success[
            condition_key
        ]
    )


    plt.plot(
        budgets,
        means,
        marker="o",
        linewidth=2.3,
        label=condition_label,
    )


    plt.fill_between(
        budgets,

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
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "True task success rate"
)

plt.title(
    "DoorKey 8x8 Reward-Delay Ablation\n"
    "Sparse vs Shaped Reward"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    budgets
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "doorkey_reward_ablation_success.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 2
# REWARD SHAPING EFFECT
#
# shaped success - sparse success
# ============================================================

option_gain = (
    mean_success[
        "option_shaped"
    ]
    - mean_success[
        "option_sparse"
    ]
)


primitive_gain = (
    mean_success[
        "primitive_shaped"
    ]
    - mean_success[
        "primitive_sparse"
    ]
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    budgets,
    option_gain,
    marker="o",
    linewidth=2.3,
    label="Option PPO shaping gain",
)


plt.plot(
    budgets,
    primitive_gain,
    marker="o",
    linewidth=2.3,
    label="Primitive PPO shaping gain",
)


plt.axhline(
    0.0,
    linewidth=1,
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Success improvement from shaping"
)

plt.title(
    "Effect of Earlier Credit on Task Success"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "doorkey_reward_ablation_shaping_gain.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 3
# OPTION VALIDITY
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    budgets,
    mean_validity[
        "option_sparse"
    ],
    marker="o",
    linewidth=2.3,
    label="Sparse reward",
)


plt.plot(
    budgets,
    mean_validity[
        "option_shaped"
    ],
    marker="o",
    linewidth=2.3,
    label="Shaped reward",
)


plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Valid-option rate"
)

plt.title(
    "Option PPO Validity Under Reward Shaping"
)

plt.ylim(
    0,
    1
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "doorkey_reward_ablation_validity.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print(
    "=" * 85
)

print(
    "DOORKEY REWARD-ABLATION SUMMARY"
)

print(
    "=" * 85
)


print(
    "Budget | Option Sparse | Option Shaped | "
    "Primitive Sparse | Primitive Shaped"
)

print(
    "-" * 85
)


for index, budget in enumerate(
    EVAL_BUDGETS
):

    print(
        f"{budget:>6} | "
        f"{mean_success['option_sparse'][index] * 100:>6.1f}% | "
        f"{mean_success['option_shaped'][index] * 100:>6.1f}% | "
        f"{mean_success['primitive_sparse'][index] * 100:>8.1f}% | "
        f"{mean_success['primitive_shaped'][index] * 100:>8.1f}%"
    )


print()
print(
    "Shaping gain at each evaluation budget:"
)


for index, budget in enumerate(
    EVAL_BUDGETS
):

    print(
        f"Budget {budget:>3}: "
        f"Option gain="
        f"{option_gain[index] * 100:>6.1f} pp | "
        f"Primitive gain="
        f"{primitive_gain[index] * 100:>6.1f} pp"
    )


# ============================================================
# Key 30-step comparison
# ============================================================

main_budget = 30

main_index = (
    EVAL_BUDGETS.index(
        main_budget
    )
)


print()
print(
    "============================================"
)

print(
    "30-STEP CREDIT-ASSIGNMENT COMPARISON"
)

print(
    "============================================"
)


print(
    "Option PPO sparse:",
    f"{mean_success['option_sparse'][main_index] * 100:.1f}% "
    f"± {sem_success['option_sparse'][main_index] * 100:.1f}%"
)


print(
    "Option PPO shaped:",
    f"{mean_success['option_shaped'][main_index] * 100:.1f}% "
    f"± {sem_success['option_shaped'][main_index] * 100:.1f}%"
)


print(
    "Primitive PPO sparse:",
    f"{mean_success['primitive_sparse'][main_index] * 100:.1f}% "
    f"± {sem_success['primitive_sparse'][main_index] * 100:.1f}%"
)


print(
    "Primitive PPO shaped:",
    f"{mean_success['primitive_shaped'][main_index] * 100:.1f}% "
    f"± {sem_success['primitive_shaped'][main_index] * 100:.1f}%"
)


print()
print(
    "Generated plots:"
)

print(
    "doorkey_reward_ablation_success.png"
)

print(
    "doorkey_reward_ablation_shaping_gain.png"
)

print(
    "doorkey_reward_ablation_validity.png"
)


print()
print(
    "Saved raw results:"
)

print(
    "doorkey_reward_ablation_1m_5seeds.npz"
)


print()
print(
    "Reward-ablation evaluation complete."
)