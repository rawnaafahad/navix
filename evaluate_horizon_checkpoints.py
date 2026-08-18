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


# ============================================================
# CONFIGURATION
# ============================================================

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
# CHECKPOINT PATHS
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

        checkpoint = pickle.load(
            file
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
# OPTION POLICY EVALUATION
# ============================================================

def evaluate_option_policy(
    env,
    params,
):

    select_option = make_option_policy(
        params
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

    valid_rates = []

    decision_counts = []

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
            < MAX_EVAL_STEPS
        ):

            policy_rng, option_rng = (
                jax.random.split(
                    policy_rng
                )
            )


            option_id = select_option(
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
                jnp.asarray(
                    option_id,
                    dtype=jnp.int32,
                ),
            )


            duration = int(
                duration
            )


            # Exact primitive evaluation budget
            if (
                primitive_steps
                + duration
                > MAX_EVAL_STEPS
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


        success = (
            bool(
                timestep.is_done()
            )
            and float(
                timestep.reward
            ) > 0.0
        )


        successes.append(
            float(
                success
            )
        )


        primitive_steps_list.append(
            primitive_steps
        )


        decision_counts.append(
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

        "mean_decisions":
            float(
                np.mean(
                    decision_counts
                )
            ),

        "mean_steps":
            float(
                np.mean(
                    primitive_steps_list
                )
            ),
    }


# ============================================================
# PRIMITIVE PPO EVALUATION
# ============================================================

def evaluate_primitive_policy(
    env,
    params,
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
            < MAX_EVAL_STEPS
        ):

            policy_rng, action_rng = (
                jax.random.split(
                    policy_rng
                )
            )


            action = select_action(
                timestep.observation,
                action_rng,
            )


            timestep = step_fn(
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

        "mean_steps":
            float(
                np.mean(
                    primitive_steps_list
                )
            ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print(
    "============================================"
)

print(
    "TEMPORAL-HORIZON CHECKPOINT EVALUATION"
)

print(
    "============================================"
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
    "Maximum primitive evaluation steps:",
    MAX_EVAL_STEPS
)


horizons = []

option_success = []

option_sem = []

primitive_success = []

primitive_sem = []

option_validity = []

option_validity_sem = []

option_decisions = []

option_decisions_sem = []


# ============================================================
# LOOP OVER TEMPORAL HORIZONS
# ============================================================

for env_label, env_info in (
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
        "Environment ID:",
        env_info[
            "env_id"
        ]
    )

    print(
        "Measured expert horizon:",
        env_info[
            "mean_expert_horizon"
        ]
    )

    print(
        "=" * 60
    )


    env = create_env(
        env_info[
            "env_id"
        ]
    )


    horizon = env_info[
        "mean_expert_horizon"
    ]


    horizons.append(
        horizon
    )


    seed_option_success = []

    seed_primitive_success = []

    seed_option_validity = []

    seed_option_decisions = []


    # ========================================================
    # FIVE INDEPENDENT TRAINING SEEDS
    # ========================================================

    for seed in TRAIN_SEEDS:

        print()
        print(
            "--------------------------------"
        )

        print(
            "TRAINING SEED:",
            seed
        )

        print(
            "--------------------------------"
        )


        # ----------------------------------------------------
        # Option PPO
        # ----------------------------------------------------

        option_file = (
            option_checkpoint_path(
                env_label,
                seed,
            )
        )


        print(
            "Loading Option PPO:",
            option_file
        )


        option_params = load_params(
            option_file
        )


        option_result = (
            evaluate_option_policy(
                env,
                option_params,
            )
        )


        seed_option_success.append(
            option_result[
                "success_rate"
            ]
        )


        seed_option_validity.append(
            option_result[
                "valid_rate"
            ]
        )


        seed_option_decisions.append(
            option_result[
                "mean_decisions"
            ]
        )


        print(
            f"Option PPO: "
            f"success="
            f"{option_result['success_rate'] * 100:.1f}% | "
            f"valid="
            f"{option_result['valid_rate'] * 100:.1f}% | "
            f"decisions="
            f"{option_result['mean_decisions']:.2f}"
        )


        del option_params


        jax.clear_caches()


        # ----------------------------------------------------
        # Primitive PPO
        # ----------------------------------------------------

        primitive_file = (
            primitive_checkpoint_path(
                env_label,
                seed,
            )
        )


        print(
            "Loading Primitive PPO:",
            primitive_file
        )


        primitive_params = (
            load_params(
                primitive_file
            )
        )


        primitive_result = (
            evaluate_primitive_policy(
                env,
                primitive_params,
            )
        )


        seed_primitive_success.append(
            primitive_result[
                "success_rate"
            ]
        )


        print(
            f"Primitive PPO: "
            f"success="
            f"{primitive_result['success_rate'] * 100:.1f}% | "
            f"mean steps="
            f"{primitive_result['mean_steps']:.2f}"
        )


        del primitive_params


        jax.clear_caches()


    # ========================================================
    # AGGREGATE OVER FIVE TRAINING SEEDS
    # ========================================================

    seed_option_success = (
        np.asarray(
            seed_option_success
        )
    )

    seed_primitive_success = (
        np.asarray(
            seed_primitive_success
        )
    )

    seed_option_validity = (
        np.asarray(
            seed_option_validity
        )
    )

    seed_option_decisions = (
        np.asarray(
            seed_option_decisions
        )
    )


    option_success.append(
        seed_option_success.mean()
    )


    option_sem.append(
        seed_option_success.std(
            ddof=1
        )
        / np.sqrt(
            len(
                TRAIN_SEEDS
            )
        )
    )


    primitive_success.append(
        seed_primitive_success.mean()
    )


    primitive_sem.append(
        seed_primitive_success.std(
            ddof=1
        )
        / np.sqrt(
            len(
                TRAIN_SEEDS
            )
        )
    )


    option_validity.append(
        seed_option_validity.mean()
    )


    option_validity_sem.append(
        seed_option_validity.std(
            ddof=1
        )
        / np.sqrt(
            len(
                TRAIN_SEEDS
            )
        )
    )


    option_decisions.append(
        seed_option_decisions.mean()
    )


    option_decisions_sem.append(
        seed_option_decisions.std(
            ddof=1
        )
        / np.sqrt(
            len(
                TRAIN_SEEDS
            )
        )
    )


    print()
    print(
        "ENVIRONMENT SUMMARY"
    )


    print(
        "Option PPO success:",
        f"{option_success[-1] * 100:.1f}% "
        f"± {option_sem[-1] * 100:.1f}%"
    )


    print(
        "Primitive PPO success:",
        f"{primitive_success[-1] * 100:.1f}% "
        f"± {primitive_sem[-1] * 100:.1f}%"
    )


    print(
        "Option valid rate:",
        f"{option_validity[-1] * 100:.1f}% "
        f"± {option_validity_sem[-1] * 100:.1f}%"
    )


    print(
        "Mean high-level decisions:",
        f"{option_decisions[-1]:.2f} "
        f"± {option_decisions_sem[-1]:.2f}"
    )


# ============================================================
# ARRAYS
# ============================================================

horizons = np.asarray(
    horizons,
    dtype=np.float32,
)


option_success = np.asarray(
    option_success
)


option_sem = np.asarray(
    option_sem
)


primitive_success = np.asarray(
    primitive_success
)


primitive_sem = np.asarray(
    primitive_sem
)


option_validity = np.asarray(
    option_validity
)


option_validity_sem = np.asarray(
    option_validity_sem
)


option_decisions = np.asarray(
    option_decisions
)


option_decisions_sem = np.asarray(
    option_decisions_sem
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

    primitive_success=(
        primitive_success
    ),

    primitive_sem=(
        primitive_sem
    ),

    option_validity=(
        option_validity
    ),

    option_validity_sem=(
        option_validity_sem
    ),

    option_decisions=(
        option_decisions
    ),

    option_decisions_sem=(
        option_decisions_sem
    ),
)


# ============================================================
# GRAPH 1
# SUCCESS VS TEMPORAL HORIZON
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.errorbar(
    horizons,
    option_success,
    yerr=option_sem,
    marker="o",
    linewidth=2.5,
    capsize=5,
    label="Option PPO",
)


plt.errorbar(
    horizons,
    primitive_success,
    yerr=primitive_sem,
    marker="o",
    linewidth=2.5,
    capsize=5,
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
    1.02
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
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 2
# OPTION VALIDITY VS TEMPORAL HORIZON
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.errorbar(
    horizons,
    option_validity,
    yerr=option_validity_sem,
    marker="o",
    linewidth=2.5,
    capsize=5,
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
    1
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
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 3
# HIGH-LEVEL DECISIONS VS TEMPORAL HORIZON
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.errorbar(
    horizons,
    option_decisions,
    yerr=option_decisions_sem,
    marker="o",
    linewidth=2.5,
    capsize=5,
)


plt.xlabel(
    "Mean expert primitive solution length"
)

plt.ylabel(
    "Mean high-level option decisions"
)

plt.title(
    "Effective High-Level Decision Horizon"
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
    "=" * 80
)

print(
    "TEMPORAL-HORIZON STRESS TEST SUMMARY"
)

print(
    "=" * 80
)

print(
    "Environment | Horizon | Option PPO | Primitive PPO | Advantage"
)

print(
    "-" * 80
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
        f"{option_success[index] * 100:>6.1f}% "
        f"± {option_sem[index] * 100:>4.1f}% | "
        f"{primitive_success[index] * 100:>6.1f}% "
        f"± {primitive_sem[index] * 100:>4.1f}% | "
        f"{advantage * 100:>7.1f} pp"
    )


print()
print(
    "Option validity:"
)

for index, env_label in enumerate(
    ENVIRONMENTS.keys()
):

    print(
        f"{env_label}: "
        f"{option_validity[index] * 100:.1f}% "
        f"± {option_validity_sem[index] * 100:.1f}%"
    )


print()
print(
    "High-level option decisions:"
)

for index, env_label in enumerate(
    ENVIRONMENTS.keys()
):

    print(
        f"{env_label}: "
        f"{option_decisions[index]:.2f} "
        f"± {option_decisions_sem[index]:.2f}"
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
    "Saved raw results:"
)

print(
    "doorkey_temporal_horizon_1m_5seeds.npz"
)


print()
print(
    "Temporal-horizon evaluation complete."
)