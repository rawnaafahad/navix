import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD RESULTS
# ============================================================

sample = np.load(
    "option_ppo_sample_efficiency_5seeds.npz",
    allow_pickle=True,
)

horizon = np.load(
    "doorkey_temporal_horizon_1m_5seeds.npz",
    allow_pickle=True,
)

reward = np.load(
    "doorkey_reward_ablation_1m_5seeds.npz",
    allow_pickle=True,
)


# ============================================================
# HELPER
# ============================================================

def get_first_existing(npz_file, names):
    """
    Return the first matching array from an NPZ file.
    Useful because some earlier scripts used slightly
    different key names.
    """

    for name in names:

        if name in npz_file.files:
            return np.asarray(
                npz_file[name]
            )

    raise KeyError(
        "Could not find any of these keys: "
        + ", ".join(names)
        + "\nAvailable keys: "
        + ", ".join(npz_file.files)
    )


# ============================================================
# PANEL A
# SAMPLE EFFICIENCY
# ============================================================

training_budgets = get_first_existing(
    sample,
    [
        "training_budgets",
        "train_budgets",
        "budgets",
    ],
)

evaluation_budgets_sample = get_first_existing(
    sample,
    [
        "evaluation_budgets",
        "eval_budgets",
    ],
)

# The raw success tensor should normally be:
# [training_budget, training_seed, evaluation_budget]
#
# If your saved file contains the mean matrix directly,
# the fallback keys below also support that.

if "success_rates" in sample.files:

    sample_success_raw = np.asarray(
        sample["success_rates"]
    )

    # Expected:
    # [training_budget, training_seed, eval_budget]
    sample_mean_matrix = (
        sample_success_raw.mean(
            axis=1
        )
    )

    sample_sem_matrix = (
        sample_success_raw.std(
            axis=1,
            ddof=1,
        )
        / np.sqrt(
            sample_success_raw.shape[1]
        )
    )

else:

    sample_mean_matrix = get_first_existing(
        sample,
        [
            "mean_success",
            "mean_success_rates",
            "success_mean",
        ],
    )

    sample_sem_matrix = get_first_existing(
        sample,
        [
            "sem_success",
            "sem_success_rates",
            "success_sem",
        ],
    )


# Use the 30-step evaluation budget.
sample_eval_target = 30

sample_eval_index = int(
    np.where(
        evaluation_budgets_sample
        == sample_eval_target
    )[0][0]
)

sample_mean_30 = (
    sample_mean_matrix[
        :,
        sample_eval_index
    ]
)

sample_sem_30 = (
    sample_sem_matrix[
        :,
        sample_eval_index
    ]
)


# ============================================================
# PANEL B
# TEMPORAL HORIZON
# ============================================================

horizons = np.asarray(
    horizon["horizons"]
)

env_labels = np.asarray(
    horizon["env_labels"]
)

option_horizon_success = np.asarray(
    horizon["option_success"]
)

option_horizon_sem = np.asarray(
    horizon["option_sem"]
)

primitive_horizon_success = np.asarray(
    horizon["primitive_success"]
)

primitive_horizon_sem = np.asarray(
    horizon["primitive_sem"]
)


# ============================================================
# PANEL C
# REWARD DELAY ABLATION
# ============================================================

reward_eval_budgets = np.asarray(
    reward["evaluation_budgets"]
)

option_sparse_mean = np.asarray(
    reward["option_sparse_mean"]
)

option_shaped_mean = np.asarray(
    reward["option_shaped_mean"]
)

primitive_sparse_mean = np.asarray(
    reward["primitive_sparse_mean"]
)

primitive_shaped_mean = np.asarray(
    reward["primitive_shaped_mean"]
)

option_sparse_sem = np.asarray(
    reward["option_sparse_sem"]
)

option_shaped_sem = np.asarray(
    reward["option_shaped_sem"]
)

primitive_sparse_sem = np.asarray(
    reward["primitive_sparse_sem"]
)

primitive_shaped_sem = np.asarray(
    reward["primitive_shaped_sem"]
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
)


# ============================================================
# PANEL A
# SAMPLE EFFICIENCY
# ============================================================

ax = axes[0]

ax.errorbar(
    training_budgets,
    sample_mean_30,
    yerr=sample_sem_30,
    marker="o",
    linewidth=2.2,
    capsize=4,
)

ax.set_xlabel(
    "Primitive training frames"
)

ax.set_ylabel(
    "Success rate"
)

ax.set_title(
    "A. Sample efficiency\n"
    "30-step evaluation budget"
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xticks(
    training_budgets
)

ax.set_xticklabels(
    [
        "100k",
        "250k",
        "500k",
        "750k",
        "1M",
    ]
)

ax.grid(
    True,
    alpha=0.3,
)


# ============================================================
# PANEL B
# TEMPORAL-HORIZON STRESS TEST
# ============================================================

ax = axes[1]

ax.errorbar(
    horizons,
    option_horizon_success,
    yerr=option_horizon_sem,
    marker="o",
    linewidth=2.2,
    capsize=4,
    label="Option PPO",
)

ax.errorbar(
    horizons,
    primitive_horizon_success,
    yerr=primitive_horizon_sem,
    marker="o",
    linewidth=2.2,
    capsize=4,
    label="Primitive PPO",
)

ax.set_xlabel(
    "Mean expert primitive solution length"
)

ax.set_ylabel(
    "Success rate"
)

ax.set_title(
    "B. Temporal-horizon stress test"
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xticks(
    horizons
)

ax.set_xticklabels(
    [
        f"{env_labels[i]}\n{horizons[i]:.1f}"
        for i in range(
            len(
                horizons
            )
        )
    ]
)

ax.legend()

ax.grid(
    True,
    alpha=0.3,
)


# ============================================================
# PANEL C
# REWARD-DELAY ABLATION
# ============================================================

ax = axes[2]

conditions = [
    (
        option_sparse_mean,
        option_sparse_sem,
        "Option sparse",
    ),
    (
        option_shaped_mean,
        option_shaped_sem,
        "Option shaped",
    ),
    (
        primitive_sparse_mean,
        primitive_sparse_sem,
        "Primitive sparse",
    ),
    (
        primitive_shaped_mean,
        primitive_shaped_sem,
        "Primitive shaped",
    ),
]


for (
    means,
    sems,
    label,
) in conditions:

    ax.plot(
        reward_eval_budgets,
        means,
        marker="o",
        linewidth=2.0,
        label=label,
    )

    ax.fill_between(
        reward_eval_budgets,

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

        alpha=0.10,
    )


ax.set_xlabel(
    "Primitive-step evaluation budget"
)

ax.set_ylabel(
    "True task success rate"
)

ax.set_title(
    "C. Reward-delay ablation"
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xticks(
    reward_eval_budgets
)

ax.legend(
    fontsize=9,
)

ax.grid(
    True,
    alpha=0.3,
)


# ============================================================
# OVERALL TITLE
# ============================================================

fig.suptitle(
    "DoorKey: Evidence for Temporal Abstraction "
    "and Credit Assignment",
    fontsize=16,
    y=1.02,
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_file = (
    "doorkey_credit_assignment_summary.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# PRINT SUMMARY TABLE
# ============================================================

print()
print(
    "============================================"
)

print(
    "DOORKEY CONSOLIDATED RESULTS"
)

print(
    "============================================"
)


print()
print(
    "SAMPLE EFFICIENCY"
)

print(
    "30-step evaluation budget"
)

print(
    "Training frames | Option PPO success"
)

print(
    "-" * 40
)


for i in range(
    len(
        training_budgets
    )
):

    print(
        f"{int(training_budgets[i]):>15} | "
        f"{sample_mean_30[i] * 100:>6.1f}% "
        f"± {sample_sem_30[i] * 100:.1f}%"
    )


print()
print(
    "TEMPORAL-HORIZON STRESS TEST"
)

print(
    "Environment | Horizon | Option PPO | Primitive PPO"
)

print(
    "-" * 60
)


for i in range(
    len(
        horizons
    )
):

    print(
        f"{env_labels[i]:>11} | "
        f"{horizons[i]:>7.2f} | "
        f"{option_horizon_success[i] * 100:>6.1f}% | "
        f"{primitive_horizon_success[i] * 100:>6.1f}%"
    )


print()
print(
    "REWARD-DELAY ABLATION"
)

print(
    "30-step evaluation budget"
)

print(
    "-" * 50
)


reward_30_index = int(
    np.where(
        reward_eval_budgets
        == 30
    )[0][0]
)


print(
    "Option PPO sparse:",
    f"{option_sparse_mean[reward_30_index] * 100:.1f}% "
    f"± {option_sparse_sem[reward_30_index] * 100:.1f}%"
)

print(
    "Option PPO shaped:",
    f"{option_shaped_mean[reward_30_index] * 100:.1f}% "
    f"± {option_shaped_sem[reward_30_index] * 100:.1f}%"
)

print(
    "Primitive PPO sparse:",
    f"{primitive_sparse_mean[reward_30_index] * 100:.1f}% "
    f"± {primitive_sparse_sem[reward_30_index] * 100:.1f}%"
)

print(
    "Primitive PPO shaped:",
    f"{primitive_shaped_mean[reward_30_index] * 100:.1f}% "
    f"± {primitive_shaped_sem[reward_30_index] * 100:.1f}%"
)


print()
print(
    "Saved:"
)

print(
    output_file
)