import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Evaluation budgets
# ============================================================

budgets = np.array([
    15,
    20,
    25,
    30,
    40,
    50,
    75,
    100,
])


# ============================================================
# 1M Option PPO
# Mean across 5 training seeds
# ============================================================

option_mean = np.array([
    13.2,
    43.6,
    76.6,
    97.0,
    99.6,
    99.8,
    100.0,
    100.0,
])

option_sem = np.array([
    0.2,
    0.5,
    1.0,
    0.5,
    0.4,
    0.2,
    0.0,
    0.0,
])


# ============================================================
# 1M Primitive PPO
# Mean across 5 training seeds
# ============================================================

primitive_mean = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.2,
])

primitive_sem = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.2,
])


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(9, 6))


# Option PPO
plt.plot(
    budgets,
    option_mean,
    marker="o",
    linewidth=2.5,
    label="Option PPO (1M frames)",
)

plt.fill_between(
    budgets,
    option_mean - option_sem,
    option_mean + option_sem,
    alpha=0.2,
)


# Primitive PPO
plt.plot(
    budgets,
    primitive_mean,
    marker="o",
    linewidth=2.5,
    label="Primitive PPO (1M frames)",
)

plt.fill_between(
    budgets,
    primitive_mean - primitive_sem,
    primitive_mean + primitive_sem,
    alpha=0.2,
)


# ============================================================
# Formatting
# ============================================================

plt.xlabel(
    "Primitive-step evaluation budget"
)

plt.ylabel(
    "Success rate (%)"
)

plt.title(
    "Option PPO vs Primitive PPO\n"
    "Navix DoorKey Random 8x8 — 5 Training Seeds"
)

plt.ylim(
    -2,
    102,
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


# ============================================================
# Save
# ============================================================

output_file = (
    "option_vs_primitive_ppo_1m_5seeds.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved comparison graph:"
)

print(
    output_file
)


plt.show()