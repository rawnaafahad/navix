import numpy as np


data = np.load(
    "option_ppo_sample_efficiency_5seeds.npz"
)


training_budgets = data[
    "training_budgets"
]

evaluation_budgets = data[
    "evaluation_budgets"
]

all_success = data[
    "all_success"
]

mean_success = data[
    "mean_success"
]

sem_success = data[
    "sem_success"
]

mean_valid = data[
    "mean_valid"
]


main_eval_budget = 30

main_index = np.where(
    evaluation_budgets
    == main_eval_budget
)[0][0]


print(
    "============================================"
)

print(
    "OPTION PPO SAMPLE-EFFICIENCY SUMMARY"
)

print(
    "============================================"
)

print(
    "Evaluation budget:",
    main_eval_budget,
    "primitive steps",
)

print()

print(
    "Training budget | Mean success | SEM | Mean validity"
)

print(
    "-----------------------------------------------------"
)


for i, budget in enumerate(
    training_budgets
):

    print(
        f"{budget:>15} | "
        f"{mean_success[i, main_index] * 100:>10.1f}% | "
        f"{sem_success[i, main_index] * 100:>4.1f}% | "
        f"{mean_valid[i, main_index] * 100:>11.1f}%"
    )


print(
    "\nPer-seed success at 30-step evaluation budget:"
)


for i, budget in enumerate(
    training_budgets
):

    values = " | ".join(
        [
            f"{value * 100:.0f}%"
            for value in all_success[
                i,
                :,
                main_index,
            ]
        ]
    )

    print(
        f"{budget:>7}: {values}"
    )


print(
    "\nFull mean success matrix:"
)

print(
    "Columns = evaluation budgets:"
)

print(
    evaluation_budgets
)

print()

print(
    np.round(
        mean_success * 100,
        1,
    )
)