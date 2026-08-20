import jax
import jax.numpy as jnp

import navix as nx

from jax_option_executor import (
    execute_option_jax,
)

from doorkey_reward_ablation import (
    sparse_doorkey_reward,
    shaped_doorkey_reward,
)


# ============================================================
# Configuration
# ============================================================

ENV_ID = (
    "Navix-DoorKey-Random-8x8-v0"
)

OPTION_SEQUENCE = [
    0,  # GO_TO_KEY
    1,  # PICKUP_KEY
    2,  # GO_TO_DOOR
    3,  # OPEN_DOOR
    4,  # GO_TO_GOAL
]

OPTION_NAMES = [
    "GO_TO_KEY",
    "PICKUP_KEY",
    "GO_TO_DOOR",
    "OPEN_DOOR",
    "GO_TO_GOAL",
]


# ============================================================
# Run one reward condition
# ============================================================

def run_condition(
    condition_name,
    reward_fn,
):

    print()
    print(
        "=" * 60
    )

    print(
        "REWARD CONDITION:",
        condition_name
    )

    print(
        "=" * 60
    )


    env = nx.make(
        ENV_ID,
        gamma=0.99,
    )


    # Replace only the reward function.
    # Dynamics, observations, termination and action space
    # remain unchanged.
    env = env.replace(
        reward_fn=reward_fn
    )


    execute_option_fn = jax.jit(
        lambda timestep, option_id:
            execute_option_jax(
                env,
                timestep,
                option_id,
                gamma=0.99,
            )
    )


    rng = jax.random.PRNGKey(
        0
    )


    timestep = env.reset(
        rng
    )


    total_undiscounted_reward = 0.0

    option_rewards = []


    for option_id in OPTION_SEQUENCE:

        (
            timestep,
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


        option_rewards.append(
            float(
                option_reward
            )
        )


        print(
            f"{OPTION_NAMES[option_id]}:"
        )

        print(
            "    Duration:",
            int(
                duration
            )
        )

        print(
            "    Discounted option reward:",
            float(
                option_reward
            )
        )

        print(
            "    Final primitive reward:",
            float(
                timestep.reward
            )
        )

        print(
            "    Valid:",
            bool(
                valid
            )
        )


        assert bool(
            valid
        )


    print()
    print(
        "Done:",
        bool(
            timestep.is_done()
        )
    )

    print(
        "Final reward:",
        float(
            timestep.reward
        )
    )

    print(
        "Option rewards:",
        option_rewards
    )


    assert bool(
        timestep.is_done()
    )


    return option_rewards


# ============================================================
# Sparse condition
# ============================================================

print(
    "Testing DoorKey reward ablation..."
)


sparse_rewards = run_condition(
    condition_name="SPARSE",
    reward_fn=sparse_doorkey_reward,
)


# ============================================================
# Shaped condition
# ============================================================

shaped_rewards = run_condition(
    condition_name="SHAPED",
    reward_fn=shaped_doorkey_reward,
)


# ============================================================
# Assertions
# ============================================================

# Sparse:
# no option should receive positive reward until GO_TO_GOAL.

assert sparse_rewards[0] == 0.0
assert sparse_rewards[1] == 0.0
assert sparse_rewards[2] == 0.0
assert sparse_rewards[3] == 0.0
assert sparse_rewards[4] > 0.0


# Shaped:
# PICKUP_KEY should now receive immediate credit.

assert shaped_rewards[1] > 0.0


# OPEN_DOOR should now receive immediate credit.

assert shaped_rewards[3] > 0.0


# GO_TO_GOAL still receives terminal credit.

assert shaped_rewards[4] > 0.0


print()
print(
    "============================================"
)

print(
    "DOORKEY REWARD ABLATION TEST PASSED"
)

print(
    "============================================"
)

print()
print(
    "Sparse:"
)

print(
    "Reward arrives only at the terminal goal."
)

print()
print(
    "Shaped:"
)

print(
    "Credit is distributed across key pickup, "
    "door opening and goal completion."
)