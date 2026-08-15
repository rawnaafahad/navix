import jax
import jax.numpy as jnp
import navix as nx

from jax_option_executor import (
    execute_option_jax,
)


GAMMA = 0.99


# ============================================================
# Create environment
# ============================================================

env = nx.make(
    "Navix-DoorKey-Random-8x8-v0"
)


def create_timestep(
    seed=0,
):
    rng = jax.random.PRNGKey(
        seed
    )

    return env.reset(
        rng
    )


# ============================================================
# JIT-compile complete dispatcher
# ============================================================

execute_option_fn = jax.jit(
    lambda timestep, option_id:
        execute_option_jax(
            env,
            timestep,
            option_id,
            gamma=GAMMA,
        )
)


print(
    "Testing reward-aware JAX option dispatcher...\n"
)


# ============================================================
# TEST 1
# Correct five-option sequence
# ============================================================

timestep = create_timestep(
    seed=0
)


sequence = [
    0,  # GO_TO_KEY
    1,  # PICKUP_KEY
    2,  # GO_TO_DOOR
    3,  # OPEN_DOOR
    4,  # GO_TO_GOAL
]


option_names = [
    "GO_TO_KEY",
    "PICKUP_KEY",
    "GO_TO_DOOR",
    "OPEN_DOOR",
    "GO_TO_GOAL",
]


total_primitive_steps = 0
option_rewards = []


print(
    "TEST 1: Correct option sequence"
)


for option_id in sequence:

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

    total_primitive_steps += int(
        duration
    )

    option_rewards.append(
        float(option_reward)
    )

    print(
        f"{option_names[option_id]}:"
    )

    print(
        "    Duration:",
        duration
    )

    print(
        "    Option reward:",
        option_reward
    )

    print(
        "    Valid:",
        valid
    )

    assert bool(
        valid
    )


print(
    "Final primitive reward:",
    timestep.reward
)

print(
    "Done:",
    timestep.is_done()
)

print(
    "Total primitive steps:",
    total_primitive_steps
)

print(
    "Option rewards:",
    option_rewards
)


assert bool(
    timestep.is_done()
)

assert float(
    timestep.reward
) > 0.0


# First four options should have no reward.
for reward in option_rewards[:4]:

    assert abs(
        reward
    ) < 1e-6


# GO_TO_GOAL should carry discounted terminal reward.
expected_goal_reward = (
    GAMMA ** 4
)


assert jnp.isclose(
    option_rewards[4],
    expected_goal_reward,
    atol=1e-6,
)


print(
    "Expected final option reward:",
    expected_goal_reward
)

print(
    "TEST 1 PASS\n"
)


# ============================================================
# TEST 2
# GO_TO_GOAL before opening door
# ============================================================

timestep = create_timestep(
    seed=0
)

time_before = int(
    timestep.t
)


(
    timestep,
    duration,
    option_reward,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        4,
        dtype=jnp.int32,
    ),
)


time_after = int(
    timestep.t
)


print(
    "TEST 2: GO_TO_GOAL before door open"
)

print(
    "Duration:",
    duration
)

print(
    "Option reward:",
    option_reward
)

print(
    "Valid:",
    valid
)

print(
    "Environment steps consumed:",
    time_after - time_before
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1

assert (
    time_after - time_before
) == 1

assert jnp.isclose(
    option_reward,
    0.0,
    atol=1e-6,
)


print(
    "TEST 2 PASS\n"
)


# ============================================================
# TEST 3
# GO_TO_DOOR before collecting key
# ============================================================

timestep = create_timestep(
    seed=0
)


(
    timestep,
    duration,
    option_reward,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        2,
        dtype=jnp.int32,
    ),
)


print(
    "TEST 3: GO_TO_DOOR without key"
)

print(
    "Duration:",
    duration
)

print(
    "Option reward:",
    option_reward
)

print(
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1

assert jnp.isclose(
    option_reward,
    0.0,
    atol=1e-6,
)


print(
    "TEST 3 PASS\n"
)


# ============================================================
# TEST 4
# OPEN_DOOR before collecting key
# ============================================================

timestep = create_timestep(
    seed=0
)


(
    timestep,
    duration,
    option_reward,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        3,
        dtype=jnp.int32,
    ),
)


print(
    "TEST 4: OPEN_DOOR without key"
)

print(
    "Duration:",
    duration
)

print(
    "Option reward:",
    option_reward
)

print(
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1

assert jnp.isclose(
    option_reward,
    0.0,
    atol=1e-6,
)


print(
    "TEST 4 PASS\n"
)


# ============================================================
# TEST 5
# GO_TO_KEY after key already collected
# ============================================================

timestep = create_timestep(
    seed=0
)


(
    timestep,
    _,
    _,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        0,
        dtype=jnp.int32,
    ),
)

assert bool(
    valid
)


(
    timestep,
    _,
    _,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        1,
        dtype=jnp.int32,
    ),
)

assert bool(
    valid
)


(
    timestep,
    duration,
    option_reward,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        0,
        dtype=jnp.int32,
    ),
)


print(
    "TEST 5: GO_TO_KEY after collection"
)

print(
    "Duration:",
    duration
)

print(
    "Option reward:",
    option_reward
)

print(
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1

assert jnp.isclose(
    option_reward,
    0.0,
    atol=1e-6,
)


print(
    "TEST 5 PASS\n"
)


# ============================================================
# TEST 6
# Invalid option ID
# ============================================================

timestep = create_timestep(
    seed=0
)


(
    timestep,
    duration,
    option_reward,
    valid,
) = execute_option_fn(
    timestep,
    jnp.asarray(
        99,
        dtype=jnp.int32,
    ),
)


print(
    "TEST 6: Invalid option ID"
)

print(
    "Duration:",
    duration
)

print(
    "Option reward:",
    option_reward
)

print(
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1

assert jnp.isclose(
    option_reward,
    0.0,
    atol=1e-6,
)


print(
    "TEST 6 PASS\n"
)


# ============================================================
# Final result
# ============================================================

print(
    "================================"
)

print(
    "REWARD-AWARE JAX DISPATCHER PASSED"
)

print(
    "================================"
)

print(
    "\nThe high-level action space is:"
)

print(
    "0 = GO_TO_KEY"
)

print(
    "1 = PICKUP_KEY"
)

print(
    "2 = GO_TO_DOOR"
)

print(
    "3 = OPEN_DOOR"
)

print(
    "4 = GO_TO_GOAL"
)

print(
    "\nEach dispatched option now returns:"
)

print(
    "timestep, duration, option_reward, valid"
)