import jax
import jax.numpy as jnp
import navix as nx

from jax_option_executor import (
    execute_option_jax,
)


# ============================================================
# Create environment
# ============================================================

env = nx.make(
    "Navix-DoorKey-Random-8x8-v0"
)


def create_timestep(seed=0):

    rng = jax.random.PRNGKey(
        seed
    )

    return env.reset(
        rng
    )


# ============================================================
# JIT-compile the complete dispatcher
# ============================================================

execute_option_fn = jax.jit(
    lambda timestep, option_id:
        execute_option_jax(
            env,
            timestep,
            option_id,
        )
)


print(
    "Testing JAX option dispatcher...\n"
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


print(
    "TEST 1: Correct option sequence"
)


for option_id in sequence:

    timestep, duration, valid = execute_option_fn(
        timestep,
        jnp.asarray(
            option_id,
            dtype=jnp.int32,
        ),
    )

    total_primitive_steps += int(
        duration
    )

    print(
        f"{option_names[option_id]}:"
    )

    print(
        "    Duration:",
        duration
    )

    print(
        "    Valid:",
        valid
    )

    assert bool(
        valid
    )


print(
    "Reward:",
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


assert bool(
    timestep.is_done()
)

assert float(
    timestep.reward
) > 0.0


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

timestep, duration, valid = execute_option_fn(
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

timestep, duration, valid = execute_option_fn(
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
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1


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

timestep, duration, valid = execute_option_fn(
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
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1


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

timestep, _, valid = execute_option_fn(
    timestep,
    jnp.asarray(
        0,
        dtype=jnp.int32,
    ),
)

assert bool(
    valid
)

timestep, _, valid = execute_option_fn(
    timestep,
    jnp.asarray(
        1,
        dtype=jnp.int32,
    ),
)

assert bool(
    valid
)


timestep, duration, valid = execute_option_fn(
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
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1


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

timestep, duration, valid = execute_option_fn(
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
    "Valid:",
    valid
)


assert bool(
    valid
) is False

assert int(
    duration
) == 1


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
    "JAX OPTION DISPATCHER PASSED"
)

print(
    "================================"
)

print(
    "\nThe high-level action space is now:"
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