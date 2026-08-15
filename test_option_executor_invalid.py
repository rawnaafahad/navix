import jax
import navix as nx

from option_executor import (
    execute_option,
    OPTION_NAMES,
)


def new_env(seed=0):
    env = nx.make(
        "Navix-DoorKey-Random-8x8-v0"
    )

    rng = jax.random.PRNGKey(seed)
    timestep = env.reset(rng)

    return env, timestep


print("Testing invalid option choices...\n")


# ============================================================
# TEST 1
# Try to open door immediately
# ============================================================

env, timestep = new_env()

t_before = int(timestep.t)

timestep, duration, valid = execute_option(
    env,
    timestep,
    3,  # OPEN_DOOR
)

print("TEST 1: OPEN_DOOR before key")
print("Valid:", valid)
print("Duration:", duration)
print("Environment steps consumed:", int(timestep.t) - t_before)

assert valid is False
assert duration == 1

print("PASS\n")


# ============================================================
# TEST 2
# Try to go to goal immediately
# ============================================================

env, timestep = new_env()

timestep, duration, valid = execute_option(
    env,
    timestep,
    4,  # GO_TO_GOAL
)

print("TEST 2: GO_TO_GOAL before door open")
print("Valid:", valid)
print("Duration:", duration)

assert valid is False
assert duration == 1

print("PASS\n")


# ============================================================
# TEST 3
# Try to go to door without key
# ============================================================

env, timestep = new_env()

timestep, duration, valid = execute_option(
    env,
    timestep,
    2,  # GO_TO_DOOR
)

print("TEST 3: GO_TO_DOOR without key")
print("Valid:", valid)
print("Duration:", duration)

assert valid is False
assert duration == 1

print("PASS\n")


# ============================================================
# TEST 4
# Correctly collect key, then try GO_TO_KEY again
# ============================================================

env, timestep = new_env()

timestep, _, valid = execute_option(
    env,
    timestep,
    0,
)

assert valid

timestep, _, valid = execute_option(
    env,
    timestep,
    1,
)

assert valid


timestep, duration, valid = execute_option(
    env,
    timestep,
    0,  # GO_TO_KEY again
)

print("TEST 4: GO_TO_KEY after key collected")
print("Valid:", valid)
print("Duration:", duration)

assert valid is False
assert duration == 1

print("PASS\n")


# ============================================================
# TEST 5
# Open door correctly, then try OPEN_DOOR again
# ============================================================

env, timestep = new_env()

for option_id in [
    0,
    1,
    2,
    3,
]:

    timestep, _, valid = execute_option(
        env,
        timestep,
        option_id,
    )

    assert valid


timestep, duration, valid = execute_option(
    env,
    timestep,
    3,  # OPEN_DOOR again
)

print("TEST 5: OPEN_DOOR after already open")
print("Valid:", valid)
print("Duration:", duration)

assert valid is False
assert duration == 1

print("PASS\n")


# ============================================================
# TEST 6
# Invalid option ID
# ============================================================

env, timestep = new_env()

timestep, duration, valid = execute_option(
    env,
    timestep,
    99,
)

print("TEST 6: Invalid option ID")
print("Valid:", valid)
print("Duration:", duration)

assert valid is False
assert duration == 1

print("PASS\n")


print(
    "================================"
)

print(
    "ALL INVALID-OPTION TESTS PASSED"
)

print(
    "================================"
)