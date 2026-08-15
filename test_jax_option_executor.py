import jax
import jax.numpy as jnp
import navix as nx

from jax_option_executor import (
    direction_to_target,
    direction_to_adjacent_target,
    action_towards_direction,
    go_to_key_jax,
    pickup_key_jax,
    go_to_door_jax,open_door_jax, go_to_goal_jax
)


print(
    "Testing JAX option utilities...\n"
)


# ============================================================
# TEST 1
# Direction toward target
# ============================================================

current = jnp.asarray(
    [6, 1]
)

target = jnp.asarray(
    [1, 1]
)

direction_fn = jax.jit(
    direction_to_target
)

direction = direction_fn(
    current,
    target,
)

print(
    "Direction toward key:",
    direction
)

assert int(direction) == 3

print(
    "TEST 1 PASS"
)


# ============================================================
# TEST 2
# Adjacent direction
# ============================================================

current = jnp.asarray(
    [2, 1]
)

target = jnp.asarray(
    [1, 1]
)

adjacent_fn = jax.jit(
    direction_to_adjacent_target
)

direction = adjacent_fn(
    current,
    target,
)

print(
    "Adjacent direction:",
    direction
)

assert int(direction) == 3

print(
    "TEST 2 PASS"
)


# ============================================================
# TEST 3
# Action selection
# ============================================================

action_fn = jax.jit(
    action_towards_direction
)

action = action_fn(
    jnp.asarray(2),
    jnp.asarray(3),
)

print(
    "Rotation action:",
    action
)

assert int(action) == 1


action = action_fn(
    jnp.asarray(3),
    jnp.asarray(3),
)

print(
    "Forward action:",
    action
)

assert int(action) == 2

print(
    "TEST 3 PASS"
)


# ============================================================
# Create environment
# ============================================================

env = nx.make(
    "Navix-DoorKey-Random-8x8-v0"
)

rng = jax.random.PRNGKey(0)
timestep = env.reset(rng)


# ============================================================
# TEST 4
# JAX GO_TO_KEY
# ============================================================

player_before = timestep.state.get_player(
    idx=0
)

key_before = timestep.state.get_keys()

print(
    "\nInitial player position:",
    player_before.position
)

print(
    "Initial player direction:",
    player_before.direction
)

print(
    "Key position:",
    key_before.position
)


go_to_key_fn = jax.jit(
    lambda ts: go_to_key_jax(
        env,
        ts,
    )
)


timestep, go_to_key_duration, go_to_key_valid = go_to_key_fn(
    timestep
)


player_after_key = timestep.state.get_player(
    idx=0
)

key_after = timestep.state.get_keys()

print(
    "\nAfter GO_TO_KEY"
)

print(
    "Player position:",
    player_after_key.position
)

print(
    "Player direction:",
    player_after_key.direction
)

print(
    "Key position:",
    key_after.position
)

print(
    "GO_TO_KEY duration:",
    go_to_key_duration
)

print(
    "GO_TO_KEY valid:",
    go_to_key_valid
)


distance = jnp.abs(
    player_after_key.position
    - key_after.position[0]
).sum()


assert int(distance) == 1
assert bool(go_to_key_valid)

print(
    "TEST 4 PASS"
)


# ============================================================
# TEST 5
# JAX PICKUP_KEY
# ============================================================

pickup_key_fn = jax.jit(
    lambda ts: pickup_key_jax(
        env,
        ts,
    )
)


player_before_pickup = timestep.state.get_player(
    idx=0
)

print(
    "\nBefore PICKUP_KEY"
)

print(
    "Player pocket:",
    player_before_pickup.pocket
)


timestep, pickup_duration, pickup_valid = pickup_key_fn(
    timestep
)


player_after_pickup = timestep.state.get_player(
    idx=0
)

key_after_pickup = timestep.state.get_keys()

print(
    "\nAfter PICKUP_KEY"
)

print(
    "Player position:",
    player_after_pickup.position
)

print(
    "Player direction:",
    player_after_pickup.direction
)

print(
    "Player pocket:",
    player_after_pickup.pocket
)

print(
    "Key position:",
    key_after_pickup.position
)

print(
    "PICKUP_KEY duration:",
    pickup_duration
)

print(
    "PICKUP_KEY valid:",
    pickup_valid
)


assert int(
    player_before_pickup.pocket
) == -1

assert int(
    player_after_pickup.pocket
) != -1

assert int(
    pickup_duration
) == 1

assert bool(
    pickup_valid
)

print(
    "TEST 5 PASS"
)


# ============================================================
# TEST 6
# JAX GO_TO_DOOR
# ============================================================

go_to_door_fn = jax.jit(
    lambda ts: go_to_door_jax(
        env,
        ts,
    )
)


timestep, go_to_door_duration, go_to_door_valid = go_to_door_fn(
    timestep
)


player_after_door = timestep.state.get_player(
    idx=0
)

doors = timestep.state.get_doors()

door_position = doors.position[0]

expected_position = jnp.asarray(
    [
        door_position[0],
        door_position[1] - 1,
    ],
    dtype=jnp.int32,
)


print(
    "\nAfter GO_TO_DOOR"
)

print(
    "Player position:",
    player_after_door.position
)

print(
    "Player direction:",
    player_after_door.direction
)

print(
    "Door position:",
    doors.position
)

print(
    "Expected approach position:",
    expected_position
)

print(
    "GO_TO_DOOR duration:",
    go_to_door_duration
)

print(
    "GO_TO_DOOR valid:",
    go_to_door_valid
)


assert bool(
    jnp.all(
        player_after_door.position
        == expected_position
    )
)

assert int(
    player_after_door.direction
) == 0

assert bool(
    go_to_door_valid
)

print(
    "TEST 6 PASS"
)

# ============================================================
# TEST 7
# JAX OPEN_DOOR
# ============================================================

open_door_fn = jax.jit(
    lambda ts: open_door_jax(
        env,
        ts,
    )
)

doors_before = timestep.state.get_doors()

print(
    "\nBefore OPEN_DOOR"
)

print(
    "Door open:",
    doors_before.open
)

timestep, open_door_duration, open_door_valid = open_door_fn(
    timestep
)

doors_after = timestep.state.get_doors()

print(
    "\nAfter OPEN_DOOR"
)

print(
    "Door open:",
    doors_after.open
)

print(
    "OPEN_DOOR duration:",
    open_door_duration
)

print(
    "OPEN_DOOR valid:",
    open_door_valid
)

assert bool(
    doors_before.open[0]
) is False

assert bool(
    doors_after.open[0]
) is True

assert int(
    open_door_duration
) == 1

assert bool(
    open_door_valid
)

print(
    "TEST 7 PASS"
)

# ============================================================
# TEST 8
# JAX GO_TO_GOAL
# ============================================================

go_to_goal_fn = jax.jit(
    lambda ts: go_to_goal_jax(
        env,
        ts,
    )
)

timestep, go_to_goal_duration, go_to_goal_valid = go_to_goal_fn(
    timestep
)

player_after_goal = timestep.state.get_player(
    idx=0
)

goals = timestep.state.get_goals()

print(
    "\nAfter GO_TO_GOAL"
)

print(
    "Player position:",
    player_after_goal.position
)

print(
    "Goal position:",
    goals.position
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
    "GO_TO_GOAL duration:",
    go_to_goal_duration
)

print(
    "GO_TO_GOAL valid:",
    go_to_goal_valid
)


assert bool(
    jnp.all(
        player_after_goal.position
        == goals.position[0]
    )
)

assert bool(
    timestep.is_done()
)

assert float(
    timestep.reward
) > 0.0

assert bool(
    go_to_goal_valid
)

print(
    "TEST 8 PASS"
)

# ============================================================
# Combined check
# ============================================================
total_duration = (
    int(go_to_key_duration)
    + int(pickup_duration)
    + int(go_to_door_duration)
    + int(open_door_duration)
    + int(go_to_goal_duration)
)

print(
    "\n==============================="
)

print(
    "ALL FIVE JAX OPTIONS PASSED"
)

print(
    "==============================="
)

print(
    "Total option decisions:",
    5
)

print(
    "Total primitive steps:",
    total_duration
)

print(
    "Mean primitive steps per option:",
    f"{total_duration / 5:.2f}"
)