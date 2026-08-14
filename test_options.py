from collections import deque

import jax
import jax.numpy as jnp
import navix as nx


# ============================================================
# Primitive action indices in Navix
# ============================================================

ROTATE_LEFT = 0
ROTATE_RIGHT = 1
FORWARD = 2
PICKUP = 3
TOGGLE = 5


# ============================================================
# Navix directions
# ============================================================

EAST = 0
SOUTH = 1
WEST = 2
NORTH = 3


DIRECTION_DELTAS = {
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1),
    NORTH: (-1, 0),
}


# ============================================================
# Utility functions
# ============================================================

def to_tuple(position):
    """
    Convert a JAX coordinate array into a normal Python tuple.
    """
    return tuple(int(x) for x in position)


def desired_direction(current_pos, target_pos):
    """
    Return the direction needed to move from current_pos
    to an adjacent target_pos.
    """

    current_row, current_col = current_pos
    target_row, target_col = target_pos

    delta = (
        target_row - current_row,
        target_col - current_col,
    )

    for direction, direction_delta in DIRECTION_DELTAS.items():
        if delta == direction_delta:
            return direction

    raise ValueError(
        f"Positions are not adjacent: "
        f"{current_pos} -> {target_pos}"
    )


def rotate_to_direction(env, timestep, target_direction):
    """
    Rotate the player until facing target_direction.

    Returns:
        timestep:
            Updated environment timestep.

        primitive_steps:
            Number of primitive rotation actions executed.
    """

    primitive_steps = 0

    while int(
        timestep.state.get_player(idx=0).direction
    ) != target_direction:

        current_direction = int(
            timestep.state.get_player(idx=0).direction
        )

        clockwise_distance = (
            target_direction - current_direction
        ) % 4

        counter_clockwise_distance = (
            current_direction - target_direction
        ) % 4

        if clockwise_distance <= counter_clockwise_distance:
            action = jnp.asarray(ROTATE_RIGHT)
        else:
            action = jnp.asarray(ROTATE_LEFT)

        timestep = env.step(
            timestep,
            action
        )

        primitive_steps += 1

    return timestep, primitive_steps


# ============================================================
# Pathfinding
# ============================================================

def get_blocked_positions(state, allow_door=False):
    """
    Build a set containing positions that the controller
    should not walk through.

    Walls are always blocked.

    Closed doors are blocked unless allow_door=True.
    """

    blocked = set()

    grid = state.grid
    height, width = grid.shape

    # Add grid walls
    for row in range(height):
        for col in range(width):

            if int(grid[row, col]) != 0:
                blocked.add((row, col))

    # Doors are entities and may not appear as blocked
    # directly in the underlying grid.
    doors = state.get_doors()

    for door_pos, is_open in zip(
        doors.position,
        doors.open
    ):

        door_tuple = to_tuple(door_pos)

        if not bool(is_open) and not allow_door:
            blocked.add(door_tuple)

    return blocked


def bfs_shortest_path(
    start,
    goal,
    blocked,
    height,
    width,
):
    """
    Perform breadth-first search on the Navix grid.

    Returns the shortest coordinate path from start
    to goal, including both endpoints.

    Returns None if no path exists.
    """

    start = tuple(start)
    goal = tuple(goal)

    if start == goal:
        return [start]

    queue = deque([start])

    parent = {
        start: None
    }

    while queue:

        current = queue.popleft()

        for delta in DIRECTION_DELTAS.values():

            next_pos = (
                current[0] + delta[0],
                current[1] + delta[1],
            )

            row, col = next_pos

            # Stay inside map
            if not (
                0 <= row < height
                and 0 <= col < width
            ):
                continue

            # Avoid obstacles
            if next_pos in blocked:
                continue

            # Avoid revisiting cells
            if next_pos in parent:
                continue

            parent[next_pos] = current

            # Goal found
            if next_pos == goal:

                path = [goal]

                while path[-1] != start:
                    path.append(
                        parent[path[-1]]
                    )

                path.reverse()

                return path

            queue.append(next_pos)

    return None


def follow_path(env, timestep, path):
    """
    Execute primitive rotation and forward actions
    required to follow a coordinate path.
    """

    primitive_steps = 0

    for next_position in path[1:]:

        player = timestep.state.get_player(idx=0)

        current_position = to_tuple(
            player.position
        )

        direction = desired_direction(
            current_position,
            next_position
        )

        # Rotate towards next cell
        timestep, rotation_steps = rotate_to_direction(
            env,
            timestep,
            direction
        )

        primitive_steps += rotation_steps

        # Move into next cell
        timestep = env.step(
            timestep,
            jnp.asarray(FORWARD)
        )

        primitive_steps += 1

    return timestep, primitive_steps


def navigate_to_adjacent_target(
    env,
    timestep,
    target_position,
    max_steps=100,
):
    """
    Navigate to the shortest reachable square adjacent
    to target_position and finish facing the target.

    Used for objects such as keys and doors because
    PICKUP and TOGGLE operate on the square directly
    in front of the player.
    """

    state = timestep.state
    player = state.get_player(idx=0)

    start = to_tuple(
        player.position
    )

    target = tuple(
        target_position
    )

    height, width = state.grid.shape

    blocked = get_blocked_positions(
        state,
        allow_door=False
    )

    candidate_positions = []

    # Find every walkable tile adjacent to target
    for delta in DIRECTION_DELTAS.values():

        candidate = (
            target[0] + delta[0],
            target[1] + delta[1],
        )

        row, col = candidate

        if not (
            0 <= row < height
            and 0 <= col < width
        ):
            continue

        if candidate in blocked:
            continue

        candidate_positions.append(
            candidate
        )

    # Find shortest reachable candidate
    best_path = None

    for candidate in candidate_positions:

        path = bfs_shortest_path(
            start=start,
            goal=candidate,
            blocked=blocked,
            height=height,
            width=width,
        )

        if path is None:
            continue

        if (
            best_path is None
            or len(path) < len(best_path)
        ):
            best_path = path

    if best_path is None:
        raise RuntimeError(
            f"No reachable adjacent position "
            f"found for target {target}."
        )

    # Follow path
    timestep, movement_steps = follow_path(
        env,
        timestep,
        best_path
    )

    primitive_steps = movement_steps

    # Finish facing target
    player = timestep.state.get_player(idx=0)

    final_position = to_tuple(
        player.position
    )

    target_direction = desired_direction(
        final_position,
        target
    )

    timestep, rotation_steps = rotate_to_direction(
        env,
        timestep,
        target_direction
    )

    primitive_steps += rotation_steps

    if primitive_steps > max_steps:
        raise RuntimeError(
            "Navigation exceeded maximum primitive steps."
        )

    return timestep, primitive_steps


def navigate_to_target(
    env,
    timestep,
    target_position,
    max_steps=100,
):
    """
    Navigate directly ONTO target_position.

    Unlike navigate_to_adjacent_target(), this is used
    when the player must occupy the target tile itself.

    GO_TO_GOAL uses this function.
    """

    state = timestep.state
    player = state.get_player(idx=0)

    start = to_tuple(
        player.position
    )

    target = tuple(
        target_position
    )

    height, width = state.grid.shape

    # At this stage the door should already be open.
    blocked = get_blocked_positions(
        state,
        allow_door=True
    )

    # Explicitly remove open doors from blocked cells.
    doors = state.get_doors()

    for door_pos, is_open in zip(
        doors.position,
        doors.open
    ):

        if bool(is_open):

            blocked.discard(
                to_tuple(door_pos)
            )

    path = bfs_shortest_path(
        start=start,
        goal=target,
        blocked=blocked,
        height=height,
        width=width,
    )

    if path is None:
        raise RuntimeError(
            f"No path found to target {target}."
        )

    timestep, primitive_steps = follow_path(
        env,
        timestep,
        path
    )

    if primitive_steps > max_steps:
        raise RuntimeError(
            "Navigation exceeded maximum primitive steps."
        )

    return timestep, primitive_steps


# ============================================================
# OPTION 1: GO_TO_KEY
# ============================================================

def go_to_key(env, timestep):
    """
    GO_TO_KEY option.

    Navigate to a square adjacent to the key
    and finish facing it.
    """

    keys = timestep.state.get_keys()

    key_position = to_tuple(
        keys.position[0]
    )

    timestep, duration = navigate_to_adjacent_target(
        env,
        timestep,
        key_position
    )

    print(
        f"GO_TO_KEY completed in "
        f"{duration} primitive steps."
    )

    return timestep, duration


# ============================================================
# OPTION 2: PICKUP_KEY
# ============================================================

def pickup_key(env, timestep):
    """
    PICKUP_KEY option.

    Assumes the player is adjacent to the key
    and facing it.
    """

    player_before = timestep.state.get_player(idx=0)

    print(
        "\nPocket before pickup:",
        player_before.pocket
    )

    timestep = env.step(
        timestep,
        jnp.asarray(PICKUP)
    )

    player_after = timestep.state.get_player(idx=0)

    print(
        "Pocket after pickup:",
        player_after.pocket
    )

    return timestep, 1


# ============================================================
# OPTION 3: GO_TO_DOOR
# ============================================================

def go_to_door(env, timestep):
    """
    GO_TO_DOOR option.

    Navigate to a square adjacent to the locked door
    and finish facing it.
    """

    doors = timestep.state.get_doors()

    door_position = to_tuple(
        doors.position[0]
    )

    timestep, duration = navigate_to_adjacent_target(
        env,
        timestep,
        door_position
    )

    print(
        f"\nGO_TO_DOOR completed in "
        f"{duration} primitive steps."
    )

    return timestep, duration


# ============================================================
# OPTION 4: OPEN_DOOR
# ============================================================

def open_door(env, timestep):
    """
    OPEN_DOOR option.

    Assumes the player is adjacent to the door,
    facing it, and holding the required key.
    """

    doors_before = timestep.state.get_doors()

    print(
        "\nDoor open before:",
        doors_before.open
    )

    timestep = env.step(
        timestep,
        jnp.asarray(TOGGLE)
    )

    doors_after = timestep.state.get_doors()

    print(
        "Door open after:",
        doors_after.open
    )

    return timestep, 1


# ============================================================
# OPTION 5: GO_TO_GOAL
# ============================================================

def go_to_goal(env, timestep):
    """
    GO_TO_GOAL option.

    Navigate through the opened door and directly
    onto the goal tile.
    """

    goals = timestep.state.get_goals()

    goal_position = to_tuple(
        goals.position[0]
    )

    timestep, duration = navigate_to_target(
        env,
        timestep,
        goal_position
    )

    print(
        f"\nGO_TO_GOAL completed in "
        f"{duration} primitive steps."
    )

    return timestep, duration


# ============================================================
# Create environment
# ============================================================

env = nx.make(
    "Navix-DoorKey-Random-8x8-v0"
)

rng = jax.random.PRNGKey(0)

timestep = env.reset(rng)


# ============================================================
# Initial state
# ============================================================

player = timestep.state.get_player(idx=0)
keys = timestep.state.get_keys()
doors = timestep.state.get_doors()
goals = timestep.state.get_goals()

print("INITIAL STATE")

print(
    "Player position:",
    player.position
)

print(
    "Player direction:",
    player.direction
)

print(
    "Player pocket:",
    player.pocket
)

print(
    "Key position:",
    keys.position
)

print(
    "Door position:",
    doors.position
)

print(
    "Goal position:",
    goals.position
)


# ============================================================
# Test OPTION 1: GO_TO_KEY
# ============================================================

timestep, go_to_key_duration = go_to_key(
    env,
    timestep
)

player = timestep.state.get_player(idx=0)
keys = timestep.state.get_keys()

print("\nAFTER GO_TO_KEY")

print(
    "Player position:",
    player.position
)

print(
    "Player direction:",
    player.direction
)

print(
    "Player pocket:",
    player.pocket
)

print(
    "Key position:",
    keys.position
)

print(
    "Option duration:",
    go_to_key_duration
)


# Verify player is adjacent to key

player_pos = to_tuple(
    player.position
)

key_pos = to_tuple(
    keys.position[0]
)

distance = (
    abs(player_pos[0] - key_pos[0])
    + abs(player_pos[1] - key_pos[1])
)

assert distance == 1

print(
    "\nGO_TO_KEY test passed."
)


# ============================================================
# Test OPTION 2: PICKUP_KEY
# ============================================================

timestep, pickup_duration = pickup_key(
    env,
    timestep
)

player = timestep.state.get_player(idx=0)
keys = timestep.state.get_keys()

print("\nAFTER PICKUP_KEY")

print(
    "Player position:",
    player.position
)

print(
    "Player direction:",
    player.direction
)

print(
    "Player pocket:",
    player.pocket
)

print(
    "Key position:",
    keys.position
)

print(
    "Option duration:",
    pickup_duration
)

assert int(player.pocket) != -1

print(
    "\nPICKUP_KEY test passed."
)


# ============================================================
# Test OPTION 3: GO_TO_DOOR
# ============================================================

timestep, go_to_door_duration = go_to_door(
    env,
    timestep
)

player = timestep.state.get_player(idx=0)
doors = timestep.state.get_doors()

print("\nAFTER GO_TO_DOOR")

print(
    "Player position:",
    player.position
)

print(
    "Player direction:",
    player.direction
)

print(
    "Player pocket:",
    player.pocket
)

print(
    "Door position:",
    doors.position
)

print(
    "Door open:",
    doors.open
)

print(
    "Option duration:",
    go_to_door_duration
)


# Verify player is adjacent to door

player_pos = to_tuple(
    player.position
)

door_pos = to_tuple(
    doors.position[0]
)

distance = (
    abs(player_pos[0] - door_pos[0])
    + abs(player_pos[1] - door_pos[1])
)

assert distance == 1

print(
    "\nGO_TO_DOOR test passed."
)


# ============================================================
# Test OPTION 4: OPEN_DOOR
# ============================================================

timestep, open_door_duration = open_door(
    env,
    timestep
)

player = timestep.state.get_player(idx=0)
doors = timestep.state.get_doors()

print("\nAFTER OPEN_DOOR")

print(
    "Player position:",
    player.position
)

print(
    "Player direction:",
    player.direction
)

print(
    "Player pocket:",
    player.pocket
)

print(
    "Door position:",
    doors.position
)

print(
    "Door open:",
    doors.open
)

print(
    "Option duration:",
    open_door_duration
)

assert bool(
    doors.open[0]
)

print(
    "\nOPEN_DOOR test passed."
)


# ============================================================
# Test OPTION 5: GO_TO_GOAL
# ============================================================

timestep, go_to_goal_duration = go_to_goal(
    env,
    timestep
)

player = timestep.state.get_player(idx=0)
goals = timestep.state.get_goals()

print("\nAFTER GO_TO_GOAL")

print(
    "Player position:",
    player.position
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
    "Option duration:",
    go_to_goal_duration
)


# Verify player reached goal

player_pos = to_tuple(
    player.position
)

goal_pos = to_tuple(
    goals.position[0]
)

assert player_pos == goal_pos

assert bool(
    timestep.is_done()
)

assert float(
    timestep.reward
) > 0.0

print(
    "\nGO_TO_GOAL test passed."
)


# ============================================================
# Final option summary
# ============================================================

total_duration = (
    go_to_key_duration
    + pickup_duration
    + go_to_door_duration
    + open_door_duration
    + go_to_goal_duration
)

print(
    "\n================================"
)

print(
    "COMPLETE OPTION SEQUENCE PASSED"
)

print(
    "================================"
)

print(
    "\nOPTION SUMMARY"
)

print(
    "GO_TO_KEY duration:",
    go_to_key_duration
)

print(
    "PICKUP_KEY duration:",
    pickup_duration
)

print(
    "GO_TO_DOOR duration:",
    go_to_door_duration
)

print(
    "OPEN_DOOR duration:",
    open_door_duration
)

print(
    "GO_TO_GOAL duration:",
    go_to_goal_duration
)

print(
    "\nTotal option decisions: 5"
)

print(
    "Total primitive steps:",
    total_duration
)

print(
    "Temporal abstraction ratio:",
    f"{total_duration / 5:.2f} primitive steps per option"
)