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
    return tuple(int(x) for x in position)


def desired_direction(current_pos, target_pos):
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

    Includes:
        - grid walls
        - Wall entities
        - closed doors

    Open doors are traversable.
    """

    blocked = set()

    grid = state.grid
    height, width = grid.shape

    # Grid-level blocked cells
    for row in range(height):
        for col in range(width):

            if int(grid[row, col]) != 0:
                blocked.add((row, col))

    # Internal DoorKey separating wall entities
    if "wall" in state.entities:

        walls = state.entities["wall"]

        for wall_pos in walls.position:
            blocked.add(
                to_tuple(wall_pos)
            )

    # Door entities
    doors = state.get_doors()

    for door_pos, is_open in zip(
        doors.position,
        doors.open
    ):

        door_tuple = to_tuple(
            door_pos
        )

        if bool(is_open):
            blocked.discard(
                door_tuple
            )
        elif not allow_door:
            blocked.add(
                door_tuple
            )

    return blocked


def bfs_shortest_path(
    start,
    goal,
    blocked,
    height,
    width,
):
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

            if not (
                0 <= row < height
                and 0 <= col < width
            ):
                continue

            if next_pos in blocked:
                continue

            if next_pos in parent:
                continue

            parent[next_pos] = current

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

        timestep, rotation_steps = rotate_to_direction(
            env,
            timestep,
            direction
        )

        primitive_steps += rotation_steps

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

    timestep, movement_steps = follow_path(
        env,
        timestep,
        best_path
    )

    primitive_steps = movement_steps

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
        allow_door=True
    )

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
# Option definitions
# ============================================================

def go_to_key(env, timestep):
    keys = timestep.state.get_keys()

    key_position = to_tuple(
        keys.position[0]
    )

    timestep, duration = navigate_to_adjacent_target(
        env,
        timestep,
        key_position
    )

    return timestep, duration


def pickup_key(env, timestep):
    timestep = env.step(
        timestep,
        jnp.asarray(PICKUP)
    )

    return timestep, 1


def go_to_door(env, timestep):
    doors = timestep.state.get_doors()

    door_position = to_tuple(
        doors.position[0]
    )

    timestep, duration = navigate_to_adjacent_target(
        env,
        timestep,
        door_position
    )

    return timestep, duration


def open_door(env, timestep):
    timestep = env.step(
        timestep,
        jnp.asarray(TOGGLE)
    )

    return timestep, 1


def go_to_goal(env, timestep):
    goals = timestep.state.get_goals()

    goal_position = to_tuple(
        goals.position[0]
    )

    timestep, duration = navigate_to_target(
        env,
        timestep,
        goal_position
    )

    return timestep, duration


# ============================================================
# Single-seed demo
# ============================================================

def run_single_seed_demo(seed=0):

    env = nx.make(
        "Navix-DoorKey-Random-8x8-v0"
    )

    rng = jax.random.PRNGKey(seed)
    timestep = env.reset(rng)

    player = timestep.state.get_player(idx=0)
    keys = timestep.state.get_keys()
    doors = timestep.state.get_doors()
    goals = timestep.state.get_goals()

    print("INITIAL STATE")
    print("Player position:", player.position)
    print("Player direction:", player.direction)
    print("Player pocket:", player.pocket)
    print("Key position:", keys.position)
    print("Door position:", doors.position)
    print("Goal position:", goals.position)

    timestep, go_to_key_duration = go_to_key(
        env,
        timestep
    )

    print(
        "\nGO_TO_KEY duration:",
        go_to_key_duration
    )

    player = timestep.state.get_player(idx=0)
    keys = timestep.state.get_keys()

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

    print("GO_TO_KEY test passed.")

    timestep, pickup_duration = pickup_key(
        env,
        timestep
    )

    player = timestep.state.get_player(idx=0)

    assert int(player.pocket) != -1

    print(
        "PICKUP_KEY duration:",
        pickup_duration
    )

    print("PICKUP_KEY test passed.")

    timestep, go_to_door_duration = go_to_door(
        env,
        timestep
    )

    player = timestep.state.get_player(idx=0)
    doors = timestep.state.get_doors()

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
        "GO_TO_DOOR duration:",
        go_to_door_duration
    )

    print("GO_TO_DOOR test passed.")

    timestep, open_door_duration = open_door(
        env,
        timestep
    )

    doors = timestep.state.get_doors()

    assert bool(
        doors.open[0]
    )

    print(
        "OPEN_DOOR duration:",
        open_door_duration
    )

    print("OPEN_DOOR test passed.")

    timestep, go_to_goal_duration = go_to_goal(
        env,
        timestep
    )

    player = timestep.state.get_player(idx=0)
    goals = timestep.state.get_goals()

    player_pos = to_tuple(
        player.position
    )

    goal_pos = to_tuple(
        goals.position[0]
    )

    assert player_pos == goal_pos
    assert bool(timestep.is_done())
    assert float(timestep.reward) > 0.0

    print(
        "GO_TO_GOAL duration:",
        go_to_goal_duration
    )

    print("GO_TO_GOAL test passed.")

    total_duration = (
        go_to_key_duration
        + pickup_duration
        + go_to_door_duration
        + open_door_duration
        + go_to_goal_duration
    )

    print("\n================================")
    print("COMPLETE OPTION SEQUENCE PASSED")
    print("================================")

    print(
        "Total option decisions: 5"
    )

    print(
        "Total primitive steps:",
        total_duration
    )

    print(
        "Temporal abstraction ratio:",
        f"{total_duration / 5:.2f}"
    )


if __name__ == "__main__":
    run_single_seed_demo()