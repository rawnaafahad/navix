import jax
import jax.numpy as jnp


# ============================================================
# Primitive Navix actions
# ============================================================

ROTATE_LEFT = 0
ROTATE_RIGHT = 1
FORWARD = 2
PICKUP = 3
DROP = 4
TOGGLE = 5
DONE = 6


# ============================================================
# Directions
#
# 0 = east
# 1 = south
# 2 = west
# 3 = north
# ============================================================

EAST = 0
SOUTH = 1
WEST = 2
NORTH = 3


# ============================================================
# JAX-safe direction utilities
# ============================================================

def direction_to_target(
    current_position,
    target_position,
):
    """
    Choose a direction that greedily reduces Manhattan
    distance to the target.

    Row movement is performed first, then column movement.
    """

    row = current_position[0]
    col = current_position[1]

    target_row = target_position[0]
    target_col = target_position[1]

    return jax.lax.cond(
        target_row < row,
        lambda _: jnp.asarray(NORTH, dtype=jnp.int32),
        lambda _: jax.lax.cond(
            target_row > row,
            lambda _: jnp.asarray(SOUTH, dtype=jnp.int32),
            lambda _: jax.lax.cond(
                target_col < col,
                lambda _: jnp.asarray(WEST, dtype=jnp.int32),
                lambda _: jnp.asarray(EAST, dtype=jnp.int32),
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )


def direction_to_adjacent_target(
    current_position,
    target_position,
):
    """
    Return the direction from current_position to an
    adjacent target_position.
    """

    difference = (
        target_position
        - current_position
    )

    return jax.lax.cond(
        difference[0] < 0,
        lambda _: jnp.asarray(NORTH, dtype=jnp.int32),
        lambda _: jax.lax.cond(
            difference[0] > 0,
            lambda _: jnp.asarray(SOUTH, dtype=jnp.int32),
            lambda _: jax.lax.cond(
                difference[1] < 0,
                lambda _: jnp.asarray(WEST, dtype=jnp.int32),
                lambda _: jnp.asarray(EAST, dtype=jnp.int32),
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )


# ============================================================
# Primitive action selection
# ============================================================

def action_towards_direction(
    current_direction,
    desired_direction,
):
    """
    Choose either ROTATE_RIGHT or FORWARD.

    Repeated calls will eventually align the player with
    desired_direction and then move forward.
    """

    facing_correctly = (
        current_direction
        == desired_direction
    )

    return jax.lax.select(
        facing_correctly,
        jnp.asarray(
            FORWARD,
            dtype=jnp.int32,
        ),
        jnp.asarray(
            ROTATE_RIGHT,
            dtype=jnp.int32,
        ),
    )


# ============================================================
# Distance utilities
# ============================================================

def manhattan_distance(
    position_a,
    position_b,
):
    """
    JAX-safe Manhattan distance between two grid positions.
    """

    return jnp.abs(
        position_a - position_b
    ).sum()


# ============================================================
# One navigation step
# ============================================================

def navigation_step(
    env,
    timestep,
    target_position,
):
    """
    Execute exactly one primitive action toward a target.
    """

    player = timestep.state.get_player(
        idx=0
    )

    current_position = player.position
    current_direction = player.direction

    desired_direction = direction_to_target(
        current_position,
        target_position,
    )

    action = action_towards_direction(
        current_direction,
        desired_direction,
    )

    new_timestep = env.step(
        timestep,
        action,
    )

    return new_timestep


# ============================================================
# JAX OPTION 1: GO_TO_KEY
# ============================================================

def go_to_key_jax(
    env,
    timestep,
    max_steps=50,
):
    """
    JAX-compatible GO_TO_KEY option.

    Navigate until adjacent to the key, then face it.
    """

    keys = timestep.state.get_keys()
    key_position = keys.position[0]

    def navigation_condition(carry):
        current_timestep, primitive_steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        distance = manhattan_distance(
            player.position,
            key_position,
        )

        not_adjacent = (
            distance > 1
        )

        within_budget = (
            primitive_steps < max_steps
        )

        episode_active = jnp.logical_not(
            current_timestep.is_done()
        )

        return jnp.logical_and(
            jnp.logical_and(
                not_adjacent,
                within_budget,
            ),
            episode_active,
        )

    def navigation_body(carry):
        current_timestep, primitive_steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        desired_direction = direction_to_target(
            player.position,
            key_position,
        )

        action = action_towards_direction(
            player.direction,
            desired_direction,
        )

        new_timestep = env.step(
            current_timestep,
            action,
        )

        return (
            new_timestep,
            primitive_steps + 1,
        )

    timestep, primitive_steps = jax.lax.while_loop(
        navigation_condition,
        navigation_body,
        (
            timestep,
            jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
        ),
    )

    player = timestep.state.get_player(
        idx=0
    )

    desired_direction = direction_to_adjacent_target(
        player.position,
        key_position,
    )

    def alignment_condition(carry):
        current_timestep, primitive_steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        not_aligned = (
            player.direction
            != desired_direction
        )

        within_budget = (
            primitive_steps < max_steps
        )

        episode_active = jnp.logical_not(
            current_timestep.is_done()
        )

        return jnp.logical_and(
            jnp.logical_and(
                not_aligned,
                within_budget,
            ),
            episode_active,
        )

    def alignment_body(carry):
        current_timestep, primitive_steps = carry

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(
                ROTATE_RIGHT,
                dtype=jnp.int32,
            ),
        )

        return (
            new_timestep,
            primitive_steps + 1,
        )

    timestep, primitive_steps = jax.lax.while_loop(
        alignment_condition,
        alignment_body,
        (
            timestep,
            primitive_steps,
        ),
    )

    player = timestep.state.get_player(
        idx=0
    )

    final_distance = manhattan_distance(
        player.position,
        key_position,
    )

    final_direction = direction_to_adjacent_target(
        player.position,
        key_position,
    )

    adjacent = (
        final_distance == 1
    )

    facing_key = (
        player.direction
        == final_direction
    )

    within_budget = (
        primitive_steps <= max_steps
    )

    valid = jnp.logical_and(
        jnp.logical_and(
            adjacent,
            facing_key,
        ),
        within_budget,
    )

    return (
        timestep,
        primitive_steps,
        valid,
    )


# ============================================================
# JAX OPTION 2: PICKUP_KEY
# ============================================================

def pickup_key_jax(
    env,
    timestep,
):
    """
    JAX-compatible PICKUP_KEY option.
    """

    player_before = timestep.state.get_player(
        idx=0
    )

    pocket_before = player_before.pocket

    new_timestep = env.step(
        timestep,
        jnp.asarray(
            PICKUP,
            dtype=jnp.int32,
        ),
    )

    player_after = new_timestep.state.get_player(
        idx=0
    )

    pocket_after = player_after.pocket

    was_empty = (
        pocket_before == -1
    )

    now_holding_item = (
        pocket_after != -1
    )

    valid = jnp.logical_and(
        was_empty,
        now_holding_item,
    )

    primitive_steps = jnp.asarray(
        1,
        dtype=jnp.int32,
    )

    return (
        new_timestep,
        primitive_steps,
        valid,
    )


# ============================================================
# JAX OPTION 3: GO_TO_DOOR
# ============================================================

def go_to_door_jax(
    env,
    timestep,
    max_steps=50,
):
    """
    JAX-compatible GO_TO_DOOR option.

    The agent navigates to the tile immediately left
    of the door and finishes facing east toward it.
    """

    doors = timestep.state.get_doors()
    door_position = doors.position[0]

    approach_position = jnp.asarray(
        [
            door_position[0],
            door_position[1] - 1,
        ],
        dtype=jnp.int32,
    )

    def navigation_condition(carry):
        current_timestep, primitive_steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        distance = manhattan_distance(
            player.position,
            approach_position,
        )

        not_at_target = (
            distance > 0
        )

        within_budget = (
            primitive_steps < max_steps
        )

        episode_active = jnp.logical_not(
            current_timestep.is_done()
        )

        return jnp.logical_and(
            jnp.logical_and(
                not_at_target,
                within_budget,
            ),
            episode_active,
        )

    def navigation_body(carry):
        current_timestep, primitive_steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        desired_direction = direction_to_target(
            player.position,
            approach_position,
        )

        action = action_towards_direction(
            player.direction,
            desired_direction,
        )

        new_timestep = env.step(
            current_timestep,
            action,
        )

        return (
            new_timestep,
            primitive_steps + 1,
        )

    timestep, primitive_steps = jax.lax.while_loop(
        navigation_condition,
        navigation_body,
        (
            timestep,
            jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
        ),
    )

    desired_direction = jnp.asarray(
        EAST,
        dtype=jnp.int32,
    )

    def alignment_condition(carry):
        current_timestep, primitive_steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        not_aligned = (
            player.direction
            != desired_direction
        )

        within_budget = (
            primitive_steps < max_steps
        )

        episode_active = jnp.logical_not(
            current_timestep.is_done()
        )

        return jnp.logical_and(
            jnp.logical_and(
                not_aligned,
                within_budget,
            ),
            episode_active,
        )

    def alignment_body(carry):
        current_timestep, primitive_steps = carry

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(
                ROTATE_RIGHT,
                dtype=jnp.int32,
            ),
        )

        return (
            new_timestep,
            primitive_steps + 1,
        )

    timestep, primitive_steps = jax.lax.while_loop(
        alignment_condition,
        alignment_body,
        (
            timestep,
            primitive_steps,
        ),
    )

    player = timestep.state.get_player(
        idx=0
    )

    correct_position = jnp.all(
        player.position
        == approach_position
    )

    facing_door = (
        player.direction
        == EAST
    )

    valid = jnp.logical_and(
        correct_position,
        facing_door,
    )

    return (
        timestep,
        primitive_steps,
        valid,
    )
# ============================================================
# JAX OPTION 4: OPEN_DOOR
# ============================================================

def open_door_jax(
    env,
    timestep,
):
    """
    JAX-compatible OPEN_DOOR option.

    Executes one primitive TOGGLE action.

    Returns:
        timestep
        primitive_steps
        valid
    """

    doors_before = timestep.state.get_doors()

    was_open = doors_before.open[0]

    new_timestep = env.step(
        timestep,
        jnp.asarray(
            TOGGLE,
            dtype=jnp.int32,
        ),
    )

    doors_after = new_timestep.state.get_doors()

    is_open = doors_after.open[0]

    valid = jnp.logical_and(
        jnp.logical_not(was_open),
        is_open,
    )

    primitive_steps = jnp.asarray(
        1,
        dtype=jnp.int32,
    )

    return (
        new_timestep,
        primitive_steps,
        valid,
    )
# ============================================================
# JAX OPTION 5: GO_TO_GOAL
# ============================================================
# ============================================================
# JAX OPTION 5: GO_TO_GOAL
# ============================================================

def go_to_goal_jax(
    env,
    timestep,
    max_steps=50,
):
    """
    JAX-compatible GO_TO_GOAL option.

    Assumes:
        - the door is already open
        - the player is on the tile immediately left of the door
        - the player is facing east

    Execution:
        1. Move onto the open door tile.
        2. Move east again into the second room.
        3. Navigate greedily to the goal.

    Returns:
        timestep
        primitive_steps
        valid
    """

    goals = timestep.state.get_goals()
    goal_position = goals.position[0]

    doors = timestep.state.get_doors()
    door_position = doors.position[0]

    primitive_steps = jnp.asarray(
        0,
        dtype=jnp.int32,
    )

    # --------------------------------------------------------
    # Important positions
    # --------------------------------------------------------

    left_of_door = jnp.asarray(
        [
            door_position[0],
            door_position[1] - 1,
        ],
        dtype=jnp.int32,
    )

    right_of_door = jnp.asarray(
        [
            door_position[0],
            door_position[1] + 1,
        ],
        dtype=jnp.int32,
    )

    # --------------------------------------------------------
    # Phase 1:
    # Move from left of door -> door tile
    # --------------------------------------------------------

    player = timestep.state.get_player(
        idx=0
    )

    door_open = doors.open[0]

    on_left_of_door = jnp.all(
        player.position
        == left_of_door
    )

    facing_east = (
        player.direction
        == EAST
    )

    can_enter_door = jnp.logical_and(
        jnp.logical_and(
            door_open,
            on_left_of_door,
        ),
        facing_east,
    )

    def enter_door(args):

        current_timestep, steps = args

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(
                FORWARD,
                dtype=jnp.int32,
            ),
        )

        return (
            new_timestep,
            steps + 1,
        )

    def skip_enter_door(args):
        return args

    timestep, primitive_steps = jax.lax.cond(
        can_enter_door,
        enter_door,
        skip_enter_door,
        (
            timestep,
            primitive_steps,
        ),
    )

    # --------------------------------------------------------
    # Phase 2:
    # Move from door tile -> second room
    #
    # This is essential. If we start greedy navigation while
    # still on the door tile, row-first movement may attempt
    # to walk along the separating wall.
    # --------------------------------------------------------

    player = timestep.state.get_player(
        idx=0
    )

    on_door = jnp.all(
        player.position
        == door_position
    )

    facing_east = (
        player.direction
        == EAST
    )

    can_exit_door = jnp.logical_and(
        on_door,
        facing_east,
    )

    def exit_door(args):

        current_timestep, steps = args

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(
                FORWARD,
                dtype=jnp.int32,
            ),
        )

        return (
            new_timestep,
            steps + 1,
        )

    def skip_exit_door(args):
        return args

    timestep, primitive_steps = jax.lax.cond(
        can_exit_door,
        exit_door,
        skip_exit_door,
        (
            timestep,
            primitive_steps,
        ),
    )

    # --------------------------------------------------------
    # Phase 3:
    # Navigate through second room to goal
    # --------------------------------------------------------

    def navigation_condition(carry):

        current_timestep, steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        distance = manhattan_distance(
            player.position,
            goal_position,
        )

        not_at_goal = (
            distance > 0
        )

        within_budget = (
            steps < max_steps
        )

        episode_active = jnp.logical_not(
            current_timestep.is_done()
        )

        return jnp.logical_and(
            jnp.logical_and(
                not_at_goal,
                within_budget,
            ),
            episode_active,
        )

    def navigation_body(carry):

        current_timestep, steps = carry

        player = current_timestep.state.get_player(
            idx=0
        )

        desired_direction = direction_to_target(
            player.position,
            goal_position,
        )

        action = action_towards_direction(
            player.direction,
            desired_direction,
        )

        new_timestep = env.step(
            current_timestep,
            action,
        )

        return (
            new_timestep,
            steps + 1,
        )

    timestep, primitive_steps = jax.lax.while_loop(
        navigation_condition,
        navigation_body,
        (
            timestep,
            primitive_steps,
        ),
    )

    # --------------------------------------------------------
    # Final validity check
    # --------------------------------------------------------

    player = timestep.state.get_player(
        idx=0
    )

    reached_goal = jnp.all(
        player.position
        == goal_position
    )

    episode_done = timestep.is_done()

    positive_reward = (
        timestep.reward > 0.0
    )

    within_budget = (
        primitive_steps <= max_steps
    )

    valid = jnp.logical_and(
        jnp.logical_and(
            reached_goal,
            episode_done,
        ),
        jnp.logical_and(
            positive_reward,
            within_budget,
        ),
    )

    return (
        timestep,
        primitive_steps,
        valid,
    )
# ============================================================
# High-level option IDs
# ============================================================

GO_TO_KEY_OPTION = 0
PICKUP_KEY_OPTION = 1
GO_TO_DOOR_OPTION = 2
OPEN_DOOR_OPTION = 3
GO_TO_GOAL_OPTION = 4

NUM_OPTIONS = 5


# ============================================================
# Failed option
# ============================================================

def failed_option_jax(
    env,
    timestep,
):
    """
    JAX-compatible failed option.

    Invalid or currently inapplicable option selections
    consume exactly one primitive environment step.

    This prevents:
        - zero-duration options
        - Python exceptions
        - PPO repeatedly selecting impossible actions
          without advancing environment time

    Returns:
        timestep
        primitive_steps = 1
        valid = False
    """

    new_timestep = env.step(
        timestep,
        jnp.asarray(
            DONE,
            dtype=jnp.int32,
        ),
    )

    return (
        new_timestep,
        jnp.asarray(
            1,
            dtype=jnp.int32,
        ),
        jnp.asarray(
            False
        ),
    )


# ============================================================
# State validity utilities
# ============================================================

def has_key_jax(
    timestep,
):
    """
    True when the player is carrying an item/key.
    """

    player = timestep.state.get_player(
        idx=0
    )

    return (
        player.pocket != -1
    )


def door_is_open_jax(
    timestep,
):
    """
    True when the DoorKey door is open.
    """

    doors = timestep.state.get_doors()

    return doors.open[0]


def key_is_available_jax(
    timestep,
):
    """
    True when the key is still present in the environment.

    Navix moves the key to [0, -1] after pickup.
    """

    keys = timestep.state.get_keys()

    key_position = keys.position[0]

    removed_position = jnp.asarray(
        [0, -1],
        dtype=key_position.dtype,
    )

    removed = jnp.all(
        key_position
        == removed_position
    )

    return jnp.logical_not(
        removed
    )


# ============================================================
# Guarded option branches
# ============================================================

def _go_to_key_branch(
    env,
    timestep,
):
    """
    Execute GO_TO_KEY only while the key still exists.
    """

    applicable = key_is_available_jax(
        timestep
    )

    return jax.lax.cond(
        applicable,
        lambda ts: go_to_key_jax(
            env,
            ts,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
        ),
        timestep,
    )


def _pickup_key_branch(
    env,
    timestep,
):
    """
    Attempt PICKUP_KEY.

    pickup_key_jax itself determines whether the
    pickup actually succeeded.
    """

    return pickup_key_jax(
        env,
        timestep,
    )


def _go_to_door_branch(
    env,
    timestep,
):
    """
    GO_TO_DOOR is only applicable after collecting
    the key and before opening the door.
    """

    holding_key = has_key_jax(
        timestep
    )

    door_open = door_is_open_jax(
        timestep
    )

    applicable = jnp.logical_and(
        holding_key,
        jnp.logical_not(
            door_open
        ),
    )

    return jax.lax.cond(
        applicable,
        lambda ts: go_to_door_jax(
            env,
            ts,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
        ),
        timestep,
    )


def _open_door_branch(
    env,
    timestep,
):
    """
    Attempt OPEN_DOOR.

    open_door_jax checks whether TOGGLE actually
    changes the door from closed to open.
    """

    holding_key = has_key_jax(
        timestep
    )

    door_open = door_is_open_jax(
        timestep
    )

    applicable = jnp.logical_and(
        holding_key,
        jnp.logical_not(
            door_open
        ),
    )

    return jax.lax.cond(
        applicable,
        lambda ts: open_door_jax(
            env,
            ts,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
        ),
        timestep,
    )


def _go_to_goal_branch(
    env,
    timestep,
):
    """
    GO_TO_GOAL is only allowed once the door is open.
    """

    applicable = door_is_open_jax(
        timestep
    )

    return jax.lax.cond(
        applicable,
        lambda ts: go_to_goal_jax(
            env,
            ts,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
        ),
        timestep,
    )


# ============================================================
# PPO-facing JAX option dispatcher
# ============================================================

def execute_option_jax(
    env,
    timestep,
    option_id,
):
    """
    Execute one high-level option selected by an integer ID.

    Option IDs:
        0 = GO_TO_KEY
        1 = PICKUP_KEY
        2 = GO_TO_DOOR
        3 = OPEN_DOOR
        4 = GO_TO_GOAL

    This function is designed to be compatible with
    jax.jit and later jax.vmap / jax.lax.scan.

    Returns
    -------
    timestep:
        Environment state after the option terminates.

    primitive_steps:
        Number of primitive environment actions consumed
        by this option.

    valid:
        True if the selected option successfully executed
        in the current state.
    """

    option_id = jnp.asarray(
        option_id,
        dtype=jnp.int32,
    )

    valid_option_id = jnp.logical_and(
        option_id >= 0,
        option_id < NUM_OPTIONS,
    )

    branches = (
        lambda ts: _go_to_key_branch(
            env,
            ts,
        ),
        lambda ts: _pickup_key_branch(
            env,
            ts,
        ),
        lambda ts: _go_to_door_branch(
            env,
            ts,
        ),
        lambda ts: _open_door_branch(
            env,
            ts,
        ),
        lambda ts: _go_to_goal_branch(
            env,
            ts,
        ),
    )

    return jax.lax.cond(
        valid_option_id,
        lambda ts: jax.lax.switch(
            option_id,
            branches,
            ts,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
        ),
        timestep,
    )