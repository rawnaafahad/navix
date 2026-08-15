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



DEFAULT_GAMMA = 0.99


# ============================================================
# Option reward utilities
# ============================================================

def update_discounted_reward(
    accumulated_reward,
    primitive_reward,
    primitive_step_index,
    gamma,
):
    """
    Add one primitive reward to an option-level discounted
    reward accumulator.

    For primitive step i:

        R_option += gamma**i * r_i

    This function is JAX-safe and can be used inside
    jax.lax.scan / jax.lax.while_loop.
    """

    accumulated_reward = jnp.asarray(
        accumulated_reward,
        dtype=jnp.float32,
    )

    primitive_reward = jnp.asarray(
        primitive_reward,
        dtype=jnp.float32,
    )

    primitive_step_index = jnp.asarray(
        primitive_step_index,
        dtype=jnp.int32,
    )

    gamma = jnp.asarray(
        gamma,
        dtype=jnp.float32,
    )

    discount = jnp.power(
        gamma,
        primitive_step_index,
    )

    return (
        accumulated_reward
        + discount * primitive_reward
    )


def option_bootstrap_discount(
    duration,
    gamma,
):
    """
    Return gamma**k for an option lasting k primitive steps.

    This is the discount applied to the next-state value in
    the semi-Markov (option-level) Bellman target.
    """

    duration = jnp.asarray(
        duration,
        dtype=jnp.int32,
    )

    gamma = jnp.asarray(
        gamma,
        dtype=jnp.float32,
    )

    return jnp.power(
        gamma,
        duration,
    )


def zero_option_reward():
    """
    Return a correctly typed zero reward accumulator.
    """

    return jnp.asarray(
        0.0,
        dtype=jnp.float32,
    )

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
    gamma=DEFAULT_GAMMA,
):
    """
    JAX-compatible GO_TO_KEY option.

    Navigate until adjacent to the key, then face it.

    Returns:
        timestep, primitive_steps, option_reward, valid
    """

    key_position = timestep.state.get_keys().position[0]

    initial_carry = (
        timestep,
        jnp.asarray(0, dtype=jnp.int32),
        zero_option_reward(),
    )

    def navigation_condition(carry):
        current_timestep, primitive_steps, _ = carry
        player = current_timestep.state.get_player(idx=0)
        distance = manhattan_distance(player.position, key_position)

        return jnp.logical_and(
            jnp.logical_and(
                distance > 1,
                primitive_steps < max_steps,
            ),
            jnp.logical_not(current_timestep.is_done()),
        )

    def navigation_body(carry):
        current_timestep, primitive_steps, option_reward = carry
        player = current_timestep.state.get_player(idx=0)

        desired_direction = direction_to_target(
            player.position,
            key_position,
        )
        action = action_towards_direction(
            player.direction,
            desired_direction,
        )
        new_timestep = env.step(current_timestep, action)

        new_option_reward = update_discounted_reward(
            option_reward,
            new_timestep.reward,
            primitive_steps,
            gamma,
        )

        return (
            new_timestep,
            primitive_steps + 1,
            new_option_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.while_loop(
        navigation_condition,
        navigation_body,
        initial_carry,
    )

    player = timestep.state.get_player(idx=0)
    desired_direction = direction_to_adjacent_target(
        player.position,
        key_position,
    )

    def alignment_condition(carry):
        current_timestep, primitive_steps, _ = carry
        current_player = current_timestep.state.get_player(idx=0)

        return jnp.logical_and(
            jnp.logical_and(
                current_player.direction != desired_direction,
                primitive_steps < max_steps,
            ),
            jnp.logical_not(current_timestep.is_done()),
        )

    def alignment_body(carry):
        current_timestep, primitive_steps, option_reward = carry
        new_timestep = env.step(
            current_timestep,
            jnp.asarray(ROTATE_RIGHT, dtype=jnp.int32),
        )

        new_option_reward = update_discounted_reward(
            option_reward,
            new_timestep.reward,
            primitive_steps,
            gamma,
        )

        return (
            new_timestep,
            primitive_steps + 1,
            new_option_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.while_loop(
        alignment_condition,
        alignment_body,
        (timestep, primitive_steps, option_reward),
    )

    player = timestep.state.get_player(idx=0)
    final_distance = manhattan_distance(
        player.position,
        key_position,
    )
    final_direction = direction_to_adjacent_target(
        player.position,
        key_position,
    )

    valid = jnp.logical_and(
        jnp.logical_and(
            final_distance == 1,
            player.direction == final_direction,
        ),
        primitive_steps <= max_steps,
    )

    return (
        timestep,
        primitive_steps,
        option_reward,
        valid,
    )


# ============================================================
# JAX OPTION 2: PICKUP_KEY
# ============================================================

def pickup_key_jax(
    env,
    timestep,
    gamma=DEFAULT_GAMMA,
):
    """
    JAX-compatible PICKUP_KEY option.

    Returns:
        timestep, primitive_steps, option_reward, valid
    """

    player_before = timestep.state.get_player(idx=0)
    pocket_before = player_before.pocket

    new_timestep = env.step(
        timestep,
        jnp.asarray(PICKUP, dtype=jnp.int32),
    )

    player_after = new_timestep.state.get_player(idx=0)
    pocket_after = player_after.pocket

    valid = jnp.logical_and(
        pocket_before == -1,
        pocket_after != -1,
    )

    primitive_steps = jnp.asarray(1, dtype=jnp.int32)

    option_reward = update_discounted_reward(
        zero_option_reward(),
        new_timestep.reward,
        jnp.asarray(0, dtype=jnp.int32),
        gamma,
    )

    return (
        new_timestep,
        primitive_steps,
        option_reward,
        valid,
    )


# ============================================================
# JAX OPTION 3: GO_TO_DOOR
# ============================================================

def go_to_door_jax(
    env,
    timestep,
    max_steps=50,
    gamma=DEFAULT_GAMMA,
):
    """
    JAX-compatible GO_TO_DOOR option.

    The agent navigates to the tile immediately left of the
    door and finishes facing east toward it.

    Returns:
        timestep, primitive_steps, option_reward, valid
    """

    door_position = timestep.state.get_doors().position[0]

    approach_position = jnp.asarray(
        [
            door_position[0],
            door_position[1] - 1,
        ],
        dtype=jnp.int32,
    )

    initial_carry = (
        timestep,
        jnp.asarray(0, dtype=jnp.int32),
        zero_option_reward(),
    )

    def navigation_condition(carry):
        current_timestep, primitive_steps, _ = carry
        player = current_timestep.state.get_player(idx=0)
        distance = manhattan_distance(
            player.position,
            approach_position,
        )

        return jnp.logical_and(
            jnp.logical_and(
                distance > 0,
                primitive_steps < max_steps,
            ),
            jnp.logical_not(current_timestep.is_done()),
        )

    def navigation_body(carry):
        current_timestep, primitive_steps, option_reward = carry
        player = current_timestep.state.get_player(idx=0)

        desired_direction = direction_to_target(
            player.position,
            approach_position,
        )
        action = action_towards_direction(
            player.direction,
            desired_direction,
        )
        new_timestep = env.step(current_timestep, action)

        new_option_reward = update_discounted_reward(
            option_reward,
            new_timestep.reward,
            primitive_steps,
            gamma,
        )

        return (
            new_timestep,
            primitive_steps + 1,
            new_option_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.while_loop(
        navigation_condition,
        navigation_body,
        initial_carry,
    )

    desired_direction = jnp.asarray(EAST, dtype=jnp.int32)

    def alignment_condition(carry):
        current_timestep, primitive_steps, _ = carry
        player = current_timestep.state.get_player(idx=0)

        return jnp.logical_and(
            jnp.logical_and(
                player.direction != desired_direction,
                primitive_steps < max_steps,
            ),
            jnp.logical_not(current_timestep.is_done()),
        )

    def alignment_body(carry):
        current_timestep, primitive_steps, option_reward = carry

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(ROTATE_RIGHT, dtype=jnp.int32),
        )

        new_option_reward = update_discounted_reward(
            option_reward,
            new_timestep.reward,
            primitive_steps,
            gamma,
        )

        return (
            new_timestep,
            primitive_steps + 1,
            new_option_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.while_loop(
        alignment_condition,
        alignment_body,
        (timestep, primitive_steps, option_reward),
    )

    player = timestep.state.get_player(idx=0)

    valid = jnp.logical_and(
        jnp.all(player.position == approach_position),
        player.direction == EAST,
    )

    return (
        timestep,
        primitive_steps,
        option_reward,
        valid,
    )


# ============================================================
# JAX OPTION 4: OPEN_DOOR
# ============================================================

def open_door_jax(
    env,
    timestep,
    gamma=DEFAULT_GAMMA,
):
    """
    JAX-compatible OPEN_DOOR option.

    Executes one primitive TOGGLE action.

    Returns:
        timestep, primitive_steps, option_reward, valid
    """

    doors_before = timestep.state.get_doors()
    was_open = doors_before.open[0]

    new_timestep = env.step(
        timestep,
        jnp.asarray(TOGGLE, dtype=jnp.int32),
    )

    doors_after = new_timestep.state.get_doors()
    is_open = doors_after.open[0]

    valid = jnp.logical_and(
        jnp.logical_not(was_open),
        is_open,
    )

    primitive_steps = jnp.asarray(1, dtype=jnp.int32)

    option_reward = update_discounted_reward(
        zero_option_reward(),
        new_timestep.reward,
        jnp.asarray(0, dtype=jnp.int32),
        gamma,
    )

    return (
        new_timestep,
        primitive_steps,
        option_reward,
        valid,
    )


# ============================================================
# JAX OPTION 5: GO_TO_GOAL
# ============================================================

def go_to_goal_jax(
    env,
    timestep,
    max_steps=50,
    gamma=DEFAULT_GAMMA,
):
    """
    JAX-compatible GO_TO_GOAL option.

    Assumes the door is open and the player is immediately
    left of it facing east. The option enters the doorway,
    exits into the second room, then navigates to the goal.

    Returns:
        timestep, primitive_steps, option_reward, valid
    """

    goal_position = timestep.state.get_goals().position[0]

    doors = timestep.state.get_doors()
    door_position = doors.position[0]

    primitive_steps = jnp.asarray(0, dtype=jnp.int32)
    option_reward = zero_option_reward()

    left_of_door = jnp.asarray(
        [
            door_position[0],
            door_position[1] - 1,
        ],
        dtype=jnp.int32,
    )

    player = timestep.state.get_player(idx=0)

    can_enter_door = jnp.logical_and(
        jnp.logical_and(
            doors.open[0],
            jnp.all(player.position == left_of_door),
        ),
        player.direction == EAST,
    )

    def enter_door(args):
        current_timestep, steps, accumulated_reward = args

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(FORWARD, dtype=jnp.int32),
        )

        new_reward = update_discounted_reward(
            accumulated_reward,
            new_timestep.reward,
            steps,
            gamma,
        )

        return (
            new_timestep,
            steps + 1,
            new_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.cond(
        can_enter_door,
        enter_door,
        lambda args: args,
        (timestep, primitive_steps, option_reward),
    )

    player = timestep.state.get_player(idx=0)

    can_exit_door = jnp.logical_and(
        jnp.all(player.position == door_position),
        player.direction == EAST,
    )

    def exit_door(args):
        current_timestep, steps, accumulated_reward = args

        new_timestep = env.step(
            current_timestep,
            jnp.asarray(FORWARD, dtype=jnp.int32),
        )

        new_reward = update_discounted_reward(
            accumulated_reward,
            new_timestep.reward,
            steps,
            gamma,
        )

        return (
            new_timestep,
            steps + 1,
            new_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.cond(
        can_exit_door,
        exit_door,
        lambda args: args,
        (timestep, primitive_steps, option_reward),
    )

    def navigation_condition(carry):
        current_timestep, steps, _ = carry
        player = current_timestep.state.get_player(idx=0)

        return jnp.logical_and(
            jnp.logical_and(
                manhattan_distance(
                    player.position,
                    goal_position,
                ) > 0,
                steps < max_steps,
            ),
            jnp.logical_not(current_timestep.is_done()),
        )

    def navigation_body(carry):
        current_timestep, steps, accumulated_reward = carry
        player = current_timestep.state.get_player(idx=0)

        desired_direction = direction_to_target(
            player.position,
            goal_position,
        )
        action = action_towards_direction(
            player.direction,
            desired_direction,
        )
        new_timestep = env.step(current_timestep, action)

        new_reward = update_discounted_reward(
            accumulated_reward,
            new_timestep.reward,
            steps,
            gamma,
        )

        return (
            new_timestep,
            steps + 1,
            new_reward,
        )

    timestep, primitive_steps, option_reward = jax.lax.while_loop(
        navigation_condition,
        navigation_body,
        (timestep, primitive_steps, option_reward),
    )

    player = timestep.state.get_player(idx=0)

    valid = jnp.logical_and(
        jnp.logical_and(
            jnp.all(player.position == goal_position),
            timestep.is_done(),
        ),
        jnp.logical_and(
            timestep.reward > 0.0,
            primitive_steps <= max_steps,
        ),
    )

    return (
        timestep,
        primitive_steps,
        option_reward,
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
    gamma=DEFAULT_GAMMA,
):
    """
    JAX-compatible failed option.

    Invalid or currently inapplicable option selections consume
    exactly one primitive environment step.

    Returns:
        timestep, primitive_steps, option_reward, valid
    """

    new_timestep = env.step(
        timestep,
        jnp.asarray(DONE, dtype=jnp.int32),
    )

    primitive_steps = jnp.asarray(1, dtype=jnp.int32)

    option_reward = update_discounted_reward(
        zero_option_reward(),
        new_timestep.reward,
        jnp.asarray(0, dtype=jnp.int32),
        gamma,
    )

    return (
        new_timestep,
        primitive_steps,
        option_reward,
        jnp.asarray(False),
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
    gamma=DEFAULT_GAMMA,
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
            gamma=gamma,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
            gamma=gamma,
        ),
        timestep,
    )


def _pickup_key_branch(
    env,
    timestep,
    gamma=DEFAULT_GAMMA,
):
    """
    Attempt PICKUP_KEY.

    pickup_key_jax itself determines whether the
    pickup actually succeeded.
    """

    return pickup_key_jax(
        env,
        timestep,
        gamma=gamma,
    )


def _go_to_door_branch(
    env,
    timestep,
    gamma=DEFAULT_GAMMA,
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
            gamma=gamma,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
            gamma=gamma,
        ),
        timestep,
    )


def _open_door_branch(
    env,
    timestep,
    gamma=DEFAULT_GAMMA,
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
            gamma=gamma,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
            gamma=gamma,
        ),
        timestep,
    )


def _go_to_goal_branch(
    env,
    timestep,
    gamma=DEFAULT_GAMMA,
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
            gamma=gamma,
        ),
        lambda ts: failed_option_jax(
            env,
            ts,
            gamma=gamma,
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
    gamma=DEFAULT_GAMMA,
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

    option_reward:
        Discounted reward accumulated across the primitive
        transitions executed by the option.

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
            gamma=gamma,
        ),
        lambda ts: _pickup_key_branch(
            env,
            ts,
            gamma=gamma,
        ),
        lambda ts: _go_to_door_branch(
            env,
            ts,
            gamma=gamma,
        ),
        lambda ts: _open_door_branch(
            env,
            ts,
            gamma=gamma,
        ),
        lambda ts: _go_to_goal_branch(
            env,
            ts,
            gamma=gamma,
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
            gamma=gamma,
        ),
        timestep,
    )