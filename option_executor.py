import jax.numpy as jnp

from test_options import (
    go_to_key,
    pickup_key,
    go_to_door,
    open_door,
    go_to_goal,
)


# ============================================================
# Option IDs
# ============================================================

GO_TO_KEY = 0
PICKUP_KEY = 1
GO_TO_DOOR = 2
OPEN_DOOR = 3
GO_TO_GOAL = 4

NUM_OPTIONS = 5


OPTION_NAMES = {
    GO_TO_KEY: "GO_TO_KEY",
    PICKUP_KEY: "PICKUP_KEY",
    GO_TO_DOOR: "GO_TO_DOOR",
    OPEN_DOOR: "OPEN_DOOR",
    GO_TO_GOAL: "GO_TO_GOAL",
}


# Primitive action index used as a harmless failed-option step.
# Navix's action space has 7 primitive actions, so this keeps
# environment time moving even when PPO selects an invalid option.
DONE_ACTION = 6


def failed_option_step(env, timestep):
    """
    Consume one primitive environment step when an option
    cannot validly execute.

    This is important because PPO will initially choose
    options in the wrong order. We do not want those choices
    to crash training or consume zero environment time.
    """

    timestep = env.step(
        timestep,
        jnp.asarray(DONE_ACTION)
    )

    return timestep, 1


def has_key(timestep):
    """
    Return True when the player currently holds a key.
    """

    player = timestep.state.get_player(idx=0)

    return int(player.pocket) != -1


def door_is_open(timestep):
    """
    Return True when the DoorKey door is open.
    """

    doors = timestep.state.get_doors()

    return bool(
        doors.open[0]
    )


def key_is_available(timestep):
    """
    Return True when the key still exists in the environment.
    """

    keys = timestep.state.get_keys()

    key_position = keys.position[0]

    # Navix moves picked-up objects off-grid.
    return not (
        int(key_position[0]) == 0
        and int(key_position[1]) == -1
    )


def execute_option(
    env,
    timestep,
    option_id,
):
    """
    Execute one high-level option.

    Returns
    -------
    timestep:
        State after the option finishes.

    duration:
        Number of primitive environment actions used.

    valid:
        Whether the option was applicable in the current state.
    """

    option_id = int(option_id)

    # --------------------------------------------------------
    # OPTION 0: GO_TO_KEY
    # --------------------------------------------------------

    if option_id == GO_TO_KEY:

        # No reason to navigate to a key already collected.
        if not key_is_available(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        timestep, duration = go_to_key(
            env,
            timestep
        )

        return timestep, duration, True

    # --------------------------------------------------------
    # OPTION 1: PICKUP_KEY
    # --------------------------------------------------------

    if option_id == PICKUP_KEY:

        # Already carrying the key.
        if has_key(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        # Try pickup.
        pocket_before = int(
            timestep.state.get_player(idx=0).pocket
        )

        timestep, duration = pickup_key(
            env,
            timestep
        )

        pocket_after = int(
            timestep.state.get_player(idx=0).pocket
        )

        valid = (
            pocket_before == -1
            and pocket_after != -1
        )

        return timestep, duration, valid

    # --------------------------------------------------------
    # OPTION 2: GO_TO_DOOR
    # --------------------------------------------------------

    if option_id == GO_TO_DOOR:

        # Door navigation only becomes useful once key is held.
        if not has_key(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        if door_is_open(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        timestep, duration = go_to_door(
            env,
            timestep
        )

        return timestep, duration, True

    # --------------------------------------------------------
    # OPTION 3: OPEN_DOOR
    # --------------------------------------------------------

    if option_id == OPEN_DOOR:

        if not has_key(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        if door_is_open(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        was_open = door_is_open(
            timestep
        )

        timestep, duration = open_door(
            env,
            timestep
        )

        is_open = door_is_open(
            timestep
        )

        valid = (
            not was_open
            and is_open
        )

        return timestep, duration, valid

    # --------------------------------------------------------
    # OPTION 4: GO_TO_GOAL
    # --------------------------------------------------------

    if option_id == GO_TO_GOAL:

        if not door_is_open(timestep):
            timestep, duration = failed_option_step(
                env,
                timestep
            )

            return timestep, duration, False

        timestep, duration = go_to_goal(
            env,
            timestep
        )

        return timestep, duration, True

    # --------------------------------------------------------
    # Invalid option ID
    # --------------------------------------------------------

    timestep, duration = failed_option_step(
        env,
        timestep
    )

    return timestep, duration, False