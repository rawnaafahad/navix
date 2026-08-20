import jax.numpy as jnp

from navix import rewards


# ============================================================
# Reward configuration
# ============================================================

KEY_REWARD = 0.25
DOOR_REWARD = 0.25
GOAL_REWARD = 0.50


# ============================================================
# Sparse reward
#
# Original DoorKey reward:
#     +1 only when the goal is reached
# ============================================================

def sparse_doorkey_reward(
    prev_state,
    action,
    new_state,
):
    return rewards.on_goal_reached(
        prev_state,
        action,
        new_state,
    )


# ============================================================
# Key-pickup event
# ============================================================

def on_key_picked_up(
    prev_state,
    action,
    new_state,
):
    """
    Reward exactly once when the player goes from carrying
    nothing to carrying an item/key.
    """

    player_before = (
        prev_state.get_player(
            idx=0
        )
    )

    player_after = (
        new_state.get_player(
            idx=0
        )
    )


    picked_up_key = (
        jnp.logical_and(
            player_before.pocket == -1,
            player_after.pocket != -1,
        )
    )


    return (
        jnp.asarray(
            KEY_REWARD,
            dtype=jnp.float32,
        )
        * picked_up_key.astype(
            jnp.float32
        )
    )


# ============================================================
# Door-opening event
# ============================================================

def on_door_opened(
    prev_state,
    action,
    new_state,
):
    """
    Reward exactly once when the DoorKey door changes from
    closed to open.
    """

    doors_before = (
        prev_state.get_doors()
    )

    doors_after = (
        new_state.get_doors()
    )


    opened_door = (
        jnp.logical_and(
            jnp.logical_not(
                doors_before.open[0]
            ),
            doors_after.open[0],
        )
    )


    return (
        jnp.asarray(
            DOOR_REWARD,
            dtype=jnp.float32,
        )
        * opened_door.astype(
            jnp.float32
        )
    )


# ============================================================
# Goal event
# ============================================================

def on_goal_reached_shaped(
    prev_state,
    action,
    new_state,
):
    """
    Terminal goal reward for the shaped condition.

    The reward is reduced to 0.50 so that:

        key + door + goal
        0.25 + 0.25 + 0.50 = 1.00

    A successful trajectory therefore receives the same total
    event reward as the sparse condition; the difference is
    when the credit signal arrives.
    """

    reached_goal = (
        rewards.on_goal_reached(
            prev_state,
            action,
            new_state,
        )
    )


    return (
        jnp.asarray(
            GOAL_REWARD,
            dtype=jnp.float32,
        )
        * reached_goal
    )


# ============================================================
# Shaped reward
# ============================================================

def shaped_doorkey_reward(
    prev_state,
    action,
    new_state,
):
    """
    Fixed-total reward shaping:

        +0.25  key pickup
        +0.25  door opening
        +0.50  goal reached

    Maximum event reward along the intended successful
    trajectory = 1.00.
    """

    key_reward = (
        on_key_picked_up(
            prev_state,
            action,
            new_state,
        )
    )


    door_reward = (
        on_door_opened(
            prev_state,
            action,
            new_state,
        )
    )


    goal_reward = (
        on_goal_reached_shaped(
            prev_state,
            action,
            new_state,
        )
    )


    return (
        key_reward
        + door_reward
        + goal_reward
    )