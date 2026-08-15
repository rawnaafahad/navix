import jax
import jax.numpy as jnp

from jax_option_executor import (
    update_discounted_reward,
    option_bootstrap_discount,
)


# ============================================================
# Configuration
# ============================================================

GAMMA = 0.99


print(
    "Testing option-level reward accounting...\n"
)


# ============================================================
# TEST 1
# One-step option reward
# ============================================================

reward_fn = jax.jit(
    update_discounted_reward
)


accumulated_reward = jnp.asarray(
    0.0,
    dtype=jnp.float32,
)


accumulated_reward = reward_fn(
    accumulated_reward,
    jnp.asarray(
        1.0,
        dtype=jnp.float32,
    ),
    jnp.asarray(
        0,
        dtype=jnp.int32,
    ),
    GAMMA,
)


print(
    "TEST 1: one-step reward"
)

print(
    "Calculated reward:",
    accumulated_reward
)

print(
    "Expected reward:",
    1.0
)


assert jnp.isclose(
    accumulated_reward,
    1.0,
    atol=1e-6,
)


print(
    "TEST 1 PASS\n"
)


# ============================================================
# Helper:
# accumulate a sequence of primitive rewards
# ============================================================

def accumulate_rewards(
    primitive_rewards,
):
    """
    Accumulate primitive rewards using:

        R_option =
            r_0
            + gamma * r_1
            + gamma^2 * r_2
            + ...

    Uses jax.lax.scan so the calculation is JAX-compatible.
    """

    primitive_rewards = jnp.asarray(
        primitive_rewards,
        dtype=jnp.float32,
    )

    step_indices = jnp.arange(
        primitive_rewards.shape[0],
        dtype=jnp.int32,
    )

    def scan_body(
        accumulated_reward,
        inputs,
    ):

        step_index, reward = inputs

        new_accumulated_reward = (
            update_discounted_reward(
                accumulated_reward,
                reward,
                step_index,
                GAMMA,
            )
        )

        return (
            new_accumulated_reward,
            None,
        )

    final_reward, _ = jax.lax.scan(
        scan_body,
        jnp.asarray(
            0.0,
            dtype=jnp.float32,
        ),
        (
            step_indices,
            primitive_rewards,
        ),
    )

    return final_reward


accumulate_rewards_jit = jax.jit(
    accumulate_rewards
)


# ============================================================
# TEST 2
# Delayed terminal reward
#
# Primitive rewards:
#
# [0, 0, 0, 0, 1]
#
# Expected:
#
# gamma^4
# ============================================================

primitive_rewards = jnp.asarray(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=jnp.float32,
)


option_reward = accumulate_rewards_jit(
    primitive_rewards
)


expected_reward = (
    GAMMA ** 4
)


print(
    "TEST 2: delayed reward"
)

print(
    "Primitive rewards:",
    primitive_rewards
)

print(
    "Option duration:",
    primitive_rewards.shape[0]
)

print(
    "Calculated discounted option reward:",
    option_reward
)

print(
    "Expected reward:",
    expected_reward
)


assert jnp.isclose(
    option_reward,
    expected_reward,
    atol=1e-6,
)


print(
    "TEST 2 PASS\n"
)


# ============================================================
# TEST 3
# Multiple non-zero primitive rewards
#
# rewards:
#
# [1, 2, 3]
#
# Expected:
#
# 1
# + gamma * 2
# + gamma^2 * 3
# ============================================================

primitive_rewards = jnp.asarray(
    [
        1.0,
        2.0,
        3.0,
    ],
    dtype=jnp.float32,
)


option_reward = accumulate_rewards_jit(
    primitive_rewards
)


expected_reward = (
    1.0
    + GAMMA * 2.0
    + (GAMMA ** 2) * 3.0
)


print(
    "TEST 3: multiple non-zero rewards"
)

print(
    "Primitive rewards:",
    primitive_rewards
)

print(
    "Calculated discounted option reward:",
    option_reward
)

print(
    "Expected reward:",
    expected_reward
)


assert jnp.isclose(
    option_reward,
    expected_reward,
    atol=1e-6,
)


print(
    "TEST 3 PASS\n"
)


# ============================================================
# TEST 4
# Option bootstrap discount
#
# For an option lasting k primitive steps:
#
# bootstrap discount = gamma^k
# ============================================================

bootstrap_discount_fn = jax.jit(
    option_bootstrap_discount
)


duration = jnp.asarray(
    5,
    dtype=jnp.int32,
)


bootstrap_discount = bootstrap_discount_fn(
    duration,
    GAMMA,
)


expected_discount = (
    GAMMA ** 5
)


print(
    "TEST 4: bootstrap discount"
)

print(
    "Duration:",
    duration
)

print(
    "Calculated gamma^k:",
    bootstrap_discount
)

print(
    "Expected gamma^k:",
    expected_discount
)


assert jnp.isclose(
    bootstrap_discount,
    expected_discount,
    atol=1e-6,
)


print(
    "TEST 4 PASS\n"
)


# ============================================================
# TEST 5
# Compare one-step and multi-step discounting
# ============================================================

one_step_discount = bootstrap_discount_fn(
    jnp.asarray(
        1,
        dtype=jnp.int32,
    ),
    GAMMA,
)


five_step_discount = bootstrap_discount_fn(
    jnp.asarray(
        5,
        dtype=jnp.int32,
    ),
    GAMMA,
)


print(
    "TEST 5: duration-aware discount comparison"
)

print(
    "gamma^1:",
    one_step_discount
)

print(
    "gamma^5:",
    five_step_discount
)


assert jnp.isclose(
    one_step_discount,
    GAMMA,
    atol=1e-6,
)

assert five_step_discount < one_step_discount


print(
    "TEST 5 PASS\n"
)


# ============================================================
# Final result
# ============================================================

print(
    "================================"
)

print(
    "OPTION REWARD ACCOUNTING PASSED"
)

print(
    "================================"
)

print(
    "\nFor an option lasting k primitive steps:"
)

print(
    "Option reward = "
    "sum(gamma^i * r_i)"
)

print(
    "Bootstrap discount = gamma^k"
)