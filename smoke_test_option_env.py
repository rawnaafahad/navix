"""Quick sanity check for OptionEnv + OptionActorCritic, before training.

Run this first:

    python smoke_test_option_env.py

It checks:
  1. OptionEnv resets and steps without crashing, for all 3 options.
  2. Observation shapes/dtypes match what the network expects.
  3. blstats-derived features look sane (not NaN, in roughly expected
     ranges) -- this is the main thing to eyeball, since the blstats
     index assumptions in option_env.py have not been verified against
     your installed NLE build in this session.
  4. The network's forward pass runs on a real observation and produces
     a valid action distribution + scalar value.

This does NOT train anything and does not require jax.jit to work
correctly (though it uses jax under the hood via the network call).
"""

import jax
import jax.numpy as jnp
import numpy as np

from option_env import OptionEnv, OptionEnvConfig, NUM_OPTIONS, NUM_GLYPHS, CROP_SIZE
from option_ppo_models import OptionActorCritic


def main():
    print("=" * 70)
    print("OPTION ENV SMOKE TEST")
    print("=" * 70)

    config = OptionEnvConfig(train_seeds=range(0, 10))
    env = OptionEnv(config)

    obs, info = env.reset(seed=0)

    print("\nInitial observation:")
    print("  crop shape:", obs["crop"].shape, "dtype:", obs["crop"].dtype)
    print("  crop value range:", obs["crop"].min(), "-", obs["crop"].max())
    print("  features shape:", obs["features"].shape)
    print("  features:", obs["features"])

    assert obs["crop"].shape == (CROP_SIZE, CROP_SIZE), "crop shape mismatch"
    assert not np.isnan(obs["features"]).any(), "NaN in feature vector"

    print("\nStepping through all 3 options once each (from the initial state):")
    for action in range(NUM_OPTIONS):
        # Re-reset each time so each option is tried from the same
        # starting state, for a clean comparison.
        obs, _ = env.reset(seed=0)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        print(
            f"  action={action} "
            f"reward={reward:+.3f} "
            f"terminated={terminated} "
            f"truncated={truncated} "
            f"info={step_info}"
        )

    print("\nRunning 20 random high-level steps from a fresh episode:")
    obs, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)

    for i in range(20):
        action = int(rng.integers(0, NUM_OPTIONS))
        obs, reward, terminated, truncated, step_info = env.step(action)
        print(
            f"  step {i:2d} action={action} reward={reward:+.3f} "
            f"terminated={terminated} truncated={truncated} "
            f"option_steps={step_info['option_steps']} "
            f"primitive_steps_used={step_info['primitive_steps_used']}"
        )
        if terminated or truncated:
            print("  episode ended, resetting")
            obs, _ = env.reset()

    env.close()

    print("\n" + "=" * 70)
    print("NETWORK FORWARD-PASS SMOKE TEST")
    print("=" * 70)

    network = OptionActorCritic(num_options=NUM_OPTIONS, num_glyphs=NUM_GLYPHS)
    rng = jax.random.PRNGKey(0)

    dummy_crop = jnp.zeros((1, CROP_SIZE, CROP_SIZE), dtype=jnp.int32)
    dummy_features = jnp.zeros((1, 9), dtype=jnp.float32)
    params = network.init(rng, dummy_crop, dummy_features)

    crop_batch = jnp.asarray(obs["crop"][None], dtype=jnp.int32)
    feature_batch = jnp.asarray(obs["features"][None], dtype=jnp.float32)

    pi, value = network.apply(params, crop_batch, feature_batch)
    print("  policy logits shape:", pi.logits.shape)
    print("  policy probs:", jax.nn.softmax(pi.logits))
    print("  value:", value)

    action = pi.sample(seed=jax.random.PRNGKey(1))
    print("  sampled action:", action)

    print("\n*** SMOKE TEST PASSED (no crashes) ***")
    print(
        "Now eyeball the feature vector values above -- if hp/max_hp, "
        "depth, or hunger look obviously wrong (e.g. always 0, or huge "
        "numbers), the blstats indices in option_env.py need adjusting "
        "for your NLE version before training."
    )


if __name__ == "__main__":
    main()