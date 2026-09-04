"""
evaluate_discovered_policy.py

Held-out evaluation harness for a trained discovered-option PPO
checkpoint. Runs deterministic episodes (greedy high-level policy,
stochastic low-level option execution -- the latter is inherent to how
the stuck-loop fix works) on a dungeon seed pool DISJOINT from every
training pool used this session ([0,10) and [0,100) so far).

Reports mean/SD/95% CI for success rate (Wilson interval, appropriate
for a proportion) and for episode return / primitive steps per episode
(normal-approximation CI).

Usage:
    python3 evaluate_discovered_policy.py --checkpoint PATH --policy-dir DIR
        [--eval-seeds-start 1000] [--eval-seeds-end 1050]
"""

import argparse
import pickle
import math
import numpy as np
import jax
import jax.numpy as jnp

from discovered_option_env import DiscoveredOptionEnv, DiscoveredOptionEnvConfig, NUM_GLYPHS
from option_ppo_models import OptionActorCritic


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)) / denom
    return p_hat, max(0.0, center - half_width), min(1.0, center + half_width)


def normal_ci(values, z=1.96):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = values.mean()
    sd = values.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return mean, sd, mean - z * se, mean + z * se


def run_episode(env, network, params, seed, jax_key, max_primitive_steps):
    obs, info = env.reset(seed=seed)
    episode_return = 0.0
    raw_reward = 0.0
    steps_before = env._primitive_steps_used

    while True:
        crop = jnp.asarray(obs["crop"], dtype=jnp.int32)[None, :, :]
        feat = jnp.asarray(obs["features"], dtype=jnp.float32)[None, :]

        pi, _ = network.apply(params, crop, feat)
        action = int(pi.mode()[0])  # greedy high-level action (distrax Categorical.mode())

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        raw_reward = reward

        if terminated or truncated:
            break

    primitive_steps = env._primitive_steps_used - steps_before
    success = raw_reward > 0.0
    return success, episode_return, primitive_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--policy-dir", type=str, default="bc_policies_cropped_v2")
    parser.add_argument("--eval-seeds-start", type=int, default=1000)
    parser.add_argument("--eval-seeds-end", type=int, default=1050)
    parser.add_argument("--max-primitive-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)

    num_options = ckpt["num_options"]
    network = OptionActorCritic(num_options=num_options, num_glyphs=NUM_GLYPHS)

    env_config = DiscoveredOptionEnvConfig(
        max_primitive_steps=args.max_primitive_steps,
        policy_dir=args.policy_dir,
        seed=args.seed,
    )
    env = DiscoveredOptionEnv(env_config)

    eval_seeds = list(range(args.eval_seeds_start, args.eval_seeds_end))
    print(f"Evaluating on {len(eval_seeds)} held-out dungeons "
          f"[{args.eval_seeds_start}, {args.eval_seeds_end})")
    print(f"Checkpoint: {args.checkpoint} | policy_dir: {args.policy_dir}\n")

    jax_key = jax.random.PRNGKey(args.seed)
    successes, returns, prim_steps = [], [], []

    for i, seed in enumerate(eval_seeds):
        success, ep_return, ep_prim_steps = run_episode(
            env, network, ckpt["params"], seed, jax_key, args.max_primitive_steps
        )
        successes.append(success)
        returns.append(ep_return)
        prim_steps.append(ep_prim_steps)
        print(f"  seed {seed:6d}: success={success} return={ep_return:+8.3f} "
              f"primitive_steps={ep_prim_steps:5d}")

    env.close()

    n = len(eval_seeds)
    n_success = sum(successes)
    p_hat, ci_lo, ci_hi = wilson_interval(n_success, n)
    ret_mean, ret_sd, ret_lo, ret_hi = normal_ci(returns)
    steps_mean, steps_sd, steps_lo, steps_hi = normal_ci(prim_steps)

    print(f"\n=== Held-out evaluation summary (n={n} episodes) ===")
    print(f"Success rate: {p_hat:.3f}  (95% Wilson CI: [{ci_lo:.3f}, {ci_hi:.3f}])  "
          f"[{n_success}/{n} episodes]")
    print(f"Episode return: mean={ret_mean:.3f} SD={ret_sd:.3f}  "
          f"(95% CI: [{ret_lo:.3f}, {ret_hi:.3f}])")
    print(f"Primitive steps/episode: mean={steps_mean:.1f} SD={steps_sd:.1f}  "
          f"(95% CI: [{steps_lo:.1f}, {steps_hi:.1f}])")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "success", "return", "primitive_steps"])
            for seed, s, r, p in zip(eval_seeds, successes, returns, prim_steps):
                writer.writerow([seed, int(s), r, p])
        print(f"\nPer-episode results saved to {args.out_csv}")


if __name__ == "__main__":
    main()
