"""
Held-out evaluation with a STOCHASTIC (sampled) high-level policy, as a
fairer companion to evaluate_policy.py's deterministic (mode) evaluation.

Motivation: for the hand-coded-options track, the high-level policy was
found (via debug_hand_coded_action.py) to place most of its probability
mass on a single action whose low-level option perpetually no-ops (stairs
location never discovered because option_explore is never selected under
greedy decoding), even though option_explore retains nonzero probability
(~3%) and *was* selected often enough during training (which samples) to
produce real progress (mean_return / success_rate improving over training).
Deterministic (argmax) evaluation suppresses this low-probability-but-load-
bearing action entirely. This script re-evaluates using the same sampling
policy used during training, to check whether that recovers the training-
time success rate.

Produces the same CSV columns and summary format as evaluate_policy.py so
it is a drop-in input to plot_ladder_results.py (use a distinct --out-csv
name, e.g. eval_handcoded_100seed_seed0_stochastic.csv, and don't overwrite
the deterministic-eval CSVs -- both are worth keeping and reporting).

Usage:
    python3 evaluate_policy_stochastic.py --track hand_coded \\
        --checkpoint checkpoints_handcoded_100seed_seed0.pkl \\
        --eval-seeds-start 1000 --eval-seeds-end 1050 \\
        --out-csv eval_handcoded_100seed_seed0_stochastic.csv

Also supports --n-repeats > 1 to run each held-out seed multiple times
with different sampling noise and average, which reduces the extra
variance sampling introduces relative to deterministic evaluation --
useful if a single-pass stochastic estimate looks noisy. Default is 1
(one episode per held-out seed, matching evaluate_policy.py's protocol
exactly except for the sampling vs. mode difference).
"""
import argparse
import csv
import math
import pickle

import jax
import jax.numpy as jnp

from option_ppo_models import OptionActorCritic


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return 0.0, (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, (max(0.0, center - margin), min(1.0, center + margin))


def normal_ci(values, z=1.96):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, (0.0, 0.0)
    mean = sum(values) / n
    if n > 1:
        sd = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    else:
        sd = 0.0
    se = sd / math.sqrt(n) if n > 0 else 0.0
    return mean, sd, (mean - z * se, mean + z * se)


def build_env(track, policy_dir, max_primitive_steps):
    if track == "baseline":
        from baseline_env import BaselineEnv, BaselineEnvConfig
        env = BaselineEnv(BaselineEnvConfig(max_primitive_steps=max_primitive_steps))
    elif track == "hand_coded":
        from option_env import OptionEnv, OptionEnvConfig
        env = OptionEnv(OptionEnvConfig(max_primitive_steps=max_primitive_steps))
    elif track in ("discovered", "random_grounding"):
        from discovered_option_env import DiscoveredOptionEnv, DiscoveredOptionEnvConfig
        # NOTE: previously hardcoded policy_dir=None here, silently ignoring
        # the --policy-dir argument -- fixed to actually pass it through.
        # For "discovered" (incl. discovered-original), this should be the
        # grounding checkpoint dir the training run used (e.g.
        # bc_policies_cropped for discovered-original, bc_policies_cropped_v2
        # for discovered-v2). For "random_grounding", this script's handling
        # remains otherwise unverified against evaluate_policy.py's real
        # random_grounding branch (no --use-random-grounding /
        # --random-grounding-seed equivalent here) -- only rely on this path
        # for "discovered"/"discovered_original" unless that's been checked.
        env = DiscoveredOptionEnv(DiscoveredOptionEnvConfig(
            max_primitive_steps=max_primitive_steps, policy_dir=policy_dir,
        ))
    else:
        raise ValueError(f"Unknown track: {track}")
    return env


def get_num_glyphs(track):
    if track in ("baseline", "hand_coded"):
        from option_env import NUM_GLYPHS
    else:
        from discovered_option_env import NUM_GLYPHS
    return NUM_GLYPHS


def run_episode_stochastic(env, network, params, seed, rng_key):
    obs, info = env.reset(seed=seed)
    episode_return = 0.0
    raw_reward = 0.0
    steps_before = env._primitive_steps_used
    while True:
        crop = jnp.asarray(obs["crop"], dtype=jnp.int32)[None, :, :]
        feat = jnp.asarray(obs["features"], dtype=jnp.float32)[None, :]
        pi, _ = network.apply(params, crop, feat)
        rng_key, subkey = jax.random.split(rng_key)
        action = int(pi.sample(seed=subkey)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward
        raw_reward = reward
        if terminated or truncated:
            break
    primitive_steps = env._primitive_steps_used - steps_before
    success = raw_reward > 0.0
    return success, episode_return, primitive_steps, rng_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, required=True,
        choices=["baseline", "hand_coded", "discovered", "random_grounding"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--policy-dir", type=str, default="bc_policies_cropped_v2")
    parser.add_argument("--eval-seeds-start", type=int, default=1000)
    parser.add_argument("--eval-seeds-end", type=int, default=1050)
    parser.add_argument("--max-primitive-steps", type=int, default=6000)
    parser.add_argument("--n-repeats", type=int, default=1,
        help="Repeat each held-out seed this many times with different "
             "sampling noise and average (default 1 = same protocol as "
             "evaluate_policy.py, one episode per seed).")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    num_options = ckpt["num_options"]
    params = ckpt["params"]

    num_glyphs = get_num_glyphs(args.track)
    network = OptionActorCritic(num_options=num_options, num_glyphs=num_glyphs)
    env = build_env(args.track, args.policy_dir, args.max_primitive_steps)

    eval_seeds = list(range(args.eval_seeds_start, args.eval_seeds_end))
    print(f"Track: {args.track} (STOCHASTIC eval) | "
          f"Evaluating on {len(eval_seeds)} held-out dungeons "
          f"[{args.eval_seeds_start}, {args.eval_seeds_end}) "
          f"x {args.n_repeats} repeat(s)")
    print(f"Checkpoint: {args.checkpoint}\n")

    rng_key = jax.random.PRNGKey(args.rng_seed)

    successes, returns, prim_steps = [], [], []
    rows = []
    for seed in eval_seeds:
        seed_successes, seed_returns, seed_steps = [], [], []
        for rep in range(args.n_repeats):
            success, ep_return, ep_prim_steps, rng_key = run_episode_stochastic(
                env, network, params, seed, rng_key
            )
            seed_successes.append(success)
            seed_returns.append(ep_return)
            seed_steps.append(ep_prim_steps)

        mean_success = sum(seed_successes) / len(seed_successes)
        mean_return = sum(seed_returns) / len(seed_returns)
        mean_steps = sum(seed_steps) / len(seed_steps)

        successes.append(mean_success >= 0.5 if args.n_repeats == 1 else mean_success)
        returns.append(mean_return)
        prim_steps.append(mean_steps)

        rows.append({
            "seed": seed,
            "success": int(seed_successes[0]) if args.n_repeats == 1 else round(mean_success, 4),
            "return": mean_return,
            "primitive_steps": mean_steps,
        })
        tag = "success" if (seed_successes[0] if args.n_repeats == 1 else mean_success >= 0.5) else "fail"
        print(f"  seed {seed:6d}: {tag:7s} mean_return={mean_return:8.3f} "
              f"mean_primitive_steps={mean_steps:7.1f} "
              f"(n_repeats={args.n_repeats})")

    n = len(eval_seeds)
    n_success = sum(1 for s in successes if (s if args.n_repeats == 1 else s >= 0.5))
    p, (lo, hi) = wilson_ci(n_success, n)
    ret_mean, ret_sd, (ret_lo, ret_hi) = normal_ci(returns)
    steps_mean, steps_sd, (steps_lo, steps_hi) = normal_ci(prim_steps)

    print(f"\n=== Held-out evaluation summary (STOCHASTIC policy): "
          f"{args.track} (n={n} episodes) ===")
    print(f"Success rate: {p:.3f}  (95% Wilson CI: [{lo:.3f}, {hi:.3f}])  "
          f"[{n_success}/{n} episodes]")
    print(f"Episode return: mean={ret_mean:.3f} SD={ret_sd:.3f}  "
          f"(95% CI: [{ret_lo:.3f}, {ret_hi:.3f}])")
    print(f"Primitive steps/episode: mean={steps_mean:.1f} SD={steps_sd:.1f}  "
          f"(95% CI: [{steps_lo:.1f}, {steps_hi:.1f}])")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "success", "return", "primitive_steps"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-episode results saved to {args.out_csv}")


if __name__ == "__main__":
    main()