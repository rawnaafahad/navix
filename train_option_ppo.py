"""CleanRL-style PPO training loop for the option-level NetHack policy.

Trains an OptionActorCritic (option_ppo_models.py) over the 3-option
action space exposed by OptionEnv (option_env.py), which wraps the real
NetHackStaircase-v0 NLE environment via the hard-coded option executor in
nethack_option_executor.py.

v2 CHANGE (matching train_discovered_option_ppo.py): budgeted by TOTAL
PRIMITIVE ENVIRONMENT STEPS, not option-selection steps. A single
hand-coded option call can consume up to explore_step_budget (150) or
navigation_step_budget (500) primitive steps, so comparing this track
against the primitive-action baseline "at the same option-step count"
silently gives it a much larger real interaction budget. This matters
directly for a sample-efficiency/credit-assignment argument.

Since the number of outer PPO updates needed to reach a fixed primitive-
step budget is not known in advance, the LR annealing schedule's
denominator is an explicit CLI parameter (--anneal-total-updates) rather
than an auto-computed constant -- set it from a short calibration run.

Also added: checkpoint saving (the original never persisted trained
weights, so no prior run of this script can be held-out evaluated).
"""

import argparse
import time
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from option_env import (
    OptionEnv,
    OptionEnvConfig,
    NUM_OPTIONS,
    NUM_GLYPHS,
    CROP_SIZE,
    NUM_FEATURES,
)
from option_ppo_models import OptionActorCritic


@dataclass
class PPOConfig:
    total_primitive_steps: int = 500_000
    rollout_length: int = 128
    anneal_total_updates: int = 1000
    num_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    anneal_lr: bool = True
    seed: int = 0
    log_every: int = 1
    train_seeds_start: int = 0
    train_seeds_end: int = 1000
    save_path: str = "option_ppo_checkpoint.pkl"


def _stack_crops(crops: List[np.ndarray]) -> jnp.ndarray:
    return jnp.asarray(np.stack(crops), dtype=jnp.int32)


def _stack_features(features: List[np.ndarray]) -> jnp.ndarray:
    return jnp.asarray(np.stack(features), dtype=jnp.float32)


def make_train_state(rng, network: OptionActorCritic, config: PPOConfig):
    dummy_crop = jnp.zeros((1, CROP_SIZE, CROP_SIZE), dtype=jnp.int32)
    dummy_features = jnp.zeros((1, NUM_FEATURES), dtype=jnp.float32)
    params = network.init(rng, dummy_crop, dummy_features)

    if config.anneal_lr:
        denom = max(config.num_minibatches * config.num_epochs, 1)

        def lr_schedule(count):
            frac = 1.0 - (count // denom) / config.anneal_total_updates
            return config.lr * jnp.clip(frac, 0.0, 1.0)

        lr = lr_schedule
    else:
        lr = config.lr

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=lr, eps=1e-5),
    )

    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    steps = rewards.shape[0]
    advantages = np.zeros(steps, dtype=np.float32)
    last_gae = 0.0
    next_value = last_value
    for t in reversed(range(steps)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_loss_fn(params, apply_fn, batch, clip_eps, ent_coef, vf_coef):
    crop, features, actions, old_log_probs, advantages, returns, old_values = batch
    pi, values = apply_fn(params, crop, features)
    log_probs = pi.log_prob(actions)

    ratio = jnp.exp(log_probs - old_log_probs)
    norm_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surrogate1 = ratio * norm_adv
    surrogate2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
    actor_loss = -jnp.minimum(surrogate1, surrogate2).mean()

    value_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    value_loss_unclipped = jnp.square(values - returns)
    value_loss_clipped = jnp.square(value_clipped - returns)
    value_loss = 0.5 * jnp.maximum(value_loss_unclipped, value_loss_clipped).mean()

    entropy = pi.entropy().mean()

    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
    logs = {
        "loss/total": total_loss,
        "loss/actor": actor_loss,
        "loss/value": value_loss,
        "loss/entropy": entropy,
    }
    return total_loss, logs


@jax.jit
def _sgd_step(train_state, batch, clip_eps, ent_coef, vf_coef):
    grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
    (_, logs), grads = grad_fn(
        train_state.params, train_state.apply_fn, batch, clip_eps, ent_coef, vf_coef,
    )
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, logs


def update(train_state, rollout, config: PPOConfig, rng):
    crops, features, actions, log_probs, advantages, returns, values = rollout
    n_samples = crops.shape[0]
    minibatch_size = max(n_samples // config.num_minibatches, 1)

    logs_accum = []
    for _ in range(config.num_epochs):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        for start in range(0, n_samples, minibatch_size):
            idx = perm[start : start + minibatch_size]
            batch = (
                crops[idx], features[idx], actions[idx], log_probs[idx],
                advantages[idx], returns[idx], values[idx],
            )
            train_state, logs = _sgd_step(
                train_state, batch, config.clip_eps, config.ent_coef, config.vf_coef,
            )
            logs_accum.append(logs)

    mean_logs = {
        k: float(np.mean([float(l[k]) for l in logs_accum])) for k in logs_accum[0]
    }
    return train_state, mean_logs, rng


def train(config: PPOConfig):
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    print(
        f"Training seed pool: [{config.train_seeds_start}, "
        f"{config.train_seeds_end}) "
        f"({config.train_seeds_end - config.train_seeds_start} dungeons)"
    )
    print(f"Primitive-step budget: {config.total_primitive_steps}")

    env_config = OptionEnvConfig(
        train_seeds=range(config.train_seeds_start, config.train_seeds_end),
    )
    env = OptionEnv(env_config)

    network = OptionActorCritic(num_options=NUM_OPTIONS, num_glyphs=NUM_GLYPHS)
    train_state = make_train_state(init_rng, network, config)

    obs, _ = env.reset()
    episode_return = 0.0
    completed_returns: List[float] = []
    completed_successes: List[bool] = []
    total_primitive_steps = 0

    start_time = time.time()
    update_idx = 0

    while total_primitive_steps < config.total_primitive_steps:
        crops_buf, features_buf, actions_buf = [], [], []
        log_probs_buf, values_buf, rewards_buf, dones_buf = [], [], [], []

        for _ in range(config.rollout_length):
            crop_arr = _stack_crops([obs["crop"]])
            feat_arr = _stack_features([obs["features"]])

            rng, sample_rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, crop_arr, feat_arr)
            action = pi.sample(seed=sample_rng)
            log_prob = pi.log_prob(action)

            action_int = int(action[0])
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            raw_reward = reward
            total_primitive_steps += info.get("option_steps", 0)

            if truncated and not terminated:
                bootstrap_crop = _stack_crops([next_obs["crop"]])
                bootstrap_feat = _stack_features([next_obs["features"]])
                _, bootstrap_value = network.apply(
                    train_state.params, bootstrap_crop, bootstrap_feat,
                )
                training_reward = reward + config.gamma * float(bootstrap_value[0])
                gae_done = False
            else:
                training_reward = reward
                gae_done = terminated

            crops_buf.append(obs["crop"])
            features_buf.append(obs["features"])
            actions_buf.append(action_int)
            log_probs_buf.append(float(log_prob[0]))
            values_buf.append(float(value[0]))
            rewards_buf.append(training_reward)
            dones_buf.append(float(gae_done))

            episode_return += raw_reward

            if done:
                completed_returns.append(episode_return)
                completed_successes.append(raw_reward > 0.0)
                episode_return = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

        crop_arr = _stack_crops([obs["crop"]])
        feat_arr = _stack_features([obs["features"]])
        _, last_value = network.apply(train_state.params, crop_arr, feat_arr)
        last_value = float(last_value[0])

        rewards_np = np.asarray(rewards_buf, dtype=np.float32)
        values_np = np.asarray(values_buf, dtype=np.float32)
        dones_np = np.asarray(dones_buf, dtype=np.float32)

        advantages_np, returns_np = compute_gae(
            rewards_np, values_np, dones_np, last_value, config.gamma, config.gae_lambda,
        )

        rollout = (
            _stack_crops(crops_buf),
            _stack_features(features_buf),
            jnp.asarray(actions_buf, dtype=jnp.int32),
            jnp.asarray(log_probs_buf, dtype=jnp.float32),
            jnp.asarray(advantages_np, dtype=jnp.float32),
            jnp.asarray(returns_np, dtype=jnp.float32),
            jnp.asarray(values_np, dtype=jnp.float32),
        )

        train_state, logs, rng = update(train_state, rollout, config, rng)

        if update_idx % config.log_every == 0:
            elapsed = time.time() - start_time
            recent_returns = completed_returns[-20:]
            recent_successes = completed_successes[-20:]
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            success_rate = float(np.mean(recent_successes)) if recent_successes else float("nan")
            pct_done = 100.0 * total_primitive_steps / config.total_primitive_steps
            print(
                f"update {update_idx} "
                f"primitive_steps={total_primitive_steps}/{config.total_primitive_steps} "
                f"({pct_done:.1f}%) "
                f"elapsed={elapsed:.1f}s "
                f"mean_return(last20)={mean_return:.3f} "
                f"success_rate(last20)={success_rate:.3f} "
                f"loss/total={logs['loss/total']:.4f} "
                f"loss/entropy={logs['loss/entropy']:.4f}"
            )

        update_idx += 1

    env.close()

    import pickle
    with open(config.save_path, "wb") as f:
        pickle.dump({
            "params": jax.device_get(train_state.params),
            "num_options": NUM_OPTIONS,
        }, f)
    print(f"Saved checkpoint to {config.save_path}")
    print(f"Total outer updates: {update_idx}")

    return train_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-primitive-steps", type=int, default=500_000)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--anneal-total-updates", type=int, default=1000,
        help="Denominator for LR annealing -- set from a calibration run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument(
        "--train-seeds-start", type=int, default=0,
        help="Start of the range of NetHack dungeon seeds used for "
             "training episodes (inclusive).",
    )
    parser.add_argument(
        "--train-seeds-end", type=int, default=1000,
        help="End of the range of NetHack dungeon seeds used for "
             "training episodes (exclusive).",
    )
    parser.add_argument("--save-path", type=str, default="option_ppo_checkpoint.pkl")
    args = parser.parse_args()

    config = PPOConfig(
        total_primitive_steps=args.total_primitive_steps,
        rollout_length=args.rollout_length,
        anneal_total_updates=args.anneal_total_updates,
        seed=args.seed,
        lr=args.lr,
        train_seeds_start=args.train_seeds_start,
        train_seeds_end=args.train_seeds_end,
        save_path=args.save_path,
    )
    train(config)


if __name__ == "__main__":
    main()
