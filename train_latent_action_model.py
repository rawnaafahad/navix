"""
train_latent_action_model.py

Training loop for the latent action discovery model. Loads (past, future)
frame pairs from NLD-AA via nld_pair_sampler (keypresses hidden), trains
LatentActionModel with optax, and logs codebook usage periodically.

Includes dead-code reset: any VQ code unused over a monitoring window is
reinitialized to a real encoder output (z_e) from the current batch, plus
small noise. This is the standard fix for VQ-VAE codebook collapse, where
gradient-only updates let one code win all assignments and starve the rest.
"""

import argparse
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
import numpy as np

from latent_action_model import LatentActionModel, NUM_CODES
from nld_pair_sampler import make_pair_batches


def create_train_state(rng, learning_rate, dummy_past, dummy_future):
    model = LatentActionModel()
    params = model.init(rng, dummy_past, dummy_future)["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, past_frames, future_frames):
    def loss_fn(params):
        loss, codes = state.apply_fn({"params": params}, past_frames, future_frames)
        return loss, codes

    (loss, codes), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, codes


@jax.jit
def get_z_e_pool(state, past_frames, future_frames):
    """Recompute pre-quantization encoder outputs (z_e) for the current
    batch, giving real vectors to reseed dead codebook entries with."""
    return state.apply_fn(
        {"params": state.params},
        past_frames,
        future_frames,
        method=LatentActionModel.encode_z_e,
    )


def reset_dead_codes(state, dead_code_mask_np, z_e_pool, rng):
    """Reinitialize any codebook row with zero usage to a random real z_e
    vector from the current batch's pool, plus small noise."""
    if not dead_code_mask_np.any():
        return state, rng

    num_codes = dead_code_mask_np.shape[0]
    pool_size = z_e_pool.shape[0]

    rng, sample_rng, noise_rng = jax.random.split(rng, 3)
    sample_idx = jax.random.randint(sample_rng, (num_codes,), 0, pool_size)
    replacement_vecs = z_e_pool[sample_idx]
    noise = jax.random.normal(noise_rng, replacement_vecs.shape) * 0.01
    dead_code_mask = jnp.asarray(dead_code_mask_np)

    # params may be a FrozenDict or plain dict depending on flax version;
    # unfreeze -> mutate -> refreeze if needed, so this works either way.
    is_frozen = isinstance(state.params, flax.core.FrozenDict)
    params = flax.core.unfreeze(state.params) if is_frozen else dict(state.params)

    codebook = params["vq"]["codebook"]
    new_codebook = jnp.where(dead_code_mask[:, None], replacement_vecs + noise, codebook)

    vq_params = dict(params["vq"])
    vq_params["codebook"] = new_codebook
    params["vq"] = vq_params

    new_params = flax.core.freeze(params) if is_frozen else params
    state = state.replace(params=new_params)
    return state, rng


def codebook_usage_str(codes, num_codes):
    counts = np.bincount(np.asarray(codes), minlength=num_codes)
    frac = counts / max(counts.sum(), 1)
    used = int((counts > 0).sum())
    return f"used {used}/{num_codes} codes | frac=" + ",".join(f"{f:.2f}" for f in frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="nld-aa-test")
    parser.add_argument("--dbfilename", type=str, default="/home/rawna/ttyrecs.db")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--reset-check-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="lam_params.pkl")
    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    dummy_past = jnp.zeros((args.batch_size, 24, 80), dtype=jnp.int32)
    dummy_future = jnp.zeros((args.batch_size, 24, 80), dtype=jnp.int32)
    state = create_train_state(rng, args.lr, dummy_past, dummy_future)

    step = 0
    running_loss = 0.0
    all_codes_since_log = []
    usage_counts_since_reset_check = np.zeros(NUM_CODES, dtype=np.int64)
    resets_done = 0

    while step < args.total_steps:
        for past, future in make_pair_batches(
            args.dataset_name, args.dbfilename, args.batch_size, args.seq_length
        ):
            if past.shape[0] != args.batch_size:
                continue  # skip ragged final batch from a short episode

            past_j = jnp.asarray(past, dtype=jnp.int32)
            future_j = jnp.asarray(future, dtype=jnp.int32)

            state, loss, codes = train_step(state, past_j, future_j)
            running_loss += float(loss)
            codes_np = np.asarray(codes)
            all_codes_since_log.append(codes_np)
            usage_counts_since_reset_check += np.bincount(codes_np, minlength=NUM_CODES)
            step += 1

            if step % args.reset_check_every == 0:
                dead_mask_np = usage_counts_since_reset_check == 0
                if dead_mask_np.any():
                    z_e_pool = get_z_e_pool(state, past_j, future_j)
                    state, rng = reset_dead_codes(state, dead_mask_np, z_e_pool, rng)
                    resets_done += int(dead_mask_np.sum())
                usage_counts_since_reset_check = np.zeros(NUM_CODES, dtype=np.int64)

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                codes_concat = np.concatenate(all_codes_since_log)
                usage_str = codebook_usage_str(codes_concat, NUM_CODES)
                print(f"step {step:5d} | loss {avg_loss:.4f} | {usage_str} | resets so far: {resets_done}")
                running_loss = 0.0
                all_codes_since_log = []

            if step >= args.total_steps:
                break

    import pickle
    with open(args.save_path, "wb") as f:
        pickle.dump(jax.device_get(state.params), f)
    print(f"Saved trained params to {args.save_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
