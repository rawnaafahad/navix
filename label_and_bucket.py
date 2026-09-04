"""
label_and_bucket.py

Uses the trained latent action model to label real NLD-AA data with
discovered codes, then buckets (state, action) pairs by code. This is
Step 1-2 of the grounding stage: turning discovered latents into training
data for per-code imitation policies.

Actions (keypresses) were hidden during discovery-model training, but are
real ground-truth data in NLD-AA and are used here now, purely for
grounding -- the discovery model itself never sees them.
"""

import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp

import nle.dataset as nld
from latent_action_model import LatentActionModel, NUM_CODES, FUTURE_HORIZON


def load_model_and_params(params_path):
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    model = LatentActionModel()
    return model, params


@jax.jit
def assign_codes_batch(params, past_frames, future_frames):
    return LatentActionModel().apply(
        {"params": params}, past_frames, future_frames, method=LatentActionModel.assign_code
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="nld-aa-test")
    parser.add_argument("--dbfilename", type=str, default="/home/rawna/ttyrecs.db")
    parser.add_argument("--params-path", type=str, default="lam_params.pkl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--max-minibatches", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="grounding_buckets")
    args = parser.parse_args()

    model, params = load_model_and_params(args.params_path)

    dataset = nld.TtyrecDataset(
        args.dataset_name,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dbfilename=args.dbfilename,
        shuffle=False,
        loop_forever=False,
    )

    # one bucket per code: lists of (state_array, action_byte)
    buckets = {c: {"states": [], "actions": []} for c in range(NUM_CODES)}

    windows_total = 0
    windows_skipped_boundary = 0

    for mb_idx, mb in enumerate(dataset):
        if mb_idx >= args.max_minibatches:
            break

        chars = mb["tty_chars"]        # (B, T, 24, 80)
        keys = mb["keypresses"]        # (B, T)
        gameids = mb["gameids"]        # (B, T)

        B, T = chars.shape[0], chars.shape[1]

        # gather all valid (non-boundary-crossing) windows in this minibatch,
        # then run code assignment in one batched call for efficiency
        window_past, window_future, window_meta = [], [], []
        for b in range(B):
            for t in range(0, T - FUTURE_HORIZON, FUTURE_HORIZON):
                if gameids[b, t] != gameids[b, t + FUTURE_HORIZON]:
                    windows_skipped_boundary += 1
                    continue
                window_past.append(chars[b, t])
                window_future.append(chars[b, t + FUTURE_HORIZON])
                window_meta.append((b, t))

        if not window_past:
            continue

        past_batch = jnp.asarray(np.stack(window_past), dtype=jnp.int32)
        future_batch = jnp.asarray(np.stack(window_future), dtype=jnp.int32)
        codes = np.asarray(assign_codes_batch(params, past_batch, future_batch))

        for (b, t), code in zip(window_meta, codes):
            code = int(code)
            for step in range(t, t + FUTURE_HORIZON):
                buckets[code]["states"].append(chars[b, step])
                buckets[code]["actions"].append(int(keys[b, step]))
            windows_total += 1

        if mb_idx % 20 == 0:
            print(f"minibatch {mb_idx}: windows labeled so far = {windows_total}, "
                  f"skipped (boundary) = {windows_skipped_boundary}")

    print(f"\nDone. Total windows labeled: {windows_total}, skipped: {windows_skipped_boundary}")

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for code in range(NUM_CODES):
        n = len(buckets[code]["states"])
        if n == 0:
            print(f"  code {code}: 0 samples -- SKIPPING (empty bucket)")
            continue
        states_arr = np.stack(buckets[code]["states"]).astype(np.uint8)
        actions_arr = np.asarray(buckets[code]["actions"], dtype=np.uint8)
        out_path = os.path.join(args.out_dir, f"bucket_code_{code}.npz")
        np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
        print(f"  code {code}: {n} samples -> {out_path}")


if __name__ == "__main__":
    main()
