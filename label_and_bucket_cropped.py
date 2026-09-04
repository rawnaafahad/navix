"""
label_and_bucket_cropped.py

Same as label_and_bucket.py (code assignment via the full-grid discovery
model, unchanged), but stores a 17x17 player-centered crop for each
(state, action) pair instead of the full 24x80 grid -- matching the
representation used by OptionEnv/OptionActorCritic elsewhere in the
pipeline, since plain full-grid BC was found to underperform a
majority-class baseline (see grounding notes).

Player position is found by searching for '@' (ASCII 64) directly in
each frame (verified empirically to be reliable), falling back to
tty_cursor (verified to agree with '@' position in ~77.5% of frames;
the remainder are menu/prompt frames where the cursor moves elsewhere),
falling back to grid center as a last resort.
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp

import nle.dataset as nld
from latent_action_model import LatentActionModel, NUM_CODES, FUTURE_HORIZON

PLAYER_CHAR = 64  # ord('@')
CROP_HALF = 8      # 17x17 crop = 2*8 + 1


def find_player_pos(grid, cursor_row, cursor_col):
    positions = np.argwhere(grid == PLAYER_CHAR)
    if len(positions) == 1:
        return int(positions[0][0]), int(positions[0][1])
    if 0 <= cursor_row < grid.shape[0] and 0 <= cursor_col < grid.shape[1]:
        return int(cursor_row), int(cursor_col)
    return grid.shape[0] // 2, grid.shape[1] // 2


def crop_centered(grid, row, col, half=CROP_HALF):
    padded = np.pad(grid, ((half, half), (half, half)), mode="constant", constant_values=0)
    r, c = row + half, col + half
    return padded[r - half:r + half + 1, c - half:c + half + 1]


def load_model_and_params(params_path):
    import pickle
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    return params


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
    parser.add_argument("--out-dir", type=str, default="grounding_buckets_cropped")
    args = parser.parse_args()

    params = load_model_and_params(args.params_path)

    dataset = nld.TtyrecDataset(
        args.dataset_name, batch_size=args.batch_size, seq_length=args.seq_length,
        dbfilename=args.dbfilename, shuffle=False, loop_forever=False,
    )

    buckets = {c: {"states": [], "actions": []} for c in range(NUM_CODES)}
    windows_total, windows_skipped_boundary = 0, 0

    for mb_idx, mb in enumerate(dataset):
        if mb_idx >= args.max_minibatches:
            break

        chars = mb["tty_chars"]
        keys = mb["keypresses"]
        gameids = mb["gameids"]
        cursor = mb["tty_cursor"]

        B, T = chars.shape[0], chars.shape[1]

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
                grid = chars[b, step]
                row, col = find_player_pos(grid, cursor[b, step][0], cursor[b, step][1])
                crop = crop_centered(grid, row, col)
                buckets[code]["states"].append(crop)
                buckets[code]["actions"].append(int(keys[b, step]))
            windows_total += 1

        if mb_idx % 20 == 0:
            print(f"minibatch {mb_idx}: windows labeled so far = {windows_total}, "
                  f"skipped (boundary) = {windows_skipped_boundary}")

    print(f"\nDone. Total windows labeled: {windows_total}, skipped: {windows_skipped_boundary}")

    os.makedirs(args.out_dir, exist_ok=True)
    for code in range(NUM_CODES):
        n = len(buckets[code]["states"])
        if n == 0:
            print(f"  code {code}: 0 samples -- SKIPPING")
            continue
        states_arr = np.stack(buckets[code]["states"]).astype(np.uint8)
        actions_arr = np.asarray(buckets[code]["actions"], dtype=np.uint8)
        out_path = os.path.join(args.out_dir, f"bucket_code_{code}.npz")
        np.savez_compressed(out_path, states=states_arr, actions=actions_arr)
        print(f"  code {code}: {n} samples, state shape {states_arr.shape[1:]} -> {out_path}")


if __name__ == "__main__":
    main()
