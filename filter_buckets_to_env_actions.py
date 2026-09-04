"""
filter_buckets_to_env_actions.py

Filters grounding_buckets/*.npz down to only samples whose raw keypress
is a valid action in NetHackStaircase-v0, and relabels the 'actions'
field from raw keypress bytes to real NLE action indices (0-22).

Run this once, then point train_bc_policy.py at the new output directory
(grounding_buckets_filtered/) -- no changes needed to the BC script itself,
since it already builds its label vocabulary dynamically from whatever is
in the 'actions' array.
"""

import glob
import os
import numpy as np
import gymnasium as gym
import nle  # noqa: F401


def main():
    env = gym.make("NetHackStaircase-v0")
    actions = env.unwrapped.actions
    keypress_to_idx = {getattr(a, "value", None): idx for idx, a in enumerate(actions)}
    env.close()

    in_dir = "grounding_buckets"
    out_dir = "grounding_buckets_filtered"
    os.makedirs(out_dir, exist_ok=True)

    bucket_files = sorted(glob.glob(os.path.join(in_dir, "bucket_code_*.npz")))
    print(f"{'code':>6} | {'before':>8} | {'after':>8} | {'pct_kept':>9}")

    for bf in bucket_files:
        code = os.path.basename(bf).replace("bucket_code_", "").replace(".npz", "")
        data = np.load(bf)
        states = data["states"]
        raw_actions = data["actions"]

        keep_mask = np.array([a in keypress_to_idx for a in raw_actions])
        n_before = len(raw_actions)
        n_after = int(keep_mask.sum())

        filtered_states = states[keep_mask]
        filtered_raw_actions = raw_actions[keep_mask]
        relabeled_actions = np.array(
            [keypress_to_idx[a] for a in filtered_raw_actions], dtype=np.uint8
        )

        out_path = os.path.join(out_dir, f"bucket_code_{code}.npz")
        np.savez_compressed(out_path, states=filtered_states, actions=relabeled_actions)

        pct = 100.0 * n_after / n_before if n_before > 0 else 0.0
        print(f"{code:>6} | {n_before:>8} | {n_after:>8} | {pct:>8.1f}%")

    print(f"\nFiltered buckets written to {out_dir}/")
    print("Actions in these files are now real NLE action indices (0-22),")
    print("not raw keypress bytes. train_bc_policy.py can be pointed at")
    print("this directory with no other changes.")


if __name__ == "__main__":
    main()
