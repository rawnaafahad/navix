"""
check_action_mapping.py

Diagnostic, run before wiring grounded BC policies into train_option_ppo.py.

Checks two things:
1. How to map a raw NLD-AA keypress byte to an NLE gym action index
   (via env.unwrapped.actions, standard NLE structure).
2. Whether the keypresses actually present in your grounding buckets are
   all valid actions in NetHackStaircase-v0 (the env your PPO pipeline
   uses) -- NLD-AA was collected on NetHackChallenge-v0 per the paper,
   which may have a wider action space. Any keypress with no match is a
   real gap: that (state, action) pair can't be used for grounding into
   this specific environment as-is.
"""

import glob
import os
import numpy as np
import gymnasium as gym
import nle  # noqa: F401  (registers NLE gym envs)


def main():
    env = gym.make("NetHackStaircase-v0")
    actions = env.unwrapped.actions
    print(f"env.unwrapped.actions has {len(actions)} entries")
    print(f"action_space.n = {env.action_space.n}")

    keypress_to_idx = {}
    for idx, act in enumerate(actions):
        val = getattr(act, "value", None)
        keypress_to_idx[val] = idx

    print("\nFirst 10 action objects and their (value -> idx) mapping:")
    for idx, act in enumerate(actions[:10]):
        print(f"  idx {idx}: {act} -> value={getattr(act, 'value', None)}")

    bucket_dir = "grounding_buckets"
    bucket_files = sorted(glob.glob(os.path.join(bucket_dir, "bucket_code_*.npz")))
    if not bucket_files:
        print(f"\nNo bucket files found in {bucket_dir}, skipping coverage check.")
        env.close()
        return

    print(f"\nChecking coverage against {len(bucket_files)} bucket files...")
    all_unique_keys = set()
    for bf in bucket_files:
        data = np.load(bf)
        unique_actions = np.unique(data["actions"])
        all_unique_keys.update(int(a) for a in unique_actions)

    covered = sorted(k for k in all_unique_keys if k in keypress_to_idx)
    uncovered = sorted(k for k in all_unique_keys if k not in keypress_to_idx)

    print(f"\nTotal distinct keypress values across all buckets: {len(all_unique_keys)}")
    print(f"Covered by NetHackStaircase-v0 action space: {len(covered)}")
    print(f"NOT covered (no matching action index): {len(uncovered)}")
    if uncovered:
        print(f"Uncovered keypress values: {uncovered}")
        print("(these bytes, likely ASCII chars, cannot currently be executed")
        print(" as an action in this environment -- affected samples would")
        print(" need to be dropped or the env's action space reconsidered)")

    env.close()


if __name__ == "__main__":
    main()
