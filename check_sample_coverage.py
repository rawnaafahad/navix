"""
check_sample_coverage.py

Follow-up to check_action_mapping.py: unique-key coverage (23/83) is the
wrong number to react to. What matters is what fraction of actual SAMPLES
in the grounding buckets have a covered (executable) action -- rare
commands each contribute few samples, so real coverage may be much higher
than the unique-key ratio suggests.
"""

import glob
import os
import numpy as np
import gymnasium as gym
import nle  # noqa: F401


def main():
    env = gym.make("NetHackStaircase-v0")
    actions = env.unwrapped.actions
    covered_values = set(getattr(a, "value", None) for a in actions)
    env.close()

    bucket_dir = "grounding_buckets"
    bucket_files = sorted(glob.glob(os.path.join(bucket_dir, "bucket_code_*.npz")))

    total_samples = 0
    total_covered = 0
    print(f"{'code':>6} | {'n_samples':>10} | {'covered':>10} | {'pct_covered':>11}")
    for bf in bucket_files:
        code = os.path.basename(bf).replace("bucket_code_", "").replace(".npz", "")
        data = np.load(bf)
        actions_arr = data["actions"]
        n = len(actions_arr)
        covered_mask = np.isin(actions_arr, list(covered_values))
        n_covered = int(covered_mask.sum())

        total_samples += n
        total_covered += n_covered
        pct = 100.0 * n_covered / n if n > 0 else 0.0
        print(f"{code:>6} | {n:>10} | {n_covered:>10} | {pct:>10.1f}%")

    overall_pct = 100.0 * total_covered / total_samples if total_samples > 0 else 0.0
    print(f"\nOverall: {total_covered}/{total_samples} samples covered ({overall_pct:.1f}%)")


if __name__ == "__main__":
    main()
