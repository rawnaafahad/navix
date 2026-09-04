"""
interpret_codes.py

Qualitative interpretability pass: for each discovered code, shows the
distribution of raw keypresses (from the original, unfiltered grounding
buckets) to see whether a recognizable behavioral pattern emerges --
e.g. one code dominated by movement keys, another by combat/interaction
commands. Cheap, no jax/training needed, just counting.

NetHack's default vi-style keybindings (for reference):
  h/j/k/l/y/u/b/n = movement (8 directions)
  H/J/K/L/Y/U/B/N = run in direction (shift + movement)
  s = search, i = inventory, , = pick up, > = descend stairs
  numbers 1-9 on some keyboards = movement in vikeys-off mode
"""

import os
import numpy as np

MOVEMENT_KEYS = set("hjklyubnHJKLYUBN")
BUCKET_DIR = "grounding_buckets"

def describe_key(k):
    if 32 <= k < 127:
        ch = chr(k)
        tag = " (movement)" if ch in MOVEMENT_KEYS else ""
        return f"'{ch}'{tag}"
    return f"<0x{k:02x}>"


def main():
    for code in range(8):
        path = os.path.join(BUCKET_DIR, f"bucket_code_{code}.npz")
        if not os.path.exists(path):
            print(f"code {code}: bucket not found, skipping")
            continue

        data = np.load(path)
        actions = data["actions"]
        n = len(actions)

        counts = np.bincount(actions, minlength=256)
        top_idx = np.argsort(counts)[::-1][:10]

        movement_total = sum(counts[k] for k in range(256) if 32 <= k < 127 and chr(k) in MOVEMENT_KEYS)
        movement_frac = movement_total / n if n > 0 else 0.0

        print(f"\n--- Code {code} (n={n}, movement-key fraction={movement_frac:.2f}) ---")
        for k in top_idx:
            if counts[k] == 0:
                continue
            frac = counts[k] / n
            print(f"  {describe_key(int(k)):18s} count={counts[k]:>7} ({frac:.1%})")


if __name__ == "__main__":
    main()
