"""
interpret_codes_context.py

Second interpretability pass: instead of looking at WHICH keys each code
presses (which turned out similar across codes -- see interpret_codes.py),
look at the STATE CONTEXT each code fires in: HP fraction, presence of
monsters nearby, and how "cluttered" the crop is (walls/features vs open
floor). This tests whether codes specialize by SITUATION even if they
don't specialize by raw action identity.

Uses the cropped, tty_chars-based states already saved in
grounding_buckets_cropped/ (17x17 crops centered on the player).
"""

import os
import numpy as np

BUCKET_DIR = "grounding_buckets_cropped"

MONSTER_CHARS = set("abcdefgijklmnopqrstuvwxyzABCDEFGIJKLMNOPQRSTUVWXYZ@&")
WALL_CHARS = set("|-#")
FLOOR_CHARS = set(".")


def analyze_crop(crop):
    flat = crop.flatten()
    chars = [chr(c) for c in flat if c != 0]

    n = max(len(chars), 1)
    monster_frac = sum(1 for c in chars if c in MONSTER_CHARS) / n
    wall_frac = sum(1 for c in chars if c in WALL_CHARS) / n
    floor_frac = sum(1 for c in chars if c in FLOOR_CHARS) / n

    center = crop[7:10, 7:10]
    center_chars = [chr(c) for c in center.flatten() if c != 0]
    adjacent_monster = any(c in MONSTER_CHARS and c != '@' for c in center_chars)

    return monster_frac, wall_frac, floor_frac, adjacent_monster


def main():
    for code in range(8):
        path = os.path.join(BUCKET_DIR, f"bucket_code_{code}.npz")
        if not os.path.exists(path):
            print(f"code {code}: bucket not found, skipping")
            continue

        data = np.load(path)
        states = data["states"]
        n = states.shape[0]

        if n > 20000:
            idx = np.random.default_rng(0).choice(n, size=20000, replace=False)
            states = states[idx]
            n = 20000

        monster_fracs, wall_fracs, floor_fracs, adjacents = [], [], [], []
        for i in range(n):
            mf, wf, ff, adj = analyze_crop(states[i])
            monster_fracs.append(mf)
            wall_fracs.append(wf)
            floor_fracs.append(ff)
            adjacents.append(adj)

        print(f"code {code}: n_sampled={n:>6} | "
              f"avg_monster_density={np.mean(monster_fracs):.3f} | "
              f"avg_wall_density={np.mean(wall_fracs):.3f} | "
              f"avg_floor_density={np.mean(floor_fracs):.3f} | "
              f"pct_with_adjacent_monster={100*np.mean(adjacents):.1f}%")


if __name__ == "__main__":
    main()
