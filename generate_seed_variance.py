"""
Generates seed_variance.png: an individual-seed strip plot showing the raw
held-out success rate for every seed, grouped by condition, with a
horizontal bar marking each condition's mean.

This complements the ladder bar chart (which only shows mean +/- CI) by
showing the actual per-seed spread directly -- e.g. that discovered-v2
(corrected) ranges from 10% to 34% across its 5 seeds, or that hand-coded's
two seeds are a stark 34%/0% split rather than two similar numbers.

All values below are the real, confirmed per-seed results from this
project's held-out evaluation (seeds 1000-1049), not read from a log file,
since these numbers were already finalised earlier in the project.

Run this from wherever you want seed_variance.png to be written (e.g. the
same directory as your other thesis figures).
Requires matplotlib (already used by the other figure-generation scripts).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# (label, per-seed success rates in %, colour, is_handcoded)
CONDITIONS = [
    ("Primitive\nbaseline",       [0, 0, 0, 0, 0],       "#888888", False),
    ("Random-\ngrounding",        [18, 22, 16, 34, 16],  "#DD8452", False),
    ("Discovered\n(original)",    [6, 16, 16, 8, 18],    "#55A868", False),
    ("Discovered\n(v2, orig.)",   [22, 12, 30, 10, 10],  "#4C72B0", False),
    ("Discovered\n(v2, corr.)",   [10, 34, 18, 24, 10],  "#4C72B0", False),
    ("Hand-coded\n(2 seeds)",     [34, 0],                "#C44E52", True),
]

OUT_PATH = "seed_variance.png"


def make_plot():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(0)  # fixed seed -> reproducible jitter

    for i, (label, vals, color, hatched) in enumerate(CONDITIONS):
        x_jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            [i + j for j in x_jitter], vals,
            s=70, color=color, edgecolor="black", linewidth=0.8, zorder=3,
            marker="D" if hatched else "o",
        )
        if not hatched:
            # Hand-coded's two seeds are deliberately NOT averaged: the
            # thesis argues they are two mechanistically distinct findings
            # (one working, one collapsed), not noise around one true rate,
            # so drawing a mean line here would visually contradict that.
            mean_val = np.mean(vals)
            ax.hlines(mean_val, i - 0.18, i + 0.18, color=color, linewidth=2.5, zorder=2)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c[0] for c in CONDITIONS], fontsize=9)
    ax.set_ylabel("Held-out success rate (%)")
    ax.set_title("Individual seed spread by condition, 100-dungeon scale")
    ax.set_ylim(-3, 40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.axvline(4.5, color="gray", linestyle=":", linewidth=1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8,
               label='Individual seed (pooled conditions)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8,
               label='Individual seed (hand-coded, not pooled)'),
        Line2D([0], [0], color='gray', linewidth=2.5, label='Condition mean (pooled conditions only)'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    make_plot()
