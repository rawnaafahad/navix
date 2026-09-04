"""
Generates three additional figures for the thesis, per reviewer feedback:
  1. training_curves.png -- success_rate(last20) vs primitive_steps, individual
     seeds shown faintly, mean shown prominently, for the four key conditions:
     primitive baseline, random-grounding, discovered-original, and
     discovered-v2 with corrected (best-checkpoint + weight-decay) grounding.
  2. bc_lift_comparison.png -- per-code BC lift over majority-class baseline,
     original grounding vs corrected (best-checkpoint) grounding, side by side.
  3. interpretability_chart.png -- monster-density and monster-adjacency per
     discovered code, as a bar chart (currently only tabulated in the thesis).

Does NOT generate a codebook-usage plot for the discovery model: that log's
format was never seen in this session, so a script for it would be guessing
blind rather than working from confirmed data, unlike everything else here.

Run this from anywhere (it reads logs from /tmp, where every training/BC/
interpretability run in this project wrote its output, and writes the three
output PNGs to your current directory).
Requires matplotlib (already used by plot_ladder_results.py).
"""
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = "/tmp"


def _p(filename):
    """Resolve a bare log filename against LOG_DIR."""
    return os.path.join(LOG_DIR, filename)


UPDATE_LINE_RE = re.compile(
    r"primitive_steps=(\d+)/(\d+)\s*\(([\d.]+)%\)\s*elapsed=([\d.]+)s\s*"
    r"mean_return\(last20\)=(-?[\d.]+)\s*success_rate\(last20\)=([\d.]+)"
)

# ---------------------------------------------------------------------------
# Figure 1: training curves
# ---------------------------------------------------------------------------

CONDITIONS = {
    "Primitive baseline": ["log_baseline_100seed_seed0.log", "log_baseline_100seed_seed1.log"],
    "Random-grounding": [f"log_randomgrounding_100seed_seed{i}.log" for i in range(5)],
    "Discovered-original": [f"log_discoveredorig_100seed_seed{i}.log" for i in range(5)],
    "Discovered-v2 (corrected)": [f"log_discoveredv2_wd_seed{i}.log" for i in range(5)],
}
COLORS = {
    "Primitive baseline": "#888888",
    "Random-grounding": "#DD8452",
    "Discovered-original": "#55A868",
    "Discovered-v2 (corrected)": "#4C72B0",
}


def parse_log(path):
    """Returns (primitive_steps array, success_rate array) from a training log."""
    steps, rates = [], []
    try:
        with open(_p(path)) as f:
            for line in f:
                m = UPDATE_LINE_RE.search(line)
                if m:
                    steps.append(int(m.group(1)))
                    rates.append(float(m.group(6)))
    except FileNotFoundError:
        print(f"  WARNING: {path} not found, skipping")
        return None, None
    if not steps:
        print(f"  WARNING: no matching lines found in {path}")
        return None, None
    return np.array(steps), np.array(rates)


def make_training_curves():
    fig, ax = plt.subplots(figsize=(9, 6))
    grid = np.linspace(0, 750_000, 200)

    for name, files in CONDITIONS.items():
        color = COLORS[name]
        seed_curves = []
        for fp in files:
            steps, rates = parse_log(fp)
            if steps is None or len(steps) < 2:
                continue
            ax.plot(steps, rates, color=color, alpha=0.15, linewidth=1)
            interp = np.interp(grid, steps, rates, left=np.nan, right=rates[-1])
            seed_curves.append(interp)

        if not seed_curves:
            print(f"  WARNING: no data at all for '{name}', skipping mean line")
            continue

        mean_curve = np.nanmean(np.vstack(seed_curves), axis=0)
        ax.plot(grid, mean_curve, color=color, linewidth=2.5,
                 label=f"{name} (n={len(seed_curves)} seeds)")

    ax.set_xlabel("Primitive environment steps")
    ax.set_ylabel("Success rate (trailing 20-episode mean)")
    ax.set_title("Training curves by condition (individual seeds faint, mean bold)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")


# ---------------------------------------------------------------------------
# Figure 2: BC grounding lift comparison
# ---------------------------------------------------------------------------

LIFT_LINE_RE = re.compile(
    r"code (\d+):.*?lift=\+([\d.]+)"
)


def parse_bc_lift(path):
    lifts = {}
    try:
        with open(_p(path)) as f:
            for line in f:
                m = LIFT_LINE_RE.search(line)
                if m:
                    lifts[int(m.group(1))] = float(m.group(2))
    except FileNotFoundError:
        print(f"  WARNING: {path} not found")
        return {}
    return lifts


def make_bc_lift_chart():
    original = parse_bc_lift("bc_train_v2.log")
    corrected = parse_bc_lift("bc_train_v2_bestckpt.log")
    if not original or not corrected:
        print("  Skipping BC lift chart: one or both source logs not found/parsed.")
        return

    codes = sorted(set(original) | set(corrected))
    orig_vals = [original.get(c, 0) for c in codes]
    corr_vals = [corrected.get(c, 0) for c in codes]

    x = np.arange(len(codes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, orig_vals, width, label="Original grounding", color="#DD8452")
    ax.bar(x + width / 2, corr_vals, width, label="Corrected (best-checkpoint) grounding", color="#4C72B0")

    ax.set_xticks(x)
    ax.set_xticklabels([f"code {c}" for c in codes])
    ax.set_ylabel("Accuracy improvement over majority-class baseline")
    ax.set_title("Behaviour-cloning performance before and after checkpoint-selection correction")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("bc_lift_comparison.png", dpi=150)
    print("Saved bc_lift_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: interpretability chart (monster density / adjacency per code)
# ---------------------------------------------------------------------------

INTERP_LINE_RE = re.compile(
    r"code (\d+): n_sampled=\s*(\d+)\s*\|\s*avg_monster_density=([\d.]+)\s*\|.*?"
    r"pct_with_adjacent_monster=([\d.]+)%"
)


def make_interpretability_chart():
    path = "interpret_context_output.txt"
    try:
        with open(_p(path)) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  WARNING: {_p(path)} not found -- if it's saved somewhere else, "
              f"edit LOG_DIR at the top of this script or copy it into /tmp.")
        return

    codes, densities, adjacencies = [], [], []
    for m in INTERP_LINE_RE.finditer(content):
        codes.append(int(m.group(1)))
        densities.append(float(m.group(3)))
        adjacencies.append(float(m.group(4)))

    if not codes:
        print("  WARNING: no matching lines found in interpret_context_output.txt")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    highlight = {2, 3, 7}
    colors = ["#DD8452" if c in highlight else "#4C72B0" for c in codes]

    ax1.bar([f"code {c}" for c in codes], densities, color=colors)
    ax1.set_ylabel("Monster density")
    ax1.set_title("Monster density per code")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)

    ax2.bar([f"code {c}" for c in codes], adjacencies, color=colors)
    ax2.set_ylabel("% episodes with adjacent monster")
    ax2.set_title("Monster adjacency per code")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Monster-related situational context by latent code")
    fig.tight_layout()
    fig.savefig("interpretability_chart.png", dpi=150)
    print("Saved interpretability_chart.png")


if __name__ == "__main__":
    print("Generating training_curves.png ...")
    make_training_curves()
    print("\nGenerating bc_lift_comparison.png ...")
    make_bc_lift_chart()
    print("\nGenerating interpretability_chart.png ...")
    make_interpretability_chart()
