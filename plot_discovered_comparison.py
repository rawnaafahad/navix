"""
plot_discovered_comparison.py

Parses a train_discovered_option_ppo.py (or train_option_ppo.py /
train_baseline_ppo.py -- they share the same log format) log file and
plots success_rate(last20) over training, with reference lines/bands for
comparison against other Phase 3 tracks. Reference values are CLI args,
not hardcoded, since different seed-pool/step-budget settings have
different comparison numbers.

Usage:
    python3 plot_discovered_comparison.py LOGFILE [--hand-coded-low X]
        [--hand-coded-high Y] [--primitive Z] [--title "..."] [--out NAME]
"""

import sys
import re
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE_RE = re.compile(
    r"option_steps=(\d+)\s+elapsed=[\d.]+s\s+mean_return\(last20\)=(-?[\d.]+|nan)\s+"
    r"success_rate\(last20\)=([\d.]+|nan)"
)


def parse_log(path, warmup_updates=20):
    UPDATE_RE = re.compile(r"update (\d+)/")
    steps, success_rates, updates = [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            um = UPDATE_RE.search(line)
            if not m or not um:
                continue
            succ = m.group(3)
            if succ == "nan":
                continue
            steps.append(int(m.group(1)))
            success_rates.append(float(succ))
            updates.append(int(um.group(1)))

    if not steps:
        raise ValueError(f"No matching log lines found in {path}")

    steps = np.array(steps)
    success_rates = np.array(success_rates)
    updates = np.array(updates)

    # exclude early updates where success_rate(last20) is computed from
    # fewer than 20 real completed episodes (degenerate small-sample
    # artifact, e.g. a single early episode inflating to 1.000)
    mask = updates >= warmup_updates
    return steps[mask], success_rates[mask]


def plot_comparison(steps, success_rates, out_path, title, hc_low, hc_high, primitive):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, success_rates, color="#1f77b4", linewidth=1.0, alpha=0.6,
              label="Discovered options (this run)")

    if len(steps) > 10:
        window = max(len(steps) // 20, 3)
        kernel = np.ones(window) / window
        smoothed = np.convolve(success_rates, kernel, mode="valid")
        smoothed_steps = steps[window - 1:]
        plt.plot(smoothed_steps, smoothed, color="#08306b", linewidth=2.2,
                  label="Discovered options (smoothed)")

    if hc_low is not None:
        if hc_high is not None and hc_high != hc_low:
            plt.axhspan(hc_low, hc_high, color="#2ca02c", alpha=0.15,
                        label=f"Hand-coded options, Phase 3 ({hc_low:.3f}-{hc_high:.3f})")
        else:
            plt.axhline(hc_low, linestyle="--", color="#2ca02c", alpha=0.8,
                        label=f"Hand-coded options, Phase 3 ({hc_low:.3f})")

    if primitive is not None:
        plt.axhline(primitive, linestyle="--", color="#d62728", alpha=0.8,
                    label=f"Primitive baseline, Phase 3 ({primitive:.3f})")

    plt.xlabel("Option steps (high-level decisions)")
    plt.ylabel("Success rate (last 20 episodes)")
    plt.title(title)
    plt.ylim(-0.02, max(0.45, success_rates.max() * 1.15))
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--hand-coded-low", type=float, default=0.150)
    parser.add_argument("--hand-coded-high", type=float, default=None)
    parser.add_argument("--primitive", type=float, default=0.050)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--warmup-updates", type=int, default=20,
        help="Exclude the first N updates, where success_rate(last20) is "
             "computed from fewer than 20 real completed episodes.")
    args = parser.parse_args()

    steps, success_rates = parse_log(args.log_path, warmup_updates=args.warmup_updates)

    print(f"Parsed {len(steps)} log points from {args.log_path}")
    print(f"Success rate: min={success_rates.min():.3f}, max={success_rates.max():.3f}, "
          f"mean={success_rates.mean():.3f}, final={success_rates[-1]:.3f}")
    late = success_rates[int(0.8 * len(success_rates)):]
    print(f"Mean of last 20% of training: {late.mean():.3f}")

    import os
    base = os.path.splitext(os.path.basename(args.log_path))[0]
    out_path = args.out or f"discovered_vs_baselines_{base}.png"
    title = args.title or f"Discovered Options PPO vs. Phase 3 Baselines ({base})"

    plot_comparison(
        steps, success_rates, out_path, title,
        args.hand_coded_low, args.hand_coded_high, args.primitive,
    )


if __name__ == "__main__":
    main()
