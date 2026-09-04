"""
plot_training_log.py

Parses a train_latent_action_model.py log file and produces two plots:
  1. loss.png            - training loss vs step
  2. codebook_usage.png  - per-code usage fraction over time (stacked area)
                            plus number of active codes over time

Usage:
    python3 plot_training_log.py /tmp/lam_full_run.log
"""

import sys
import re
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import numpy as np

LINE_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*used\s+(\d+)/(\d+)\s+codes\s*\|\s*frac=([\d.,]+)"
)


def parse_log(path):
    steps, losses, used_counts, num_codes, frac_rows = [], [], [], None, []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            loss = float(m.group(2))
            used = int(m.group(3))
            total_codes = int(m.group(4))
            fracs = [float(x) for x in m.group(5).split(",")]

            steps.append(step)
            losses.append(loss)
            used_counts.append(used)
            num_codes = total_codes
            frac_rows.append(fracs)

    if not steps:
        raise ValueError(f"No matching log lines found in {path}")

    return (
        np.array(steps),
        np.array(losses),
        np.array(used_counts),
        num_codes,
        np.array(frac_rows),
    )


def plot_loss(steps, losses, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, color="#1f77b4")
    plt.xlabel("Training step")
    plt.ylabel("Loss (reconstruction + VQ)")
    plt.title("Latent Action Model — Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_codebook_usage(steps, frac_rows, num_codes, used_counts, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    colors = plt.cm.tab10(np.linspace(0, 1, num_codes))
    ax1.stackplot(
        steps,
        frac_rows.T,
        labels=[f"code {i}" for i in range(num_codes)],
        colors=colors,
        alpha=0.85,
    )
    ax1.set_ylabel("Usage fraction")
    ax1.set_title("Codebook Usage Over Training")
    ax1.legend(loc="upper right", ncol=4, fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, used_counts, marker="o", markersize=3, color="#d62728")
    ax2.axhline(num_codes, linestyle="--", color="gray", alpha=0.6, label=f"max ({num_codes})")
    ax2.set_ylabel("Active codes")
    ax2.set_xlabel("Training step")
    ax2.set_ylim(0, num_codes + 1)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lam_full_run.log"
    steps, losses, used_counts, num_codes, frac_rows = parse_log(log_path)

    print(f"Parsed {len(steps)} log points from {log_path}")
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Active codes: min={used_counts.min()}, max={used_counts.max()}, final={used_counts[-1]}")

    plot_loss(steps, losses, "loss.png")
    plot_codebook_usage(steps, frac_rows, num_codes, used_counts, "codebook_usage.png")


if __name__ == "__main__":
    main()
