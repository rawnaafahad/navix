#!/usr/bin/env python3
"""
Aggregates held-out evaluation CSVs (from evaluate_policy.py) across seeds and
tracks, and plots a success-rate comparison bar chart with 95% CIs.

Usage:
    python3 plot_ladder_results.py

Expects files matching: eval_<track>_100seed_v2_seed<N>.csv (or similar)
in the current directory. Adjust TRACK_PATTERNS below if your filenames
differ from the ones used in the handoff doc / prior commands.

Each CSV is expected to have per-episode rows with a 'success' column
(True/False or 1/0). If evaluate_policy.py's CSV uses different column
names, edit SUCCESS_COL below.
"""

import glob
import re
import csv
import math
import statistics
import sys

SUCCESS_COL_CANDIDATES = ["success", "success_bool", "is_success"]

# BUGFIX: the previous single-pattern-per-track approach (e.g.
# "eval_handcoded_100seed_seed*.csv") also matched *_stochastic.csv files,
# silently pooling deterministic and stochastic evaluations of the SAME
# checkpoint together as if they were independent seeds -- this corrupted
# both n_seeds and the resulting mean/CI for hand-coded and discovered
# (original). Deterministic and stochastic patterns are now kept separate,
# and PRIMARY_PROTOCOL below states, per track, which one is the number
# actually reported -- matching the decision already written into
# main.tex's Methodology ("Deterministic-vs-stochastic evaluation bug"
# paragraph): hand-coded reports stochastic as primary; every other track
# reports deterministic as primary (matches evaluate_policy.py's default
# protocol, and for discovered-original the stochastic/deterministic gap
# was small and non-significant when checked, so deterministic stays
# primary there).

# Map: display name -> (deterministic glob, stochastic glob or None)
# The [0-9] character class deliberately excludes "_stochastic" suffixes,
# unlike a bare seed*.csv wildcard.
TRACK_PATTERNS = {
    "discovered (v2)": ("eval_100seed_v2_seed[0-9].csv", None),
    "discovered (v2 corrected)": ("eval_discoveredv2_wd_seed[0-9].csv", None),
    "primitive baseline": ("eval_baseline_100seed_seed[0-9].csv", None),
    "hand-coded options": ("eval_handcoded_100seed_seed[0-9].csv",
                            "eval_handcoded_100seed_seed[0-9]_stochastic.csv"),
    "random grounding": ("eval_randomgrounding_100seed_seed[0-9].csv", None),
    "discovered (original)": ("eval_discoveredorig_100seed_seed[0-9].csv",
                               "eval_discoveredorig_100seed_seed[0-9]_stochastic.csv"),
}

# Tracks whose seeds must NOT be pooled together into one Wilson CI, because
# main.tex's Methodology already commits to reporting them separately (e.g.
# hand-coded: seed 0's 34% recovery and seed 1's genuine 0% collapse are
# mechanistically distinct findings, not noise around one true rate --
# pooling them would silently average over that distinction).
# Left-to-right order for the bar chart, per reviewer feedback: the four
# pooled ladder conditions in ascending pipeline order, with discovered-v2's
# original and corrected grounding shown as adjacent bars so the correction's
# effect is directly visible, hand-coded appended separately at the end.
LADDER_ORDER = [
    "primitive baseline",
    "random grounding",
    "discovered (original)",
    "discovered (v2)",
    "discovered (v2 corrected)",
]

NO_POOL_TRACKS = {"hand-coded options"}

# Which protocol's numbers are the one actually reported/plotted per track.
# Falls back to "deterministic" if no stochastic files exist for a track.
PRIMARY_PROTOCOL = {
    "discovered (v2)": "deterministic",
    "discovered (v2 corrected)": "deterministic",
    "primitive baseline": "deterministic",
    "hand-coded options": "stochastic",
    "random grounding": "deterministic",
    "discovered (original)": "deterministic",
}

# Reviewer feedback (correct point): the unit of independent replication for
# a trained RL policy is the training seed, not the held-out episode -- 250
# pooled episodes across 5 seeds are NOT 250 independent experiments, they
# are 5, each evaluated 50 times. Wherever enough seeds exist for a
# defensible across-seed CI (t-distribution needs a sane df), that is now
# the number PLOTTED. Pooled-episode Wilson CI remains available (printed,
# and usable in the table) as a fallback for the 2-seed conditions where an
# across-seed CI's df is too low to defend on its own.
MIN_SEEDS_FOR_ACROSS_SEED_CI = 3


def find_success_col(fieldnames):
    for cand in SUCCESS_COL_CANDIDATES:
        if cand in fieldnames:
            return cand
    # fall back: any column containing 'success'
    for f in fieldnames:
        if "success" in f.lower():
            return f
    raise ValueError(f"No success column found in {fieldnames}")


def parse_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("true", "1", "yes")


def per_seed_success_rate(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        col = find_success_col(reader.fieldnames)
        vals = [parse_bool(row[col]) for row in reader]
    if not vals:
        return None
    return sum(vals) / len(vals)


def raw_success_values(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        col = find_success_col(reader.fieldnames)
        return [parse_bool(row[col]) for row in reader]


def wilson_ci(successes, n, z=1.96):
    # Same formula as evaluate_policy.py / evaluate_policy_stochastic.py's
    # wilson_ci -- kept in sync deliberately, since this is the number
    # that should match what those scripts already printed per-seed.
    if n == 0:
        return 0.0, (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, (max(0.0, center - margin), min(1.0, center + margin))


def t_multiplier_95(df):
    # Two-sided 95% t-multipliers for small df (1..10); falls back to 1.96 (z) beyond that.
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 1.96)


def aggregate_track(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    rates = []
    for fp in files:
        r = per_seed_success_rate(fp)
        if r is not None:
            rates.append(r)
    if not rates:
        return None
    n = len(rates)
    mean = statistics.mean(rates)
    sd = statistics.stdev(rates) if n > 1 else 0.0
    if n > 1:
        se = sd / math.sqrt(n)
        margin = t_multiplier_95(n - 1) * se
    else:
        margin = 0.0

    # Pooled-episode Wilson CI, computed over every individual episode across
    # all matched files for this track -- NOT per-seed. With only 2-5 seeds
    # per track, the across-seed t-based CI above has very low df (as low as
    # df=1) and produces intervals that can extend below 0% or above 100%,
    # which is not statistically defensible as "significant" evidence. The
    # pooled Wilson CI (same formula evaluate_policy.py already uses
    # per-seed) is the number that should actually be reported/plotted.
    pooled_vals = []
    for fp in files:
        pooled_vals.extend(raw_success_values(fp))
    pooled_n = len(pooled_vals)
    pooled_successes = sum(pooled_vals)
    pooled_p, (pooled_lo, pooled_hi) = wilson_ci(pooled_successes, pooled_n)

    return {
        "files": files,
        "n_seeds": n,
        "per_seed_rates": rates,
        "mean": mean,
        "sd": sd,
        "ci95_margin": margin,
        "pooled_n": pooled_n,
        "pooled_successes": pooled_successes,
        "pooled_p": pooled_p,
        "pooled_ci": (pooled_lo, pooled_hi),
    }


def main():
    results = {}
    both_protocols = {}  # name -> {"deterministic": agg or None, "stochastic": agg or None}
    for name, (det_pattern, stoch_pattern) in TRACK_PATTERNS.items():
        det_agg = aggregate_track(det_pattern)
        stoch_agg = aggregate_track(stoch_pattern) if stoch_pattern else None
        both_protocols[name] = {"deterministic": det_agg, "stochastic": stoch_agg}

        primary = PRIMARY_PROTOCOL.get(name, "deterministic")
        chosen = stoch_agg if (primary == "stochastic" and stoch_agg is not None) else det_agg
        if chosen is not None:
            results[name] = chosen

    if not results:
        print("No matching eval CSVs found. Check TRACK_PATTERNS against your actual filenames.")
        sys.exit(1)

    print("=== Ladder summary ===")
    print("(headline number = across-seed t-CI when n_seeds >= 3, since the")
    print(" training seed -- not the held-out episode -- is the correct unit")
    print(" of independent replication for a trained policy; pooled-episode")
    print(" Wilson CI is shown for transparency and used as the fallback for")
    print(" 2-seed conditions, where an across-seed CI's df is too low to")
    print(" defend on its own)\n")
    for name, agg in results.items():
        primary = PRIMARY_PROTOCOL.get(name, "deterministic")
        if name in NO_POOL_TRACKS:
            print(f"{name:24s} [{primary} -- primary, NOT POOLED per main.tex Methodology]")
            for fp, rate in zip(agg["files"], agg["per_seed_rates"]):
                print(f"{'':24s}   {fp}: {rate:.3f}")
            print(f"{'':24s}   (report these per-seed in Results/Discussion -- "
                  f"do not average them into one figure)")
            continue
        use_across_seed = agg["n_seeds"] >= MIN_SEEDS_FOR_ACROSS_SEED_CI
        seed_lo, seed_hi = agg['mean'] - agg['ci95_margin'], agg['mean'] + agg['ci95_margin']
        lo, hi = agg["pooled_ci"]
        headline_tag = "ACROSS-SEED -- plotted" if use_across_seed else "POOLED-EPISODE -- plotted"
        print(f"{name:24s} [{primary}] [{headline_tag}]")
        flag = "  <-- out of [0,1], not usable as-is" if (seed_lo < 0 or seed_hi > 1) else ""
        print(f"{'':24s} across-seed (t, df={agg['n_seeds']-1}): n_seeds={agg['n_seeds']} "
              f"mean={agg['mean']:.3f} 95% CI=[{seed_lo:.3f}, {seed_hi:.3f}]{flag} "
              f"per-seed={['%.3f' % r for r in agg['per_seed_rates']]}")
        print(f"{'':24s} pooled-episode (Wilson): "
              f"{agg['pooled_successes']}/{agg['pooled_n']} = {agg['pooled_p']:.3f} "
              f"(95% CI=[{lo:.3f}, {hi:.3f}])")
        other = "stochastic" if primary == "deterministic" else "deterministic"
        other_agg = both_protocols[name][other]
        if other_agg is not None:
            o_lo, o_hi = other_agg["pooled_ci"]
            print(f"{'':24s} [{other} -- not plotted] pooled: "
                  f"{other_agg['pooled_successes']}/{other_agg['pooled_n']} = {other_agg['pooled_p']:.3f} "
                  f"(95% Wilson CI=[{o_lo:.3f}, {o_hi:.3f}])")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping plot. "
              "Install with: pip install matplotlib --break-system-packages")
        return

    # Build one flat list of bars to draw, in ladder order, with hand-coded's
    # two seeds appended at the end as clearly-marked separate bars rather
    # than silently omitted (as the first version of this plot did) or
    # incorrectly averaged into one bar (which would hide the 34%-vs-0%
    # finding the whole point of the Methodology section is to preserve).
    bars = []  # list of dicts: label, value, lo, hi, n, color, hatch
    POOLED_COLOR = "#4C72B0"
    NOPOOL_COLOR = "#DD8452"

    for name in LADDER_ORDER:
        if name not in results:
            continue
        agg = results[name]
        use_across_seed = agg["n_seeds"] >= MIN_SEEDS_FOR_ACROSS_SEED_CI
        if use_across_seed:
            value = agg["mean"]
            lo_margin = min(agg["ci95_margin"], value)
            hi_margin = min(agg["ci95_margin"], 1.0 - value)
            ci_note = "across-seed t-CI"
        else:
            lo, hi = agg["pooled_ci"]
            value = agg["pooled_p"]
            lo_margin = value - lo
            hi_margin = hi - value
            ci_note = "pooled-episode Wilson CI"
        bars.append({
            "label": f"{name}\n(n={agg['n_seeds']}, {ci_note.replace('across-seed t-CI', 'seed CI').replace('pooled-episode Wilson CI', 'pooled CI')})",
            "value": value,
            "lo": lo_margin,
            "hi": hi_margin,
            "color": POOLED_COLOR,
            "hatch": None,
        })

    for name in NO_POOL_TRACKS:
        if name not in results:
            continue
        agg = results[name]
        for fp, rate in zip(agg["files"], agg["per_seed_rates"]):
            seed_label = fp.split("seed")[-1].split("_")[0].split(".")[0]
            n_eps = 50  # matches this project's fixed held-out protocol
            successes = round(rate * n_eps)
            _, (lo, hi) = wilson_ci(successes, n_eps)
            bars.append({
                "label": f"hand-coded\nseed {seed_label} (not pooled)",
                "value": rate,
                "lo": rate - lo,
                "hi": hi - rate,
                "color": NOPOOL_COLOR,
                "hatch": "//",
            })

    if not bars:
        print("\nNothing to plot.")
        return

    labels = [b["label"] for b in bars]
    values = [b["value"] for b in bars]
    lo_margins = [b["lo"] for b in bars]
    hi_margins = [b["hi"] for b in bars]
    colors = [b["color"] for b in bars]
    hatches = [b["hatch"] for b in bars]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = list(range(len(bars)))
    bar_containers = ax.bar(x, values, yerr=[lo_margins, hi_margins], capsize=5,
                             color=colors, edgecolor="black", linewidth=0.6)
    for rect, hatch in zip(bar_containers, hatches):
        if hatch:
            rect.set_hatch(hatch)

    # A visual gap + divider between the pooled ladder and the not-pooled
    # hand-coded bars, so the chart itself communicates that hand-coded is
    # being shown differently, not just the legend.
    n_pooled = sum(1 for b in bars if b["hatch"] is None)
    if 0 < n_pooled < len(bars):
        ax.axvline(n_pooled - 0.5, color="grey", linestyle=":", linewidth=1)

    # Value + CI label above each bar, matching NAVIX's own convention of
    # printing the actual number on/near each bar rather than leaving the
    # reader to read it off the axis.
    for xi, b in zip(x, bars):
        top = b["value"] + b["hi"]
        ax.text(xi, top + 0.012, f"{b['value']*100:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7.5)
    ax.set_ylabel("Held-out success rate")
    ax.set_title("Ablation ladder: held-out success rate by track (100-dungeon protocol)")

    y_top = max(b["value"] + b["hi"] for b in bars)
    ax.set_ylim(0, min(1.0, y_top * 1.35 + 0.03))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=POOLED_COLOR, edgecolor="black",
              label=f"$\\geq${MIN_SEEDS_FOR_ACROSS_SEED_CI} seeds: across-seed t-CI; "
                    f"<{MIN_SEEDS_FOR_ACROSS_SEED_CI}: pooled-episode Wilson CI"),
        Patch(facecolor=NOPOOL_COLOR, edgecolor="black", hatch="//",
              label="Hand-coded: reported per-seed, not pooled (see Methodology)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()

    out_path = "ladder_success_rate.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()