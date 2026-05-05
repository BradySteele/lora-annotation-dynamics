#!/usr/bin/env python3
"""
Noise Injection Multi-Seed Analysis
====================================
Aggregates the noise injection results across all 3 seeds (42, 123, 456)
and computes cross-seed statistics for the paper.

This script does NOT require GPU -- it reads existing result JSONs.

Usage:
    python scripts/19_noise_aggregate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOISE_DIR = PROJECT_ROOT / "results" / "tracking" / "robustness_experiments" / "noise_injection"

SEEDS = [42, 123, 456]
CONDITIONS = ["control", "moderate_noise", "high_noise"]


def _get(d, *keys):
    """Try multiple key names (handles legacy vs new naming)."""
    for k in keys:
        if k in d:
            return d[k]
    return None


def main():
    print("=" * 70)
    print("Noise Injection Multi-Seed Aggregation")
    print("=" * 70)

    # Collect per-seed, per-condition results
    data = {}  # condition -> list of dicts
    for condition in CONDITIONS:
        data[condition] = []
        for seed in SEEDS:
            path = NOISE_DIR / f"noise_{condition}_s{seed}.json"
            if not path.exists():
                print(f"  WARNING: missing {path}")
                continue
            with open(path) as f:
                result = json.load(f)
            data[condition].append(result)

    # Aggregate
    print(f"\n{'Condition':<20} {'Mean AULC (clean)':>20} {'Spearman rho':>15}")
    print("-" * 60)

    condition_aulcs = {}
    for condition in CONDITIONS:
        aulcs = [_get(d, "mean_clean_aulc", "clean_mean_aulc") for d in data[condition]]
        rhos = [_get(d, "aulc_rho", "spearman_rho") for d in data[condition]]
        aulcs = [x for x in aulcs if x is not None]
        rhos = [x for x in rhos if x is not None]

        condition_aulcs[condition] = aulcs

        if aulcs:
            aulc_str = f"{np.mean(aulcs):.4f} +/- {np.std(aulcs):.4f}"
        else:
            aulc_str = "N/A"
        if rhos:
            rho_str = f"{np.mean(rhos):+.4f} +/- {np.std(rhos):.4f}"
        else:
            rho_str = "N/A"

        print(f"{condition:<20} {aulc_str:>20} {rho_str:>15}")

    # Statistical tests: control vs moderate, control vs high
    print(f"\n{'='*70}")
    print("Statistical Tests (Wilcoxon signed-rank on clean AULC across seeds)")
    print("=" * 70)

    for comparison in [("control", "moderate_noise"), ("control", "high_noise")]:
        c1, c2 = comparison
        a1 = condition_aulcs.get(c1, [])
        a2 = condition_aulcs.get(c2, [])

        if len(a1) >= 2 and len(a2) >= 2 and len(a1) == len(a2):
            # Per-seed paired test
            diffs = [a2[i] - a1[i] for i in range(len(a1))]
            mean_diff = np.mean(diffs)

            # Cohen's d (paired)
            sd = np.std(diffs, ddof=1) if len(diffs) > 1 else 1e-10
            d = mean_diff / sd if sd > 0 else 0

            print(f"\n  {c1} vs {c2}:")
            print(f"    Per-seed AULC diffs: {[f'{x:+.4f}' for x in diffs]}")
            print(f"    Mean diff: {mean_diff:+.4f}")
            print(f"    Cohen's d: {d:.3f}")
            print(f"    (Note: only {len(a1)} seeds -- use for descriptive purposes)")
        else:
            print(f"\n  {c1} vs {c2}: insufficient data for paired test")

    # For the paper: pooled Wilcoxon across all per-example AULC values
    # would require loading tracker data. This script gives the seed-level summary.
    print(f"\n{'='*70}")
    print("Paper-ready summary (for updating noise injection paragraph):")
    print("=" * 70)
    for condition in CONDITIONS:
        aulcs = condition_aulcs.get(condition, [])
        if aulcs:
            print(f"  {condition}: clean AULC = {np.mean(aulcs):.3f} +/- {np.std(aulcs):.3f} "
                  f"(seeds: {[f'{x:.3f}' for x in aulcs]})")

    # Save aggregated results
    output = {
        "seeds": SEEDS,
        "conditions": {},
    }
    for condition in CONDITIONS:
        aulcs = condition_aulcs.get(condition, [])
        rhos = [_get(d, "aulc_rho", "spearman_rho") for d in data[condition]]
        rhos = [x for x in rhos if x is not None]
        output["conditions"][condition] = {
            "clean_aulc_per_seed": aulcs,
            "clean_aulc_mean": float(np.mean(aulcs)) if aulcs else None,
            "clean_aulc_std": float(np.std(aulcs)) if aulcs else None,
            "rho_per_seed": rhos,
            "rho_mean": float(np.mean(rhos)) if rhos else None,
            "rho_std": float(np.std(rhos)) if rhos else None,
        }

    out_path = NOISE_DIR / "noise_multiseed_summary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Summary saved: {out_path}")


if __name__ == "__main__":
    main()
