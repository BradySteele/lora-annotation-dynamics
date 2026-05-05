#!/usr/bin/env python3
"""
Post-hoc analysis for decoder experiments.

Loads the per-run JSON result files produced by 16_decoder_experiments.py,
prints the summary table, saves summary.json, generates the summary bar
chart, and prints the cross-architecture comparison with encoder baselines.

Usage:
    python scripts/16b_decoder_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results" / "tracking" / "decoder_experiments"
FIGURE_DIR = PROJECT_ROOT / "figures" / "decoder_experiments"


def load_results() -> List[Dict[str, Any]]:
    """Load all per-run result JSON files (excluding tracker files)."""
    results = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        if path.name.endswith("_tracker.json") or path.name == "summary.json":
            continue
        with open(path) as f:
            results.append(json.load(f))
    return results


def plot_summary_figure(
    all_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a summary bar chart of AULC-entropy rho across all runs."""
    groups: Dict[Tuple[str, str], List[float]] = {}
    for r in all_results:
        key = (r["model_short"], r["config"])
        groups.setdefault(key, []).append(r["aulc_rho"])

    if not groups:
        return

    labels = []
    means = []
    stds = []
    for (model, config), rhos in sorted(groups.items()):
        labels.append(f"{model}\n{config}")
        means.append(np.mean(rhos))
        stds.append(np.std(rhos) if len(rhos) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0",
                  edgecolor="black", alpha=0.85, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AULC-Entropy Spearman $\\rho$", fontsize=11)
    ax.set_title("Decoder-Only Models: AULC-Entropy Correlation", fontsize=12)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{m:.3f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary figure: {output_path}")


def print_cross_architecture_comparison(
    decoder_results: List[Dict[str, Any]],
) -> None:
    """Print comparison table including encoder baselines if available."""
    encoder_summary_paths = [
        PROJECT_ROOT / "results" / "tracking" / "expanded" / "summary.json",
        PROJECT_ROOT / "results" / "tracking" / "deberta_extended" / "summary.json",
    ]

    encoder_rhos: Dict[str, List[float]] = {}
    for path in encoder_summary_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for r in data:
                        model = r.get("model", r.get("model_name", "unknown"))
                        rho = r.get("aulc_rho", r.get("spearman_aulc_rho"))
                        if rho is not None:
                            encoder_rhos.setdefault(model, []).append(float(rho))
            except (json.JSONDecodeError, KeyError):
                pass

    if not encoder_rhos and not decoder_results:
        return

    print(f"\n{'=' * 70}")
    print("Cross-Architecture Comparison (Encoder vs Decoder)")
    print(f"{'=' * 70}")

    if encoder_rhos:
        print("\n  Encoder models (from existing experiments):")
        for model, rhos in sorted(encoder_rhos.items()):
            mean_rho = np.mean(rhos)
            if len(rhos) > 1:
                print(f"    {model}: rho={mean_rho:+.4f} +/- {np.std(rhos):.4f} "
                      f"(n={len(rhos)} runs)")
            else:
                print(f"    {model}: rho={mean_rho:+.4f} (n=1)")
    else:
        print("\n  No encoder baseline results found for comparison.")

    if decoder_results:
        print("\n  Decoder models (this experiment):")
        groups: Dict[str, List[float]] = {}
        for r in decoder_results:
            key = r.get("model_short", r.get("model", "?"))
            if "aulc_rho" in r:
                groups.setdefault(key, []).append(r["aulc_rho"])
        for model, rhos in sorted(groups.items()):
            mean_rho = np.mean(rhos)
            if len(rhos) > 1:
                print(f"    {model}: rho={mean_rho:+.4f} +/- {np.std(rhos):.4f} "
                      f"(n={len(rhos)} runs)")
            else:
                print(f"    {model}: rho={mean_rho:+.4f} (n=1)")


def main() -> None:
    all_results = load_results()
    if not all_results:
        print("No result files found in", RESULT_DIR)
        sys.exit(1)

    print(f"Loaded {len(all_results)} result files.\n")

    # ---- Summary table ----
    print(f"{'=' * 70}")
    print("Decoder-Only Experiments: Summary")
    print(f"{'=' * 70}\n")

    header = (
        f"{'Model':>10} {'Config':>6} {'Seed':>6} "
        f"{'AULC rho':>10} {'p-value':>12} "
        f"{'Kendall':>8} {'Val Acc':>8} "
        f"{'dl clean':>10} {'dl contest':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        delta_c = (
            f"{r.get('mean_delta_ell_clean', 0):+.4f}"
            if r.get("mean_delta_ell_clean") is not None
            else "N/A"
        )
        delta_t = (
            f"{r.get('mean_delta_ell_contested', 0):+.4f}"
            if r.get("mean_delta_ell_contested") is not None
            else "N/A"
        )
        val_acc = (
            f"{r['final_val_acc']:.4f}"
            if r.get("final_val_acc") is not None
            else "N/A"
        )
        print(
            f"{r.get('model_short', '?'):>10} "
            f"{r.get('config', '?'):>6} "
            f"{r.get('seed', '?'):>6} "
            f"{r.get('aulc_rho', 0):>10.4f} "
            f"{r.get('aulc_p', 1.0):>12.2e} "
            f"{r.get('kendall_tau', 0):>8.4f} "
            f"{val_acc:>8} "
            f"{delta_c:>10} "
            f"{delta_t:>10}"
        )

    # ---- Per-model/config aggregation ----
    print()
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in all_results:
        key = (r.get("model_short", "?"), r.get("config", "?"))
        groups.setdefault(key, []).append(r)

    for (model_short, config), runs in sorted(groups.items()):
        if len(runs) > 1:
            rhos = [r["aulc_rho"] for r in runs if "aulc_rho" in r]
            taus = [r["kendall_tau"] for r in runs if "kendall_tau" in r]
            accs = [r["final_val_acc"] for r in runs
                    if r.get("final_val_acc") is not None]
            if rhos:
                print(
                    f"  {model_short} {config}: "
                    f"rho={np.mean(rhos):+.4f} +/- {np.std(rhos):.4f}, "
                    f"tau={np.mean(taus):+.4f} +/- {np.std(taus):.4f}"
                    + (f", val_acc={np.mean(accs):.4f}" if accs else "")
                )

    # ---- Save summary.json ----
    summary_path = RESULT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved summary to {summary_path}")

    # ---- Summary figure ----
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary_figure(all_results, FIGURE_DIR / "decoder_summary.png")

    # ---- Cross-architecture comparison ----
    print_cross_architecture_comparison(all_results)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
