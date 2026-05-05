#!/usr/bin/env python3
"""
Phase 5: Generate Publication Figures
======================================
Generate all figures for the paper, formatted for ACL 2026 proceedings.

Figures:
    1. Per-category loss curves at best rank (hero figure)
    2. Spearman rho vs. LoRA rank with 5-seed error bars
    3. Scatter of learning time vs. entropy at rank 4
    4. Learning order consistency heatmap across seeds (optional)

All figures are saved as both PDF (vector, for paper) and PNG (raster,
for quick preview).

Usage:
    python scripts/06_generate_figures.py
    python scripts/06_generate_figures.py --analysis-dir results/analysis/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.temporal_tracker import TemporalTracker


# --------------------------------------------------------------------------- #
# ACL figure styling
# --------------------------------------------------------------------------- #

# ACL column width is 3.25 inches; full page width is 6.75 inches
ACL_COLUMN_WIDTH = 3.25
ACL_FULL_WIDTH = 6.75

# Color palette: colorblind-safe, print-friendly
COLORS = {
    "clean": "#2166AC",       # blue
    "ambiguous": "#F4A582",   # peach/salmon
    "contested": "#B2182B",   # red
    "primary": "#2166AC",     # blue
    "secondary": "#B2182B",   # red
    "tertiary": "#4DAF4A",    # green
    "gray": "#666666",
}

MARKERS = {
    "clean": "o",
    "ambiguous": "s",
    "contested": "^",
}


def apply_acl_style() -> None:
    """Apply publication-quality matplotlib styling for ACL papers."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


def save_figure(fig: plt.Figure, path: Path, name: str) -> None:
    """Save figure in both PDF and PNG formats."""
    path.mkdir(parents=True, exist_ok=True)

    pdf_path = path / f"{name}.pdf"
    png_path = path / f"{name}.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# --------------------------------------------------------------------------- #
# Tracker loading (shared with 05_analyze_results.py)
# --------------------------------------------------------------------------- #

def discover_trackers(tracking_dir: Path) -> Dict[Tuple[int, int], Path]:
    """Discover all tracker files and parse rank/seed from filenames."""
    pattern = re.compile(r"(?:full|sweep|pilot)_r(\d+)_s(\d+)\.json")
    trackers = {}
    for p in sorted(tracking_dir.glob("*.json")):
        match = pattern.match(p.name)
        if match:
            rank = int(match.group(1))
            seed = int(match.group(2))
            key = (rank, seed)
            if key not in trackers or p.name.startswith("full"):
                trackers[key] = p
    return trackers


def load_tracker_with_times(
    path: Path, threshold: float = 0.693,
) -> Tuple[TemporalTracker, np.ndarray, np.ndarray, np.ndarray]:
    """Load tracker and extract learning times and entropies."""
    tracker = TemporalTracker.load(path)
    ids, times, entropies = [], [], []
    for eid, record in tracker.records.items():
        t = tracker.get_learning_time(eid, threshold=threshold)
        ids.append(eid)
        times.append(float(t) if t is not None else np.inf)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )
    return tracker, np.array(ids), np.array(times), np.array(entropies)


# --------------------------------------------------------------------------- #
# Figure 1: Per-category loss curves (hero figure)
# --------------------------------------------------------------------------- #

def figure_1_hero_loss_curves(
    tracker_map: Dict[Tuple[int, int], Path],
    figure_dir: Path,
    preferred_rank: int = 4,
    preferred_seed: int = 42,
    threshold: float = 0.693,
) -> None:
    """Figure 1: Per-category mean loss curves over training.

    This is the hero figure that visually demonstrates the core finding:
    clean examples are learned before contested ones under LoRA's rank
    constraint.
    """
    print("\nFigure 1: Per-category loss curves (hero figure)")

    # Find the best tracker for this figure
    key = (preferred_rank, preferred_seed)
    if key not in tracker_map:
        # Fall back to any available tracker at the preferred rank
        fallbacks = [(r, s) for (r, s) in tracker_map if r == preferred_rank]
        if fallbacks:
            key = fallbacks[0]
        else:
            key = next(iter(tracker_map))
        print(f"  Using fallback: rank={key[0]}, seed={key[1]}")

    tracker = TemporalTracker.load(tracker_map[key])
    rank, seed = key

    # Group examples by entropy category
    categories = {"clean": [], "ambiguous": [], "contested": []}
    for eid, record in tracker.records.items():
        h = record.annotation_entropy
        if h is None:
            continue
        if h < 0.5:
            categories["clean"].append(eid)
        elif h < 1.0:
            categories["ambiguous"].append(eid)
        else:
            categories["contested"].append(eid)

    mean_losses = tracker.get_mean_loss_by_category(categories)

    fig, ax = plt.subplots(figsize=(ACL_COLUMN_WIDTH, 2.5))

    for cat_name in ["clean", "ambiguous", "contested"]:
        if cat_name not in mean_losses or len(mean_losses[cat_name]) == 0:
            continue
        losses = mean_losses[cat_name]
        steps = np.arange(len(losses))
        n_ex = len(categories[cat_name])

        ax.plot(
            steps, losses,
            color=COLORS[cat_name],
            marker=MARKERS[cat_name],
            markersize=3,
            linewidth=1.5,
            markevery=max(1, len(steps) // 10),
            label=f"{cat_name} ($n$={n_ex})",
            alpha=0.9,
        )

    ax.axhline(
        threshold, color=COLORS["gray"], linestyle="--", linewidth=0.8,
        label=r"$\theta = -\ln(0.5)$",
    )

    ax.set_xlabel("Tracking Step")
    ax.set_ylabel("Mean Cross-Entropy Loss")
    ax.set_title(f"Learning Dynamics by Entropy Category (rank={rank})")
    ax.legend(loc="upper right", framealpha=0.9)

    save_figure(fig, figure_dir, "fig1_hero_loss_curves")


# --------------------------------------------------------------------------- #
# Figure 2: Spearman rho vs rank with error bars
# --------------------------------------------------------------------------- #

def figure_2_spearman_vs_rank(
    tracker_map: Dict[Tuple[int, int], Path],
    figure_dir: Path,
    threshold: float = 0.693,
) -> None:
    """Figure 2: Spearman rho vs LoRA rank with 5-seed error bars.

    Shows that temporal separation (measured by Spearman correlation)
    decreases with increasing rank, as predicted by the theory.
    """
    print("\nFigure 2: Spearman rho vs. LoRA rank")

    # Compute rho for every (rank, seed) combination
    by_rank: Dict[int, List[float]] = {}
    for (rank, seed), path in sorted(tracker_map.items()):
        _, _, times, entropies = load_tracker_with_times(path, threshold)
        valid = np.isfinite(times) & np.isfinite(entropies)
        if valid.sum() >= 3:
            rho, _ = stats.spearmanr(times[valid], entropies[valid])
        else:
            rho = 0.0
        if rank not in by_rank:
            by_rank[rank] = []
        by_rank[rank].append(float(rho))

    ranks = sorted(by_rank.keys())
    means = [np.mean(by_rank[r]) for r in ranks]
    stds = [np.std(by_rank[r], ddof=1) if len(by_rank[r]) > 1 else 0.0 for r in ranks]

    fig, ax = plt.subplots(figsize=(ACL_COLUMN_WIDTH, 2.5))

    ax.errorbar(
        ranks, means, yerr=stds,
        color=COLORS["primary"],
        marker="o",
        markersize=5,
        linewidth=1.5,
        capsize=3,
        capthick=1,
        label="Spearman $\\rho$ (mean $\\pm$ std)",
    )

    # Individual seed points as faint dots
    for rank in ranks:
        for rho_val in by_rank[rank]:
            ax.scatter(
                rank, rho_val,
                color=COLORS["primary"],
                alpha=0.2,
                s=15,
                zorder=1,
            )

    ax.axhline(0, color=COLORS["gray"], linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("LoRA Rank $r$")
    ax.set_ylabel("Spearman $\\rho$(learning time, entropy)")
    ax.set_title("Temporal Separation vs. LoRA Rank")
    ax.set_xticks(ranks)
    ax.legend(loc="upper right", framealpha=0.9)

    save_figure(fig, figure_dir, "fig2_spearman_vs_rank")


# --------------------------------------------------------------------------- #
# Figure 3: Scatter of learning time vs entropy
# --------------------------------------------------------------------------- #

def figure_3_scatter_time_vs_entropy(
    tracker_map: Dict[Tuple[int, int], Path],
    figure_dir: Path,
    preferred_rank: int = 4,
    preferred_seed: int = 42,
    threshold: float = 0.693,
) -> None:
    """Figure 3: Scatter plot of learning time vs annotation entropy.

    Each point is one training example. Shows the positive correlation
    between entropy and learning time that underlies the Spearman rho.
    """
    print("\nFigure 3: Learning time vs. entropy scatter")

    key = (preferred_rank, preferred_seed)
    if key not in tracker_map:
        fallbacks = [(r, s) for (r, s) in tracker_map if r == preferred_rank]
        key = fallbacks[0] if fallbacks else next(iter(tracker_map))

    _, _, times, entropies = load_tracker_with_times(tracker_map[key], threshold)
    rank, seed = key

    # Only plot finite learning times
    valid = np.isfinite(times) & np.isfinite(entropies)
    t_valid = times[valid]
    h_valid = entropies[valid]

    # Color by category
    cat_colors = np.array([COLORS["gray"]] * len(h_valid))
    for i in range(len(h_valid)):
        if h_valid[i] < 0.5:
            cat_colors[i] = COLORS["clean"]
        elif h_valid[i] < 1.0:
            cat_colors[i] = COLORS["ambiguous"]
        else:
            cat_colors[i] = COLORS["contested"]

    fig, ax = plt.subplots(figsize=(ACL_COLUMN_WIDTH, 2.5))

    # Plot each category separately for legend
    for cat_name, color in [("clean", COLORS["clean"]),
                             ("ambiguous", COLORS["ambiguous"]),
                             ("contested", COLORS["contested"])]:
        if cat_name == "clean":
            mask = h_valid < 0.5
        elif cat_name == "ambiguous":
            mask = (h_valid >= 0.5) & (h_valid < 1.0)
        else:
            mask = h_valid >= 1.0

        if mask.sum() > 0:
            ax.scatter(
                h_valid[mask], t_valid[mask],
                c=color, alpha=0.4, s=8,
                label=f"{cat_name} ($n$={mask.sum()})",
                marker=MARKERS[cat_name],
                edgecolors="none",
            )

    # Add regression line
    if len(h_valid) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(h_valid, t_valid)
        x_line = np.linspace(h_valid.min(), h_valid.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line,
            color="black", linewidth=1, linestyle="--",
            alpha=0.7,
            label=f"OLS ($r$={r_value:.3f})",
        )

    # Spearman annotation
    rho, p = stats.spearmanr(h_valid, t_valid)
    ax.annotate(
        f"$\\rho_s$ = {rho:.3f}\n$p$ = {p:.1e}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        ha="left", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.set_xlabel("Annotation Entropy $H_i$ (nats)")
    ax.set_ylabel("Learning Time $t_i$ (tracking step)")
    ax.set_title(f"Learning Time vs. Entropy (rank={rank})")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    save_figure(fig, figure_dir, "fig3_scatter_time_entropy")


# --------------------------------------------------------------------------- #
# Figure 4: Learning order consistency heatmap (optional)
# --------------------------------------------------------------------------- #

def figure_4_consistency_heatmap(
    tracker_map: Dict[Tuple[int, int], Path],
    figure_dir: Path,
    threshold: float = 0.693,
) -> None:
    """Figure 4: Pairwise rank-correlation heatmap of learning orders.

    For each pair of seeds at a given rank, compute Spearman correlation
    of learning orders. High values mean the learning order is stable
    across seeds.
    """
    print("\nFigure 4: Learning order consistency heatmap")

    # Group by rank
    by_rank: Dict[int, Dict[int, np.ndarray]] = {}
    by_rank_ids: Dict[int, Dict[int, np.ndarray]] = {}
    for (rank, seed), path in sorted(tracker_map.items()):
        _, ids_arr, times, _ = load_tracker_with_times(path, threshold)
        if rank not in by_rank:
            by_rank[rank] = {}
            by_rank_ids[rank] = {}
        by_rank[rank][seed] = times
        by_rank_ids[rank][seed] = ids_arr

    # Pick one rank with the most seeds
    best_rank = max(by_rank.keys(), key=lambda r: len(by_rank[r]))
    seeds = sorted(by_rank[best_rank].keys())

    if len(seeds) < 2:
        print("  Skipping: need at least 2 seeds for heatmap.")
        return

    # Find common examples
    common_ids = set(by_rank_ids[best_rank][seeds[0]])
    for s in seeds[1:]:
        common_ids &= set(by_rank_ids[best_rank][s])
    common_ids = sorted(common_ids)

    if len(common_ids) < 10:
        print(f"  Skipping: only {len(common_ids)} common examples.")
        return

    # Build time vectors for common examples
    seed_time_vecs = {}
    for s in seeds:
        id_to_time = dict(zip(by_rank_ids[best_rank][s], by_rank[best_rank][s]))
        times = np.array([id_to_time[eid] for eid in common_ids])
        times = np.where(np.isinf(times), 1e6, times)
        seed_time_vecs[s] = stats.rankdata(times)

    # Pairwise Spearman correlation
    n_seeds = len(seeds)
    corr_matrix = np.ones((n_seeds, n_seeds))
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            rho, _ = stats.spearmanr(seed_time_vecs[seeds[i]], seed_time_vecs[seeds[j]])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    fig, ax = plt.subplots(figsize=(ACL_COLUMN_WIDTH, ACL_COLUMN_WIDTH * 0.8))

    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, aspect="equal")

    # Annotate cells
    for i in range(n_seeds):
        for j in range(n_seeds):
            text_color = "white" if corr_matrix[i, j] > 0.7 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xticks(range(n_seeds))
    ax.set_yticks(range(n_seeds))
    seed_labels = [f"s={s}" for s in seeds]
    ax.set_xticklabels(seed_labels, fontsize=7)
    ax.set_yticklabels(seed_labels, fontsize=7)
    ax.set_title(f"Learning Order Consistency (rank={best_rank})")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman $\\rho$ of learning order", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    save_figure(fig, figure_dir, "fig4_consistency_heatmap")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: Generate publication figures."
    )
    parser.add_argument(
        "--tracking-dir", type=str, default=None,
        help="Directory containing tracker JSON files.",
    )
    parser.add_argument(
        "--analysis-dir", type=str, default=None,
        help="Directory containing analysis JSON files.",
    )
    parser.add_argument(
        "--figure-dir", type=str, default=None,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--loss-threshold", type=float, default=0.693,
        help="Loss threshold for learning time computation.",
    )
    parser.add_argument(
        "--preferred-rank", type=int, default=4,
        help="Preferred rank for single-rank figures.",
    )
    parser.add_argument(
        "--preferred-seed", type=int, default=42,
        help="Preferred seed for single-seed figures.",
    )
    parser.add_argument(
        "--skip-fig4", action="store_true",
        help="Skip generating Figure 4 (consistency heatmap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    tracking_dir = Path(args.tracking_dir) if args.tracking_dir else PROJECT_ROOT / "results" / "tracking"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 5: Generate Publication Figures")
    print("=" * 70)
    print(f"  Tracking dir: {tracking_dir}")
    print(f"  Figure dir:   {figure_dir}")
    print()

    apply_acl_style()

    # Discover trackers
    tracker_map = discover_trackers(tracking_dir)

    if not tracker_map:
        print("  ERROR: No tracker files found. Run experiments first.")
        return

    ranks = sorted(set(r for r, _ in tracker_map))
    seeds = sorted(set(s for _, s in tracker_map))
    print(f"  Found {len(tracker_map)} trackers: ranks={ranks}, seeds={seeds}")

    # Generate all figures
    figure_1_hero_loss_curves(
        tracker_map, figure_dir,
        preferred_rank=args.preferred_rank,
        preferred_seed=args.preferred_seed,
        threshold=args.loss_threshold,
    )

    figure_2_spearman_vs_rank(
        tracker_map, figure_dir,
        threshold=args.loss_threshold,
    )

    figure_3_scatter_time_vs_entropy(
        tracker_map, figure_dir,
        preferred_rank=args.preferred_rank,
        preferred_seed=args.preferred_seed,
        threshold=args.loss_threshold,
    )

    if not args.skip_fig4:
        figure_4_consistency_heatmap(
            tracker_map, figure_dir,
            threshold=args.loss_threshold,
        )

    elapsed = time.time() - t0
    print(f"\nPhase 5 complete ({elapsed:.1f}s)")

    # List generated files
    print("\nGenerated figures:")
    for ext in ["pdf", "png"]:
        for f in sorted(figure_dir.glob(f"fig*.{ext}")):
            print(f"  {f}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
