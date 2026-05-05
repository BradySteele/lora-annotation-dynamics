#!/usr/bin/env python3
"""
Phase 0: Data Preparation and Validation
=========================================
Download/load ChaosNLI dataset, compute annotation entropy H_i for each
example, plot entropy distribution, compute difficulty proxies, create
stratified train/val split, and save processed data.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --subset snli --max-examples 5000
    python scripts/01_prepare_data.py --synthetic  # use synthetic data for development
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_entropy import (
    categorize_by_entropy,
    compute_annotation_entropy_from_distribution,
)
from src.data.chaosnli import create_synthetic_chaosnli, load_chaosnli
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Difficulty proxies
# --------------------------------------------------------------------------- #

def compute_difficulty_proxies(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Compute difficulty proxies for each example.

    These are used to control for confounds: we want to show that entropy
    predicts learning time above and beyond simple difficulty measures.

    Returns:
        Dictionary mapping proxy name to array of shape (n_examples,).
    """
    proxies = {}

    # Sentence length (premise + hypothesis in tokens, approximated by whitespace)
    premise_lengths = np.array([len(p.split()) for p in data["premises"]], dtype=np.float64)
    hyp_lengths = np.array([len(h.split()) for h in data["hypotheses"]], dtype=np.float64)
    proxies["premise_length"] = premise_lengths
    proxies["hypothesis_length"] = hyp_lengths
    proxies["total_length"] = premise_lengths + hyp_lengths

    # Majority vote confidence (max probability in label distribution)
    label_dists = data["label_distributions"].astype(np.float64)
    row_sums = label_dists.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    probs = label_dists / row_sums
    proxies["majority_confidence"] = probs.max(axis=1)

    # Label distribution skewness
    proxies["label_skewness"] = np.array([
        float(stats.skew(row)) if row.sum() > 0 else 0.0
        for row in label_dists
    ])

    return proxies


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_entropy_distribution(
    entropies: np.ndarray,
    categories: np.ndarray,
    category_names: List[str],
    thresholds: List[float],
    output_path: Path,
) -> None:
    """Plot entropy distribution with category boundaries.

    Creates a publication-quality histogram of H_i values with vertical
    lines at the category thresholds and per-category counts.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Color palette safe for colorblind readers
    colors = ["#2166AC", "#F4A582", "#B2182B"]  # blue, peach, red
    category_colors = {name: colors[i] for i, name in enumerate(category_names)}

    # Histogram
    bins = np.linspace(0, max(entropies.max(), np.log(3) + 0.1), 50)
    for cat_idx, cat_name in enumerate(category_names):
        mask = categories == cat_idx
        if mask.sum() > 0:
            ax.hist(
                entropies[mask],
                bins=bins,
                alpha=0.7,
                label=f"{cat_name} (n={mask.sum()})",
                color=colors[cat_idx],
                edgecolor="white",
                linewidth=0.5,
            )

    # Threshold lines
    for thresh in thresholds:
        ax.axvline(
            thresh, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
            label=f"threshold = {thresh:.1f}",
        )

    # Max entropy reference line
    max_h = np.log(3)
    ax.axvline(
        max_h, color="gray", linestyle=":", linewidth=1.0, alpha=0.5,
        label=f"max H = log(3) = {max_h:.3f}",
    )

    ax.set_xlabel("Annotation Entropy $H_i$ (nats)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Annotation Entropy (ChaosNLI)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved entropy distribution plot to {output_path}")


# --------------------------------------------------------------------------- #
# Stratified splitting
# --------------------------------------------------------------------------- #

def create_stratified_split(
    n_examples: int,
    categories: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stratified train/val split by entropy category.

    Ensures each category is proportionally represented in both splits.

    Args:
        n_examples: Total number of examples.
        categories: Integer category per example.
        val_fraction: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_indices, val_indices) as numpy arrays.
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed,
    )
    indices = np.arange(n_examples)
    train_idx, val_idx = next(splitter.split(indices, categories))
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# Gate check
# --------------------------------------------------------------------------- #

def check_entropy_distribution(
    entropies: np.ndarray,
    categories: np.ndarray,
    category_names: List[str],
) -> bool:
    """Check if the entropy distribution is suitable for the experiment.

    A degenerate distribution (e.g., all examples in one category) would
    make the temporal separation hypothesis untestable.

    Returns:
        True if the distribution passes the gate check.
    """
    passed = True

    # Check that each category has at least 5% of examples
    n = len(entropies)
    for cat_idx, name in enumerate(category_names):
        count = (categories == cat_idx).sum()
        fraction = count / n
        if fraction < 0.05:
            print(f"  WARNING: Category '{name}' has only {count} examples ({fraction:.1%})")
            passed = False

    # Check that entropy range is reasonable
    entropy_range = entropies.max() - entropies.min()
    if entropy_range < 0.3:
        print(f"  WARNING: Entropy range is very narrow ({entropy_range:.3f})")
        passed = False

    # Check for degenerate distribution (very low variance)
    if entropies.std() < 0.1:
        print(f"  WARNING: Entropy std is very low ({entropies.std():.3f})")
        passed = False

    return passed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: Data preparation and validation for ChaosNLI."
    )
    parser.add_argument(
        "--subset", type=str, default="snli",
        choices=["snli", "mnli"],
        help="ChaosNLI subset to use (default: snli).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to manually downloaded ChaosNLI data.",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Maximum number of examples to load (for debugging).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of real ChaosNLI (for development).",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=1500,
        help="Number of synthetic examples to generate (default: 1500).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of data for validation (default: 0.2).",
    )
    parser.add_argument(
        "--entropy-thresholds", type=float, nargs="+", default=[0.4, 0.7],
        help="Entropy thresholds for categorization (default: 0.4 0.7, paper setting).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/data/).",
    )
    parser.add_argument(
        "--figure-dir", type=str, default=None,
        help="Figure directory (default: figures/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    set_seed(args.seed)

    # Resolve output directories relative to project root
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "data"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 0: Data Preparation and Validation")
    print("=" * 70)
    print(f"  Subset:       {args.subset}")
    print(f"  Synthetic:    {args.synthetic}")
    print(f"  Seed:         {args.seed}")
    print(f"  Val fraction: {args.val_fraction}")
    print(f"  Thresholds:   {args.entropy_thresholds}")
    print(f"  Output dir:   {output_dir}")
    print()

    # ------------------------------------------------------------------ #
    # Step 1: Load data
    # ------------------------------------------------------------------ #
    print("Step 1: Loading data...")

    if args.synthetic:
        print("  Using synthetic ChaosNLI data for development.")
        data = create_synthetic_chaosnli(
            n_examples=args.n_synthetic,
            n_annotators=100,
            seed=args.seed,
        )
    else:
        print(f"  Loading real ChaosNLI data (subset={args.subset})...")
        data = load_chaosnli(
            subset=args.subset,
            data_dir=args.data_dir,
            max_examples=args.max_examples,
        )

    n_examples = len(data["premises"])
    print(f"  Loaded {n_examples} examples.")

    # ------------------------------------------------------------------ #
    # Step 2: Compute annotation entropy
    # ------------------------------------------------------------------ #
    print("\nStep 2: Computing annotation entropy...")

    entropies = np.array([
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ])

    print(f"  Mean entropy:   {entropies.mean():.4f}")
    print(f"  Std entropy:    {entropies.std():.4f}")
    print(f"  Min entropy:    {entropies.min():.4f}")
    print(f"  Max entropy:    {entropies.max():.4f}")
    print(f"  Median entropy: {np.median(entropies):.4f}")

    # ------------------------------------------------------------------ #
    # Step 3: Categorize by entropy
    # ------------------------------------------------------------------ #
    print("\nStep 3: Categorizing examples by entropy...")

    cats = categorize_by_entropy(entropies, thresholds=args.entropy_thresholds)

    for name, count in cats.counts.items():
        frac = count / n_examples
        mean_h = cats.mean_entropy_per_category[name]
        print(f"  {name:12s}: {count:5d} ({frac:5.1%}), mean H = {mean_h:.4f}")

    # ------------------------------------------------------------------ #
    # Step 4: Plot entropy distribution
    # ------------------------------------------------------------------ #
    print("\nStep 4: Plotting entropy distribution...")

    plot_entropy_distribution(
        entropies=entropies,
        categories=cats.categories,
        category_names=cats.category_names,
        thresholds=args.entropy_thresholds,
        output_path=figure_dir / "entropy_distribution.png",
    )

    # ------------------------------------------------------------------ #
    # Step 5: Compute difficulty proxies
    # ------------------------------------------------------------------ #
    print("\nStep 5: Computing difficulty proxies...")

    proxies = compute_difficulty_proxies(data)

    for proxy_name, proxy_vals in proxies.items():
        print(f"  {proxy_name:25s}: mean={proxy_vals.mean():.2f}, std={proxy_vals.std():.2f}")

    # ------------------------------------------------------------------ #
    # Step 6: Check entropy-difficulty correlation
    # ------------------------------------------------------------------ #
    print("\nStep 6: Checking entropy-difficulty correlations...")

    correlation_flag = False
    for proxy_name, proxy_vals in proxies.items():
        rho, p = stats.spearmanr(entropies, proxy_vals)
        flag = " *** FLAGGED (r > 0.5)" if abs(rho) > 0.5 else ""
        print(f"  Spearman(H, {proxy_name:25s}): rho = {rho:+.4f}, p = {p:.2e}{flag}")
        if abs(rho) > 0.5:
            correlation_flag = True

    if correlation_flag:
        print("\n  WARNING: High correlation between entropy and difficulty proxy detected.")
        print("  Will need partial correlation analysis to control for this confound.")
    else:
        print("\n  No strong entropy-difficulty correlations detected. Good.")

    # ------------------------------------------------------------------ #
    # Step 7: Create stratified train/val split
    # ------------------------------------------------------------------ #
    print("\nStep 7: Creating stratified train/val split...")

    train_idx, val_idx = create_stratified_split(
        n_examples=n_examples,
        categories=cats.categories,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"  Train: {len(train_idx)} examples")
    print(f"  Val:   {len(val_idx)} examples")

    # Verify stratification
    for name_idx, name in enumerate(cats.category_names):
        train_frac = (cats.categories[train_idx] == name_idx).mean()
        val_frac = (cats.categories[val_idx] == name_idx).mean()
        print(f"  {name:12s}: train={train_frac:.1%}, val={val_frac:.1%}")

    # ------------------------------------------------------------------ #
    # Step 8: Save processed data
    # ------------------------------------------------------------------ #
    print("\nStep 8: Saving processed data...")

    processed = {
        "premises": data["premises"],
        "hypotheses": data["hypotheses"],
        "label_distributions": data["label_distributions"].tolist(),
        "majority_labels": data["majority_labels"].tolist(),
        "example_ids": data["example_ids"],
        "entropies": entropies.tolist(),
        "entropy_categories": cats.categories.tolist(),
        "category_names": cats.category_names,
        "entropy_thresholds": args.entropy_thresholds,
        "category_counts": cats.counts,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "difficulty_proxies": {k: v.tolist() for k, v in proxies.items()},
        "metadata": {
            "subset": args.subset,
            "synthetic": args.synthetic,
            "n_examples": n_examples,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        },
    }

    save_path = output_dir / "processed_chaosnli.json"
    with open(save_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"  Saved to {save_path}")

    # Also save a compact version with just the essential arrays for training
    compact = {
        "example_ids": data["example_ids"],
        "premises": data["premises"],
        "hypotheses": data["hypotheses"],
        "majority_labels": data["majority_labels"].tolist(),
        "entropies": entropies.tolist(),
        "entropy_categories": cats.categories.tolist(),
        "category_names": cats.category_names,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "metadata": {
            "subset": args.subset,
            "synthetic": args.synthetic,
            "n_examples": n_examples,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "seed": args.seed,
        },
    }
    compact_path = output_dir / "train_data.json"
    with open(compact_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"  Saved compact version to {compact_path}")

    # ------------------------------------------------------------------ #
    # Gate check
    # ------------------------------------------------------------------ #
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 0 complete ({elapsed:.1f}s)")

    gate_passed = check_entropy_distribution(
        entropies, cats.categories, cats.category_names
    )

    if gate_passed:
        print("\nPHASE 0 GATE: entropy distribution looks suitable")
        print("  All categories have sufficient representation.")
        print("  Entropy range and variance are adequate.")
    else:
        print("\nPHASE 0 GATE WARNING: entropy distribution may be degenerate")
        print("  Review the distribution plot and consider adjusting thresholds.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
