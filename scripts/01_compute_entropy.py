#!/usr/bin/env python3
"""
Script 01: Compute Annotation Entropy
======================================
Pre-compute per-example annotation entropy for all datasets.
Saves entropy values and category assignments for downstream analysis.

Usage:
    python scripts/01_compute_entropy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.annotation_entropy import (
    categorize_by_entropy,
    compute_annotation_entropy_batch,
    compute_annotation_entropy_from_distribution,
)
from src.data.chaosnli import create_synthetic_chaosnli
from src.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    output_dir = Path("results/entropy")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic ChaosNLI data...")
    data = create_synthetic_chaosnli(n_examples=1000, n_annotators=100, seed=42)

    entropies = np.array([
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ])

    print(f"  Mean entropy: {entropies.mean():.4f}")
    print(f"  Std entropy:  {entropies.std():.4f}")
    print(f"  Min entropy:  {entropies.min():.4f}")
    print(f"  Max entropy:  {entropies.max():.4f}")

    cats = categorize_by_entropy(entropies, thresholds=[0.4, 0.7])
    print(f"\n  Category counts: {cats.counts}")
    print(f"  Mean entropy per category: {cats.mean_entropy_per_category}")

    results = {
        "n_examples": len(entropies),
        "entropies": entropies.tolist(),
        "categories": cats.categories.tolist(),
        "category_names": cats.category_names,
        "counts": cats.counts,
        "mean_entropy_per_category": cats.mean_entropy_per_category,
    }
    with open(output_dir / "chaosnli_entropy.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir / 'chaosnli_entropy.json'}")


if __name__ == "__main__":
    main()
