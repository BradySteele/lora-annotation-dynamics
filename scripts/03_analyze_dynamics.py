#!/usr/bin/env python3
"""
Script 03: Analyze Learning Dynamics
=====================================
Analyze temporal separation between clean and contested examples.
Produces the main results for the paper:
    - Correlation between H_i and learning time t_i (Table 1)
    - Temporal separation gap across ranks (Figure 2)
    - Bias-variance-interaction decomposition (Figure 3)

Usage:
    python scripts/03_analyze_dynamics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.theory.bias_variance import decompose_across_ranks
from src.theory.temporal_separation import (
    compute_separation_gap,
    entropy_learning_time_correlation,
)


def demo_theoretical_predictions() -> None:
    """Demonstrate theoretical predictions with synthetic data."""

    print("="*60)
    print("Demo: Bias-Variance-Interaction Decomposition")
    print("="*60)

    ranks = [1, 2, 4, 8, 16, 32]

    # Low entropy dataset (clean)
    print("\n--- Low entropy (clean data, H_bar = 0.2) ---")
    decomps_clean = decompose_across_ranks(
        ranks=ranks,
        n_samples=5000,
        entropy=0.2,
        bayes_error=0.05,
        d_model=768,
        sigma_noise=0.3,
        gradient_variance=0.5,
        spectral_decay_rate=1.0,
    )
    for d in decomps_clean:
        print(f"  r={d.rank:3d}: bias^2={d.bias_squared:.4f}  "
              f"var={d.variance:.4f}  C(r,H)={d.interaction_C:.4f}  "
              f"total={d.total_error:.4f}")

    # High entropy dataset (contested)
    print("\n--- High entropy (contested data, H_bar = 1.0) ---")
    decomps_contested = decompose_across_ranks(
        ranks=ranks,
        n_samples=5000,
        entropy=1.0,
        bayes_error=0.15,
        d_model=768,
        sigma_noise=0.8,
        gradient_variance=1.5,
        spectral_decay_rate=1.0,
    )
    for d in decomps_contested:
        print(f"  r={d.rank:3d}: bias^2={d.bias_squared:.4f}  "
              f"var={d.variance:.4f}  C(r,H)={d.interaction_C:.4f}  "
              f"total={d.total_error:.4f}")

    print("\n" + "="*60)
    print("Demo: Temporal Separation Test")
    print("="*60)

    # Simulate learning times
    rng = np.random.RandomState(42)
    clean_times = rng.exponential(scale=2.0, size=200)    # learned early
    contested_times = rng.exponential(scale=5.0, size=200) # learned late

    result = compute_separation_gap(clean_times, contested_times)
    print(f"\n  {result}")

    # Correlation analysis
    entropies = np.concatenate([
        rng.uniform(0, 0.5, size=200),    # clean
        rng.uniform(0.5, 1.1, size=200),  # contested
    ])
    times = np.concatenate([clean_times, contested_times])

    for method in ["spearman", "kendall"]:
        corr, p = entropy_learning_time_correlation(entropies, times, method)
        print(f"  {method.capitalize()} correlation: rho={corr:.4f}, p={p:.6f}")


if __name__ == "__main__":
    demo_theoretical_predictions()
