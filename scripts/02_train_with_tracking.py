#!/usr/bin/env python3
"""
Script 02: Train with Temporal Tracking
========================================
Train LoRA models across multiple ranks and seeds while recording
per-example losses at every epoch.

Usage:
    python scripts/02_train_with_tracking.py
"""

from __future__ import annotations

from pathlib import Path

from src.utils.seed import set_seed


def main() -> None:
    ranks = [1, 2, 4, 8, 16, 32]
    seeds = [42, 123, 456, 789, 1024]
    output_dir = Path("results/tracking")
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank in ranks:
        for seed in seeds:
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Training: rank={rank}, seed={seed}")
            print(f"{'='*60}")

            print("  [STUB] Training not yet implemented.")

    print("\nDone. See results/tracking/ for tracker outputs.")


if __name__ == "__main__":
    main()
