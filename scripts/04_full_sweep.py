#!/usr/bin/env python3
"""
Phase 3: Full Experiment -- 6 Ranks x 5 Seeds
==============================================
Run the complete experiment grid: all LoRA ranks crossed with all random
seeds. Supports resumption by checking for existing tracker files.

Usage:
    python scripts/04_full_sweep.py
    python scripts/04_full_sweep.py --ranks 4 8 --seeds 42 123
    python scripts/04_full_sweep.py --synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed


def _import_pilot():
    """Import functions from the pilot experiment script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Full experiment (all ranks x all seeds)."
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
        help="LoRA ranks to train.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
        help="Random seeds.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--eval-every-n-steps", type=int, default=50, help="Tracking interval.")
    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--loss-threshold", type=float, default=0.693, help="Threshold.")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--model-name", type=str, default="roberta-base", help="Base model.")
    parser.add_argument("--device", type=str, default=None, help="Device.")
    parser.add_argument("--data-path", type=str, default=None, help="Processed data path.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data.")
    parser.add_argument("--n-synthetic", type=int, default=800, help="Synthetic example count.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--figure-dir", type=str, default=None, help="Figure directory.")
    return parser.parse_args()


def find_existing_tracker(
    output_dir: Path, rank: int, seed: int,
) -> Optional[Path]:
    """Check if a tracker file already exists for this (rank, seed).

    Searches across all naming conventions (full, sweep, pilot).

    Returns:
        Path to existing tracker, or None.
    """
    candidates = [
        output_dir / f"full_r{rank}_s{seed}.json",
        output_dir / f"sweep_r{rank}_s{seed}.json",
        output_dir / f"pilot_r{rank}_s{seed}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    args = parse_args()
    t0 = time.time()

    pilot = _import_pilot()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "tracking"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures"
    analysis_dir = PROJECT_ROOT / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = pilot.detect_device(args.device)

    all_combos = [(r, s) for r in args.ranks for s in args.seeds]
    total_runs = len(all_combos)

    print("=" * 70)
    print("Phase 3: Full Experiment")
    print("=" * 70)
    print(f"  Ranks:    {args.ranks}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Total:    {total_runs} combinations")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.learning_rate}")
    print(f"  Device:   {device}")
    print()

    # Check how many already exist
    existing = []
    pending = []
    for rank, seed in all_combos:
        path = find_existing_tracker(output_dir, rank, seed)
        if path is not None:
            existing.append((rank, seed, path))
        else:
            pending.append((rank, seed))

    print(f"  Already completed: {len(existing)} / {total_runs}")
    print(f"  Pending:           {len(pending)} / {total_runs}")

    if len(pending) == 0:
        print("\n  All combinations already completed. Skipping to analysis.")
    else:
        print()

    # ------------------------------------------------------------------ #
    # Load data once (shared across all runs)
    # ------------------------------------------------------------------ #
    print("Loading data...")
    # Use seed=42 for data loading (deterministic split)
    args_copy = argparse.Namespace(**vars(args))
    args_copy.seed = 42
    (premises, hypotheses, example_ids, labels, entropies,
     train_indices, val_indices) = pilot.load_data_for_training(args_copy)

    print(f"  Total examples: {len(premises)}")
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = pilot.ChaosNLIDataset(
        premises=[premises[i] for i in train_indices],
        hypotheses=[hypotheses[i] for i in train_indices],
        labels=[labels[i] for i in train_indices],
        example_ids=[example_ids[i] for i in train_indices],
        entropies=[entropies[i] for i in train_indices],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = pilot.ChaosNLIDataset(
        premises=[premises[i] for i in val_indices],
        hypotheses=[hypotheses[i] for i in val_indices],
        labels=[labels[i] for i in val_indices],
        example_ids=[example_ids[i] for i in val_indices],
        entropies=[entropies[i] for i in val_indices],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # ------------------------------------------------------------------ #
    # Run pending combinations
    # ------------------------------------------------------------------ #
    import torch
    from torch.utils.data import DataLoader
    from src.training.temporal_tracker import TemporalTracker

    completed = 0
    failed = 0

    for run_idx, (rank, seed) in enumerate(pending):
        print(f"\n{'=' * 60}")
        print(f"  Run {run_idx + 1}/{len(pending)}: rank={rank}, seed={seed}")
        print(f"{'=' * 60}")

        set_seed(seed)

        # Create data loaders with the specific seed for shuffling
        use_mps = device == "mps"

        # Create a generator with the specific seed for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0 if use_mps else 2, pin_memory=not use_mps,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=0 if use_mps else 2, pin_memory=not use_mps,
        )

        try:
            run_t0 = time.time()

            model = pilot.create_lora_model(
                model_name=args.model_name,
                num_labels=3,
                rank=rank,
                lora_alpha=2 * rank,
                lora_dropout=0.05,
            )

            tracker = TemporalTracker(loss_threshold=args.loss_threshold)
            train_example_ids = [example_ids[i] for i in train_indices]
            train_entropies = [entropies[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]

            tracker.register_examples(
                example_ids=train_example_ids,
                true_labels=train_labels,
                annotation_entropies=train_entropies,
            )

            history = pilot.train_with_tracking(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                tracker=tracker,
                n_epochs=args.epochs,
                learning_rate=args.learning_rate,
                eval_every_n_steps=args.eval_every_n_steps,
                device=device,
            )

            # Compute correlation
            _, times_arr, entropies_arr = pilot.compute_learning_times(
                tracker, threshold=args.loss_threshold,
            )
            rho, p_val = pilot.compute_spearman_correlation(times_arr, entropies_arr)
            n_learned = int(np.isfinite(times_arr).sum())

            run_elapsed = time.time() - run_t0

            # Save tracker
            tracker_path = output_dir / f"full_r{rank}_s{seed}.json"
            tracker.save(tracker_path)

            print(f"  rank={rank}, seed={seed}: rho={rho:.4f}, p={p_val:.2e}, "
                  f"learned={n_learned}/{len(times_arr)}, time={run_elapsed:.0f}s")

            completed += 1

            # Clean up
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: rank={rank}, seed={seed}: {e}")
            failed += 1

    # ------------------------------------------------------------------ #
    # Aggregate all results (including previously existing ones)
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("Aggregating all results...")
    print(f"{'=' * 70}")

    from src.training.temporal_tracker import TemporalTracker

    all_results = []
    for rank in sorted(args.ranks):
        for seed in args.seeds:
            tracker_path = find_existing_tracker(output_dir, rank, seed)
            if tracker_path is None:
                print(f"  WARNING: No tracker for rank={rank}, seed={seed}")
                continue

            tracker = TemporalTracker.load(tracker_path)
            _, times_arr, entropies_arr = pilot.compute_learning_times(
                tracker, threshold=args.loss_threshold,
            )
            rho, p_val = pilot.compute_spearman_correlation(times_arr, entropies_arr)
            n_learned = int(np.isfinite(times_arr).sum())

            all_results.append({
                "rank": rank,
                "seed": seed,
                "spearman_rho": rho,
                "spearman_p": p_val,
                "n_learned": n_learned,
                "n_total": len(times_arr),
                "tracker_path": str(tracker_path),
            })

    # Compute mean +/- std per rank
    print(f"\n{'Rank':>6} {'mean rho':>10} {'std rho':>10} {'min rho':>10} {'max rho':>10} {'n_seeds':>8}")
    print("-" * 60)

    summary_per_rank = {}
    for rank in sorted(args.ranks):
        rank_results = [r for r in all_results if r["rank"] == rank]
        if not rank_results:
            print(f"{rank:>6} {'N/A':>10}")
            continue

        rhos = [r["spearman_rho"] for r in rank_results]
        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos, ddof=1) if len(rhos) > 1 else 0.0

        print(f"{rank:>6} {mean_rho:>10.4f} {std_rho:>10.4f} "
              f"{min(rhos):>10.4f} {max(rhos):>10.4f} {len(rhos):>8}")

        summary_per_rank[rank] = {
            "mean_rho": float(mean_rho),
            "std_rho": float(std_rho),
            "min_rho": float(min(rhos)),
            "max_rho": float(max(rhos)),
            "n_seeds": len(rhos),
            "per_seed": rank_results,
        }

    # Save full sweep summary
    full_summary = {
        "ranks": sorted(args.ranks),
        "seeds": args.seeds,
        "loss_threshold": args.loss_threshold,
        "per_rank": summary_per_rank,
        "all_results": all_results,
        "completed_new": completed,
        "failed": failed,
        "reused_existing": len(existing),
    }

    summary_path = analysis_dir / "full_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)
    print(f"\nSaved full sweep summary to {summary_path}")

    elapsed = time.time() - t0
    print(f"\nPhase 3 complete ({elapsed:.1f}s)")
    print(f"  Completed: {completed} new runs")
    print(f"  Reused:    {len(existing)} existing runs")
    print(f"  Failed:    {failed} runs")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
