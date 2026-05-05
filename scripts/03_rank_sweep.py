#!/usr/bin/env python3
"""
Phase 2: Rank Sweep -- All 6 Ranks, Seed 42 Only
=================================================
Sweep over LoRA ranks {1, 2, 4, 8, 16, 32} with seed=42 to characterize
how temporal separation varies with rank.

Uses the same SNLI+ChaosNLI combined training approach as the pilot:
  - Train on 20K SNLI + 1211 ChaosNLI combined
  - Track per-example losses only for ChaosNLI examples
  - AULC (Area Under Loss Curve) as primary metric

Skips rank/seed combinations that already have tracker files (from the
pilot or previous partial runs).

Usage:
    python scripts/03_rank_sweep.py
    python scripts/03_rank_sweep.py --ranks 1 2 8 16 32
    python scripts/03_rank_sweep.py --no-skip-existing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Import shared functions from pilot script
# --------------------------------------------------------------------------- #

def _import_pilot():
    """Import functions from the pilot experiment script (02_pilot_experiment.py).

    We reuse the pilot's data loading, model creation, training, and analysis
    functions rather than duplicating them.
    """
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Rank sweep (all ranks, single seed)."
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[4, 8, 1, 16, 2, 32],
        help="LoRA ranks to sweep, in desired execution order.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--eval-every-n-steps", type=int, default=100, help="Tracking interval.")
    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--loss-threshold", type=float, default=0.693, help="Learning time threshold.")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--model-name", type=str, default="roberta-base", help="Base model.")
    parser.add_argument(
        "--snli-size", type=int, default=20000,
        help="Number of SNLI training examples to subsample.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device.")
    parser.add_argument("--data-path", type=str, default=None, help="Processed ChaosNLI data path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--figure-dir", type=str, default=None, help="Figure directory.")
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip ranks with existing tracker files (default: True).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_false", dest="skip_existing",
        help="Force re-run even if tracker exists.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def tracker_exists(output_dir: Path, rank: int, seed: int) -> bool:
    """Check if a tracker file already exists for this (rank, seed)."""
    paths = [
        output_dir / f"sweep_r{rank}_s{seed}.json",
        output_dir / f"pilot_r{rank}_s{seed}.json",
    ]
    return any(p.exists() for p in paths)


def get_existing_tracker_path(output_dir: Path, rank: int, seed: int) -> Optional[Path]:
    """Return the path of an existing tracker file, or None."""
    for name in [f"sweep_r{rank}_s{seed}.json", f"pilot_r{rank}_s{seed}.json"]:
        p = output_dir / name
        if p.exists():
            return p
    return None


def load_existing_results(
    output_dir: Path, rank: int, seed: int, pilot_mod: Any,
) -> Dict[str, Any]:
    """Load an existing tracker and compute metrics for the summary table."""
    existing_path = get_existing_tracker_path(output_dir, rank, seed)
    tracker = TemporalTracker.load(existing_path)

    # AULC (primary)
    _, aulc_arr, aulc_ent = pilot_mod.compute_aulc(tracker)
    valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
    if valid.sum() >= 3:
        rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
    else:
        rho_aulc, p_aulc = 0.0, 1.0

    # Final loss
    _, final_arr, final_ent = pilot_mod.compute_final_loss(tracker)
    valid_f = np.isfinite(final_arr) & np.isfinite(final_ent)
    if valid_f.sum() >= 3:
        rho_final, p_final = stats.spearmanr(final_arr[valid_f], final_ent[valid_f])
    else:
        rho_final, p_final = 0.0, 1.0

    # Threshold (legacy)
    _, times_arr, entropies_arr = pilot_mod.compute_learning_times(tracker, threshold=0.693)
    rho_thresh, p_thresh = pilot_mod.compute_spearman_correlation(times_arr, entropies_arr)
    n_learned = int(np.isfinite(times_arr).sum())

    # Also load pilot_results if available for val_acc
    results_json = None
    for prefix in ["pilot_results", "sweep_results"]:
        rp = output_dir / f"{prefix}_r{rank}_s{seed}.json"
        if rp.exists():
            with open(rp) as f:
                results_json = json.load(f)
            break

    return {
        "rank": rank,
        "seed": seed,
        "aulc_rho": float(rho_aulc),
        "aulc_p": float(p_aulc),
        "final_loss_rho": float(rho_final),
        "final_loss_p": float(p_final),
        "threshold_rho": float(rho_thresh),
        "threshold_p": float(p_thresh),
        "n_learned": n_learned,
        "n_total": len(times_arr),
        "final_val_acc": results_json.get("final_val_accuracy") if results_json else None,
        "status": "skipped (existing)",
        "tracker_path": str(existing_path),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Use cached models/datasets
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    pilot = _import_pilot()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "tracking"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = pilot.detect_device(args.device)

    print("=" * 70)
    print("Phase 2: Rank Sweep (SNLI + ChaosNLI tracking)")
    print("=" * 70)
    print(f"  Ranks:    {args.ranks}")
    print(f"  Seed:     {args.seed}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.learning_rate}")
    print(f"  SNLI size: {args.snli_size}")
    print(f"  Device:   {device}")
    print(f"  Skip existing: {args.skip_existing}")
    print()

    # ------------------------------------------------------------------ #
    # Load data once (shared across all ranks)
    # ------------------------------------------------------------------ #
    print("Loading data...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load ChaosNLI data
    # Build a minimal namespace for _load_chaosnli_data
    chaosnli_args = argparse.Namespace(
        data_path=args.data_path,
        seed=args.seed,
    )
    chaosnli = pilot._load_chaosnli_data(chaosnli_args)

    tracking_premises = [chaosnli["premises"][i] for i in chaosnli["train_indices"]]
    tracking_hypotheses = [chaosnli["hypotheses"][i] for i in chaosnli["train_indices"]]
    tracking_labels = [chaosnli["majority_labels"][i] for i in chaosnli["train_indices"]]
    tracking_example_ids = [chaosnli["example_ids"][i] for i in chaosnli["train_indices"]]
    tracking_entropies = [chaosnli["entropies"][i] for i in chaosnli["train_indices"]]

    val_premises = [chaosnli["premises"][i] for i in chaosnli["val_indices"]]
    val_hypotheses = [chaosnli["hypotheses"][i] for i in chaosnli["val_indices"]]
    val_labels = [chaosnli["majority_labels"][i] for i in chaosnli["val_indices"]]
    val_example_ids = [chaosnli["example_ids"][i] for i in chaosnli["val_indices"]]
    val_entropies = [chaosnli["entropies"][i] for i in chaosnli["val_indices"]]

    print(f"  ChaosNLI train: {len(tracking_premises)}, val: {len(val_premises)}")

    # Load SNLI data
    snli = pilot._load_snli_data(n_examples=args.snli_size, seed=args.seed)
    snli_premises = snli["premises"]
    snli_hypotheses = snli["hypotheses"]
    snli_labels = snli["labels"]
    snli_example_ids = [f"snli_{i}" for i in range(len(snli_premises))]

    print(f"  SNLI train: {len(snli_premises)}")

    # Combine for training
    combined_premises = list(snli_premises) + tracking_premises
    combined_hypotheses = list(snli_hypotheses) + tracking_hypotheses
    combined_labels = list(snli_labels) + tracking_labels
    combined_example_ids = snli_example_ids + tracking_example_ids
    combined_entropies = [None] * len(snli_premises) + list(tracking_entropies)

    print(f"  Combined training set: {len(combined_premises)} examples")

    # Create datasets
    train_dataset = pilot.NLIDataset(
        premises=combined_premises,
        hypotheses=combined_hypotheses,
        labels=combined_labels,
        example_ids=combined_example_ids,
        entropies=combined_entropies,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    tracking_dataset = pilot.ChaosNLIDataset(
        premises=tracking_premises,
        hypotheses=tracking_hypotheses,
        labels=tracking_labels,
        example_ids=tracking_example_ids,
        entropies=tracking_entropies,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    val_dataset = pilot.ChaosNLIDataset(
        premises=val_premises,
        hypotheses=val_hypotheses,
        labels=val_labels,
        example_ids=val_example_ids,
        entropies=val_entropies,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    use_mps = device == "mps"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )
    tracking_loader = DataLoader(
        tracking_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )

    # Compute class weights once
    all_train_labels = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_train_labels, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"  Class weights: {class_weights.tolist()}")

    # ------------------------------------------------------------------ #
    # Run sweep
    # ------------------------------------------------------------------ #
    results_table = []

    for rank_idx, rank in enumerate(args.ranks):
        print(f"\n{'=' * 60}")
        print(f"  Rank {rank} [{rank_idx + 1}/{len(args.ranks)}]")
        print(f"{'=' * 60}")

        # Check if already completed
        if args.skip_existing and tracker_exists(output_dir, rank, args.seed):
            existing_path = get_existing_tracker_path(output_dir, rank, args.seed)
            print(f"  SKIPPED: Tracker already exists at {existing_path}")

            result = load_existing_results(output_dir, rank, args.seed, pilot)
            results_table.append(result)

            print(f"  Loaded: AULC rho={result['aulc_rho']:.4f}, "
                  f"p={result['aulc_p']:.2e}")
            continue

        # Fresh training run
        set_seed(args.seed)
        rank_t0 = time.time()

        # Create model for this rank
        model = pilot.create_lora_model(
            model_name=args.model_name,
            num_labels=3,
            rank=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.05,
        )

        # Initialize tracker
        tracker = TemporalTracker(loss_threshold=args.loss_threshold)
        tracker.register_examples(
            example_ids=tracking_example_ids,
            true_labels=tracking_labels,
            annotation_entropies=tracking_entropies,
        )

        # Train with the full SNLI+ChaosNLI pipeline
        history = pilot.train_with_tracking(
            model=model,
            train_loader=train_loader,
            tracking_loader=tracking_loader,
            val_loader=val_loader,
            tracker=tracker,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            eval_every_n_steps=args.eval_every_n_steps,
            device=device,
            max_grad_norm=1.0,
            class_weights=class_weights,
        )

        # Compute AULC correlation (primary)
        _, aulc_arr, aulc_ent = pilot.compute_aulc(tracker)
        valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
        if valid.sum() >= 3:
            rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
        else:
            rho_aulc, p_aulc = 0.0, 1.0

        # Final loss correlation
        _, final_arr, final_ent = pilot.compute_final_loss(tracker)
        valid_f = np.isfinite(final_arr) & np.isfinite(final_ent)
        if valid_f.sum() >= 3:
            rho_final, p_final = stats.spearmanr(final_arr[valid_f], final_ent[valid_f])
        else:
            rho_final, p_final = 0.0, 1.0

        # Threshold (legacy)
        _, times_arr, entropies_arr = pilot.compute_learning_times(
            tracker, threshold=args.loss_threshold,
        )
        rho_thresh, p_thresh = pilot.compute_spearman_correlation(times_arr, entropies_arr)
        n_learned = int(np.isfinite(times_arr).sum())

        rank_elapsed = time.time() - rank_t0

        # Save tracker
        tracker_path = output_dir / f"sweep_r{rank}_s{args.seed}.json"
        tracker.save(tracker_path)

        # Save per-rank results
        rank_results = {
            "rank": rank,
            "seed": args.seed,
            "aulc_rho": float(rho_aulc),
            "aulc_p": float(p_aulc),
            "final_loss_rho": float(rho_final),
            "final_loss_p": float(p_final),
            "threshold_rho": float(rho_thresh),
            "threshold_p": float(p_thresh),
            "n_learned": n_learned,
            "n_total": len(times_arr),
            "final_val_acc": history["val_accuracy"][-1] if history["val_accuracy"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "elapsed_seconds": rank_elapsed,
            "status": "completed",
            "tracker_path": str(tracker_path),
            "tracking_steps": history["tracking_steps"],
            "train_loss_history": history["train_loss"],
            "val_loss_history": history["val_loss"],
            "val_accuracy_history": history["val_accuracy"],
        }
        results_table.append(rank_results)

        # Save individual results JSON
        results_path = output_dir / f"sweep_results_r{rank}_s{args.seed}.json"
        with open(results_path, "w") as f:
            json.dump(rank_results, f, indent=2)

        print(f"\n  Rank {rank}: AULC rho={rho_aulc:+.4f} (p={p_aulc:.2e}), "
              f"final_loss rho={rho_final:+.4f}, "
              f"val_acc={history['val_accuracy'][-1]:.4f}, "
              f"time={rank_elapsed:.0f}s")

        # Generate hero figure for this rank
        pilot.plot_hero_figure(
            tracker=tracker,
            category_names=["clean", "ambiguous", "contested"],
            tracking_steps=history["tracking_steps"],
            output_path=figure_dir / f"hero_loss_curves_r{rank}_s{args.seed}.png",
            title_suffix=f" (rank={rank}, seed={args.seed})",
            loss_threshold=args.loss_threshold,
        )

        # Clean up memory between ranks
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("Phase 2 Summary: AULC Spearman rho vs. LoRA Rank")
    print(f"{'=' * 70}")
    print(f"{'Rank':>6} {'AULC rho':>10} {'p-value':>12} {'FinalL rho':>12} {'Val Acc':>10} {'Status':>20}")
    print("-" * 72)

    rhos_by_rank = []
    for r in sorted(results_table, key=lambda x: x["rank"]):
        rho = r["aulc_rho"]
        p_val = r["aulc_p"]
        rho_f = r.get("final_loss_rho", float("nan"))
        val_acc = r.get("final_val_acc")
        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        print(f"{r['rank']:>6} {rho:>10.4f} {p_val:>12.2e} {rho_f:>12.4f} {val_acc_str:>10} {r['status']:>20}")
        rhos_by_rank.append((r["rank"], rho))

    # Save sweep summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved sweep summary to {summary_path}")

    # ------------------------------------------------------------------ #
    # Gate check: does temporal separation decrease with rank?
    # ------------------------------------------------------------------ #
    sorted_rhos = sorted(rhos_by_rank, key=lambda x: x[0])
    ranks_sorted = [r for r, _ in sorted_rhos]
    rhos_sorted = [rho for _, rho in sorted_rhos]

    # Spearman correlation between rank and AULC rho
    if len(ranks_sorted) >= 3:
        rank_rho_corr, rank_rho_p = stats.spearmanr(ranks_sorted, rhos_sorted)
    else:
        rank_rho_corr, rank_rho_p = 0.0, 1.0

    elapsed = time.time() - t0
    print(f"\nPhase 2 complete ({elapsed:.1f}s)")
    print(f"\nSpearman(rank, AULC_rho) = {rank_rho_corr:.4f} (p = {rank_rho_p:.4f})")
    print(f"  Expected: negative (higher rank => weaker temporal separation)")

    if rank_rho_corr < -0.5:
        print(f"\nPHASE 2 GATE PASSED: Strong negative trend (r = {rank_rho_corr:.3f})")
        print("  Higher rank reduces temporal separation, as predicted by theory.")
        print("  Proceed to Phase 3 (multi-seed sweep).")
    elif rank_rho_corr < 0:
        print(f"\nPHASE 2 GATE MARGINAL: Negative but weak trend (r = {rank_rho_corr:.3f})")
        print("  Proceed with caution. Multi-seed sweep may clarify.")
    else:
        print(f"\nPHASE 2 GATE WARNING: Non-negative trend (r = {rank_rho_corr:.3f})")
        print("  Review per-rank results before proceeding to Phase 3.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
