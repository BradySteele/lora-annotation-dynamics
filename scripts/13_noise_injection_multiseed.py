#!/usr/bin/env python3
"""
Noise Injection Multi-Seed Extension
=====================================
Runs the noise injection experiment on seeds 123 and 456 (seed 42 already
exists in results/tracking/robustness_experiments/noise_injection/).

Uses the same setup as 10_robustness_experiments.py --noise-injection.

Usage:
    python scripts/13_noise_injection_multiseed.py
    python scripts/13_noise_injection_multiseed.py --seeds 123 456
    python scripts/13_noise_injection_multiseed.py --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed

ENTROPY_LOW = 0.4
ENTROPY_HIGH = 0.7
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking" / "robustness_experiments" / "noise_injection"
FIGURE_DIR = PROJECT_ROOT / "figures" / "robustness_experiments" / "noise_injection"


def _import_robustness():
    spec = importlib.util.spec_from_file_location(
        "robustness", str(PROJECT_ROOT / "scripts" / "10_robustness_experiments.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_pilot():
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(description="Noise injection multi-seed.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456],
                        help="Seeds to run (42 already done).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    robustness = _import_robustness()
    pilot = _import_pilot()
    device = args.device or pilot.detect_device(None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load data (use seed 42 for data split consistency)
    chaosnli = robustness.load_chaosnli_data(subset="snli", seed=42)
    bulk = robustness.load_bulk_training_data(dataset="snli", n_examples=20000, seed=42)

    train_idx = chaosnli["train_indices"]
    cn_premises = [chaosnli["premises"][i] for i in train_idx]
    cn_hypotheses = [chaosnli["hypotheses"][i] for i in train_idx]
    cn_labels = [chaosnli["majority_labels"][i] for i in train_idx]
    cn_eids = [chaosnli["example_ids"][i] for i in train_idx]
    cn_entropies = [chaosnli["entropies"][i] for i in train_idx]

    val_premises = [chaosnli["premises"][i] for i in chaosnli["val_indices"]]
    val_hypotheses = [chaosnli["hypotheses"][i] for i in chaosnli["val_indices"]]
    val_labels = [chaosnli["majority_labels"][i] for i in chaosnli["val_indices"]]
    val_eids = [chaosnli["example_ids"][i] for i in chaosnli["val_indices"]]
    val_entropies = [chaosnli["entropies"][i] for i in chaosnli["val_indices"]]

    clean_mask = np.array(cn_entropies) < ENTROPY_LOW
    clean_indices = np.where(clean_mask)[0]

    noise_levels = {
        "control": 0.0,
        "moderate_noise": 0.3,
        "high_noise": 0.6,
    }

    print("=" * 70)
    print("Noise Injection Multi-Seed Extension")
    print("=" * 70)
    print(f"  Seeds: {args.seeds}")
    print(f"  Clean examples: {len(clean_indices)}")
    print(f"  Device: {device}")

    all_results = []

    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        seed_results = {}

        for condition_name, noise_frac in noise_levels.items():
            run_id = f"noise_{condition_name}_s{seed}"
            result_path = OUTPUT_DIR / f"{run_id}.json"

            if result_path.exists() and not args.force:
                print(f"\n  Skipping {run_id} (exists).")
                with open(result_path) as f:
                    seed_results[condition_name] = json.load(f)
                continue

            print(f"\n  Running: {run_id} (noise_frac={noise_frac})")
            set_seed(seed)

            # Create noisy labels
            noisy_cn_labels = list(cn_labels)
            if noise_frac > 0:
                n_to_flip = int(len(clean_indices) * noise_frac)
                flip_indices = rng.choice(clean_indices, size=n_to_flip, replace=False)
                for idx in flip_indices:
                    original_label = cn_labels[idx]
                    other_labels = [l for l in range(3) if l != original_label]
                    noisy_cn_labels[idx] = rng.choice(other_labels)
                print(f"  Flipped {n_to_flip}/{len(clean_indices)} clean labels")

            combined_premises = list(bulk["premises"]) + cn_premises
            combined_hypotheses = list(bulk["hypotheses"]) + cn_hypotheses
            combined_labels = list(bulk["labels"]) + noisy_cn_labels
            combined_eids = [f"snli_{i}" for i in range(len(bulk["premises"]))] + cn_eids
            combined_entropies = [None] * len(bulk["premises"]) + cn_entropies

            train_dataset = pilot.NLIDataset(
                premises=combined_premises, hypotheses=combined_hypotheses,
                labels=combined_labels, example_ids=combined_eids,
                entropies=combined_entropies, tokenizer=tokenizer, max_length=128,
            )
            tracking_dataset = pilot.ChaosNLIDataset(
                premises=cn_premises, hypotheses=cn_hypotheses,
                labels=cn_labels, example_ids=cn_eids,
                entropies=cn_entropies, tokenizer=tokenizer, max_length=128,
            )
            val_dataset = pilot.ChaosNLIDataset(
                premises=val_premises, hypotheses=val_hypotheses,
                labels=val_labels, example_ids=val_eids,
                entropies=val_entropies, tokenizer=tokenizer, max_length=128,
            )

            use_mps = device == "mps"
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True,
                num_workers=0 if use_mps else 2, pin_memory=not use_mps,
            )
            tracking_loader = DataLoader(
                tracking_dataset, batch_size=64, shuffle=False,
                num_workers=0 if use_mps else 2, pin_memory=not use_mps,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=64, shuffle=False,
                num_workers=0 if use_mps else 2, pin_memory=not use_mps,
            )

            model = robustness.create_lora_model(
                model_name="roberta-base", num_labels=3, rank=4,
                target_modules=["query", "value"],
            )

            tracker = TemporalTracker(loss_threshold=0.693)
            tracker.register_examples(
                example_ids=cn_eids, true_labels=cn_labels,
                annotation_entropies=cn_entropies,
            )

            all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
            label_counts = torch.bincount(all_labels_t, minlength=3).float()
            class_weights = (1.0 / label_counts.clamp(min=1))
            class_weights = class_weights / class_weights.sum() * 3

            run_t0 = time.time()
            history = robustness.train_with_tracking(
                model=model, train_loader=train_loader,
                tracking_loader=tracking_loader, val_loader=val_loader,
                tracker=tracker, n_epochs=5, learning_rate=2e-5,
                eval_every_n_steps=100, device=device,
                class_weights=class_weights,
            )
            elapsed = time.time() - run_t0

            # Compute metrics
            aulc_arr, aulc_ent = robustness.compute_aulc_from_tracker(tracker)
            valid_mask = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
            if valid_mask.sum() >= 3:
                rho, p = stats.spearmanr(aulc_arr[valid_mask], aulc_ent[valid_mask])
            else:
                rho, p = 0.0, 1.0

            # Per-group AULC for clean examples
            clean_aulcs = [float(aulc_arr[i]) for i in range(len(cn_eids))
                          if i in clean_indices.tolist() and np.isfinite(aulc_arr[i])]

            tracker.save(OUTPUT_DIR / f"{run_id}_tracker.json")

            result = {
                "experiment": "noise_injection",
                "condition": condition_name,
                "noise_fraction": noise_frac,
                "seed": seed,
                "aulc_rho": float(rho),
                "aulc_p": float(p),
                "mean_clean_aulc": float(np.mean(clean_aulcs)) if clean_aulcs else None,
                "final_val_acc": history["val_accuracy"][-1],
                "elapsed_seconds": elapsed,
                "tracking_steps": history["tracking_steps"],
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            seed_results[condition_name] = result

            print(f"  {run_id}: rho={rho:+.4f}, clean_aulc={result['mean_clean_aulc']:.4f}")

            robustness.plot_hero_figure(
                tracker, history["tracking_steps"],
                FIGURE_DIR / f"hero_{run_id}.png",
                title_suffix=f" (noise={condition_name}, seed={seed})",
            )

            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        all_results.append(seed_results)

    # Cross-seed summary
    print(f"\n{'='*70}")
    print("Noise Injection Cross-Seed Summary")
    print(f"{'='*70}")

    # Load seed 42 results if available
    # Note: seed 42 files from 10_robustness_experiments.py use 'spearman_rho' / 'clean_mean_aulc',
    # while new seeds use 'aulc_rho' / 'mean_clean_aulc'.
    def _get_rho(d):
        return d.get("aulc_rho", d.get("spearman_rho", float("nan")))

    def _get_clean_aulc(d):
        return d.get("mean_clean_aulc", d.get("clean_mean_aulc", "N/A"))

    for condition in noise_levels:
        s42_path = OUTPUT_DIR / f"noise_{condition}_s42.json"
        if s42_path.exists():
            with open(s42_path) as f:
                s42_result = json.load(f)
            print(f"\n  {condition} (seed 42): rho={_get_rho(s42_result):+.4f}, "
                  f"clean_aulc={_get_clean_aulc(s42_result)}")

    for seed_results in all_results:
        for condition, result in seed_results.items():
            print(f"  {condition} (seed {result['seed']}): rho={_get_rho(result):+.4f}, "
                  f"clean_aulc={_get_clean_aulc(result)}")

    elapsed = time.time() - t0
    print(f"\nNoise injection multi-seed complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
