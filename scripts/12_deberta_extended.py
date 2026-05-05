#!/usr/bin/env python3
"""
DeBERTa v3-base Extended Experiments
=====================================
Extends DeBERTa experiments to LoRA r=16 and Full Fine-Tuning on SNLI
(3 seeds each) to strengthen the four-architectures claim.

Existing r=4 results are in results/tracking/robustness_experiments/deberta/.

Usage:
    python scripts/12_deberta_extended.py
    python scripts/12_deberta_extended.py --configs r16
    python scripts/12_deberta_extended.py --configs fullft
    python scripts/12_deberta_extended.py --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_entropy import (
    categorize_by_entropy,
    compute_annotation_entropy_from_distribution,
)
from src.data.chaosnli import load_chaosnli
from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed

CHAOSNLI_DATA_DIR = "/Users/bradysteele/Documents/research/ChaosNLI/data/chaosNLI_v1.0"
SEEDS = [42, 123, 456]
ENTROPY_LOW = 0.4
ENTROPY_HIGH = 0.7
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking" / "deberta_extended"
FIGURE_DIR = PROJECT_ROOT / "figures" / "deberta_extended"


def _import_pilot():
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import shared functions from robustness experiments
def _import_robustness():
    spec = importlib.util.spec_from_file_location(
        "robustness", str(PROJECT_ROOT / "scripts" / "10_robustness_experiments.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(description="DeBERTa v3 extended experiments.")
    parser.add_argument("--configs", nargs="+", default=["r16", "fullft"],
                        choices=["r16", "fullft"],
                        help="Configurations to run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def create_deberta_model(num_labels: int = 3, rank: int = 16) -> nn.Module:
    """Create DeBERTa v3-base with LoRA."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=num_labels,
    )
    base_model = base_model.float()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=["query_proj", "value_proj"],
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  DeBERTa v3-base LoRA rank={rank}, alpha={2*rank}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def create_deberta_fullft(num_labels: int = 3) -> nn.Module:
    """Create DeBERTa v3-base for full fine-tuning (all params trainable)."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=num_labels,
    )
    model = model.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  DeBERTa v3-base Full FT")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def main():
    args = parse_args()
    t0 = time.time()

    robustness = _import_robustness()
    pilot = _import_pilot()
    device = args.device or robustness.detect_device() if hasattr(robustness, 'detect_device') else pilot.detect_device(None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Load data
    chaosnli = robustness.load_chaosnli_data(subset="snli", seed=SEEDS[0])
    bulk = robustness.load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

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

    combined_premises = list(bulk["premises"]) + cn_premises
    combined_hypotheses = list(bulk["hypotheses"]) + cn_hypotheses
    combined_labels = list(bulk["labels"]) + cn_labels
    combined_eids = [f"snli_{i}" for i in range(len(bulk["premises"]))] + cn_eids
    combined_entropies = [None] * len(bulk["premises"]) + cn_entropies

    print("=" * 70)
    print("DeBERTa v3-base Extended Experiments (SNLI)")
    print("=" * 70)
    print(f"  Configs:  {args.configs}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Device:   {device}")

    all_results = []

    for config in args.configs:
        for seed in args.seeds:
            run_id = f"deberta-v3-base_snli_{config}_s{seed}"
            result_path = OUTPUT_DIR / f"{run_id}.json"

            if result_path.exists() and not args.force:
                print(f"\n  Skipping {run_id} (exists). Use --force to rerun.")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n{'='*60}")
            print(f"  Running: {run_id}")
            print(f"{'='*60}")

            set_seed(seed)

            # Create datasets
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

            # Create model
            if config == "r16":
                model = create_deberta_model(num_labels=3, rank=16)
            elif config == "fullft":
                model = create_deberta_fullft(num_labels=3)

            tracker = TemporalTracker(loss_threshold=0.693)
            tracker.register_examples(
                example_ids=cn_eids, true_labels=cn_labels,
                annotation_entropies=cn_entropies,
            )

            # Class weights
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

            # Correlations
            aulc_arr, aulc_ent = robustness.compute_aulc_from_tracker(tracker)
            valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
            if valid.sum() >= 3:
                rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
                tau, p_tau = stats.kendalltau(aulc_arr[valid], aulc_ent[valid])
            else:
                rho, p, tau, p_tau = 0.0, 1.0, 0.0, 1.0

            # Delta-ell (start-to-end loss change) by entropy category
            delta_ells = {"clean": [], "contested": []}
            for eid, record in tracker.records.items():
                if record.annotation_entropy is None or len(record.losses) < 2:
                    continue
                delta = record.losses[-1] - record.losses[0]
                if record.annotation_entropy < ENTROPY_LOW:
                    delta_ells["clean"].append(delta)
                elif record.annotation_entropy >= ENTROPY_HIGH:
                    delta_ells["contested"].append(delta)

            # Save tracker
            tracker.save(OUTPUT_DIR / f"{run_id}_tracker.json")

            result = {
                "experiment": "deberta_v3_extended",
                "model": "deberta-v3-base",
                "dataset": "snli",
                "config": config,
                "seed": seed,
                "aulc_rho": float(rho),
                "aulc_p": float(p),
                "kendall_tau": float(tau),
                "kendall_p": float(p_tau),
                "final_val_acc": history["val_accuracy"][-1],
                "final_train_loss": history["train_loss"][-1],
                "mean_delta_ell_clean": float(np.mean(delta_ells["clean"])) if delta_ells["clean"] else None,
                "mean_delta_ell_contested": float(np.mean(delta_ells["contested"])) if delta_ells["contested"] else None,
                "elapsed_seconds": elapsed,
                "tracking_steps": history["tracking_steps"],
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

            print(f"  {run_id}: rho={rho:+.4f} (p={p:.2e}), "
                  f"val_acc={result['final_val_acc']:.4f}, "
                  f"delta_ell contested={result.get('mean_delta_ell_contested', 'N/A')}")

            robustness.plot_hero_figure(
                tracker, history["tracking_steps"],
                FIGURE_DIR / f"hero_{run_id}.png",
                title_suffix=f" (DeBERTa v3, SNLI, {config}, seed={seed})",
            )

            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("DeBERTa v3-base Extended Summary")
    print(f"{'='*70}")
    print(f"{'Config':>8} {'Seed':>6} {'AULC rho':>10} {'p-value':>12} {'Val Acc':>10} {'Δℓ contest':>12}")
    print("-" * 60)
    for r in all_results:
        delta_str = f"{r.get('mean_delta_ell_contested', 0):+.4f}" if r.get('mean_delta_ell_contested') is not None else "N/A"
        print(f"{r['config']:>8} {r['seed']:>6} {r['aulc_rho']:>10.4f} {r['aulc_p']:>12.2e} "
              f"{r['final_val_acc']:>10.4f} {delta_str:>12}")

    # Per-config summary
    for config in args.configs:
        config_results = [r for r in all_results if r["config"] == config]
        if len(config_results) > 1:
            mean_rho = np.mean([r["aulc_rho"] for r in config_results])
            std_rho = np.std([r["aulc_rho"] for r in config_results])
            mean_acc = np.mean([r["final_val_acc"] for r in config_results])
            print(f"\n  {config} mean: rho={mean_rho:+.4f} ± {std_rho:.4f}, val_acc={mean_acc:.4f}")

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    elapsed = time.time() - t0
    print(f"\nDeBERTa extended experiments complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
