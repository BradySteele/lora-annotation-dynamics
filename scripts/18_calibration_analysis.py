#!/usr/bin/env python3
"""
Prediction Entropy and ECE: Calibration Analysis
=================================================
Computes prediction entropy and Expected Calibration Error (ECE) at the
final checkpoint for LoRA r=4, LoRA r=16, and full fine-tuning, broken
down by annotation entropy group (clean/ambiguous/contested).

If both LoRA and full FT produce similarly peaked predictions (low
prediction entropy) but only LoRA exhibits un-learning, then
calibration cannot explain the LoRA-specific un-learning pattern.

Output:
    results/tracking/robustness_experiments/calibration/
    figures/robustness_experiments/calibration/

Usage:
    python scripts/18_calibration_analysis.py
    python scripts/18_calibration_analysis.py --device cuda
    python scripts/18_calibration_analysis.py --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed

ENTROPY_LOW = 0.4
ENTROPY_HIGH = 0.7
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking" / "robustness_experiments" / "calibration"
FIGURE_DIR = PROJECT_ROOT / "figures" / "robustness_experiments" / "calibration"


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


@torch.no_grad()
def compute_calibration_metrics(model, data_loader, entropies_map, device, n_bins=15):
    """Compute prediction entropy and ECE per entropy group.

    Args:
        model: Trained model.
        data_loader: DataLoader for ChaosNLI examples.
        entropies_map: Dict mapping example_id -> annotation entropy.
        device: Device string.
        n_bins: Number of bins for ECE.

    Returns:
        Dict with per-group prediction entropy, ECE, and confidence stats.
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_ann_entropies = []
    all_eids = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        eids = batch["example_id"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        all_probs.append(probs)
        all_labels.extend(labels.numpy())
        all_eids.extend(eids)
        all_ann_entropies.extend([entropies_map.get(eid, None) for eid in eids])

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    all_ann_entropies = np.array(all_ann_entropies, dtype=float)

    # Prediction entropy: H(p) = -sum p_c log p_c
    pred_entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)

    # Max confidence
    max_conf = np.max(all_probs, axis=1)

    # Correctness
    preds = np.argmax(all_probs, axis=1)
    correct = (preds == all_labels).astype(float)

    # Group by annotation entropy
    groups = {
        "clean": all_ann_entropies < ENTROPY_LOW,
        "ambiguous": (all_ann_entropies >= ENTROPY_LOW) & (all_ann_entropies < ENTROPY_HIGH),
        "contested": all_ann_entropies >= ENTROPY_HIGH,
    }

    results = {}
    for group_name, mask in groups.items():
        if mask.sum() == 0:
            continue

        g_pred_ent = pred_entropy[mask]
        g_max_conf = max_conf[mask]
        g_correct = correct[mask]

        # ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (g_max_conf > lo) & (g_max_conf <= hi)
            if in_bin.sum() == 0:
                continue
            bin_acc = g_correct[in_bin].mean()
            bin_conf = g_max_conf[in_bin].mean()
            ece += (in_bin.sum() / len(g_max_conf)) * abs(bin_acc - bin_conf)

        results[group_name] = {
            "n": int(mask.sum()),
            "mean_pred_entropy": float(np.mean(g_pred_ent)),
            "std_pred_entropy": float(np.std(g_pred_ent)),
            "mean_max_confidence": float(np.mean(g_max_conf)),
            "std_max_confidence": float(np.std(g_max_conf)),
            "accuracy": float(g_correct.mean()),
            "ece": float(ece),
        }

    # Overall ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    overall_ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (max_conf > lo) & (max_conf <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = max_conf[in_bin].mean()
        overall_ece += (in_bin.sum() / len(max_conf)) * abs(bin_acc - bin_conf)

    results["overall"] = {
        "n": len(all_labels),
        "mean_pred_entropy": float(np.mean(pred_entropy)),
        "mean_max_confidence": float(np.mean(max_conf)),
        "accuracy": float(correct.mean()),
        "ece": float(overall_ece),
    }

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Calibration analysis.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    robustness = _import_robustness()
    pilot = _import_pilot()
    device = args.device or robustness.detect_device(None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load data
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

    # Entropy map for tracked examples
    entropies_map = dict(zip(cn_eids, cn_entropies))

    combined_premises = list(bulk["premises"]) + cn_premises
    combined_hypotheses = list(bulk["hypotheses"]) + cn_hypotheses
    combined_labels = list(bulk["labels"]) + cn_labels
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
    loader_kwargs = dict(num_workers=0 if use_mps else 2, pin_memory=not use_mps)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **loader_kwargs)
    tracking_loader = DataLoader(tracking_dataset, batch_size=64, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, **loader_kwargs)

    # Class weights
    all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_labels_t, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * 3

    configs = [
        ("lora_r4", {"type": "lora", "rank": 4}),
        ("lora_r16", {"type": "lora", "rank": 16}),
        ("fullft", {"type": "fullft"}),
    ]

    all_results = {}

    for config_name, config in configs:
        result_path = OUTPUT_DIR / f"calibration_{config_name}_s{args.seed}.json"
        if result_path.exists() and not args.force:
            print(f"\n  Skipping {config_name} (exists). Use --force to rerun.")
            with open(result_path) as f:
                all_results[config_name] = json.load(f)
            continue

        print(f"\n{'='*70}")
        print(f"  Training and evaluating: {config_name}")
        print(f"{'='*70}")

        set_seed(args.seed)

        if config["type"] == "lora":
            model = robustness.create_lora_model(
                model_name="roberta-base", num_labels=3, rank=config["rank"],
                target_modules=["query", "value"],
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=3,
            )

        tracker = TemporalTracker(loss_threshold=0.693)
        tracker.register_examples(
            example_ids=cn_eids, true_labels=cn_labels,
            annotation_entropies=cn_entropies,
        )

        history = robustness.train_with_tracking(
            model=model, train_loader=train_loader,
            tracking_loader=tracking_loader, val_loader=val_loader,
            tracker=tracker, n_epochs=5, learning_rate=2e-5,
            eval_every_n_steps=100, device=device,
            class_weights=class_weights,
        )

        # Compute calibration metrics on training (tracked) examples
        cal_results = compute_calibration_metrics(
            model, tracking_loader, entropies_map, device,
        )

        result = {
            "config": config_name,
            "seed": args.seed,
            "val_accuracy": history["val_accuracy"][-1],
            "calibration": cal_results,
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results[config_name] = result

        print(f"\n  {config_name} calibration:")
        for group in ["clean", "ambiguous", "contested"]:
            if group in cal_results:
                g = cal_results[group]
                print(f"    {group}: pred_entropy={g['mean_pred_entropy']:.3f}, "
                      f"max_conf={g['mean_max_confidence']:.3f}, "
                      f"ECE={g['ece']:.3f}, acc={g['accuracy']:.3f}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # --- Plot: Prediction entropy by group across methods ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = ["clean", "ambiguous", "contested"]
    config_labels = {"lora_r4": "LoRA r=4", "lora_r16": "LoRA r=16", "fullft": "Full FT"}
    colors = {"lora_r4": "#e74c3c", "lora_r16": "#f39c12", "fullft": "#3498db"}
    x = np.arange(len(groups))
    width = 0.25

    # Panel 1: Mean prediction entropy
    ax = axes[0]
    for i, (cfg, label) in enumerate(config_labels.items()):
        if cfg not in all_results:
            continue
        cal = all_results[cfg]["calibration"]
        vals = [cal.get(g, {}).get("mean_pred_entropy", 0) for g in groups]
        ax.bar(x + i * width, vals, width, label=label, color=colors[cfg], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel("Mean Prediction Entropy (nats)")
    ax.set_title("Prediction Entropy by Group")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: ECE
    ax = axes[1]
    for i, (cfg, label) in enumerate(config_labels.items()):
        if cfg not in all_results:
            continue
        cal = all_results[cfg]["calibration"]
        vals = [cal.get(g, {}).get("ece", 0) for g in groups]
        ax.bar(x + i * width, vals, width, label=label, color=colors[cfg], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("ECE by Group")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Calibration Analysis (RoBERTa, SNLI, seed {args.seed})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"calibration_s{args.seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {FIGURE_DIR / f'calibration_s{args.seed}.png'}")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("Calibration Summary")
    print(f"{'='*70}")
    print(f"{'Config':<12} {'Group':<12} {'PredEnt':>8} {'MaxConf':>8} {'ECE':>8} {'Acc':>8}")
    print("-" * 60)
    for cfg in config_labels:
        if cfg not in all_results:
            continue
        cal = all_results[cfg]["calibration"]
        for group in groups:
            if group in cal:
                g = cal[group]
                print(f"{config_labels[cfg]:<12} {group:<12} "
                      f"{g['mean_pred_entropy']:>8.3f} {g['mean_max_confidence']:>8.3f} "
                      f"{g['ece']:>8.3f} {g['accuracy']:>8.3f}")

    elapsed = time.time() - t0
    print(f"\nCalibration analysis complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
