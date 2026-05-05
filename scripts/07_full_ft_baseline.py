#!/usr/bin/env python3
"""
Full Fine-Tuning Baseline for Temporal Separation Comparison
=============================================================
Runs the same SNLI+ChaosNLI training pipeline as the pilot experiment
but with FULL fine-tuning (all parameters trainable, no LoRA) to determine
whether temporal separation is LoRA-specific or a general property of
gradient-based fine-tuning.

Usage:
    python scripts/07_full_ft_baseline.py
    python scripts/07_full_ft_baseline.py --seed 42 --epochs 5
    python scripts/07_full_ft_baseline.py --seeds 42 123 456
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
from typing import Any, Dict, List, Optional, Tuple

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

from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Import shared functions from pilot
# --------------------------------------------------------------------------- #

def _import_pilot():
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Full fine-tuning model creation
# --------------------------------------------------------------------------- #

def create_full_ft_model(
    model_name: str = "roberta-base",
    num_labels: int = 3,
) -> nn.Module:
    """Create a standard (non-LoRA) RoBERTa model for sequence classification.

    All parameters are trainable. This serves as the baseline for
    comparing whether temporal separation is LoRA-specific.
    """
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Full FT model: {model_name}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# --------------------------------------------------------------------------- #
# Training loop (adapted from pilot, no LoRA-specific handling)
# --------------------------------------------------------------------------- #

def train_full_ft_with_tracking(
    model: nn.Module,
    train_loader: DataLoader,
    tracking_loader: DataLoader,
    val_loader: DataLoader,
    tracker: TemporalTracker,
    n_epochs: int = 5,
    learning_rate: float = 2e-5,
    eval_every_n_steps: int = 100,
    device: str = "mps",
    max_grad_norm: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Train full FT model while recording per-example losses.

    Identical to the LoRA pilot's train_with_tracking but operates on
    a standard (non-PEFT) model with all parameters trainable.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_fn_mean = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "tracking_steps": [],
    }

    global_step = 0
    tracking_step = 0

    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Tracking every {eval_every_n_steps} steps")

    # Initial tracking pass
    print("  Recording initial per-example losses (step 0)...")
    _record_pass(model, tracking_loader, tracker, tracking_step, loss_fn, device)
    history["tracking_steps"].append(0)
    tracking_step += 1

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{n_epochs}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn_mean(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % eval_every_n_steps == 0:
                _record_pass(model, tracking_loader, tracker, tracking_step, loss_fn, device)
                history["tracking_steps"].append(global_step)
                tracking_step += 1

        if global_step % eval_every_n_steps != 0:
            _record_pass(model, tracking_loader, tracker, tracking_step, loss_fn, device)
            history["tracking_steps"].append(global_step)
            tracking_step += 1

        train_loss = np.mean(epoch_losses)
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn_mean, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        print(f"  Epoch {epoch+1}/{n_epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    history["total_tracking_steps"] = tracking_step
    return history


@torch.no_grad()
def _record_pass(model, data_loader, tracker, step, loss_fn, device):
    model.eval()
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        example_ids = batch["example_id"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        losses = loss_fn(outputs.logits, labels)

        tracker.record_epoch_losses(
            example_ids=list(example_ids),
            losses=losses.cpu().numpy(),
            epoch=step,
        )
    model.train()


@torch.no_grad()
def _evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Full FT baseline for temporal separation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval-every-n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--loss-threshold", type=float, default=0.693)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--snli-size", type=int, default=20000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    pilot = _import_pilot()
    device = pilot.detect_device(args.device)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "tracking"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full Fine-Tuning Baseline (temporal separation comparison)")
    print("=" * 70)
    print(f"  Seeds:    {args.seeds}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LR:       {args.learning_rate}")
    print(f"  SNLI size: {args.snli_size}")
    print(f"  Device:   {device}")
    print()

    # Load data once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # We only need one seed's data split -- use the first seed for consistency
    chaosnli_args = argparse.Namespace(data_path=args.data_path, seed=args.seeds[0])
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

    snli = pilot._load_snli_data(n_examples=args.snli_size, seed=args.seeds[0])
    snli_premises = snli["premises"]
    snli_hypotheses = snli["hypotheses"]
    snli_labels = snli["labels"]
    snli_example_ids = [f"snli_{i}" for i in range(len(snli_premises))]

    combined_premises = list(snli_premises) + tracking_premises
    combined_hypotheses = list(snli_hypotheses) + tracking_hypotheses
    combined_labels = list(snli_labels) + tracking_labels
    combined_example_ids = snli_example_ids + tracking_example_ids
    combined_entropies = [None] * len(snli_premises) + list(tracking_entropies)

    print(f"  Combined training set: {len(combined_premises)} examples")

    # Create datasets
    train_dataset = pilot.NLIDataset(
        premises=combined_premises, hypotheses=combined_hypotheses,
        labels=combined_labels, example_ids=combined_example_ids,
        entropies=combined_entropies, tokenizer=tokenizer, max_length=args.max_length,
    )
    tracking_dataset = pilot.ChaosNLIDataset(
        premises=tracking_premises, hypotheses=tracking_hypotheses,
        labels=tracking_labels, example_ids=tracking_example_ids,
        entropies=tracking_entropies, tokenizer=tokenizer, max_length=args.max_length,
    )
    val_dataset = pilot.ChaosNLIDataset(
        premises=val_premises, hypotheses=val_hypotheses,
        labels=val_labels, example_ids=val_example_ids,
        entropies=val_entropies, tokenizer=tokenizer, max_length=args.max_length,
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

    # Class weights
    all_train_labels = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_train_labels, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    # Run for each seed
    results_table = []

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'=' * 60}")
        print(f"  Full FT -- Seed {seed} [{seed_idx+1}/{len(args.seeds)}]")
        print(f"{'=' * 60}")

        # Check if already exists
        tracker_path = output_dir / f"fullft_s{seed}.json"
        if tracker_path.exists():
            print(f"  SKIPPED: {tracker_path} already exists")
            tracker = TemporalTracker.load(tracker_path)
            _, aulc_arr, aulc_ent = pilot.compute_aulc(tracker)
            valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
            rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid], aulc_ent[valid]) if valid.sum() >= 3 else (0.0, 1.0)
            _, final_arr, final_ent = pilot.compute_final_loss(tracker)
            valid_f = np.isfinite(final_arr) & np.isfinite(final_ent)
            rho_final, p_final = stats.spearmanr(final_arr[valid_f], final_ent[valid_f]) if valid_f.sum() >= 3 else (0.0, 1.0)
            results_table.append({
                "method": "full_ft", "seed": seed,
                "aulc_rho": float(rho_aulc), "aulc_p": float(p_aulc),
                "final_loss_rho": float(rho_final), "final_loss_p": float(p_final),
                "status": "skipped",
            })
            continue

        set_seed(seed)
        seed_t0 = time.time()

        model = create_full_ft_model(model_name=args.model_name, num_labels=3)

        tracker = TemporalTracker(loss_threshold=args.loss_threshold)
        tracker.register_examples(
            example_ids=tracking_example_ids,
            true_labels=tracking_labels,
            annotation_entropies=tracking_entropies,
        )

        history = train_full_ft_with_tracking(
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

        # Compute AULC
        _, aulc_arr, aulc_ent = pilot.compute_aulc(tracker)
        valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
        rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid], aulc_ent[valid]) if valid.sum() >= 3 else (0.0, 1.0)

        # Final loss
        _, final_arr, final_ent = pilot.compute_final_loss(tracker)
        valid_f = np.isfinite(final_arr) & np.isfinite(final_ent)
        rho_final, p_final = stats.spearmanr(final_arr[valid_f], final_ent[valid_f]) if valid_f.sum() >= 3 else (0.0, 1.0)

        seed_elapsed = time.time() - seed_t0

        # Save tracker
        tracker.save(tracker_path)

        # Save results
        seed_results = {
            "method": "full_ft", "seed": seed,
            "aulc_rho": float(rho_aulc), "aulc_p": float(p_aulc),
            "final_loss_rho": float(rho_final), "final_loss_p": float(p_final),
            "final_val_acc": history["val_accuracy"][-1] if history["val_accuracy"] else None,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "elapsed_seconds": seed_elapsed,
            "status": "completed",
            "tracking_steps": history["tracking_steps"],
            "train_loss_history": history["train_loss"],
            "val_loss_history": history["val_loss"],
            "val_accuracy_history": history["val_accuracy"],
        }
        results_table.append(seed_results)

        results_path = output_dir / f"fullft_results_s{seed}.json"
        with open(results_path, "w") as f:
            json.dump(seed_results, f, indent=2)

        print(f"\n  Full FT seed {seed}: AULC rho={rho_aulc:+.4f} (p={p_aulc:.2e}), "
              f"val_acc={history['val_accuracy'][-1]:.4f}, time={seed_elapsed:.0f}s")

        # Hero figure
        pilot.plot_hero_figure(
            tracker=tracker,
            category_names=["clean", "ambiguous", "contested"],
            tracking_steps=history["tracking_steps"],
            output_path=figure_dir / f"hero_loss_curves_fullft_s{seed}.png",
            title_suffix=f" (Full FT, seed={seed})",
            loss_threshold=args.loss_threshold,
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("Full FT Baseline Summary")
    print(f"{'=' * 70}")
    print(f"{'Seed':>6} {'AULC rho':>10} {'p-value':>12} {'FinalL rho':>12} {'Val Acc':>10}")
    print("-" * 52)

    for r in results_table:
        val_acc = r.get("final_val_acc")
        val_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        print(f"{r['seed']:>6} {r['aulc_rho']:>10.4f} {r['aulc_p']:>12.2e} "
              f"{r['final_loss_rho']:>12.4f} {val_str:>10}")

    # Mean across seeds
    rhos = [r["aulc_rho"] for r in results_table]
    print(f"\n  Mean AULC rho: {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}")

    # Save summary
    summary_path = output_dir / "fullft_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_table, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nFull FT baseline complete ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
