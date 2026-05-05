#!/usr/bin/env python3
"""
Gradient Cosine Similarity Between Entropy Groups
==================================================
Computes cosine similarity between mean gradients of clean vs. contested
examples at multiple checkpoints during LoRA training. If similarity
decreases over training, this provides direct evidence of competitive
dynamics between entropy groups.

Runs RoBERTa-base LoRA r=4 on SNLI (seed 42) and also full FT for comparison.

Output:
    results/tracking/robustness_experiments/gradient_cosine/
    figures/robustness_experiments/gradient_cosine/

Usage:
    python scripts/17_gradient_cosine.py
    python scripts/17_gradient_cosine.py --device cuda
    python scripts/17_gradient_cosine.py --force
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking" / "robustness_experiments" / "gradient_cosine"
FIGURE_DIR = PROJECT_ROOT / "figures" / "robustness_experiments" / "gradient_cosine"


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


def compute_group_mean_gradient(model, data_loader, device, loss_fn):
    """Compute mean gradient vector for a set of examples.

    Returns a single 1-D tensor: the mean of per-example gradients
    across all LoRA parameters.
    """
    model.train()
    grad_sum = None
    n_examples = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        bsz = labels.size(0)

        # Accumulate per-example gradients via batch mean
        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        # Collect gradients from LoRA parameters only
        grads = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.detach().reshape(-1))
        if grads:
            batch_grad = torch.cat(grads)
            if grad_sum is None:
                grad_sum = batch_grad * bsz
            else:
                grad_sum += batch_grad * bsz
            n_examples += bsz

    model.zero_grad()
    if grad_sum is None or n_examples == 0:
        return None
    return grad_sum / n_examples


def train_with_gradient_cosine(
    model, train_loader, clean_loader, contested_loader,
    val_loader, n_epochs, learning_rate, device, class_weights,
    measure_every_n_steps=100,
):
    """Train model, measuring gradient cosine similarity at regular intervals."""
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn_train = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    loss_fn_per_example = nn.CrossEntropyLoss(reduction="mean")

    cosine_history = {"steps": [], "cosine_sim": []}
    global_step = 0

    # Measure at step 0
    clean_grad = compute_group_mean_gradient(model, clean_loader, device, loss_fn_per_example)
    contested_grad = compute_group_mean_gradient(model, contested_loader, device, loss_fn_per_example)
    if clean_grad is not None and contested_grad is not None:
        cos = torch.nn.functional.cosine_similarity(
            clean_grad.unsqueeze(0), contested_grad.unsqueeze(0),
        ).item()
        cosine_history["steps"].append(0)
        cosine_history["cosine_sim"].append(cos)
        print(f"  Step 0: cosine_sim={cos:.4f}")

    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{n_epochs}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn_train(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % measure_every_n_steps == 0:
                clean_grad = compute_group_mean_gradient(
                    model, clean_loader, device, loss_fn_per_example,
                )
                contested_grad = compute_group_mean_gradient(
                    model, contested_loader, device, loss_fn_per_example,
                )
                if clean_grad is not None and contested_grad is not None:
                    cos = torch.nn.functional.cosine_similarity(
                        clean_grad.unsqueeze(0), contested_grad.unsqueeze(0),
                    ).item()
                    cosine_history["steps"].append(global_step)
                    cosine_history["cosine_sim"].append(cos)
                    pbar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{cos:.3f}")

        # End-of-epoch measurement
        if global_step % measure_every_n_steps != 0:
            clean_grad = compute_group_mean_gradient(
                model, clean_loader, device, loss_fn_per_example,
            )
            contested_grad = compute_group_mean_gradient(
                model, contested_loader, device, loss_fn_per_example,
            )
            if clean_grad is not None and contested_grad is not None:
                cos = torch.nn.functional.cosine_similarity(
                    clean_grad.unsqueeze(0), contested_grad.unsqueeze(0),
                ).item()
                cosine_history["steps"].append(global_step)
                cosine_history["cosine_sim"].append(cos)

        print(f"  Epoch {epoch+1}: latest cosine_sim={cosine_history['cosine_sim'][-1]:.4f}")

    return cosine_history


def parse_args():
    parser = argparse.ArgumentParser(description="Gradient cosine similarity experiment.")
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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    set_seed(args.seed)

    # Load data
    chaosnli = robustness.load_chaosnli_data(subset="snli", seed=42)
    bulk = robustness.load_bulk_training_data(dataset="snli", n_examples=20000, seed=42)

    train_idx = chaosnli["train_indices"]
    cn_premises = [chaosnli["premises"][i] for i in train_idx]
    cn_hypotheses = [chaosnli["hypotheses"][i] for i in train_idx]
    cn_labels = [chaosnli["majority_labels"][i] for i in train_idx]
    cn_eids = [chaosnli["example_ids"][i] for i in train_idx]
    cn_entropies = [chaosnli["entropies"][i] for i in train_idx]

    # Split tracked examples by entropy category
    clean_idx = [i for i, h in enumerate(cn_entropies) if h < ENTROPY_LOW]
    contested_idx = [i for i, h in enumerate(cn_entropies) if h >= ENTROPY_HIGH]
    print(f"  Clean examples: {len(clean_idx)}, Contested examples: {len(contested_idx)}")

    # Build datasets
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
    clean_dataset = pilot.ChaosNLIDataset(
        premises=[cn_premises[i] for i in clean_idx],
        hypotheses=[cn_hypotheses[i] for i in clean_idx],
        labels=[cn_labels[i] for i in clean_idx],
        example_ids=[cn_eids[i] for i in clean_idx],
        entropies=[cn_entropies[i] for i in clean_idx],
        tokenizer=tokenizer, max_length=128,
    )
    contested_dataset = pilot.ChaosNLIDataset(
        premises=[cn_premises[i] for i in contested_idx],
        hypotheses=[cn_hypotheses[i] for i in contested_idx],
        labels=[cn_labels[i] for i in contested_idx],
        example_ids=[cn_eids[i] for i in contested_idx],
        entropies=[cn_entropies[i] for i in contested_idx],
        tokenizer=tokenizer, max_length=128,
    )

    use_mps = device == "mps"
    loader_kwargs = dict(num_workers=0 if use_mps else 2, pin_memory=not use_mps)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **loader_kwargs)
    clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=False, **loader_kwargs)
    contested_loader = DataLoader(contested_dataset, batch_size=64, shuffle=False, **loader_kwargs)

    # Class weights
    all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_labels_t, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * 3

    # --- Run LoRA r=4 ---
    results = {}
    for config_name, config in [
        ("lora_r4", {"rank": 4, "type": "lora"}),
        ("fullft", {"type": "fullft"}),
    ]:
        result_path = OUTPUT_DIR / f"gradient_cosine_{config_name}_s{args.seed}.json"
        if result_path.exists() and not args.force:
            print(f"\n  Skipping {config_name} (exists). Use --force to rerun.")
            with open(result_path) as f:
                results[config_name] = json.load(f)
            continue

        print(f"\n{'='*70}")
        print(f"  Training {config_name} with gradient cosine tracking")
        print(f"{'='*70}")

        set_seed(args.seed)

        if config["type"] == "lora":
            model = robustness.create_lora_model(
                model_name="roberta-base", num_labels=3, rank=config["rank"],
                target_modules=["query", "value"],
            )
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=3,
            )

        cosine_history = train_with_gradient_cosine(
            model=model, train_loader=train_loader,
            clean_loader=clean_loader, contested_loader=contested_loader,
            val_loader=None, n_epochs=5, learning_rate=2e-5,
            device=device, class_weights=class_weights,
            measure_every_n_steps=200,
        )

        result = {
            "config": config_name,
            "seed": args.seed,
            **cosine_history,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        results[config_name] = result

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for config_name, label, color in [
        ("lora_r4", "LoRA r=4", "#e74c3c"),
        ("fullft", "Full FT", "#3498db"),
    ]:
        if config_name in results:
            data = results[config_name]
            ax.plot(data["steps"], data["cosine_sim"],
                    marker="o", markersize=3, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cosine Similarity\n(Clean vs. Contested Gradients)")
    ax.set_title(f"Gradient Alignment Between Entropy Groups (RoBERTa, SNLI, seed {args.seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"gradient_cosine_s{args.seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {FIGURE_DIR / f'gradient_cosine_s{args.seed}.png'}")

    elapsed = time.time() - t0
    print(f"\nGradient cosine experiment complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
