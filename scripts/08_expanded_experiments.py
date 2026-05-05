#!/usr/bin/env python3
"""
Expanded Experiments: Multi-Model, Multi-Dataset
=================================================
Runs the temporal separation experiment across:
  - Models: roberta-base, bert-base-uncased, distilbert-base-uncased
  - Datasets: ChaosNLI-SNLI, ChaosNLI-MNLI
  - Configurations: LoRA r=4, LoRA r=16, Full fine-tuning
  - Seeds: 42, 123, 456

Total: 3 models × 2 datasets × 3 configs × 3 seeds = 54 runs
Existing: 9 runs (RoBERTa × SNLI × {r4,r16,fullft} × {42,123,456})
New runs: 45

Skips existing results for resumability.

Usage:
    python scripts/08_expanded_experiments.py
    python scripts/08_expanded_experiments.py --dry-run
    python scripts/08_expanded_experiments.py --filter-model bert-base-uncased
    python scripts/08_expanded_experiments.py --filter-dataset mnli
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
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_entropy import compute_annotation_entropy_from_distribution
from src.data.chaosnli import load_chaosnli
from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Import pilot functions
# --------------------------------------------------------------------------- #

def _import_pilot():
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CHAOSNLI_DATA_DIR = "/Users/bradysteele/Documents/research/ChaosNLI/data/chaosNLI_v1.0"

MODELS = [
    {
        "name": "roberta-base",
        "lora_targets": ["query", "value"],
    },
    {
        "name": "bert-base-uncased",
        "lora_targets": ["query", "value"],
    },
    {
        "name": "distilbert-base-uncased",
        "lora_targets": ["q_lin", "v_lin"],
    },
]

DATASETS = ["snli", "mnli"]

CONFIGS = [
    {"type": "lora", "rank": 4},
    {"type": "lora", "rank": 16},
    {"type": "fullft"},
]

SEEDS = [42, 123, 456]


# --------------------------------------------------------------------------- #
# Map existing results to expanded naming convention
# --------------------------------------------------------------------------- #

EXISTING_RESULTS_MAP = {
    # pilot r4: pilot_r4_s{seed}.json -> roberta-base_snli_r4_s{seed}.json
    # pilot r16: pilot_r16_s{seed}.json -> roberta-base_snli_r16_s{seed}.json
    # fullft: fullft_s{seed}.json -> roberta-base_snli_fullft_s{seed}.json
}


def _result_filename(model_name: str, dataset: str, config: dict, seed: int) -> str:
    """Generate the result filename for a given configuration."""
    if config["type"] == "lora":
        return f"{model_name}_{dataset}_r{config['rank']}_s{seed}.json"
    else:
        return f"{model_name}_{dataset}_fullft_s{seed}.json"


def _tracker_filename(model_name: str, dataset: str, config: dict, seed: int) -> str:
    """Generate the tracker filename for a given configuration."""
    if config["type"] == "lora":
        return f"{model_name}_{dataset}_r{config['rank']}_s{seed}_tracker.json"
    else:
        return f"{model_name}_{dataset}_fullft_s{seed}_tracker.json"


def _figure_filename(model_name: str, dataset: str, config: dict, seed: int) -> str:
    """Generate the figure filename for a given configuration."""
    if config["type"] == "lora":
        return f"hero_{model_name}_{dataset}_r{config['rank']}_s{seed}.png"
    else:
        return f"hero_{model_name}_{dataset}_fullft_s{seed}.png"


def _existing_result_path(output_dir: Path, model_name: str, dataset: str, config: dict, seed: int) -> Optional[Path]:
    """Check if results already exist under the old naming convention."""
    if model_name != "roberta-base" or dataset != "snli":
        return None

    tracking_dir = PROJECT_ROOT / "results" / "tracking"

    if config["type"] == "lora":
        rank = config["rank"]
        # Check old pilot naming
        old_tracker = tracking_dir / f"pilot_r{rank}_s{seed}.json"
        old_results = tracking_dir / f"pilot_results_r{rank}_s{seed}.json"
        if old_tracker.exists() and old_results.exists():
            return old_results
    else:
        old_tracker = tracking_dir / f"fullft_s{seed}.json"
        old_results = tracking_dir / f"fullft_results_s{seed}.json"
        if old_tracker.exists() and old_results.exists():
            return old_results

    return None


def _existing_tracker_path(model_name: str, dataset: str, config: dict, seed: int) -> Optional[Path]:
    """Check if tracker already exists under the old naming convention."""
    if model_name != "roberta-base" or dataset != "snli":
        return None

    tracking_dir = PROJECT_ROOT / "results" / "tracking"

    if config["type"] == "lora":
        rank = config["rank"]
        old_tracker = tracking_dir / f"pilot_r{rank}_s{seed}.json"
        if old_tracker.exists():
            return old_tracker
    else:
        old_tracker = tracking_dir / f"fullft_s{seed}.json"
        if old_tracker.exists():
            return old_tracker

    return None


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_chaosnli_data(
    subset: str,
    seed: int,
) -> Dict[str, Any]:
    """Load ChaosNLI data for a given subset, with train/val split.

    Args:
        subset: "snli" or "mnli"
        seed: Seed for train/val split (uses first seed for consistency)

    Returns:
        Dict with premises, hypotheses, example_ids, majority_labels,
        entropies, train_indices, val_indices.
    """
    data = load_chaosnli(subset=subset, data_dir=CHAOSNLI_DATA_DIR)

    entropies = [
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ]

    from src.data.annotation_entropy import categorize_by_entropy
    cats = categorize_by_entropy(np.array(entropies), thresholds=[0.4, 0.7])
    n = len(data["premises"])

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(n), cats.categories))

    return {
        "premises": data["premises"],
        "hypotheses": data["hypotheses"],
        "example_ids": data["example_ids"],
        "majority_labels": data["majority_labels"].tolist(),
        "entropies": entropies,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }


def load_bulk_training_data(
    dataset: str,
    n_examples: int,
    seed: int,
) -> Dict[str, Any]:
    """Load bulk training data (SNLI or MNLI) from HuggingFace.

    Args:
        dataset: "snli" or "mnli"
        n_examples: Number of examples to subsample.
        seed: Random seed for subsampling.

    Returns:
        Dict with premises, hypotheses, labels.
    """
    from datasets import load_dataset

    if dataset == "snli":
        print(f"  Loading SNLI training data (subsampling to {n_examples})...")
        ds = load_dataset("stanfordnlp/snli", split="train")
        ds = ds.filter(lambda x: x["label"] != -1)
        premise_key, hypothesis_key = "premise", "hypothesis"
    elif dataset == "mnli":
        print(f"  Loading MNLI training data (subsampling to {n_examples})...")
        ds = load_dataset("glue", "mnli", split="train")
        premise_key, hypothesis_key = "premise", "hypothesis"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"  {dataset.upper()} after filtering: {len(ds)} examples")

    if len(ds) > n_examples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), size=n_examples, replace=False)
        indices.sort()
        ds = ds.select(indices.tolist())
    print(f"  {dataset.upper()} subsampled to {len(ds)} examples")

    return {
        "premises": ds[premise_key],
        "hypotheses": ds[hypothesis_key],
        "labels": ds["label"],
    }


# --------------------------------------------------------------------------- #
# Model creation
# --------------------------------------------------------------------------- #

def create_lora_model(
    model_name: str,
    num_labels: int,
    rank: int,
    target_modules: List[str],
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
) -> nn.Module:
    """Create a model with LoRA adapters for sequence classification."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    if lora_alpha is None:
        lora_alpha = 2 * rank

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={rank}, alpha={lora_alpha}, targets={target_modules}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def create_full_ft_model(
    model_name: str,
    num_labels: int = 3,
) -> nn.Module:
    """Create a standard (non-LoRA) model for sequence classification."""
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
# Training loop
# --------------------------------------------------------------------------- #

def train_with_tracking(
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
    is_peft: bool = True,
) -> Dict[str, Any]:
    """Train model while recording per-example losses at regular intervals.

    Works for both PEFT and full FT models.
    """
    model = model.to(device)

    if is_peft:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()

    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

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

    # Initial tracking pass (step 0)
    print("  Recording initial per-example losses (step 0)...")
    _record_tracking_pass(model, tracking_loader, tracker, tracking_step, loss_fn, device)
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
                _record_tracking_pass(
                    model, tracking_loader, tracker, tracking_step, loss_fn, device,
                )
                history["tracking_steps"].append(global_step)
                tracking_step += 1

        if global_step % eval_every_n_steps != 0:
            _record_tracking_pass(
                model, tracking_loader, tracker, tracking_step, loss_fn, device,
            )
            history["tracking_steps"].append(global_step)
            tracking_step += 1

        train_loss = np.mean(epoch_losses)
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn_mean, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        print(
            f"  Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    history["total_tracking_steps"] = tracking_step
    return history


@torch.no_grad()
def _record_tracking_pass(model, data_loader, tracker, step, loss_fn, device):
    """Record per-example losses for ChaosNLI examples."""
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
    """Evaluate model on validation set."""
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
# Analysis helpers (from pilot)
# --------------------------------------------------------------------------- #

def compute_aulc(tracker: TemporalTracker):
    """Compute AULC for each example."""
    ids = []
    aulcs = []
    entropies = []

    for eid, record in tracker.records.items():
        losses = record.losses
        valid_losses = [l for l in losses if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            ids.append(eid)
            aulcs.append(np.nan)
            entropies.append(
                record.annotation_entropy if record.annotation_entropy is not None else np.nan
            )
            continue

        aulc = float(np.mean(valid_losses))
        ids.append(eid)
        aulcs.append(aulc)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )

    return np.array(ids), np.array(aulcs), np.array(entropies)


def compute_final_loss(tracker: TemporalTracker):
    """Extract final-checkpoint loss and entropy for each example."""
    ids = []
    final_losses = []
    entropies = []

    for eid, record in tracker.records.items():
        losses = record.losses
        valid_losses = [l for l in losses if not (isinstance(l, float) and np.isnan(l))]
        final_loss = valid_losses[-1] if valid_losses else np.nan

        ids.append(eid)
        final_losses.append(final_loss)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )

    return np.array(ids), np.array(final_losses), np.array(entropies)


# --------------------------------------------------------------------------- #
# Hero figure (from pilot)
# --------------------------------------------------------------------------- #

def plot_hero_figure(tracker, tracking_steps, output_path, title_suffix="", loss_threshold=1.0):
    """Generate per-category mean loss curves over training."""
    import matplotlib.pyplot as plt

    categories = {}
    for eid, record in tracker.records.items():
        h = record.annotation_entropy
        if h is None:
            continue
        if h < 0.4:
            cat = "clean"
        elif h < 0.7:
            cat = "ambiguous"
        else:
            cat = "contested"

        if cat not in categories:
            categories[cat] = []
        categories[cat].append(eid)

    mean_losses = tracker.get_mean_loss_by_category(categories)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"clean": "#2166AC", "ambiguous": "#F4A582", "contested": "#B2182B"}
    markers = {"clean": "o", "ambiguous": "s", "contested": "^"}

    for cat_name in ["clean", "ambiguous", "contested"]:
        if cat_name not in mean_losses or len(mean_losses[cat_name]) == 0:
            continue

        losses = mean_losses[cat_name]
        n_steps = len(losses)
        steps = tracking_steps[:n_steps] if len(tracking_steps) >= n_steps else list(range(n_steps))
        n_examples = len(categories.get(cat_name, []))

        ax.plot(
            steps, losses,
            color=colors.get(cat_name, "gray"),
            marker=markers.get(cat_name, "."),
            markersize=4, linewidth=2,
            label=f"{cat_name} (n={n_examples})",
            alpha=0.9,
        )

    ax.axhline(
        loss_threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
        label=f"threshold $\\theta = {loss_threshold:.2f}$",
    )

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean Cross-Entropy Loss", fontsize=11)
    ax.set_title(f"Per-Category Learning Dynamics{title_suffix}", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved hero figure to {output_path}")


# --------------------------------------------------------------------------- #
# Single run
# --------------------------------------------------------------------------- #

def run_single_experiment(
    model_name: str,
    model_lora_targets: List[str],
    dataset: str,
    config: dict,
    seed: int,
    output_dir: Path,
    figure_dir: Path,
    device: str,
    n_epochs: int = 5,
    snli_size: int = 20000,
    eval_every_n_steps: int = 100,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    eval_batch_size: int = 64,
    max_length: int = 128,
    loss_threshold: float = 0.693,
) -> Dict[str, Any]:
    """Run a single experiment and return results dict."""
    from transformers import AutoTokenizer

    config_str = f"r{config['rank']}" if config["type"] == "lora" else "fullft"
    run_id = f"{model_name}_{dataset}_{config_str}_s{seed}"

    print(f"\n{'=' * 70}")
    print(f"  Running: {run_id}")
    print(f"{'=' * 70}")

    t0 = time.time()
    set_seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load ChaosNLI data
    print(f"  Loading ChaosNLI-{dataset.upper()} data...")
    chaosnli = load_chaosnli_data(subset=dataset, seed=SEEDS[0])  # Use first seed for consistent splits

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

    print(f"  ChaosNLI-{dataset.upper()} train: {len(tracking_premises)}, val: {len(val_premises)}")

    # Load bulk training data
    bulk = load_bulk_training_data(dataset=dataset, n_examples=snli_size, seed=SEEDS[0])

    bulk_example_ids = [f"{dataset}_{i}" for i in range(len(bulk["premises"]))]

    combined_premises = list(bulk["premises"]) + tracking_premises
    combined_hypotheses = list(bulk["hypotheses"]) + tracking_hypotheses
    combined_labels = list(bulk["labels"]) + tracking_labels
    combined_example_ids = bulk_example_ids + tracking_example_ids
    combined_entropies = [None] * len(bulk["premises"]) + list(tracking_entropies)

    print(f"  Combined training set: {len(combined_premises)} examples")

    # Import dataset classes from pilot
    pilot = _import_pilot()

    # Create datasets
    train_dataset = pilot.NLIDataset(
        premises=combined_premises, hypotheses=combined_hypotheses,
        labels=combined_labels, example_ids=combined_example_ids,
        entropies=combined_entropies, tokenizer=tokenizer, max_length=max_length,
    )
    tracking_dataset = pilot.ChaosNLIDataset(
        premises=tracking_premises, hypotheses=tracking_hypotheses,
        labels=tracking_labels, example_ids=tracking_example_ids,
        entropies=tracking_entropies, tokenizer=tokenizer, max_length=max_length,
    )
    val_dataset = pilot.ChaosNLIDataset(
        premises=val_premises, hypotheses=val_hypotheses,
        labels=val_labels, example_ids=val_example_ids,
        entropies=val_entropies, tokenizer=tokenizer, max_length=max_length,
    )

    use_mps = device == "mps"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )
    tracking_loader = DataLoader(
        tracking_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )

    # Create model
    if config["type"] == "lora":
        model = create_lora_model(
            model_name=model_name,
            num_labels=3,
            rank=config["rank"],
            target_modules=model_lora_targets,
        )
        is_peft = True
    else:
        model = create_full_ft_model(model_name=model_name, num_labels=3)
        is_peft = False

    # Initialize tracker
    tracker = TemporalTracker(loss_threshold=loss_threshold)
    tracker.register_examples(
        example_ids=tracking_example_ids,
        true_labels=tracking_labels,
        annotation_entropies=tracking_entropies,
    )

    # Class weights
    all_train_labels = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_train_labels, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    # Train
    history = train_with_tracking(
        model=model,
        train_loader=train_loader,
        tracking_loader=tracking_loader,
        val_loader=val_loader,
        tracker=tracker,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        eval_every_n_steps=eval_every_n_steps,
        device=device,
        max_grad_norm=1.0,
        class_weights=class_weights,
        is_peft=is_peft,
    )

    # Compute AULC correlation
    _, aulc_arr, aulc_ent = compute_aulc(tracker)
    valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
    if valid.sum() >= 3:
        rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
    else:
        rho_aulc, p_aulc = 0.0, 1.0

    # Final loss correlation
    _, final_arr, final_ent = compute_final_loss(tracker)
    valid_f = np.isfinite(final_arr) & np.isfinite(final_ent)
    if valid_f.sum() >= 3:
        rho_final, p_final = stats.spearmanr(final_arr[valid_f], final_ent[valid_f])
    else:
        rho_final, p_final = 0.0, 1.0

    elapsed = time.time() - t0

    # Compute contested loss change
    contested_initial = []
    contested_final = []
    clean_initial = []
    clean_final = []
    for eid, record in tracker.records.items():
        h = record.annotation_entropy
        if h is None:
            continue
        valid_losses = [l for l in record.losses if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            continue
        if h >= 0.7:
            contested_initial.append(valid_losses[0])
            contested_final.append(valid_losses[-1])
        elif h < 0.4:
            clean_initial.append(valid_losses[0])
            clean_final.append(valid_losses[-1])

    contested_loss_change = float(np.mean(contested_final) - np.mean(contested_initial)) if contested_final else None
    clean_loss_change = float(np.mean(clean_final) - np.mean(clean_initial)) if clean_final else None

    # Save tracker
    tracker_path = output_dir / _tracker_filename(model_name, dataset, config, seed)
    tracker.save(tracker_path)

    # Save results
    results = {
        "model": model_name,
        "dataset": dataset,
        "config_type": config["type"],
        "rank": config.get("rank"),
        "seed": seed,
        "aulc_rho": float(rho_aulc),
        "aulc_p": float(p_aulc),
        "final_loss_rho": float(rho_final),
        "final_loss_p": float(p_final),
        "final_val_acc": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "contested_loss_change": contested_loss_change,
        "clean_loss_change": clean_loss_change,
        "n_train_chaosnli": len(tracking_premises),
        "n_val_chaosnli": len(val_premises),
        "elapsed_seconds": elapsed,
        "tracking_steps": history["tracking_steps"],
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "val_accuracy_history": history["val_accuracy"],
    }

    results_path = output_dir / _result_filename(model_name, dataset, config, seed)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {run_id}: AULC rho={rho_aulc:+.4f} (p={p_aulc:.2e}), "
          f"val_acc={results['final_val_acc']:.4f}, time={elapsed:.0f}s")

    # Hero figure
    title_suffix = f" ({model_name}, {dataset.upper()}, {config_str}, seed={seed})"
    figure_path = figure_dir / _figure_filename(model_name, dataset, config, seed)
    plot_hero_figure(
        tracker=tracker,
        tracking_steps=history["tracking_steps"],
        output_path=figure_path,
        title_suffix=title_suffix,
        loss_threshold=loss_threshold,
    )

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    return results


# --------------------------------------------------------------------------- #
# Collect existing results
# --------------------------------------------------------------------------- #

def collect_existing_result(
    model_name: str,
    dataset: str,
    config: dict,
    seed: int,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Try to load an existing result, checking both old and new naming."""
    # Check new naming
    new_path = output_dir / _result_filename(model_name, dataset, config, seed)
    if new_path.exists():
        with open(new_path, "r") as f:
            return json.load(f)

    # Check old naming for roberta-base SNLI
    old_path = _existing_result_path(output_dir, model_name, dataset, config, seed)
    if old_path is not None:
        with open(old_path, "r") as f:
            old_data = json.load(f)

        # Convert old format to new format
        result = {
            "model": model_name,
            "dataset": dataset,
            "config_type": config["type"],
            "rank": config.get("rank"),
            "seed": seed,
            "aulc_rho": old_data.get("spearman_aulc_rho", old_data.get("aulc_rho")),
            "aulc_p": old_data.get("spearman_aulc_p", old_data.get("aulc_p")),
            "final_loss_rho": old_data.get("spearman_final_loss_rho", old_data.get("final_loss_rho")),
            "final_loss_p": old_data.get("spearman_final_loss_p", old_data.get("final_loss_p")),
            "final_val_acc": old_data.get("final_val_accuracy", old_data.get("final_val_acc")),
            "final_train_loss": old_data.get("final_train_loss"),
            "n_train_chaosnli": old_data.get("n_train_chaosnli"),
            "n_val_chaosnli": old_data.get("n_val_chaosnli"),
            "tracking_steps": old_data.get("tracking_steps"),
            "status": "existing",
        }

        # Compute contested loss change from old tracker if available
        old_tracker_path = _existing_tracker_path(model_name, dataset, config, seed)
        if old_tracker_path is not None:
            tracker = TemporalTracker.load(old_tracker_path)
            contested_initial = []
            contested_final = []
            clean_initial = []
            clean_final = []
            for eid, record in tracker.records.items():
                h = record.annotation_entropy
                if h is None:
                    continue
                valid_losses = [l for l in record.losses if not (isinstance(l, float) and np.isnan(l))]
                if len(valid_losses) < 2:
                    continue
                if h >= 0.7:
                    contested_initial.append(valid_losses[0])
                    contested_final.append(valid_losses[-1])
                elif h < 0.4:
                    clean_initial.append(valid_losses[0])
                    clean_final.append(valid_losses[-1])
            result["contested_loss_change"] = float(np.mean(contested_final) - np.mean(contested_initial)) if contested_final else None
            result["clean_loss_change"] = float(np.mean(clean_final) - np.mean(clean_initial)) if clean_final else None

        return result

    return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="Expanded experiments: multi-model, multi-dataset.")
    parser.add_argument("--dry-run", action="store_true", help="List runs without executing.")
    parser.add_argument("--filter-model", type=str, default=None, help="Run only this model.")
    parser.add_argument("--filter-dataset", type=str, default=None, help="Run only this dataset.")
    parser.add_argument("--filter-config", type=str, default=None, help="Filter config: r4, r16, fullft.")
    parser.add_argument("--filter-seed", type=int, default=None, help="Run only this seed.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--snli-size", type=int, default=20000)
    parser.add_argument("--eval-every-n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--loss-threshold", type=float, default=0.693)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--figure-dir", type=str, default=None)
    return parser.parse_args()


def detect_device(requested=None):
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    args = parse_args()
    t0 = time.time()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = detect_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "tracking" / "expanded"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures" / "expanded"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Expanded Experiments: Multi-Model, Multi-Dataset")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Figures: {figure_dir}")
    print()

    # Build run list
    runs = []
    for model_info in MODELS:
        if args.filter_model and model_info["name"] != args.filter_model:
            continue
        for dataset in DATASETS:
            if args.filter_dataset and dataset != args.filter_dataset:
                continue
            for config in CONFIGS:
                if args.filter_config:
                    config_str = f"r{config['rank']}" if config["type"] == "lora" else "fullft"
                    if config_str != args.filter_config:
                        continue
                for seed in SEEDS:
                    if args.filter_seed and seed != args.filter_seed:
                        continue
                    runs.append((model_info, dataset, config, seed))

    # Classify runs as existing or new
    existing_runs = []
    new_runs = []
    for model_info, dataset, config, seed in runs:
        result = collect_existing_result(
            model_info["name"], dataset, config, seed, output_dir,
        )
        if result is not None:
            existing_runs.append((model_info, dataset, config, seed, result))
        else:
            new_runs.append((model_info, dataset, config, seed))

    print(f"  Total configurations: {len(runs)}")
    print(f"  Already completed:    {len(existing_runs)}")
    print(f"  New runs needed:      {len(new_runs)}")
    print()

    if args.dry_run:
        print("DRY RUN -- listing new runs:")
        for i, (model_info, dataset, config, seed) in enumerate(new_runs):
            config_str = f"r{config['rank']}" if config["type"] == "lora" else "fullft"
            print(f"  {i+1:3d}. {model_info['name']} | {dataset} | {config_str} | seed={seed}")
        print(f"\n  Existing results:")
        for model_info, dataset, config, seed, result in existing_runs:
            config_str = f"r{config['rank']}" if config["type"] == "lora" else "fullft"
            rho = result.get("aulc_rho", "?")
            print(f"       {model_info['name']} | {dataset} | {config_str} | seed={seed} | rho={rho}")
        return

    # Run new experiments
    all_results = [r for _, _, _, _, r in existing_runs]

    for run_idx, (model_info, dataset, config, seed) in enumerate(new_runs):
        config_str = f"r{config['rank']}" if config["type"] == "lora" else "fullft"
        print(f"\n[{run_idx+1}/{len(new_runs)}] {model_info['name']} | {dataset} | {config_str} | seed={seed}")

        result = run_single_experiment(
            model_name=model_info["name"],
            model_lora_targets=model_info["lora_targets"],
            dataset=dataset,
            config=config,
            seed=seed,
            output_dir=output_dir,
            figure_dir=figure_dir,
            device=device,
            n_epochs=args.epochs,
            snli_size=args.snli_size,
            eval_every_n_steps=args.eval_every_n_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            loss_threshold=args.loss_threshold,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 70}")
    print("Expanded Experiments Summary")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Dataset':<8} {'Config':<8} {'Seed':>5} {'AULC rho':>10} {'p-value':>12} {'Val Acc':>10}")
    print("-" * 80)

    for r in sorted(all_results, key=lambda x: (x.get("model", ""), x.get("dataset", ""), str(x.get("rank", "")), x.get("seed", 0))):
        model = r.get("model", "?")
        dataset = r.get("dataset", "?")
        config_str = f"r{r['rank']}" if r.get("config_type") == "lora" else "fullft"
        seed = r.get("seed", "?")
        rho = r.get("aulc_rho", 0)
        p = r.get("aulc_p", 1)
        val_acc = r.get("final_val_acc")
        val_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
        print(f"{model:<25} {dataset:<8} {config_str:<8} {seed:>5} {rho:>+10.4f} {p:>12.2e} {val_str:>10}")

    # Save master results
    master_path = output_dir / "all_results.json"
    with open(master_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved all results to {master_path}")

    elapsed = time.time() - t0
    print(f"\nExpanded experiments complete ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
