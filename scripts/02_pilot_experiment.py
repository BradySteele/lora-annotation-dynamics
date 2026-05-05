#!/usr/bin/env python3
"""
Phase 1: Pilot Experiment -- Single Rank, Single Seed
=====================================================
Run a pilot experiment with rank=4, seed=42 to validate the experimental
pipeline and test the core hypothesis before committing to the full sweep.

Trains RoBERTa-base + LoRA on a subsample of the full SNLI training set
(default 20K examples) while tracking per-example losses ONLY for the
1211 ChaosNLI examples that have annotation entropy values.

The key insight: the model needs enough training data to actually learn
NLI (1211 ChaosNLI examples alone is insufficient -- val accuracy stays
at majority-class 54%). By training on 20K SNLI examples, the model
learns the task properly, and we can then measure whether ChaosNLI
examples with low entropy are learned before those with high entropy.

Usage:
    python scripts/02_pilot_experiment.py
    python scripts/02_pilot_experiment.py --rank 4 --seed 42 --epochs 5
    python scripts/02_pilot_experiment.py --snli-size 50000
    python scripts/02_pilot_experiment.py --synthetic  # synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import math
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_entropy import compute_annotation_entropy_from_distribution
from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

class NLIDataset(Dataset):
    """PyTorch Dataset for NLI fine-tuning (combined SNLI + ChaosNLI).

    Each example provides tokenized premise-hypothesis pairs, the label,
    and optionally an example_id and entropy (for ChaosNLI examples).
    SNLI-only examples have entropy=None and synthetic example_ids.
    """

    def __init__(
        self,
        premises: List[str],
        hypotheses: List[str],
        labels: List[int],
        example_ids: List[str],
        entropies: List[Optional[float]],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.example_ids = example_ids
        self.entropies = entropies
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.premises)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoding = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "example_id": self.example_ids[idx],
        }
        # Entropy is only present for ChaosNLI examples
        if self.entropies[idx] is not None:
            item["entropy"] = torch.tensor(self.entropies[idx], dtype=torch.float32)
        else:
            item["entropy"] = torch.tensor(float("nan"), dtype=torch.float32)
        return item


class ChaosNLIDataset(Dataset):
    """PyTorch Dataset wrapping ONLY ChaosNLI examples for tracking passes.

    This dataset is used exclusively for the tracking DataLoader --
    computing per-example losses for the subset of examples that have
    entropy annotations. This avoids wasting time on the ~20K SNLI-only
    examples during tracking passes.
    """

    def __init__(
        self,
        premises: List[str],
        hypotheses: List[str],
        labels: List[int],
        example_ids: List[str],
        entropies: List[float],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.example_ids = example_ids
        self.entropies = entropies
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.premises)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoding = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "example_id": self.example_ids[idx],
            "entropy": torch.tensor(self.entropies[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Model creation
# --------------------------------------------------------------------------- #

def create_lora_model(
    model_name: str = "roberta-base",
    num_labels: int = 3,
    rank: int = 4,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Create RoBERTa + LoRA model for sequence classification.

    Uses the PEFT library to apply LoRA adapters. Alpha defaults to 2*rank
    per the experimental protocol (scaling that preserves effective learning
    rate across ranks).

    Args:
        model_name: HuggingFace model name.
        num_labels: Number of output classes (3 for NLI).
        rank: LoRA rank r.
        lora_alpha: LoRA scaling. Defaults to 2 * rank.
        lora_dropout: Dropout in LoRA layers.
        target_modules: Which attention matrices to adapt.

    Returns:
        PEFT-wrapped model.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    if lora_alpha is None:
        lora_alpha = 2 * rank

    if target_modules is None:
        target_modules = ["query", "value"]

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
        modules_to_save=["classifier"],  # Keep classification head trainable
    )

    model = get_peft_model(base_model, lora_config)

    # Report parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={rank}, alpha={lora_alpha}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# --------------------------------------------------------------------------- #
# Training with per-example tracking
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
) -> Dict[str, Any]:
    """Train model while recording per-example losses at regular intervals.

    The training loop uses the full combined dataset (SNLI + ChaosNLI)
    for gradient updates. Tracking passes use a SEPARATE DataLoader
    containing only ChaosNLI examples (those with entropy annotations),
    avoiding unnecessary computation on the ~20K SNLI-only examples.

    Args:
        model: PEFT model to train.
        train_loader: Training data loader (full SNLI + ChaosNLI).
        tracking_loader: DataLoader with ChaosNLI examples only (for tracking).
        val_loader: Validation data loader (ChaosNLI val set).
        tracker: TemporalTracker for recording per-example losses.
        n_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        eval_every_n_steps: Record per-example losses every N steps.
        device: Device string ("mps", "cuda", "cpu").
        max_grad_norm: Gradient clipping norm.
        class_weights: Pre-computed class weights for weighted CE loss.

    Returns:
        Dictionary with training history: per-step metrics, final metrics.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Cosine annealing with warmup (6% warmup, min LR = 10% of peak)
    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Class-weighted loss for training (handles label imbalance)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"  Class weights: {class_weights.tolist()}")

    loss_fn = nn.CrossEntropyLoss(reduction="none")  # per-example losses (unweighted, for tracking)
    loss_fn_mean = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "tracking_steps": [],
        "learning_rates": [],
    }

    global_step = 0
    tracking_step = 0

    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Tracking every {eval_every_n_steps} steps ({total_steps // eval_every_n_steps} checkpoints expected)")
    print(f"  Training examples: {len(train_loader.dataset)}")
    print(f"  Tracking examples: {len(tracking_loader.dataset)} (ChaosNLI only)")

    # Initial tracking pass (step 0, before any training)
    print("  Recording initial per-example losses (step 0)...")
    _record_tracking_pass(model, tracking_loader, tracker, tracking_step, loss_fn, device)
    history["tracking_steps"].append(0)
    tracking_step += 1

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{n_epochs}",
            leave=False,
        )

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

            # Periodic tracking pass (on ChaosNLI examples only)
            if global_step % eval_every_n_steps == 0:
                _record_tracking_pass(
                    model, tracking_loader, tracker, tracking_step, loss_fn, device,
                )
                history["tracking_steps"].append(global_step)
                tracking_step += 1

        # End-of-epoch tracking pass (if not already done at this step)
        if global_step % eval_every_n_steps != 0:
            _record_tracking_pass(
                model, tracking_loader, tracker, tracking_step, loss_fn, device,
            )
            history["tracking_steps"].append(global_step)
            tracking_step += 1

        # Record epoch-level metrics
        train_loss = np.mean(epoch_losses)
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn_mean, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))
        history["learning_rates"].append(float(scheduler.get_last_lr()[0]))

        print(
            f"  Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    history["total_tracking_steps"] = tracking_step
    return history


@torch.no_grad()
def _record_tracking_pass(
    model: nn.Module,
    data_loader: DataLoader,
    tracker: TemporalTracker,
    step: int,
    loss_fn: nn.Module,
    device: str,
) -> None:
    """Do a full pass over ChaosNLI examples to record per-example losses.

    This is the instrumentation core: we evaluate every ChaosNLI training
    example and store its loss in the tracker, indexed by step number.
    Only ChaosNLI examples (those with entropy annotations) are evaluated
    here -- the ~20K SNLI-only examples are skipped.

    We re-purpose TemporalTracker.record_epoch_losses with step as the
    "epoch" index, since our tracking granularity is finer than epochs.
    """
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
def _evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Returns:
        (mean_loss, accuracy).
    """
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
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #

def compute_learning_times(
    tracker: TemporalTracker,
    threshold: float = 0.693,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract learning times and entropies from tracker.

    Args:
        tracker: Trained TemporalTracker with recorded losses.
        threshold: Loss threshold for "learned" (default: -log(0.5)).

    Returns:
        (example_ids, learning_times, entropies) as parallel arrays.
        learning_times is np.inf for unlearned examples.
    """
    ids = []
    times = []
    entropies = []

    for eid, record in tracker.records.items():
        t = tracker.get_learning_time(eid, threshold=threshold)
        ids.append(eid)
        times.append(float(t) if t is not None else np.inf)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )

    return np.array(ids), np.array(times), np.array(entropies)


def compute_aulc(
    tracker: TemporalTracker,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Area Under the Loss Curve (AULC) for each example.

    AULC is a continuous measure of learning speed that uses the full loss
    trajectory rather than a single threshold crossing. Lower AULC = the
    model learned this example faster/better.

    Uses the mean loss across all tracked steps (equivalent to normalized
    trapezoidal integral with uniform spacing).

    Returns:
        (example_ids, aulc_values, entropies) as parallel arrays.
    """
    ids = []
    aulcs = []
    entropies = []

    for eid, record in tracker.records.items():
        losses = record.losses
        # Filter out NaN values
        valid_losses = [l for l in losses if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            ids.append(eid)
            aulcs.append(np.nan)
            entropies.append(
                record.annotation_entropy if record.annotation_entropy is not None else np.nan
            )
            continue

        # Mean loss across all tracked steps
        aulc = float(np.mean(valid_losses))

        ids.append(eid)
        aulcs.append(aulc)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )

    return np.array(ids), np.array(aulcs), np.array(entropies)


def compute_final_loss(
    tracker: TemporalTracker,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract final-checkpoint loss and entropy for each example.

    Returns:
        (example_ids, final_losses, entropies) as parallel arrays.
    """
    ids = []
    final_losses = []
    entropies = []

    for eid, record in tracker.records.items():
        losses = record.losses
        # Get last non-NaN loss
        valid_losses = [l for l in losses if not (isinstance(l, float) and np.isnan(l))]
        final_loss = valid_losses[-1] if valid_losses else np.nan

        ids.append(eid)
        final_losses.append(final_loss)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )

    return np.array(ids), np.array(final_losses), np.array(entropies)


def compute_spearman_correlation(
    learning_times: np.ndarray,
    entropies: np.ndarray,
) -> Tuple[float, float]:
    """Compute Spearman correlation between learning time and entropy.

    Filters out examples that were never learned (inf) or have no entropy.

    Returns:
        (rho, p_value).
    """
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    if valid.sum() < 3:
        return 0.0, 1.0

    rho, p = stats.spearmanr(learning_times[valid], entropies[valid])
    return float(rho), float(p)


# --------------------------------------------------------------------------- #
# Hero figure: per-category loss curves
# --------------------------------------------------------------------------- #

def plot_hero_figure(
    tracker: TemporalTracker,
    category_names: List[str],
    tracking_steps: List[int],
    output_path: Path,
    title_suffix: str = "",
    loss_threshold: float = 1.0,
) -> None:
    """Generate the hero figure: per-category mean loss curves over training.

    This is Figure 1 in the paper, showing that clean examples (low H_i)
    have losses that drop earlier than contested examples (high H_i).

    Args:
        tracker: Trained TemporalTracker.
        category_names: Names of entropy categories.
        tracking_steps: Global step numbers corresponding to each tracking index.
        output_path: Where to save the figure.
        title_suffix: Optional text to append to the title.
    """
    # Group examples by entropy category
    categories = {}
    for eid, record in tracker.records.items():
        h = record.annotation_entropy
        if h is None:
            continue
        # Categorize using the same thresholds as data prep
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

    # Compute SEM for CI bands
    sem_losses = {}
    for cat_name, eids in categories.items():
        trajectories = []
        for eid in eids:
            if eid in tracker.records and len(tracker.records[eid].losses) > 0:
                trajectories.append(tracker.records[eid].losses)
        if trajectories:
            max_len = max(len(t) for t in trajectories)
            padded = np.full((len(trajectories), max_len), np.nan)
            for i, t in enumerate(trajectories):
                padded[i, : len(t)] = t
            std = np.nanstd(padded, axis=0, ddof=1)
            n_valid = np.maximum(np.sum(~np.isnan(padded), axis=0).astype(float), 1.0)
            sem_losses[cat_name] = std / np.sqrt(n_valid)

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
        color = colors.get(cat_name, "gray")

        ax.plot(
            steps,
            losses,
            color=color,
            marker=markers.get(cat_name, "."),
            markersize=4,
            linewidth=2,
            label=f"{cat_name} (n={n_examples})",
            alpha=0.9,
        )

        # 95% CI band
        if cat_name in sem_losses:
            sem = sem_losses[cat_name][:n_steps]
            ci_lower = losses - 1.96 * sem
            ci_upper = losses + 1.96 * sem
            ax.fill_between(steps, ci_lower, ci_upper, color=color, alpha=0.15)

    # Loss threshold reference line
    ax.axhline(
        loss_threshold, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
        label=f"threshold $\\theta = {loss_threshold:.2f}$",
    )

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean Cross-Entropy Loss", fontsize=11)
    ax.set_title(
        f"Per-Category Learning Dynamics{title_suffix}", fontsize=12
    )
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
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Pilot experiment (single rank, single seed)."
    )
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5).")
    parser.add_argument("--eval-every-n-steps", type=int, default=100, help="Tracking interval in training steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--loss-threshold", type=float, default=0.693, help="Learning time threshold (default: -log(0.5) for confident prediction).")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--model-name", type=str, default="roberta-base", help="Base model.")
    parser.add_argument(
        "--snli-size", type=int, default=20000,
        help="Number of SNLI training examples to subsample (default: 20000).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if not specified).",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to processed ChaosNLI data JSON (from 01_prepare_data.py).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for development.",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=800,
        help="Number of synthetic examples.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for tracker files.",
    )
    parser.add_argument(
        "--figure-dir", type=str, default=None,
        help="Directory for figures.",
    )
    return parser.parse_args()


def detect_device(requested: Optional[str] = None) -> str:
    """Detect the best available device."""
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_chaosnli_data(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Load ChaosNLI data from pre-processed JSON or directly.

    Returns a dict with keys: premises, hypotheses, example_ids,
    majority_labels, entropies, train_indices, val_indices.
    """
    data_path = args.data_path
    if data_path is None:
        default_path = str(PROJECT_ROOT / "results" / "data" / "train_data.json")
        if Path(default_path).exists():
            data_path = default_path

    if data_path is not None:
        print(f"  Loading ChaosNLI data from {data_path}...")
        with open(data_path, "r") as f:
            data = json.load(f)

        # Validate that this is real data, not synthetic leftovers
        metadata = data.get("metadata", {})
        if metadata.get("synthetic", False):
            raise RuntimeError(
                f"The pre-processed data at {data_path} was generated from "
                f"SYNTHETIC data (metadata.synthetic=True). This means Phase 0 "
                f"silently fell back to synthetic data.\n\n"
                f"Fix: re-run Phase 0 with real ChaosNLI data:\n"
                f"  python scripts/01_prepare_data.py\n"
                f"or pass --synthetic explicitly if you intend to use synthetic data."
            )

        # Spot-check for placeholder text that indicates synthetic data
        sample_premises = data["premises"][:5]
        if all(p.startswith("premise_") for p in sample_premises):
            raise RuntimeError(
                f"The pre-processed data at {data_path} contains placeholder "
                f"text (e.g., 'premise_0'). This is synthetic data, not real "
                f"ChaosNLI examples.\n\n"
                f"Fix: re-run Phase 0 with real ChaosNLI data:\n"
                f"  python scripts/01_prepare_data.py"
            )

        return {
            "premises": data["premises"],
            "hypotheses": data["hypotheses"],
            "example_ids": data["example_ids"],
            "majority_labels": data["majority_labels"],
            "entropies": data["entropies"],
            "train_indices": data["train_indices"],
            "val_indices": data["val_indices"],
        }

    # No pre-processed data available -- load directly from ChaosNLI
    print("  No pre-processed ChaosNLI data found. Loading ChaosNLI directly...")
    from src.data.chaosnli import load_chaosnli
    from src.data.annotation_entropy import categorize_by_entropy

    data = load_chaosnli(subset="snli")
    entropies = [
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ]
    cats = categorize_by_entropy(np.array(entropies), thresholds=[0.4, 0.7])
    n = len(data["premises"])

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
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


def _load_snli_data(
    n_examples: int,
    seed: int,
) -> Dict[str, Any]:
    """Load and subsample the full SNLI training set from HuggingFace.

    Filters out examples with label == -1 (unlabeled), then subsamples
    to n_examples. SNLI labels: 0=entailment, 1=neutral, 2=contradiction
    (same encoding as ChaosNLI majority labels).

    Args:
        n_examples: Number of SNLI examples to subsample.
        seed: Random seed for reproducible subsampling.

    Returns:
        Dict with keys: premises, hypotheses, labels.
    """
    from datasets import load_dataset

    print(f"  Loading SNLI training data from HuggingFace (subsampling to {n_examples})...")
    snli = load_dataset("stanfordnlp/snli", split="train")

    # Filter out unlabeled examples (label == -1)
    snli = snli.filter(lambda x: x["label"] != -1)
    print(f"  SNLI after filtering label=-1: {len(snli)} examples")

    # Subsample
    if len(snli) > n_examples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(snli), size=n_examples, replace=False)
        indices.sort()  # Keep deterministic order
        snli = snli.select(indices.tolist())
    print(f"  SNLI subsampled to {len(snli)} examples")

    return {
        "premises": snli["premise"],
        "hypotheses": snli["hypothesis"],
        "labels": snli["label"],
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # Use cached models/datasets to avoid HuggingFace connectivity delays
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    set_seed(args.seed)

    device = detect_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "tracking"
    figure_dir = Path(args.figure_dir) if args.figure_dir else PROJECT_ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1: Pilot Experiment (SNLI + ChaosNLI tracking)")
    print("=" * 70)
    print(f"  Rank:       {args.rank}")
    print(f"  Seed:       {args.seed}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Device:     {device}")
    print(f"  SNLI size:  {args.snli_size}")
    print(f"  Eval every: {args.eval_every_n_steps} steps")
    print(f"  Loss threshold: {args.loss_threshold}")
    print()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.synthetic:
        # ------------------------------------------------------------- #
        # Synthetic mode: use synthetic data for pipeline testing
        # ------------------------------------------------------------- #
        print("Step 1: Loading SYNTHETIC data (--synthetic flag set)...")

        from src.data.chaosnli import create_synthetic_chaosnli
        from src.data.annotation_entropy import categorize_by_entropy

        data = create_synthetic_chaosnli(
            n_examples=args.n_synthetic,
            n_annotators=100,
            seed=args.seed,
        )
        entropies = [
            compute_annotation_entropy_from_distribution(dist)
            for dist in data["label_distributions"]
        ]
        cats = categorize_by_entropy(np.array(entropies), thresholds=[0.4, 0.7])
        n = len(data["premises"])

        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, val_idx = next(splitter.split(np.arange(n), cats.categories))

        # In synthetic mode, all examples are "tracked" (no SNLI supplement)
        train_premises = [data["premises"][i] for i in train_idx]
        train_hypotheses = [data["hypotheses"][i] for i in train_idx]
        train_labels = [int(data["majority_labels"][i]) for i in train_idx]
        train_example_ids = [data["example_ids"][i] for i in train_idx]
        train_entropies = [entropies[i] for i in train_idx]

        val_premises = [data["premises"][i] for i in val_idx]
        val_hypotheses = [data["hypotheses"][i] for i in val_idx]
        val_labels = [int(data["majority_labels"][i]) for i in val_idx]
        val_example_ids = [data["example_ids"][i] for i in val_idx]
        val_entropies = [entropies[i] for i in val_idx]

        # In synthetic mode, training and tracking datasets are the same
        combined_premises = train_premises
        combined_hypotheses = train_hypotheses
        combined_labels = train_labels
        combined_example_ids = train_example_ids
        combined_entropies: List[Optional[float]] = list(train_entropies)

        # Tracking dataset is the same as training dataset
        tracking_premises = train_premises
        tracking_hypotheses = train_hypotheses
        tracking_labels = train_labels
        tracking_example_ids = train_example_ids
        tracking_entropies = train_entropies

        print(f"  Synthetic train: {len(train_premises)}, val: {len(val_premises)}")

    else:
        # ------------------------------------------------------------- #
        # Real mode: SNLI for training + ChaosNLI for tracking
        # ------------------------------------------------------------- #
        print("Step 1: Loading data...")

        # 1a. Load ChaosNLI data (with entropy annotations)
        chaosnli = _load_chaosnli_data(args)
        cn_premises = chaosnli["premises"]
        cn_hypotheses = chaosnli["hypotheses"]
        cn_example_ids = chaosnli["example_ids"]
        cn_labels = chaosnli["majority_labels"]
        cn_entropies = chaosnli["entropies"]
        cn_train_idx = chaosnli["train_indices"]
        cn_val_idx = chaosnli["val_indices"]

        # Extract ChaosNLI train and val splits
        tracking_premises = [cn_premises[i] for i in cn_train_idx]
        tracking_hypotheses = [cn_hypotheses[i] for i in cn_train_idx]
        tracking_labels = [cn_labels[i] for i in cn_train_idx]
        tracking_example_ids = [cn_example_ids[i] for i in cn_train_idx]
        tracking_entropies = [cn_entropies[i] for i in cn_train_idx]

        val_premises = [cn_premises[i] for i in cn_val_idx]
        val_hypotheses = [cn_hypotheses[i] for i in cn_val_idx]
        val_labels = [cn_labels[i] for i in cn_val_idx]
        val_example_ids = [cn_example_ids[i] for i in cn_val_idx]
        val_entropies = [cn_entropies[i] for i in cn_val_idx]

        print(f"  ChaosNLI train: {len(tracking_premises)}, val: {len(val_premises)}")

        # 1b. Load SNLI training data (the bulk of the training set)
        snli = _load_snli_data(n_examples=args.snli_size, seed=args.seed)
        snli_premises = snli["premises"]
        snli_hypotheses = snli["hypotheses"]
        snli_labels = snli["labels"]
        snli_example_ids = [f"snli_{i}" for i in range(len(snli_premises))]

        print(f"  SNLI train: {len(snli_premises)}")

        # 1c. Combine SNLI + ChaosNLI into one training dataset
        # ChaosNLI examples have real entropies; SNLI examples have None
        combined_premises = list(snli_premises) + tracking_premises
        combined_hypotheses = list(snli_hypotheses) + tracking_hypotheses
        combined_labels = list(snli_labels) + tracking_labels
        combined_example_ids = snli_example_ids + tracking_example_ids
        combined_entropies: List[Optional[float]] = (
            [None] * len(snli_premises) + list(tracking_entropies)
        )

        print(f"  Combined training set: {len(combined_premises)} examples")
        print(f"    SNLI-only (no tracking): {len(snli_premises)}")
        print(f"    ChaosNLI (tracked):      {len(tracking_premises)}")

    # ------------------------------------------------------------------ #
    # Step 2: Create data loaders
    # ------------------------------------------------------------------ #
    print("\nStep 2: Creating data loaders...")

    # Training dataset: full combined set (SNLI + ChaosNLI)
    train_dataset = NLIDataset(
        premises=combined_premises,
        hypotheses=combined_hypotheses,
        labels=combined_labels,
        example_ids=combined_example_ids,
        entropies=combined_entropies,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Tracking dataset: ChaosNLI train examples only (for efficient tracking passes)
    tracking_dataset = ChaosNLIDataset(
        premises=tracking_premises,
        hypotheses=tracking_hypotheses,
        labels=tracking_labels,
        example_ids=tracking_example_ids,
        entropies=tracking_entropies,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # Validation dataset: ChaosNLI val examples
    val_dataset = ChaosNLIDataset(
        premises=val_premises,
        hypotheses=val_hypotheses,
        labels=val_labels,
        example_ids=val_example_ids,
        entropies=val_entropies,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # MPS-specific data loading settings: no multiprocessing, no pinned memory
    use_mps = device == "mps"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if use_mps else 2,
        pin_memory=not use_mps,
        drop_last=False,
    )
    tracking_loader = DataLoader(
        tracking_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0 if use_mps else 2,
        pin_memory=not use_mps,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0 if use_mps else 2,
        pin_memory=not use_mps,
    )

    print(f"  Train batches:    {len(train_loader)} (combined SNLI + ChaosNLI)")
    print(f"  Tracking batches: {len(tracking_loader)} (ChaosNLI only)")
    print(f"  Val batches:      {len(val_loader)}")

    # ------------------------------------------------------------------ #
    # Step 3: Create model
    # ------------------------------------------------------------------ #
    print("\nStep 3: Creating LoRA model...")

    model = create_lora_model(
        model_name=args.model_name,
        num_labels=3,
        rank=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Initialize tracker (ChaosNLI examples only)
    # ------------------------------------------------------------------ #
    print("\nStep 4: Initializing temporal tracker...")

    tracker = TemporalTracker(loss_threshold=args.loss_threshold)

    # Pre-register ONLY ChaosNLI training examples (they have entropy)
    tracker.register_examples(
        example_ids=tracking_example_ids,
        true_labels=tracking_labels,
        annotation_entropies=tracking_entropies,
    )

    print(f"  Registered {len(tracking_example_ids)} ChaosNLI examples in tracker.")

    # ------------------------------------------------------------------ #
    # Step 5: Compute class weights from combined training labels
    # ------------------------------------------------------------------ #
    print("\nStep 5: Computing class weights...")

    all_train_labels = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_train_labels, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"  Label distribution: {label_counts.tolist()}")
    print(f"  Class weights: {class_weights.tolist()}")

    # ------------------------------------------------------------------ #
    # Step 6: Train with tracking
    # ------------------------------------------------------------------ #
    print("\nStep 6: Training with per-example tracking...")

    history = train_with_tracking(
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

    # ------------------------------------------------------------------ #
    # Step 7: Compute correlations (three metrics)
    # ------------------------------------------------------------------ #
    print("\nStep 7: Computing entropy correlations...")

    # 7a. AULC (primary metric -- continuous, uses full trajectory)
    _, aulc_arr, aulc_ent = compute_aulc(tracker)
    valid_aulc = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
    if valid_aulc.sum() >= 3:
        rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid_aulc], aulc_ent[valid_aulc])
    else:
        rho_aulc, p_aulc = 0.0, 1.0
    print(f"  [AULC]       Spearman rho = {rho_aulc:+.4f}, p = {p_aulc:.2e}  (n={valid_aulc.sum()})")

    # 7b. Final loss (sanity check -- do clean examples end with lower loss?)
    _, final_arr, final_ent = compute_final_loss(tracker)
    valid_final = np.isfinite(final_arr) & np.isfinite(final_ent)
    if valid_final.sum() >= 3:
        rho_final, p_final = stats.spearmanr(final_arr[valid_final], final_ent[valid_final])
    else:
        rho_final, p_final = 0.0, 1.0
    print(f"  [Final loss] Spearman rho = {rho_final:+.4f}, p = {p_final:.2e}  (n={valid_final.sum()})")

    # 7c. Threshold crossing (legacy metric -- for comparison)
    ids_arr, times_arr, entropies_arr = compute_learning_times(
        tracker, threshold=args.loss_threshold,
    )
    rho, p_value = compute_spearman_correlation(times_arr, entropies_arr)
    n_learned = np.isfinite(times_arr).sum()
    n_unlearned = (~np.isfinite(times_arr)).sum()
    print(f"  [Threshold]  Spearman rho = {rho:+.4f}, p = {p_value:.2e}  (learned={n_learned}/{len(times_arr)})")

    # Use AULC as the primary gate metric
    primary_rho = rho_aulc
    primary_p = p_aulc

    # ------------------------------------------------------------------ #
    # Step 8: Save tracker
    # ------------------------------------------------------------------ #
    print("\nStep 8: Saving tracker and results...")

    tracker_path = output_dir / f"pilot_r{args.rank}_s{args.seed}.json"
    tracker.save(tracker_path)
    print(f"  Saved tracker to {tracker_path}")

    # Save pilot results summary
    pilot_results = {
        "rank": args.rank,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "loss_threshold": args.loss_threshold,
        "eval_every_n_steps": args.eval_every_n_steps,
        "snli_size": args.snli_size,
        "n_train_combined": len(combined_premises),
        "n_train_chaosnli": len(tracking_premises),
        "n_val_chaosnli": len(val_premises),
        "spearman_aulc_rho": rho_aulc,
        "spearman_aulc_p": p_aulc,
        "spearman_final_loss_rho": rho_final,
        "spearman_final_loss_p": p_final,
        "spearman_threshold_rho": rho,
        "spearman_threshold_p": p_value,
        "n_learned": int(n_learned),
        "n_unlearned": int(n_unlearned),
        "n_total": len(times_arr),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "tracking_steps": history["tracking_steps"],
        "train_loss_history": history["train_loss"],
        "val_loss_history": history["val_loss"],
        "val_accuracy_history": history["val_accuracy"],
    }

    results_path = output_dir / f"pilot_results_r{args.rank}_s{args.seed}.json"
    with open(results_path, "w") as f:
        json.dump(pilot_results, f, indent=2)
    print(f"  Saved results to {results_path}")

    # ------------------------------------------------------------------ #
    # Step 9: Generate hero figure
    # ------------------------------------------------------------------ #
    print("\nStep 9: Generating hero figure...")

    plot_hero_figure(
        tracker=tracker,
        category_names=["clean", "ambiguous", "contested"],
        tracking_steps=history["tracking_steps"],
        output_path=figure_dir / f"hero_loss_curves_r{args.rank}_s{args.seed}.png",
        title_suffix=f" (rank={args.rank}, seed={args.seed})",
        loss_threshold=args.loss_threshold,
    )

    # ------------------------------------------------------------------ #
    # Gate check (uses AULC as primary metric)
    # ------------------------------------------------------------------ #
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 1 complete ({elapsed:.1f}s)")

    # For AULC: positive rho means higher entropy -> higher mean loss (slower learning)
    # For final loss: positive rho means higher entropy -> higher final loss
    # Both are the predicted direction.
    best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}" if history["val_accuracy"] else "")

    if primary_rho > 0.10 and primary_p < 0.05:
        print(f"\nPHASE 1 GATE PASSED: AULC Spearman rho = {primary_rho:.3f} (p = {primary_p:.3e})")
        print("  Positive correlation: higher entropy -> higher mean loss (slower learning).")
        print("  This confirms the temporal separation hypothesis.")
        print("  Proceed to Phase 2 (rank sweep).")
    else:
        print(f"\nPHASE 1 GATE FAILED: AULC Spearman rho = {primary_rho:.3f} (p = {primary_p:.3e})")
        print("  Diagnostics:")
        print(f"    AULC rho > 0.10?     {'YES' if primary_rho > 0.10 else 'NO'} (rho = {primary_rho:+.4f})")
        print(f"    AULC p < 0.05?       {'YES' if primary_p < 0.05 else 'NO'} (p = {primary_p:.2e})")
        print(f"    Final-loss rho:      {rho_final:+.4f} (p = {p_final:.2e})")
        print(f"    Threshold rho:       {rho:+.4f} (p = {p_value:.2e})")
        print(f"    Best val accuracy:   {best_val_acc:.4f}")
        if best_val_acc < 0.55:
            print("    Model may not be learning the task. Check training config.")
        if primary_rho < 0:
            print("    Negative correlation: contested examples have LOWER mean loss.")
            print("    This contradicts the hypothesis.")
        print("  Do NOT proceed to Phase 2 without diagnosing the failure.")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
