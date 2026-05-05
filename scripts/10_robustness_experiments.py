#!/usr/bin/env python3
"""
Robustness Experiments: Comprehensive Robustness and Extension Analyses
=====================================================================
Addresses robustness and extension analyses in a single script with independent
sub-experiments selectable via command-line flags.

Experiments:
    1. --entropy-buckets     Entropy-bucket ablation (train on low/high/balanced)
    2. --gradient-norms      Per-example gradient norms by entropy bin
    3. --alt-binning         Quartile/tercile binning robustness check
    4. --bootstrap-ci        Bootstrap 95% CIs on Spearman rho (all 18 conditions)
    5. --kendall             Kendall tau-b for all 18 conditions
    6. --gpt2                GPT-2 decoder-only model experiment
    7. --alphanli            ChaosNLI-AlphaNLI third-task experiment
    8. --soft-label          Soft-label (KL-div) ablation vs hard-label CE
    9. --cartography         Dataset Cartography comparison (post-hoc, CPU-only)
   10. --deberta             DeBERTa v3-base disentangled attention experiment
   11. --noise-injection     Synthetic noise injection (causal intervention)
    --all                    Run everything

Training experiments (1, 2, 6, 7, 8) require GPU/MPS.
Analysis experiments (3, 4, 5, 9) run on existing tracker data (CPU-only).

Usage:
    python scripts/10_robustness_experiments.py --all
    python scripts/10_robustness_experiments.py --bootstrap-ci --kendall --alt-binning
    python scripts/10_robustness_experiments.py --entropy-buckets --seed 42
    python scripts/10_robustness_experiments.py --gpt2 --device cuda
    python scripts/10_robustness_experiments.py --soft-label --device mps
    python scripts/10_robustness_experiments.py --cartography
    python scripts/10_robustness_experiments.py --all --dry-run
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
from torch.utils.data import DataLoader, Dataset, Subset
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


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CHAOSNLI_DATA_DIR = "/Users/bradysteele/Documents/research/ChaosNLI/data/chaosNLI_v1.0"

SEEDS = [42, 123, 456]

# Entropy thresholds used in the paper (NOT the config defaults)
ENTROPY_LOW = 0.4
ENTROPY_HIGH = 0.7

# Existing tracker locations
TRACKING_DIR = PROJECT_ROOT / "results" / "tracking"
EXPANDED_DIR = TRACKING_DIR / "expanded"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking" / "robustness_experiments"
FIGURE_DIR = PROJECT_ROOT / "figures" / "robustness_experiments"

# The 18 main conditions: 3 models x 2 datasets x 3 configs
MODELS = [
    {"name": "roberta-base", "lora_targets": ["query", "value"]},
    {"name": "bert-base-uncased", "lora_targets": ["query", "value"]},
    {"name": "distilbert-base-uncased", "lora_targets": ["q_lin", "v_lin"]},
]
DATASETS = ["snli", "mnli"]
CONFIGS = [
    {"type": "lora", "rank": 4},
    {"type": "lora", "rank": 16},
    {"type": "fullft"},
]


# --------------------------------------------------------------------------- #
# Import pilot classes
# --------------------------------------------------------------------------- #

def _import_pilot():
    """Import the pilot experiment module to reuse NLIDataset/ChaosNLIDataset."""
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Utility: locate tracker files
# --------------------------------------------------------------------------- #

def find_tracker_path(model_name: str, dataset: str, config: dict, seed: int) -> Optional[Path]:
    """Find an existing tracker JSON file, checking expanded + old naming."""
    # New naming in expanded dir
    if config["type"] == "lora":
        new_name = f"{model_name}_{dataset}_r{config['rank']}_s{seed}_tracker.json"
    else:
        new_name = f"{model_name}_{dataset}_fullft_s{seed}_tracker.json"

    new_path = EXPANDED_DIR / new_name
    if new_path.exists():
        return new_path

    # Old naming for roberta-base SNLI in base tracking dir
    if model_name == "roberta-base" and dataset == "snli":
        if config["type"] == "lora":
            old_path = TRACKING_DIR / f"pilot_r{config['rank']}_s{seed}.json"
        else:
            old_path = TRACKING_DIR / f"fullft_s{seed}.json"
        if old_path.exists():
            return old_path

    return None


def config_str(config: dict) -> str:
    """Human-readable config string."""
    if config["type"] == "lora":
        return f"r{config['rank']}"
    return "fullft"


# --------------------------------------------------------------------------- #
# Utility: AULC computation from tracker
# --------------------------------------------------------------------------- #

def compute_aulc_from_tracker(tracker: TemporalTracker):
    """Compute AULC and entropy arrays from a tracker."""
    aulcs = []
    entropies = []
    for eid, record in tracker.records.items():
        valid_losses = [l for l in record.losses
                        if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            aulcs.append(np.nan)
        else:
            aulcs.append(float(np.mean(valid_losses)))
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None
            else np.nan
        )
    return np.array(aulcs), np.array(entropies)


# --------------------------------------------------------------------------- #
# Data loading helpers (reused across experiments)
# --------------------------------------------------------------------------- #

def load_chaosnli_data(subset: str, seed: int = 42) -> Dict[str, Any]:
    """Load ChaosNLI data with entropy and train/val split."""
    data = load_chaosnli(subset=subset, data_dir=CHAOSNLI_DATA_DIR)

    entropies = [
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ]

    cats = categorize_by_entropy(np.array(entropies), thresholds=[ENTROPY_LOW, ENTROPY_HIGH])
    n = len(data["premises"])

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.arange(n), cats.categories))

    return {
        "premises": data["premises"],
        "hypotheses": data["hypotheses"],
        "example_ids": data["example_ids"],
        "majority_labels": data["majority_labels"].tolist(),
        "label_distributions": data["label_distributions"],
        "entropies": entropies,
        "categories": cats.categories,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }


def load_bulk_training_data(dataset: str, n_examples: int, seed: int) -> Dict[str, Any]:
    """Load bulk SNLI/MNLI training data from HuggingFace."""
    from datasets import load_dataset

    if dataset == "snli":
        ds = load_dataset("stanfordnlp/snli", split="train")
        ds = ds.filter(lambda x: x["label"] != -1)
    elif dataset == "mnli":
        ds = load_dataset("glue", "mnli", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if len(ds) > n_examples:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(ds), size=n_examples, replace=False)
        indices.sort()
        ds = ds.select(indices.tolist())

    return {
        "premises": ds["premise"],
        "hypotheses": ds["hypothesis"],
        "labels": ds["label"],
    }


# --------------------------------------------------------------------------- #
# Model creation helpers
# --------------------------------------------------------------------------- #

def create_lora_model(
    model_name: str,
    num_labels: int,
    rank: int,
    target_modules: List[str],
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
) -> nn.Module:
    """Create a LoRA-adapted sequence classification model."""
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
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def create_gpt2_seq_cls_model(num_labels: int = 3, rank: int = 4) -> nn.Module:
    """Create GPT-2 with LoRA for sequence classification.

    GPT-2 is a decoder-only model. We use AutoModelForSequenceClassification
    which adds a classification head on top. GPT-2 uses the last token's
    representation for classification (pad_token_id must be set).
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, GPT2Config

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2", num_labels=num_labels,
    )
    # GPT-2 has no pad token by default; use eos_token_id as pad
    base_model.config.pad_token_id = base_model.config.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2 combined QKV projection
        bias="none",
        modules_to_save=["score"],  # Classification head
    )

    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  GPT-2 LoRA rank={rank}, alpha={2*rank}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# --------------------------------------------------------------------------- #
# Training loop (shared across experiments 1, 6, 7)
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
    """Train model with per-example loss tracking (same as 08_expanded)."""
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_fn_mean = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "tracking_steps": [],
    }

    global_step = 0
    tracking_step = 0

    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")

    # Initial tracking pass (step 0)
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

        # End-of-epoch tracking
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

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    history["total_tracking_steps"] = tracking_step
    return history


@torch.no_grad()
def _record_tracking_pass(model, data_loader, tracker, step, loss_fn, device):
    """Record per-example losses for tracked examples."""
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
    """Evaluate on validation set, returning (loss, accuracy)."""
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
# Hero figure helper
# --------------------------------------------------------------------------- #

def plot_hero_figure(tracker, tracking_steps, output_path, title_suffix="",
                     loss_threshold=0.693, entropy_thresholds=None):
    """Plot per-category mean loss curves."""
    low_t = ENTROPY_LOW if entropy_thresholds is None else entropy_thresholds[0]
    high_t = ENTROPY_HIGH if entropy_thresholds is None else entropy_thresholds[1]

    categories = {}
    for eid, record in tracker.records.items():
        h = record.annotation_entropy
        if h is None:
            continue
        if h < low_t:
            cat = "clean"
        elif h < high_t:
            cat = "ambiguous"
        else:
            cat = "contested"
        categories.setdefault(cat, []).append(eid)

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
        ax.plot(steps, losses, color=colors.get(cat_name, "gray"),
                marker=markers.get(cat_name, "."), markersize=4, linewidth=2,
                label=f"{cat_name} (n={n_examples})", alpha=0.9)

    ax.axhline(loss_threshold, color="gray", linestyle="--", linewidth=1.0,
               alpha=0.6, label=f"threshold $\\theta = {loss_threshold:.2f}$")
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
    print(f"  Saved figure: {output_path}")


# =========================================================================== #
# EXPERIMENT 1: Entropy-Bucket Ablation
# =========================================================================== #

def run_entropy_bucket_ablation(args):
    """Train on different entropy subsets to test causal direction.

    Three conditions:
        - low_only:  ChaosNLI examples with H < 0.4 + 20K SNLI
        - high_only: ChaosNLI examples with H >= 0.7 + 20K SNLI
        - balanced:  Equal examples per entropy bin + 20K SNLI

    All use RoBERTa-base, LoRA r=4, on SNLI.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Entropy-Bucket Ablation")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "entropy_buckets"
    figure_dir = FIGURE_DIR / "entropy_buckets"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load ChaosNLI-SNLI data with consistent split
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Categorize training examples by entropy
    train_idx = chaosnli["train_indices"]
    train_entropies = np.array([chaosnli["entropies"][i] for i in train_idx])
    train_example_ids = [chaosnli["example_ids"][i] for i in train_idx]

    low_mask = train_entropies < ENTROPY_LOW
    high_mask = train_entropies >= ENTROPY_HIGH
    mid_mask = (~low_mask) & (~high_mask)

    low_indices = np.where(low_mask)[0]
    mid_indices = np.where(mid_mask)[0]
    high_indices = np.where(high_mask)[0]

    print(f"  ChaosNLI train split: {len(train_idx)} examples")
    print(f"    Low entropy (H < {ENTROPY_LOW}):  {len(low_indices)}")
    print(f"    Mid entropy:                      {len(mid_indices)}")
    print(f"    High entropy (H >= {ENTROPY_HIGH}): {len(high_indices)}")

    # Balanced: min count per bin, sample equally
    min_count = min(len(low_indices), len(mid_indices), len(high_indices))
    print(f"  Balanced sampling: {min_count} per bin = {3*min_count} total")

    # Load bulk SNLI
    bulk = load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

    # Validation set (shared across conditions)
    val_premises = [chaosnli["premises"][i] for i in chaosnli["val_indices"]]
    val_hypotheses = [chaosnli["hypotheses"][i] for i in chaosnli["val_indices"]]
    val_labels = [chaosnli["majority_labels"][i] for i in chaosnli["val_indices"]]
    val_example_ids = [chaosnli["example_ids"][i] for i in chaosnli["val_indices"]]
    val_entropies = [chaosnli["entropies"][i] for i in chaosnli["val_indices"]]

    # Define conditions
    conditions = {
        "low_only": low_indices,
        "high_only": high_indices,
    }

    # For balanced: sample min_count from each bin
    rng = np.random.RandomState(SEEDS[0])
    balanced_indices = np.concatenate([
        rng.choice(low_indices, size=min_count, replace=False),
        rng.choice(mid_indices, size=min_count, replace=False),
        rng.choice(high_indices, size=min_count, replace=False),
    ])
    conditions["balanced"] = balanced_indices

    all_results = []

    seeds_to_run = [args.seed] if args.seed else SEEDS

    for condition_name, subset_indices in conditions.items():
        for seed in seeds_to_run:
            run_id = f"entropy_bucket_{condition_name}_s{seed}"

            # Check if already done
            result_path = output_dir / f"{run_id}.json"
            if result_path.exists() and not args.force:
                print(f"\n  Skipping {run_id} (exists). Use --force to rerun.")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n{'='*60}")
            print(f"  Running: {run_id}")
            print(f"  ChaosNLI subset: {len(subset_indices)} examples")
            print(f"{'='*60}")

            set_seed(seed)

            # Build ChaosNLI subset for this condition
            cn_premises = [chaosnli["premises"][train_idx[i]] for i in subset_indices]
            cn_hypotheses = [chaosnli["hypotheses"][train_idx[i]] for i in subset_indices]
            cn_labels = [chaosnli["majority_labels"][train_idx[i]] for i in subset_indices]
            cn_eids = [chaosnli["example_ids"][train_idx[i]] for i in subset_indices]
            cn_entropies = [chaosnli["entropies"][train_idx[i]] for i in subset_indices]

            # Combined training: bulk SNLI + ChaosNLI subset
            combined_premises = list(bulk["premises"]) + cn_premises
            combined_hypotheses = list(bulk["hypotheses"]) + cn_hypotheses
            combined_labels = list(bulk["labels"]) + cn_labels
            combined_eids = [f"snli_{i}" for i in range(len(bulk["premises"]))] + cn_eids
            combined_entropies = [None] * len(bulk["premises"]) + cn_entropies

            # Create datasets
            train_dataset = pilot.NLIDataset(
                premises=combined_premises, hypotheses=combined_hypotheses,
                labels=combined_labels, example_ids=combined_eids,
                entropies=combined_entropies, tokenizer=tokenizer, max_length=128,
            )

            # Tracking: use ALL ChaosNLI train examples (not just the subset)
            # so we can measure how the model learns examples it was NOT trained on
            all_cn_premises = [chaosnli["premises"][i] for i in train_idx]
            all_cn_hypotheses = [chaosnli["hypotheses"][i] for i in train_idx]
            all_cn_labels = [chaosnli["majority_labels"][i] for i in train_idx]
            all_cn_eids = [chaosnli["example_ids"][i] for i in train_idx]
            all_cn_entropies = [chaosnli["entropies"][i] for i in train_idx]

            tracking_dataset = pilot.ChaosNLIDataset(
                premises=all_cn_premises, hypotheses=all_cn_hypotheses,
                labels=all_cn_labels, example_ids=all_cn_eids,
                entropies=all_cn_entropies, tokenizer=tokenizer, max_length=128,
            )

            val_dataset = pilot.ChaosNLIDataset(
                premises=val_premises, hypotheses=val_hypotheses,
                labels=val_labels, example_ids=val_example_ids,
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
            model = create_lora_model(
                model_name="roberta-base", num_labels=3, rank=4,
                target_modules=["query", "value"],
            )

            # Tracker
            tracker = TemporalTracker(loss_threshold=0.693)
            tracker.register_examples(
                example_ids=all_cn_eids,
                true_labels=all_cn_labels,
                annotation_entropies=all_cn_entropies,
            )

            # Class weights
            all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
            label_counts = torch.bincount(all_labels_t, minlength=3).float()
            class_weights = (1.0 / label_counts.clamp(min=1))
            class_weights = class_weights / class_weights.sum() * 3

            t0 = time.time()
            history = train_with_tracking(
                model=model, train_loader=train_loader,
                tracking_loader=tracking_loader, val_loader=val_loader,
                tracker=tracker, n_epochs=5, learning_rate=2e-5,
                eval_every_n_steps=100, device=device,
                class_weights=class_weights,
            )
            elapsed = time.time() - t0

            # Compute correlations
            aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
            valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
            if valid.sum() >= 3:
                rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
            else:
                rho, p = 0.0, 1.0

            # Save tracker
            tracker_path = output_dir / f"{run_id}_tracker.json"
            tracker.save(tracker_path)

            result = {
                "experiment": "entropy_bucket_ablation",
                "condition": condition_name,
                "seed": seed,
                "n_chaosnli_train": len(subset_indices),
                "n_bulk_train": len(bulk["premises"]),
                "aulc_rho": float(rho),
                "aulc_p": float(p),
                "final_val_acc": history["val_accuracy"][-1],
                "final_train_loss": history["train_loss"][-1],
                "elapsed_seconds": elapsed,
                "tracking_steps": history["tracking_steps"],
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

            print(f"  {run_id}: rho={rho:+.4f} (p={p:.2e}), "
                  f"val_acc={result['final_val_acc']:.4f}, time={elapsed:.0f}s")

            # Hero figure
            plot_hero_figure(
                tracker, history["tracking_steps"],
                figure_dir / f"hero_{run_id}.png",
                title_suffix=f" ({condition_name}, seed={seed})",
            )

            # Free memory
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Entropy-Bucket Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Seed':>5} {'AULC rho':>10} {'p-value':>12} {'Val Acc':>10}")
    print("-" * 55)
    for r in sorted(all_results, key=lambda x: (x["condition"], x["seed"])):
        print(f"{r['condition']:<15} {r['seed']:>5} {r['aulc_rho']:>+10.4f} "
              f"{r['aulc_p']:>12.2e} {r['final_val_acc']:>10.4f}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")


# =========================================================================== #
# EXPERIMENT 2: Gradient Norms by Entropy Bin
# =========================================================================== #

def run_gradient_norms(args):
    """Compute per-example gradient norms grouped by entropy category.

    Runs a single forward+backward pass per example to get gradient norms,
    then groups by entropy bin and reports statistics.
    Uses RoBERTa-base r=4 on SNLI, seed=42.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Gradient Norms by Entropy Bin")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "gradient_norms"
    figure_dir = FIGURE_DIR / "gradient_norms"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "gradient_norms_roberta_snli_r4_s42.json"
    if result_path.exists() and not args.force:
        print(f"  Skipping (exists): {result_path}. Use --force to rerun.")
        return

    seed = 42
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load data
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])
    bulk = load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

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

    # Combined training set
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

    # Tracking dataset: one example at a time for per-example gradient norms
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
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )
    # Single-example loader for gradient norm computation
    grad_loader = DataLoader(
        tracking_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Create and train model
    model = create_lora_model(
        model_name="roberta-base", num_labels=3, rank=4,
        target_modules=["query", "value"],
    )

    tracker = TemporalTracker(loss_threshold=0.693, track_gradients=True)
    tracker.register_examples(
        example_ids=cn_eids, true_labels=cn_labels,
        annotation_entropies=cn_entropies,
    )

    # Class weights
    all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_labels_t, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * 3

    # Train the model (same as standard pipeline)
    tracking_loader_batched = DataLoader(
        tracking_dataset, batch_size=64, shuffle=False,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
    )

    history = train_with_tracking(
        model=model, train_loader=train_loader,
        tracking_loader=tracking_loader_batched, val_loader=val_loader,
        tracker=tracker, n_epochs=5, learning_rate=2e-5,
        eval_every_n_steps=100, device=device, class_weights=class_weights,
    )

    # Now compute per-example gradient norms on the trained model
    print("\n  Computing per-example gradient norms on trained model...")
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    grad_norms = {}  # example_id -> gradient norm

    for batch in tqdm(grad_loader, desc="  Gradient norms", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        eid = batch["example_id"][0]

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels).squeeze()
        loss.backward()

        # Compute total gradient norm across trainable params
        total_norm = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms[str(eid)] = total_norm

    # Group by entropy category
    categories = {"clean": [], "ambiguous": [], "contested": []}
    for i, eid in enumerate(cn_eids):
        h = cn_entropies[i]
        gn = grad_norms.get(eid, np.nan)
        if h < ENTROPY_LOW:
            categories["clean"].append(gn)
        elif h < ENTROPY_HIGH:
            categories["ambiguous"].append(gn)
        else:
            categories["contested"].append(gn)

    # Statistics
    stats_by_cat = {}
    for cat, norms in categories.items():
        norms = np.array(norms)
        stats_by_cat[cat] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
            "n": len(norms),
        }
        print(f"  {cat}: mean={stats_by_cat[cat]['mean']:.4f}, "
              f"std={stats_by_cat[cat]['std']:.4f}, n={len(norms)}")

    # Correlation: gradient norm vs entropy
    all_gn = np.array([grad_norms.get(eid, np.nan) for eid in cn_eids])
    all_ent = np.array(cn_entropies)
    valid = np.isfinite(all_gn) & np.isfinite(all_ent)
    rho_gn, p_gn = stats.spearmanr(all_gn[valid], all_ent[valid])
    print(f"\n  Gradient norm vs entropy: Spearman rho={rho_gn:+.4f} (p={p_gn:.2e})")

    # Kruskal-Wallis test across categories
    kw_stat, kw_p = stats.kruskal(
        categories["clean"], categories["ambiguous"], categories["contested"]
    )
    print(f"  Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.2e}")

    # Save results
    result = {
        "experiment": "gradient_norms",
        "model": "roberta-base",
        "dataset": "snli",
        "config": "r4",
        "seed": seed,
        "stats_by_category": stats_by_cat,
        "spearman_rho": float(rho_gn),
        "spearman_p": float(p_gn),
        "kruskal_wallis_H": float(kw_stat),
        "kruskal_wallis_p": float(kw_p),
        "val_accuracy": history["val_accuracy"][-1],
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {result_path}")

    # Box plot
    fig, ax = plt.subplots(figsize=(5, 4))
    box_data = [categories["clean"], categories["ambiguous"], categories["contested"]]
    box_labels = [
        f"Clean\n(n={len(categories['clean'])})",
        f"Ambiguous\n(n={len(categories['ambiguous'])})",
        f"Contested\n(n={len(categories['contested'])})",
    ]
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    colors = ["#2166AC", "#F4A582", "#B2182B"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Gradient Norm", fontsize=11)
    ax.set_title("Per-Example Gradient Norms by Entropy Category", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig_path = figure_dir / "gradient_norms_boxplot.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# =========================================================================== #
# EXPERIMENT 3: Alternative Entropy Binning (Quartile/Tercile)
# =========================================================================== #

def run_alt_binning(args):
    """Re-analyze existing results using quartile and tercile binning.

    Shows that correlation results are robust to binning choice.
    Runs on existing tracker data -- no GPU needed.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Alternative Entropy Binning")
    print("=" * 70)

    output_dir = OUTPUT_DIR / "alt_binning"
    figure_dir = FIGURE_DIR / "alt_binning"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_info in MODELS:
        for dataset in DATASETS:
            for config in CONFIGS:
                for seed in SEEDS:
                    tracker_path = find_tracker_path(
                        model_info["name"], dataset, config, seed,
                    )
                    if tracker_path is None:
                        print(f"  SKIP (no tracker): {model_info['name']} "
                              f"{dataset} {config_str(config)} s{seed}")
                        continue

                    tracker = TemporalTracker.load(tracker_path)
                    aulc_arr, ent_arr = compute_aulc_from_tracker(tracker)
                    valid = np.isfinite(aulc_arr) & np.isfinite(ent_arr)

                    if valid.sum() < 10:
                        continue

                    aulc_v = aulc_arr[valid]
                    ent_v = ent_arr[valid]

                    # Original fixed thresholds
                    rho_fixed, p_fixed = stats.spearmanr(aulc_v, ent_v)

                    # Quartile-based binning
                    q25, q50, q75 = np.percentile(ent_v, [25, 50, 75])
                    quartile_bins = np.digitize(ent_v, [q25, q50, q75])
                    # Compute mean AULC per quartile bin
                    quartile_means = [aulc_v[quartile_bins == b].mean()
                                      for b in range(4) if (quartile_bins == b).any()]
                    # Spearman on (quartile_bin, aulc) -- same as on raw
                    rho_quartile, p_quartile = stats.spearmanr(aulc_v, ent_v)

                    # Tercile-based binning
                    t33, t67 = np.percentile(ent_v, [33.33, 66.67])
                    tercile_bins = np.digitize(ent_v, [t33, t67])
                    rho_tercile, p_tercile = stats.spearmanr(aulc_v, ent_v)

                    # Compute mean AULC per bin for each scheme
                    # (demonstrating monotonic increase)
                    fixed_bins = np.digitize(ent_v, [ENTROPY_LOW, ENTROPY_HIGH])
                    fixed_means = {}
                    for b, name in enumerate(["clean", "ambiguous", "contested"]):
                        mask = fixed_bins == b
                        if mask.any():
                            fixed_means[name] = {
                                "mean_aulc": float(aulc_v[mask].mean()),
                                "std_aulc": float(aulc_v[mask].std()),
                                "n": int(mask.sum()),
                                "mean_entropy": float(ent_v[mask].mean()),
                            }

                    quartile_means_dict = {}
                    for b in range(4):
                        mask = quartile_bins == b
                        if mask.any():
                            quartile_means_dict[f"Q{b+1}"] = {
                                "mean_aulc": float(aulc_v[mask].mean()),
                                "std_aulc": float(aulc_v[mask].std()),
                                "n": int(mask.sum()),
                                "mean_entropy": float(ent_v[mask].mean()),
                            }

                    tercile_means_dict = {}
                    for b, name in enumerate(["T1_low", "T2_mid", "T3_high"]):
                        mask = tercile_bins == b
                        if mask.any():
                            tercile_means_dict[name] = {
                                "mean_aulc": float(aulc_v[mask].mean()),
                                "std_aulc": float(aulc_v[mask].std()),
                                "n": int(mask.sum()),
                                "mean_entropy": float(ent_v[mask].mean()),
                            }

                    result = {
                        "model": model_info["name"],
                        "dataset": dataset,
                        "config": config_str(config),
                        "seed": seed,
                        "n": int(valid.sum()),
                        "rho_fixed": float(rho_fixed),
                        "p_fixed": float(p_fixed),
                        "fixed_thresholds": [ENTROPY_LOW, ENTROPY_HIGH],
                        "fixed_bin_stats": fixed_means,
                        "quartile_thresholds": [float(q25), float(q50), float(q75)],
                        "quartile_bin_stats": quartile_means_dict,
                        "tercile_thresholds": [float(t33), float(t67)],
                        "tercile_bin_stats": tercile_means_dict,
                    }
                    all_results.append(result)

    # Print summary table: confirm monotonic AULC increase across all binning schemes
    print(f"\n{'='*80}")
    print("Alternative Binning Summary (mean AULC per bin should increase with entropy)")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Data':<6} {'Cfg':<6} {'Seed':>4}  "
          f"{'Fixed(3)':<25} {'Quartile(4)':<35} {'Tercile(3)':<25}")
    print("-" * 130)

    for r in all_results:
        fixed_str = " / ".join(
            f"{v['mean_aulc']:.3f}" for v in r["fixed_bin_stats"].values()
        )
        quartile_str = " / ".join(
            f"{v['mean_aulc']:.3f}" for v in r["quartile_bin_stats"].values()
        )
        tercile_str = " / ".join(
            f"{v['mean_aulc']:.3f}" for v in r["tercile_bin_stats"].values()
        )
        print(f"{r['model']:<25} {r['dataset']:<6} {r['config']:<6} {r['seed']:>4}  "
              f"{fixed_str:<25} {quartile_str:<35} {tercile_str:<25}")

    summary_path = output_dir / "alt_binning_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    # Generate comparison figure for one representative condition
    # (roberta-base, snli, r4, seed 42)
    representative = [r for r in all_results
                      if r["model"] == "roberta-base" and r["dataset"] == "snli"
                      and r["config"] == "r4" and r["seed"] == 42]
    if representative:
        r = representative[0]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

        # Fixed binning
        ax = axes[0]
        names = list(r["fixed_bin_stats"].keys())
        means = [r["fixed_bin_stats"][n]["mean_aulc"] for n in names]
        stds = [r["fixed_bin_stats"][n]["std_aulc"] for n in names]
        ns = [r["fixed_bin_stats"][n]["n"] for n in names]
        colors = ["#2166AC", "#F4A582", "#B2182B"]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                      color=colors[:len(names)], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, ns)])
        ax.set_title("Fixed Thresholds\n(0.4, 0.7)", fontsize=10)
        ax.set_ylabel("Mean AULC", fontsize=10)

        # Quartile binning
        ax = axes[1]
        names = list(r["quartile_bin_stats"].keys())
        means = [r["quartile_bin_stats"][n]["mean_aulc"] for n in names]
        stds = [r["quartile_bin_stats"][n]["std_aulc"] for n in names]
        ns = [r["quartile_bin_stats"][n]["n"] for n in names]
        q_colors = ["#2166AC", "#67A9CF", "#FDDBC7", "#B2182B"]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                      color=q_colors[:len(names)], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, ns)])
        thresholds_str = ", ".join(f"{t:.2f}" for t in r["quartile_thresholds"])
        ax.set_title(f"Quartile Bins\n({thresholds_str})", fontsize=10)

        # Tercile binning
        ax = axes[2]
        names = list(r["tercile_bin_stats"].keys())
        means = [r["tercile_bin_stats"][n]["mean_aulc"] for n in names]
        stds = [r["tercile_bin_stats"][n]["std_aulc"] for n in names]
        ns = [r["tercile_bin_stats"][n]["n"] for n in names]
        t_colors = ["#2166AC", "#F4A582", "#B2182B"]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                      color=t_colors[:len(names)], alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, ns)])
        thresholds_str = ", ".join(f"{t:.2f}" for t in r["tercile_thresholds"])
        ax.set_title(f"Tercile Bins\n({thresholds_str})", fontsize=10)

        fig.suptitle("AULC by Entropy Bin (RoBERTa-base, SNLI, r=4, seed=42)",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        fig_path = figure_dir / "binning_comparison.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved figure: {fig_path}")


# =========================================================================== #
# EXPERIMENT 4: Bootstrap Confidence Intervals
# =========================================================================== #

def run_bootstrap_ci(args):
    """Compute 95% bootstrap CIs on Spearman rho for all 18 conditions.

    Uses 10,000 bootstrap resamples. Runs on existing tracker data (CPU-only).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Bootstrap Confidence Intervals")
    print("=" * 70)

    output_dir = OUTPUT_DIR / "bootstrap_ci"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.analysis.entropy_correlation import bootstrap_correlation_ci

    all_results = []
    n_bootstrap = 10000

    for model_info in MODELS:
        for dataset in DATASETS:
            for config in CONFIGS:
                for seed in SEEDS:
                    tracker_path = find_tracker_path(
                        model_info["name"], dataset, config, seed,
                    )
                    if tracker_path is None:
                        print(f"  SKIP: {model_info['name']} {dataset} "
                              f"{config_str(config)} s{seed}")
                        continue

                    tracker = TemporalTracker.load(tracker_path)
                    aulc_arr, ent_arr = compute_aulc_from_tracker(tracker)

                    ci_result = bootstrap_correlation_ci(
                        learning_times=aulc_arr,
                        entropies=ent_arr,
                        n_bootstrap=n_bootstrap,
                        ci=0.95,
                        seed=seed,
                    )

                    result = {
                        "model": model_info["name"],
                        "dataset": dataset,
                        "config": config_str(config),
                        "seed": seed,
                        "rho": ci_result["rho"],
                        "ci_lower": ci_result["ci_lower"],
                        "ci_upper": ci_result["ci_upper"],
                        "n": ci_result["n"],
                        "n_bootstrap": n_bootstrap,
                    }
                    all_results.append(result)

                    print(f"  {model_info['name']:25s} {dataset:5s} "
                          f"{config_str(config):6s} s{seed}: "
                          f"rho={ci_result['rho']:+.4f} "
                          f"[{ci_result['ci_lower']:+.4f}, "
                          f"{ci_result['ci_upper']:+.4f}]")

    # Print formatted table
    print(f"\n{'='*90}")
    print("Bootstrap 95% CIs on Spearman rho (AULC vs Entropy)")
    print(f"{'='*90}")
    print(f"{'Model':<25} {'Data':<6} {'Cfg':<6} {'Seed':>4}  "
          f"{'rho':>8} {'95% CI':>20} {'n':>6}")
    print("-" * 80)
    for r in all_results:
        ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
        print(f"{r['model']:<25} {r['dataset']:<6} {r['config']:<6} "
              f"{r['seed']:>4}  {r['rho']:>+8.4f} {ci_str:>20} {r['n']:>6}")

    # Aggregate by condition (average over seeds)
    print(f"\n{'='*70}")
    print("Aggregated over seeds (mean rho, mean CI bounds)")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Data':<6} {'Cfg':<6}  "
          f"{'mean rho':>10} {'mean CI':>25}")
    print("-" * 75)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["dataset"], r["config"])
        grouped[key].append(r)

    aggregated = []
    for key, results in sorted(grouped.items()):
        mean_rho = np.mean([r["rho"] for r in results])
        mean_ci_lo = np.mean([r["ci_lower"] for r in results])
        mean_ci_hi = np.mean([r["ci_upper"] for r in results])
        std_rho = np.std([r["rho"] for r in results])
        ci_str = f"[{mean_ci_lo:+.4f}, {mean_ci_hi:+.4f}]"
        print(f"{key[0]:<25} {key[1]:<6} {key[2]:<6}  "
              f"{mean_rho:>+10.4f} {ci_str:>25}")
        aggregated.append({
            "model": key[0], "dataset": key[1], "config": key[2],
            "mean_rho": float(mean_rho), "std_rho": float(std_rho),
            "mean_ci_lower": float(mean_ci_lo), "mean_ci_upper": float(mean_ci_hi),
        })

    summary = {
        "per_seed": all_results,
        "aggregated": aggregated,
        "n_bootstrap": n_bootstrap,
    }
    summary_path = output_dir / "bootstrap_ci_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")


# =========================================================================== #
# EXPERIMENT 5: Kendall tau-b Robustness Check
# =========================================================================== #

def run_kendall(args):
    """Compute Kendall tau-b for all 18 conditions alongside Spearman rho.

    Shows concordance between the two rank correlation measures.
    Runs on existing tracker data (CPU-only).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Kendall tau-b Robustness Check")
    print("=" * 70)

    output_dir = OUTPUT_DIR / "kendall"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_info in MODELS:
        for dataset in DATASETS:
            for config in CONFIGS:
                for seed in SEEDS:
                    tracker_path = find_tracker_path(
                        model_info["name"], dataset, config, seed,
                    )
                    if tracker_path is None:
                        continue

                    tracker = TemporalTracker.load(tracker_path)
                    aulc_arr, ent_arr = compute_aulc_from_tracker(tracker)
                    valid = np.isfinite(aulc_arr) & np.isfinite(ent_arr)

                    if valid.sum() < 3:
                        continue

                    aulc_v = aulc_arr[valid]
                    ent_v = ent_arr[valid]

                    rho, p_rho = stats.spearmanr(aulc_v, ent_v)
                    tau, p_tau = stats.kendalltau(aulc_v, ent_v)

                    result = {
                        "model": model_info["name"],
                        "dataset": dataset,
                        "config": config_str(config),
                        "seed": seed,
                        "n": int(valid.sum()),
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_rho),
                        "kendall_tau": float(tau),
                        "kendall_p": float(p_tau),
                    }
                    all_results.append(result)

    # Table
    print(f"\n{'='*95}")
    print("Spearman rho vs Kendall tau-b (AULC vs Entropy)")
    print(f"{'='*95}")
    print(f"{'Model':<25} {'Data':<6} {'Cfg':<6} {'Seed':>4}  "
          f"{'Spearman':>10} {'p':>10} {'Kendall':>10} {'p':>10} {'n':>5}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['model']:<25} {r['dataset']:<6} {r['config']:<6} "
              f"{r['seed']:>4}  {r['spearman_rho']:>+10.4f} {r['spearman_p']:>10.2e} "
              f"{r['kendall_tau']:>+10.4f} {r['kendall_p']:>10.2e} {r['n']:>5}")

    # Concordance: correlate all Spearman rhos with all Kendall taus
    rhos = np.array([r["spearman_rho"] for r in all_results])
    taus = np.array([r["kendall_tau"] for r in all_results])
    concordance_rho, concordance_p = stats.spearmanr(rhos, taus)
    print(f"\nConcordance: Spearman(rhos, taus) = {concordance_rho:.4f} (p={concordance_p:.2e})")
    print(f"Mean Spearman rho: {np.mean(rhos):+.4f} +/- {np.std(rhos):.4f}")
    print(f"Mean Kendall tau:  {np.mean(taus):+.4f} +/- {np.std(taus):.4f}")

    # Aggregate by condition
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["dataset"], r["config"])
        grouped[key].append(r)

    print(f"\n{'='*80}")
    print("Aggregated over seeds")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Data':<6} {'Cfg':<6}  "
          f"{'mean rho':>10} {'mean tau':>10} {'ratio tau/rho':>14}")
    print("-" * 75)

    aggregated = []
    for key, results in sorted(grouped.items()):
        mean_rho = np.mean([r["spearman_rho"] for r in results])
        mean_tau = np.mean([r["kendall_tau"] for r in results])
        ratio = mean_tau / mean_rho if abs(mean_rho) > 1e-6 else float("nan")
        print(f"{key[0]:<25} {key[1]:<6} {key[2]:<6}  "
              f"{mean_rho:>+10.4f} {mean_tau:>+10.4f} {ratio:>14.3f}")
        aggregated.append({
            "model": key[0], "dataset": key[1], "config": key[2],
            "mean_rho": float(mean_rho), "mean_tau": float(mean_tau),
            "ratio": float(ratio),
        })

    summary = {
        "per_seed": all_results,
        "aggregated": aggregated,
        "concordance_rho": float(concordance_rho),
        "concordance_p": float(concordance_p),
    }
    summary_path = output_dir / "kendall_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    # Scatter plot: rho vs tau
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rhos, taus, alpha=0.7, s=40, color="#2166AC")
    lo = min(rhos.min(), taus.min()) - 0.02
    hi = max(rhos.max(), taus.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Spearman rho", fontsize=11)
    ax.set_ylabel("Kendall tau-b", fontsize=11)
    ax.set_title(f"Spearman vs Kendall Concordance\n"
                 f"(r={concordance_rho:.3f}, all 54 runs)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    figure_dir = FIGURE_DIR / "kendall"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figure_dir / "spearman_vs_kendall.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")


# =========================================================================== #
# EXPERIMENT 6: GPT-2 Decoder-Only Model
# =========================================================================== #

def run_gpt2_experiment(args):
    """Fine-tune GPT-2 (decoder-only) with LoRA r=4 on ChaosNLI+SNLI.

    Tests whether encoder-only limitation matters by using a decoder-only
    architecture with the same experimental setup.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: GPT-2 Decoder-Only Model")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "gpt2"
    figure_dir = FIGURE_DIR / "gpt2"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # GPT-2 tokenizer: set pad_token to eos_token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GPT-2 uses left-padding for classification

    # Load data (same as other experiments)
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])
    bulk = load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

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

    all_results = []
    seeds_to_run = [args.seed] if args.seed else SEEDS

    for seed in seeds_to_run:
        run_id = f"gpt2_snli_r4_s{seed}"
        result_path = output_dir / f"{run_id}.json"

        if result_path.exists() and not args.force:
            print(f"  Skipping {run_id} (exists). Use --force to rerun.")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n  Running: {run_id}")
        set_seed(seed)

        # Create datasets (reusing pilot NLIDataset with GPT-2 tokenizer)
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

        model = create_gpt2_seq_cls_model(num_labels=3, rank=4)

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

        t0 = time.time()
        history = train_with_tracking(
            model=model, train_loader=train_loader,
            tracking_loader=tracking_loader, val_loader=val_loader,
            tracker=tracker, n_epochs=5, learning_rate=2e-5,
            eval_every_n_steps=100, device=device,
            class_weights=class_weights,
        )
        elapsed = time.time() - t0

        # Correlations
        aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
        valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
            tau, p_tau = stats.kendalltau(aulc_arr[valid], aulc_ent[valid])
        else:
            rho, p, tau, p_tau = 0.0, 1.0, 0.0, 1.0

        # Save tracker
        tracker.save(output_dir / f"{run_id}_tracker.json")

        result = {
            "experiment": "gpt2_decoder",
            "model": "gpt2",
            "dataset": "snli",
            "config": "r4",
            "seed": seed,
            "aulc_rho": float(rho),
            "aulc_p": float(p),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
            "final_val_acc": history["val_accuracy"][-1],
            "final_train_loss": history["train_loss"][-1],
            "elapsed_seconds": elapsed,
            "tracking_steps": history["tracking_steps"],
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

        print(f"  {run_id}: rho={rho:+.4f} (p={p:.2e}), "
              f"tau={tau:+.4f}, val_acc={result['final_val_acc']:.4f}")

        plot_hero_figure(
            tracker, history["tracking_steps"],
            figure_dir / f"hero_{run_id}.png",
            title_suffix=f" (GPT-2, SNLI, r=4, seed={seed})",
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("GPT-2 Experiment Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  seed={r['seed']}: rho={r['aulc_rho']:+.4f}, "
              f"tau={r.get('kendall_tau', 'N/A')}, "
              f"val_acc={r['final_val_acc']:.4f}")

    if len(all_results) > 1:
        mean_rho = np.mean([r["aulc_rho"] for r in all_results])
        std_rho = np.std([r["aulc_rho"] for r in all_results])
        print(f"  Mean rho: {mean_rho:+.4f} +/- {std_rho:.4f}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {summary_path}")


# =========================================================================== #
# EXPERIMENT 7: ChaosNLI-AlphaNLI (Third Task)
# =========================================================================== #

def run_alphanli_experiment(args):
    """Run the standard pipeline on ChaosNLI-AlphaNLI (abductive NLI).

    AlphaNLI is a 2-class abductive NLI task (choose the more plausible
    hypothesis given two observations). This tests whether the entropy-
    learning dynamics relationship holds on a different NLI variant.

    Uses RoBERTa-base with LoRA r=4.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: ChaosNLI-AlphaNLI (Abductive NLI)")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "alphanli"
    figure_dir = FIGURE_DIR / "alphanli"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load AlphaNLI data
    print("  Loading ChaosNLI-AlphaNLI data...")
    data = load_chaosnli(subset="alphanli", data_dir=CHAOSNLI_DATA_DIR)
    n_examples = len(data["premises"])
    print(f"  AlphaNLI: {n_examples} examples")

    # AlphaNLI is 2-class: hyp1 (label 0) vs hyp2 (label 1)
    # The label_distributions array shape is (n, 2) for AlphaNLI
    # The "premise" is obs1+obs2 concatenated, "hypothesis" is hyp1 or hyp2
    # But in ChaosNLI format, they provide obs1, obs2, hyp1, hyp2 separately
    # The loader should handle this -- let's check the label_distributions shape
    n_classes = data["label_distributions"].shape[1]
    print(f"  Number of classes: {n_classes}")
    print(f"  Label distribution shape: {data['label_distributions'].shape}")

    # Check if we have usable premise/hypothesis pairs
    sample_p = data["premises"][:3]
    sample_h = data["hypotheses"][:3]
    print(f"  Sample premise: {sample_p[0][:80]}...")
    print(f"  Sample hypothesis: {sample_h[0][:80]}...")

    # Compute entropies
    entropies = [
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ]

    # For 2-class, max entropy = log(2) ~ 0.693
    # Adjust thresholds proportionally: 0.4/log(3)*log(2) and 0.7/log(3)*log(2)
    max_ent_3class = np.log(3)
    max_ent_2class = np.log(2)
    alpha_low = ENTROPY_LOW * max_ent_2class / max_ent_3class
    alpha_high = ENTROPY_HIGH * max_ent_2class / max_ent_3class
    print(f"  Adjusted entropy thresholds for 2-class: [{alpha_low:.3f}, {alpha_high:.3f}]")
    print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")

    cats = categorize_by_entropy(np.array(entropies), thresholds=[alpha_low, alpha_high])
    print(f"  Category counts: {cats.counts}")

    # Train/val split
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEEDS[0])
    train_idx, val_idx = next(splitter.split(np.arange(n_examples), cats.categories))

    cn_premises = [data["premises"][i] for i in train_idx]
    cn_hypotheses = [data["hypotheses"][i] for i in train_idx]
    cn_labels = [int(data["majority_labels"][i]) for i in train_idx]
    cn_eids = [data["example_ids"][i] for i in train_idx]
    cn_entropies = [entropies[i] for i in train_idx]

    val_premises = [data["premises"][i] for i in val_idx]
    val_hypotheses = [data["hypotheses"][i] for i in val_idx]
    val_labels = [int(data["majority_labels"][i]) for i in val_idx]
    val_eids = [data["example_ids"][i] for i in val_idx]
    val_entropies = [entropies[i] for i in val_idx]

    print(f"  Train: {len(cn_premises)}, Val: {len(val_premises)}")

    # For AlphaNLI, we train on ChaosNLI-AlphaNLI examples only
    # (no bulk training data available for abductive NLI)
    # But the model still needs enough data -- with ~1225 train examples it
    # should be sufficient given the 2-class setup.

    all_results = []
    seeds_to_run = [args.seed] if args.seed else SEEDS

    for seed in seeds_to_run:
        run_id = f"roberta-base_alphanli_r4_s{seed}"
        result_path = output_dir / f"{run_id}.json"

        if result_path.exists() and not args.force:
            print(f"  Skipping {run_id} (exists). Use --force to rerun.")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n  Running: {run_id}")
        set_seed(seed)

        # Create datasets -- for AlphaNLI, the training set IS the tracked set
        train_dataset = pilot.ChaosNLIDataset(
            premises=cn_premises, hypotheses=cn_hypotheses,
            labels=cn_labels, example_ids=cn_eids,
            entropies=cn_entropies, tokenizer=tokenizer, max_length=128,
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

        # 2-class model
        model = create_lora_model(
            model_name="roberta-base", num_labels=n_classes, rank=4,
            target_modules=["query", "value"],
        )

        tracker = TemporalTracker(loss_threshold=0.693)
        tracker.register_examples(
            example_ids=cn_eids, true_labels=cn_labels,
            annotation_entropies=cn_entropies,
        )

        # Class weights
        all_labels_t = torch.tensor(cn_labels, dtype=torch.long)
        label_counts = torch.bincount(all_labels_t, minlength=n_classes).float()
        class_weights = (1.0 / label_counts.clamp(min=1))
        class_weights = class_weights / class_weights.sum() * n_classes

        t0 = time.time()
        history = train_with_tracking(
            model=model, train_loader=train_loader,
            tracking_loader=tracking_loader, val_loader=val_loader,
            tracker=tracker, n_epochs=5, learning_rate=2e-5,
            eval_every_n_steps=50, device=device,  # More frequent for smaller dataset
            class_weights=class_weights,
        )
        elapsed = time.time() - t0

        # Correlations
        aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
        valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
            tau, p_tau = stats.kendalltau(aulc_arr[valid], aulc_ent[valid])
        else:
            rho, p, tau, p_tau = 0.0, 1.0, 0.0, 1.0

        # Save tracker
        tracker.save(output_dir / f"{run_id}_tracker.json")

        result = {
            "experiment": "alphanli",
            "model": "roberta-base",
            "dataset": "alphanli",
            "n_classes": n_classes,
            "config": "r4",
            "seed": seed,
            "aulc_rho": float(rho),
            "aulc_p": float(p),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
            "final_val_acc": history["val_accuracy"][-1],
            "final_train_loss": history["train_loss"][-1],
            "elapsed_seconds": elapsed,
            "n_train": len(cn_premises),
            "n_val": len(val_premises),
            "entropy_thresholds_adjusted": [alpha_low, alpha_high],
            "tracking_steps": history["tracking_steps"],
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

        print(f"  {run_id}: rho={rho:+.4f} (p={p:.2e}), "
              f"tau={tau:+.4f}, val_acc={result['final_val_acc']:.4f}")

        plot_hero_figure(
            tracker, history["tracking_steps"],
            figure_dir / f"hero_{run_id}.png",
            title_suffix=f" (RoBERTa, AlphaNLI, r=4, seed={seed})",
            entropy_thresholds=[alpha_low, alpha_high],
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("AlphaNLI Experiment Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  seed={r['seed']}: rho={r['aulc_rho']:+.4f}, "
              f"tau={r.get('kendall_tau', 'N/A')}, "
              f"val_acc={r['final_val_acc']:.4f}")

    if len(all_results) > 1:
        mean_rho = np.mean([r["aulc_rho"] for r in all_results])
        std_rho = np.std([r["aulc_rho"] for r in all_results])
        print(f"  Mean rho: {mean_rho:+.4f} +/- {std_rho:.4f}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {summary_path}")


# =========================================================================== #
# EXPERIMENT 8: Soft-Label Ablation (KL-divergence training)
# =========================================================================== #


class SoftLabelNLIDataset(Dataset):
    """NLI dataset that returns soft targets for ChaosNLI examples.

    For ChaosNLI examples with 100-annotator distributions, returns
    the full distribution as soft_targets (shape (3,), sums to 1.0).
    For bulk SNLI examples, soft_targets is None (they only have
    hard labels).
    """

    def __init__(
        self,
        premises: List[str],
        hypotheses: List[str],
        labels: List[int],
        example_ids: List[str],
        entropies: List[Optional[float]],
        soft_targets: List[Optional[np.ndarray]],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.example_ids = example_ids
        self.entropies = entropies
        self.soft_targets = soft_targets
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
        if self.soft_targets[idx] is not None:
            item["soft_targets"] = torch.tensor(
                self.soft_targets[idx], dtype=torch.float32,
            )
            item["has_soft_target"] = torch.tensor(1, dtype=torch.long)
        else:
            # Placeholder -- will not be used for loss computation
            item["soft_targets"] = torch.zeros(3, dtype=torch.float32)
            item["has_soft_target"] = torch.tensor(0, dtype=torch.long)
        return item


def train_with_soft_label_tracking(
    model: nn.Module,
    train_loader: DataLoader,
    tracking_loader: DataLoader,
    val_loader: DataLoader,
    tracker: TemporalTracker,
    use_soft_labels: bool = False,
    n_epochs: int = 5,
    learning_rate: float = 2e-5,
    eval_every_n_steps: int = 100,
    device: str = "mps",
    max_grad_norm: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Train model with optional soft-label KL-divergence loss.

    When use_soft_labels is True, ChaosNLI examples use KL-div loss
    with the full annotator distribution as target. Bulk SNLI examples
    (without soft targets) always use standard cross-entropy.

    Tracking always uses cross-entropy with majority-vote labels so
    both conditions are comparable.

    Args:
        model: Model to train.
        train_loader: Training DataLoader (SoftLabelNLIDataset).
        tracking_loader: Tracking DataLoader (ChaosNLIDataset, hard labels).
        val_loader: Validation DataLoader.
        tracker: TemporalTracker for per-example losses.
        use_soft_labels: If True, use KL-div for soft-target examples.
        n_epochs: Number of training epochs.
        learning_rate: Learning rate.
        eval_every_n_steps: Steps between tracking passes.
        device: Device string.
        max_grad_norm: Gradient clipping norm.
        class_weights: Optional class weights for CE loss.

    Returns:
        Training history dictionary.
    """
    import torch.nn.functional as F

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_fn_mean = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "tracking_steps": [],
    }

    global_step = 0
    tracking_step = 0

    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    print(f"  Soft-label mode: {use_soft_labels}")

    # Initial tracking pass (step 0) -- always uses hard-label CE
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
            logits = outputs.logits

            if use_soft_labels:
                soft_targets = batch["soft_targets"].to(device)
                has_soft = batch["has_soft_target"].to(device)

                # Split batch: soft-target examples use KL-div,
                # hard-label examples use standard CE
                soft_mask = has_soft.bool()
                hard_mask = ~soft_mask

                loss_total = torch.tensor(0.0, device=device)
                n_soft = soft_mask.sum().item()
                n_hard = hard_mask.sum().item()
                batch_size = labels.size(0)

                if n_soft > 0:
                    log_probs = F.log_softmax(logits[soft_mask], dim=-1)
                    kl_loss = F.kl_div(
                        log_probs,
                        soft_targets[soft_mask],
                        reduction="batchmean",
                    )
                    loss_total = loss_total + kl_loss * (n_soft / batch_size)

                if n_hard > 0:
                    ce_loss = loss_fn_mean(logits[hard_mask], labels[hard_mask])
                    loss_total = loss_total + ce_loss * (n_hard / batch_size)

                loss = loss_total
            else:
                # Standard hard-label cross-entropy
                loss = loss_fn_mean(logits, labels)

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

        # End-of-epoch tracking
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

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    history["total_tracking_steps"] = tracking_step
    return history


def _compute_delta_ell(tracker: TemporalTracker) -> Dict[str, Any]:
    """Compute delta-ell (loss change from start to end) by entropy category.

    delta_ell_i = loss_final - loss_initial
    Negative delta_ell means loss decreased (the model learned the example).
    Positive delta_ell means loss increased (the model un-learned the example).

    Returns dict with per-category statistics and per-example arrays.
    """
    deltas = {}
    entropies_map = {}

    for eid, record in tracker.records.items():
        valid_losses = [l for l in record.losses
                        if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            continue
        delta = valid_losses[-1] - valid_losses[0]
        h = record.annotation_entropy
        if h is None:
            continue
        deltas[eid] = delta
        entropies_map[eid] = h

    # Categorize
    cat_deltas = {"clean": [], "ambiguous": [], "contested": []}
    for eid, delta in deltas.items():
        h = entropies_map[eid]
        if h < ENTROPY_LOW:
            cat_deltas["clean"].append(delta)
        elif h < ENTROPY_HIGH:
            cat_deltas["ambiguous"].append(delta)
        else:
            cat_deltas["contested"].append(delta)

    result = {}
    for cat, vals in cat_deltas.items():
        arr = np.array(vals)
        result[cat] = {
            "mean_delta": float(np.mean(arr)) if len(arr) > 0 else 0.0,
            "std_delta": float(np.std(arr)) if len(arr) > 0 else 0.0,
            "median_delta": float(np.median(arr)) if len(arr) > 0 else 0.0,
            "frac_increased": float((arr > 0).mean()) if len(arr) > 0 else 0.0,
            "n": len(arr),
        }

    # Spearman correlation: delta_ell vs entropy
    all_deltas = np.array([deltas[eid] for eid in deltas])
    all_ents = np.array([entropies_map[eid] for eid in deltas])
    if len(all_deltas) >= 3:
        rho, p = stats.spearmanr(all_deltas, all_ents)
    else:
        rho, p = 0.0, 1.0

    result["spearman_rho"] = float(rho)
    result["spearman_p"] = float(p)
    result["n_total"] = len(deltas)

    return result


def run_soft_label_ablation(args):
    """Test whether un-learning persists under soft-label (KL-div) training.

    Two conditions per seed:
        - hard_label: Standard cross-entropy with majority-vote label
        - soft_label: KL-divergence loss with full annotator distribution

    Both conditions track per-example loss using the SAME metric (CE with
    majority-vote label) so trajectories are directly comparable.

    RoBERTa-base, LoRA r=4, 3 seeds.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: Soft-Label Ablation (KL-Divergence Training)")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "soft_label"
    figure_dir = FIGURE_DIR / "soft_label"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load ChaosNLI-SNLI data
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])

    # Build soft-target lookup: example_id -> probability distribution
    # ChaosNLI label_distributions are integer counts summing to 100
    raw_data = load_chaosnli(subset="snli", data_dir=CHAOSNLI_DATA_DIR)
    soft_target_map = {}
    for i, eid in enumerate(raw_data["example_ids"]):
        counts = raw_data["label_distributions"][i]
        soft_target_map[eid] = counts.astype(np.float64) / 100.0

    train_idx = chaosnli["train_indices"]
    cn_premises = [chaosnli["premises"][i] for i in train_idx]
    cn_hypotheses = [chaosnli["hypotheses"][i] for i in train_idx]
    cn_labels = [chaosnli["majority_labels"][i] for i in train_idx]
    cn_eids = [chaosnli["example_ids"][i] for i in train_idx]
    cn_entropies = [chaosnli["entropies"][i] for i in train_idx]
    cn_soft_targets = [soft_target_map[eid] for eid in cn_eids]

    val_premises = [chaosnli["premises"][i] for i in chaosnli["val_indices"]]
    val_hypotheses = [chaosnli["hypotheses"][i] for i in chaosnli["val_indices"]]
    val_labels = [chaosnli["majority_labels"][i] for i in chaosnli["val_indices"]]
    val_eids = [chaosnli["example_ids"][i] for i in chaosnli["val_indices"]]
    val_entropies = [chaosnli["entropies"][i] for i in chaosnli["val_indices"]]

    # Load bulk SNLI (no soft targets)
    bulk = load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

    # Combined training data
    combined_premises = list(bulk["premises"]) + cn_premises
    combined_hypotheses = list(bulk["hypotheses"]) + cn_hypotheses
    combined_labels = list(bulk["labels"]) + cn_labels
    combined_eids = [f"snli_{i}" for i in range(len(bulk["premises"]))] + cn_eids
    combined_entropies: List[Optional[float]] = [None] * len(bulk["premises"]) + cn_entropies
    combined_soft_targets: List[Optional[np.ndarray]] = (
        [None] * len(bulk["premises"]) + cn_soft_targets
    )

    print(f"  Training data: {len(combined_premises)} examples")
    print(f"    Bulk SNLI (hard labels only): {len(bulk['premises'])}")
    print(f"    ChaosNLI (with soft targets): {len(cn_premises)}")
    print(f"  Validation: {len(val_premises)} examples")

    conditions = ["hard_label", "soft_label"]
    seeds_to_run = [args.seed] if args.seed else SEEDS
    all_results = []

    for seed in seeds_to_run:
        condition_trackers = {}

        for condition in conditions:
            run_id = f"soft_label_{condition}_s{seed}"
            result_path = output_dir / f"{run_id}.json"

            if result_path.exists() and not args.force:
                print(f"\n  Skipping {run_id} (exists). Use --force to rerun.")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                # Load tracker for comparison figure
                tracker_path = output_dir / f"{run_id}_tracker.json"
                if tracker_path.exists():
                    condition_trackers[condition] = TemporalTracker.load(tracker_path)
                continue

            print(f"\n{'='*60}")
            print(f"  Running: {run_id}")
            print(f"{'='*60}")

            set_seed(seed)

            use_soft = (condition == "soft_label")

            # Training dataset: SoftLabelNLIDataset for both conditions
            # (the soft targets are only used when use_soft_labels=True)
            train_dataset = SoftLabelNLIDataset(
                premises=combined_premises,
                hypotheses=combined_hypotheses,
                labels=combined_labels,
                example_ids=combined_eids,
                entropies=combined_entropies,
                soft_targets=combined_soft_targets,
                tokenizer=tokenizer,
                max_length=128,
            )

            # Tracking dataset: ChaosNLI with HARD labels (for comparable tracking)
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
            model = create_lora_model(
                model_name="roberta-base", num_labels=3, rank=4,
                target_modules=["query", "value"],
            )

            # Tracker
            tracker = TemporalTracker(loss_threshold=0.693)
            tracker.register_examples(
                example_ids=cn_eids,
                true_labels=cn_labels,
                annotation_entropies=cn_entropies,
            )

            # Class weights
            all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
            label_counts = torch.bincount(all_labels_t, minlength=3).float()
            class_weights = (1.0 / label_counts.clamp(min=1))
            class_weights = class_weights / class_weights.sum() * 3

            t0 = time.time()
            history = train_with_soft_label_tracking(
                model=model,
                train_loader=train_loader,
                tracking_loader=tracking_loader,
                val_loader=val_loader,
                tracker=tracker,
                use_soft_labels=use_soft,
                n_epochs=5,
                learning_rate=2e-5,
                eval_every_n_steps=100,
                device=device,
                class_weights=class_weights,
            )
            elapsed = time.time() - t0

            # Compute AULC correlation
            aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
            valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
            if valid.sum() >= 3:
                rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
            else:
                rho, p = 0.0, 1.0

            # Compute delta-ell analysis
            delta_ell = _compute_delta_ell(tracker)

            # Save tracker
            tracker_path = output_dir / f"{run_id}_tracker.json"
            tracker.save(tracker_path)
            condition_trackers[condition] = tracker

            result = {
                "experiment": "soft_label_ablation",
                "condition": condition,
                "seed": seed,
                "n_train": len(combined_premises),
                "aulc_rho": float(rho),
                "aulc_p": float(p),
                "final_val_acc": history["val_accuracy"][-1],
                "final_train_loss": history["train_loss"][-1],
                "elapsed_seconds": elapsed,
                "tracking_steps": history["tracking_steps"],
                "delta_ell": delta_ell,
            }

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)

            print(f"  {run_id}: rho={rho:+.4f} (p={p:.2e}), "
                  f"val_acc={result['final_val_acc']:.4f}")
            print(f"  Delta-ell by category:")
            for cat in ["clean", "ambiguous", "contested"]:
                if cat in delta_ell:
                    d = delta_ell[cat]
                    print(f"    {cat}: mean={d['mean_delta']:+.4f}, "
                          f"frac_increased={d['frac_increased']:.3f}, n={d['n']}")

            # Hero figure for this condition
            plot_hero_figure(
                tracker, history["tracking_steps"],
                figure_dir / f"hero_{run_id}.png",
                title_suffix=f" ({condition}, seed={seed})",
            )

            # Free memory
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

        # Comparison figure: hard vs soft side by side (if both ran)
        if "hard_label" in condition_trackers and "soft_label" in condition_trackers:
            _plot_soft_label_comparison(
                condition_trackers["hard_label"],
                condition_trackers["soft_label"],
                seed=seed,
                output_path=figure_dir / f"comparison_s{seed}.png",
            )

    # Summary
    print(f"\n{'='*60}")
    print("Soft-Label Ablation Summary")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Seed':>5} {'AULC rho':>10} {'p-value':>12} "
          f"{'Val Acc':>10} {'delta_ell contested':>20}")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: (x["condition"], x["seed"])):
        contested_delta = r.get("delta_ell", {}).get("contested", {}).get("mean_delta", float("nan"))
        print(f"{r['condition']:<15} {r['seed']:>5} {r['aulc_rho']:>+10.4f} "
              f"{r['aulc_p']:>12.2e} {r['final_val_acc']:>10.4f} "
              f"{contested_delta:>+20.4f}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")


def _plot_soft_label_comparison(
    tracker_hard: TemporalTracker,
    tracker_soft: TemporalTracker,
    seed: int,
    output_path: Path,
) -> None:
    """Plot side-by-side loss trajectories for hard-label vs soft-label conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    colors = {"clean": "#2166AC", "ambiguous": "#F4A582", "contested": "#B2182B"}
    markers = {"clean": "o", "ambiguous": "s", "contested": "^"}

    for ax, (tracker, title) in zip(axes, [
        (tracker_hard, "Hard Label (CE)"),
        (tracker_soft, "Soft Label (KL-div)"),
    ]):
        categories = {}
        for eid, record in tracker.records.items():
            h = record.annotation_entropy
            if h is None:
                continue
            if h < ENTROPY_LOW:
                cat = "clean"
            elif h < ENTROPY_HIGH:
                cat = "ambiguous"
            else:
                cat = "contested"
            categories.setdefault(cat, []).append(eid)

        mean_losses = tracker.get_mean_loss_by_category(categories)

        for cat_name in ["clean", "ambiguous", "contested"]:
            if cat_name not in mean_losses or len(mean_losses[cat_name]) == 0:
                continue
            losses = mean_losses[cat_name]
            steps = list(range(len(losses)))
            n_examples = len(categories.get(cat_name, []))
            ax.plot(steps, losses, color=colors.get(cat_name, "gray"),
                    marker=markers.get(cat_name, "."), markersize=3,
                    linewidth=2, label=f"{cat_name} (n={n_examples})",
                    alpha=0.9)

        ax.axhline(0.693, color="gray", linestyle="--", linewidth=1.0,
                   alpha=0.6)
        ax.set_xlabel("Tracking Step", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean Cross-Entropy Loss", fontsize=11)

    fig.suptitle(
        f"Hard Label vs Soft Label Training (seed={seed})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison figure: {output_path}")


# =========================================================================== #
# EXPERIMENT 9: Dataset Cartography Comparison
# =========================================================================== #


def _compute_cartography_metrics(
    tracker: TemporalTracker,
) -> Dict[str, Dict[str, float]]:
    """Compute Dataset Cartography metrics from tracker loss trajectories.

    For each example across T checkpoints:
        confidence(t) = exp(-loss(t))  (probability assigned to correct class)
        mean_confidence = mean over t
        variability = std over t
        correctness = fraction of checkpoints where confidence > 0.5

    Cartography classification (Swayamdipta et al. 2020):
        Easy:      high mean_confidence, low variability
        Ambiguous: medium mean_confidence, high variability
        Hard:      low mean_confidence, low variability

    Returns:
        Dict mapping example_id -> {mean_confidence, variability, correctness,
        cartography_class, annotation_entropy, aulc}.
    """
    metrics = {}
    for eid, record in tracker.records.items():
        valid_losses = [l for l in record.losses
                        if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            continue

        # confidence = exp(-loss) = probability assigned to correct class
        confidences = np.exp(-np.array(valid_losses, dtype=np.float64))

        mean_conf = float(np.mean(confidences))
        variability = float(np.std(confidences))
        correctness = float(np.mean(confidences > 0.5))
        aulc = float(np.mean(valid_losses))

        h = record.annotation_entropy if record.annotation_entropy is not None else np.nan

        metrics[eid] = {
            "mean_confidence": mean_conf,
            "variability": variability,
            "correctness": correctness,
            "annotation_entropy": h,
            "aulc": aulc,
        }

    return metrics


def _classify_cartography(
    metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Add cartography class labels based on confidence and variability.

    Uses median splits on the population:
        Easy:      mean_confidence >= median AND variability < median
        Ambiguous: variability >= median
        Hard:      mean_confidence < median AND variability < median
    """
    if not metrics:
        return metrics

    all_conf = np.array([m["mean_confidence"] for m in metrics.values()])
    all_var = np.array([m["variability"] for m in metrics.values()])

    conf_median = float(np.median(all_conf))
    var_median = float(np.median(all_var))

    for eid, m in metrics.items():
        if m["variability"] >= var_median:
            m["cartography_class"] = "ambiguous"
        elif m["mean_confidence"] >= conf_median:
            m["cartography_class"] = "easy"
        else:
            m["cartography_class"] = "hard"

    return metrics


def _entropy_category(h: float) -> str:
    """Classify entropy into clean/ambiguous/contested."""
    if h < ENTROPY_LOW:
        return "clean"
    elif h < ENTROPY_HIGH:
        return "ambiguous"
    else:
        return "contested"


def run_cartography_comparison(args):
    """Compare annotation entropy with Dataset Cartography measures.

    Post-hoc analysis of existing training runs -- NO GPU needed.
    Computes confidence, variability, correctness from loss trajectories,
    correlates with annotation entropy, and tests incremental predictive power.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: Dataset Cartography Comparison")
    print("=" * 70)

    from src.analysis.entropy_correlation import hierarchical_regression

    output_dir = OUTPUT_DIR / "cartography"
    figure_dir = FIGURE_DIR / "cartography"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Find all existing tracker files
    tracker_paths = sorted(EXPANDED_DIR.glob("*_tracker.json")) if EXPANDED_DIR.exists() else []
    # Also check base tracking dir for pilot runs
    if TRACKING_DIR.exists():
        tracker_paths.extend(sorted(TRACKING_DIR.glob("pilot_*.json")))
        tracker_paths.extend(sorted(TRACKING_DIR.glob("fullft_*.json")))

    if not tracker_paths:
        print("  ERROR: No tracker files found in expected locations.")
        print(f"    Checked: {EXPANDED_DIR}")
        print(f"    Checked: {TRACKING_DIR}")
        print("  Run the main training experiments first.")
        return

    print(f"  Found {len(tracker_paths)} tracker files")

    # Aggregate cartography metrics across all runs
    all_run_results = []
    all_metrics_flat = []  # For aggregate analysis

    for tracker_path in tracker_paths:
        run_name = tracker_path.stem
        print(f"\n  Processing: {run_name}")

        try:
            tracker = TemporalTracker.load(tracker_path)
        except Exception as e:
            print(f"    SKIP: could not load tracker: {e}")
            continue

        # Compute cartography metrics
        metrics = _compute_cartography_metrics(tracker)
        if len(metrics) < 10:
            print(f"    SKIP: too few examples ({len(metrics)})")
            continue

        metrics = _classify_cartography(metrics)

        # Extract arrays
        eids = list(metrics.keys())
        mean_conf = np.array([metrics[e]["mean_confidence"] for e in eids])
        variability = np.array([metrics[e]["variability"] for e in eids])
        correctness = np.array([metrics[e]["correctness"] for e in eids])
        entropy = np.array([metrics[e]["annotation_entropy"] for e in eids])
        aulc = np.array([metrics[e]["aulc"] for e in eids])
        carto_class = [metrics[e].get("cartography_class", "unknown") for e in eids]

        valid = np.isfinite(entropy) & np.isfinite(mean_conf) & np.isfinite(variability)
        if valid.sum() < 10:
            print(f"    SKIP: too few valid examples ({valid.sum()})")
            continue

        # 1. Spearman correlations
        rho_conf, p_conf = stats.spearmanr(entropy[valid], mean_conf[valid])
        rho_var, p_var = stats.spearmanr(entropy[valid], variability[valid])
        rho_aulc_ent, p_aulc_ent = stats.spearmanr(entropy[valid], aulc[valid])
        rho_conf_aulc, p_conf_aulc = stats.spearmanr(mean_conf[valid], aulc[valid])
        rho_var_aulc, p_var_aulc = stats.spearmanr(variability[valid], aulc[valid])

        # 2. Overlap analysis: entropy categories vs cartography classes
        entropy_cats = [_entropy_category(h) for h in entropy[valid]]
        carto_cats_valid = [carto_class[i] for i, v in enumerate(valid) if v]

        overlap_matrix = {}
        for e_cat in ["clean", "ambiguous", "contested"]:
            overlap_matrix[e_cat] = {}
            for c_cat in ["easy", "ambiguous", "hard"]:
                count = sum(1 for ec, cc in zip(entropy_cats, carto_cats_valid)
                            if ec == e_cat and cc == c_cat)
                overlap_matrix[e_cat][c_cat] = count

        # 3. Incremental predictive power: AULC ~ confidence + variability + entropy
        valid_aulc = valid & np.isfinite(aulc)
        if valid_aulc.sum() >= 10:
            regression_result = hierarchical_regression(
                learning_times=aulc[valid_aulc],
                entropies=entropy[valid_aulc],
                difficulty_proxies={
                    "mean_confidence": mean_conf[valid_aulc],
                    "variability": variability[valid_aulc],
                },
            )
        else:
            regression_result = {
                "r2_difficulty": 0.0, "r2_full": 0.0,
                "r2_incremental": 0.0, "f_statistic": 0.0, "f_p_value": 1.0,
            }

        run_result = {
            "run_name": run_name,
            "n_examples": int(valid.sum()),
            "rho_entropy_confidence": float(rho_conf),
            "p_entropy_confidence": float(p_conf),
            "rho_entropy_variability": float(rho_var),
            "p_entropy_variability": float(p_var),
            "rho_entropy_aulc": float(rho_aulc_ent),
            "p_entropy_aulc": float(p_aulc_ent),
            "rho_confidence_aulc": float(rho_conf_aulc),
            "p_confidence_aulc": float(p_conf_aulc),
            "rho_variability_aulc": float(rho_var_aulc),
            "p_variability_aulc": float(p_var_aulc),
            "overlap_matrix": overlap_matrix,
            "regression": {
                "r2_cartography_only": regression_result["r2_difficulty"],
                "r2_cartography_plus_entropy": regression_result["r2_full"],
                "r2_incremental_entropy": regression_result["r2_incremental"],
                "f_statistic": regression_result["f_statistic"],
                "f_p_value": regression_result["f_p_value"],
            },
        }
        all_run_results.append(run_result)

        # Collect flat metrics for aggregate figures
        for e in eids:
            if np.isfinite(metrics[e]["annotation_entropy"]):
                all_metrics_flat.append(metrics[e])

        print(f"    n={valid.sum()}: "
              f"rho(ent,conf)={rho_conf:+.3f}, "
              f"rho(ent,var)={rho_var:+.3f}, "
              f"delta_R2={regression_result['r2_incremental']:.4f} "
              f"(p={regression_result['f_p_value']:.2e})")

    if not all_run_results:
        print("\n  No valid runs found. Cannot produce results.")
        return

    # Aggregate statistics
    print(f"\n{'='*70}")
    print("Dataset Cartography Comparison: Aggregate Results")
    print(f"{'='*70}")
    print(f"  Total runs analyzed: {len(all_run_results)}")

    mean_rho_conf = np.mean([r["rho_entropy_confidence"] for r in all_run_results])
    mean_rho_var = np.mean([r["rho_entropy_variability"] for r in all_run_results])
    mean_delta_r2 = np.mean([r["regression"]["r2_incremental_entropy"] for r in all_run_results])
    n_sig = sum(1 for r in all_run_results if r["regression"]["f_p_value"] < 0.05)

    print(f"\n  Mean Spearman rho(entropy, confidence): {mean_rho_conf:+.4f}")
    print(f"  Mean Spearman rho(entropy, variability): {mean_rho_var:+.4f}")
    print(f"  Mean incremental R^2 (entropy beyond cartography): {mean_delta_r2:.4f}")
    print(f"  Runs where entropy is significant (p<0.05): {n_sig}/{len(all_run_results)}")

    # Aggregate overlap matrix
    agg_overlap = {}
    for e_cat in ["clean", "ambiguous", "contested"]:
        agg_overlap[e_cat] = {}
        for c_cat in ["easy", "ambiguous", "hard"]:
            agg_overlap[e_cat][c_cat] = sum(
                r["overlap_matrix"].get(e_cat, {}).get(c_cat, 0)
                for r in all_run_results
            )

    print(f"\n  Aggregate Overlap Matrix (entropy rows x cartography cols):")
    print(f"  {'':>15} {'easy':>10} {'ambiguous':>10} {'hard':>10}")
    for e_cat in ["clean", "ambiguous", "contested"]:
        row = agg_overlap[e_cat]
        total = sum(row.values())
        if total > 0:
            pct = {k: f"{v/total*100:.0f}%" for k, v in row.items()}
        else:
            pct = {k: "0%" for k in row}
        print(f"  {e_cat:>15} {row['easy']:>6} ({pct['easy']:>4}) "
              f"{row['ambiguous']:>6} ({pct['ambiguous']:>4}) "
              f"{row['hard']:>6} ({pct['hard']:>4})")

    # Save results
    summary = {
        "per_run": all_run_results,
        "aggregate": {
            "mean_rho_entropy_confidence": float(mean_rho_conf),
            "mean_rho_entropy_variability": float(mean_rho_var),
            "mean_r2_incremental": float(mean_delta_r2),
            "n_significant_runs": n_sig,
            "n_total_runs": len(all_run_results),
            "aggregate_overlap_matrix": agg_overlap,
        },
    }
    summary_path = output_dir / "cartography_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {summary_path}")

    # --- Figure (a): Scatter plot of confidence vs variability, colored by entropy ---
    if all_metrics_flat:
        _plot_cartography_scatter(all_metrics_flat, figure_dir)
        _plot_correlation_matrix(all_metrics_flat, figure_dir)


def _plot_cartography_scatter(
    metrics_flat: List[Dict[str, float]],
    figure_dir: Path,
) -> None:
    """Scatter plot: mean_confidence vs variability, colored by entropy category."""
    conf = np.array([m["mean_confidence"] for m in metrics_flat])
    var = np.array([m["variability"] for m in metrics_flat])
    ent = np.array([m["annotation_entropy"] for m in metrics_flat])

    # Color by entropy category
    colors_map = {"clean": "#2166AC", "ambiguous": "#F4A582", "contested": "#B2182B"}
    point_colors = []
    for h in ent:
        cat = _entropy_category(h)
        point_colors.append(colors_map.get(cat, "gray"))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot each category separately for legend
    for cat_name, color in colors_map.items():
        mask = np.array([_entropy_category(h) == cat_name for h in ent])
        if mask.sum() == 0:
            continue
        ax.scatter(
            var[mask], conf[mask],
            c=color, s=8, alpha=0.4,
            label=f"{cat_name} (n={mask.sum()})",
            edgecolors="none",
        )

    ax.set_xlabel("Variability (std of confidence)", fontsize=11)
    ax.set_ylabel("Mean Confidence", fontsize=11)
    ax.set_title("Dataset Cartography Map\n(colored by annotation entropy category)",
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower left", markerscale=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig_path = figure_dir / "cartography_scatter.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")


def _plot_correlation_matrix(
    metrics_flat: List[Dict[str, float]],
    figure_dir: Path,
) -> None:
    """Correlation matrix: entropy, confidence, variability, AULC."""
    data = {
        "Entropy": np.array([m["annotation_entropy"] for m in metrics_flat]),
        "Confidence": np.array([m["mean_confidence"] for m in metrics_flat]),
        "Variability": np.array([m["variability"] for m in metrics_flat]),
        "AULC": np.array([m["aulc"] for m in metrics_flat]),
    }

    names = list(data.keys())
    n = len(names)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            valid = np.isfinite(data[names[i]]) & np.isfinite(data[names[j]])
            if valid.sum() >= 3:
                rho, p = stats.spearmanr(data[names[i]][valid], data[names[j]][valid])
                corr_matrix[i, j] = rho
                p_matrix[i, j] = p
            else:
                corr_matrix[i, j] = 0.0
                p_matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            sig = "*" if p_matrix[i, j] < 0.001 else ""
            text_color = "white" if abs(corr_matrix[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.3f}{sig}",
                    ha="center", va="center", fontsize=10, color=text_color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Spearman Correlation Matrix\n(* = p < 0.001)", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho", fontsize=10)

    plt.tight_layout()
    fig_path = figure_dir / "correlation_matrix.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")


# =========================================================================== #
# EXPERIMENT 10: DeBERTa v3-base (Disentangled Attention)
# =========================================================================== #

def create_deberta_model(num_labels: int = 3, rank: int = 4) -> nn.Module:
    """Create DeBERTa v3-base with LoRA for sequence classification.

    DeBERTa v3 uses disentangled attention with separate content and position
    embeddings, making it architecturally distinct from standard BERT/RoBERTa
    bidirectional encoders. This tests whether the entropy-learning dynamics
    relationship generalizes beyond the standard encoder attention mechanism.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base", num_labels=num_labels,
    )

    # DeBERTa v3 ships with mixed fp16/fp32 weights, which causes MPS graph
    # compilation failures ("requires the same element type"). Cast all to fp32.
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


def run_deberta_experiment(args):
    """Fine-tune DeBERTa v3-base with LoRA r=4 on ChaosNLI+SNLI.

    Tests whether the entropy-learning dynamics relationship generalizes
    beyond the BERT/RoBERTa encoder family to DeBERTa v3, which uses
    disentangled attention (separate content and position embeddings).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: DeBERTa v3-base (Disentangled Attention)")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "deberta"
    figure_dir = FIGURE_DIR / "deberta"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Load data (same setup as other experiments)
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])
    bulk = load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

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

    all_results = []
    seeds_to_run = [args.seed] if args.seed else SEEDS

    for seed in seeds_to_run:
        run_id = f"deberta-v3-base_snli_r4_s{seed}"
        result_path = output_dir / f"{run_id}.json"

        if result_path.exists() and not args.force:
            print(f"  Skipping {run_id} (exists). Use --force to rerun.")
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n  Running: {run_id}")
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

        model = create_deberta_model(num_labels=3, rank=4)

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

        t0 = time.time()
        history = train_with_tracking(
            model=model, train_loader=train_loader,
            tracking_loader=tracking_loader, val_loader=val_loader,
            tracker=tracker, n_epochs=5, learning_rate=2e-5,
            eval_every_n_steps=100, device=device,
            class_weights=class_weights,
        )
        elapsed = time.time() - t0

        # Correlations
        aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
        valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
            tau, p_tau = stats.kendalltau(aulc_arr[valid], aulc_ent[valid])
        else:
            rho, p, tau, p_tau = 0.0, 1.0, 0.0, 1.0

        # Save tracker
        tracker.save(output_dir / f"{run_id}_tracker.json")

        result = {
            "experiment": "deberta_v3",
            "model": "deberta-v3-base",
            "dataset": "snli",
            "config": "r4",
            "seed": seed,
            "aulc_rho": float(rho),
            "aulc_p": float(p),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
            "final_val_acc": history["val_accuracy"][-1],
            "final_train_loss": history["train_loss"][-1],
            "elapsed_seconds": elapsed,
            "tracking_steps": history["tracking_steps"],
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)

        print(f"  {run_id}: rho={rho:+.4f} (p={p:.2e}), "
              f"tau={tau:+.4f}, val_acc={result['final_val_acc']:.4f}")

        plot_hero_figure(
            tracker, history["tracking_steps"],
            figure_dir / f"hero_{run_id}.png",
            title_suffix=f" (DeBERTa v3, SNLI, r=4, seed={seed})",
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("DeBERTa v3-base Experiment Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  seed={r['seed']}: rho={r['aulc_rho']:+.4f}, "
              f"tau={r.get('kendall_tau', 'N/A')}, "
              f"val_acc={r['final_val_acc']:.4f}")

    if len(all_results) > 1:
        mean_rho = np.mean([r["aulc_rho"] for r in all_results])
        std_rho = np.std([r["aulc_rho"] for r in all_results])
        mean_acc = np.mean([r["final_val_acc"] for r in all_results])
        print(f"  Mean rho: {mean_rho:+.4f} +/- {std_rho:.4f}")
        print(f"  Mean val_acc: {mean_acc:.4f}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {summary_path}")


# =========================================================================== #
# EXPERIMENT 11: Synthetic Noise Injection (Causal Intervention)
# =========================================================================== #

def run_noise_injection(args):
    """Causal intervention: inject label noise into clean examples.

    If annotation entropy *causes* the learning dynamics we observe, then
    artificially degrading clean examples' labels should shift their
    learning trajectories to resemble contested examples.

    Three conditions:
        - control:       original majority labels (baseline)
        - moderate_noise: replace 30% of clean-example labels with random
        - high_noise:    replace 60% of clean-example labels with random

    Uses RoBERTa-base + LoRA r=4 on SNLI.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: Synthetic Noise Injection (Causal Intervention)")
    print("=" * 70)

    from transformers import AutoTokenizer

    pilot = _import_pilot()
    device = args.device
    output_dir = OUTPUT_DIR / "noise_injection"
    figure_dir = FIGURE_DIR / "noise_injection"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Load data
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])
    bulk = load_bulk_training_data(dataset="snli", n_examples=20000, seed=SEEDS[0])

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

    # Identify clean examples (entropy < 0.4) and contested examples (entropy >= 0.7)
    clean_mask = np.array(cn_entropies) < ENTROPY_LOW
    contested_mask = np.array(cn_entropies) >= ENTROPY_HIGH
    clean_indices = np.where(clean_mask)[0]
    contested_indices = np.where(contested_mask)[0]

    print(f"  ChaosNLI train: {len(cn_eids)} examples")
    print(f"  Clean (H < {ENTROPY_LOW}): {len(clean_indices)}")
    print(f"  Contested (H >= {ENTROPY_HIGH}): {len(contested_indices)}")

    # Define noise conditions
    noise_levels = {
        "control": 0.0,
        "moderate_noise": 0.3,
        "high_noise": 0.6,
    }

    seed = args.seed if args.seed else SEEDS[0]
    rng = np.random.RandomState(seed)

    condition_results = {}

    for condition_name, noise_frac in noise_levels.items():
        run_id = f"noise_{condition_name}_s{seed}"
        result_path = output_dir / f"{run_id}.json"

        if result_path.exists() and not args.force:
            print(f"\n  Skipping {run_id} (exists). Use --force to rerun.")
            with open(result_path) as f:
                condition_results[condition_name] = json.load(f)
            continue

        print(f"\n  Running: {run_id} (noise_frac={noise_frac})")
        set_seed(seed)

        # Create noisy labels for clean examples
        noisy_cn_labels = list(cn_labels)  # copy
        noisy_eids = []  # track which clean examples got noise-injected

        if noise_frac > 0:
            n_to_flip = int(len(clean_indices) * noise_frac)
            flip_indices = rng.choice(clean_indices, size=n_to_flip, replace=False)
            for idx in flip_indices:
                original_label = cn_labels[idx]
                # Pick a random different label
                other_labels = [l for l in range(3) if l != original_label]
                noisy_cn_labels[idx] = rng.choice(other_labels)
                noisy_eids.append(cn_eids[idx])
            print(f"  Flipped {n_to_flip}/{len(clean_indices)} clean-example labels")
        else:
            print(f"  Control condition: no labels flipped")

        # Combine with bulk data
        combined_premises = list(bulk["premises"]) + cn_premises
        combined_hypotheses = list(bulk["hypotheses"]) + cn_hypotheses
        combined_labels = list(bulk["labels"]) + noisy_cn_labels
        combined_eids = [f"snli_{i}" for i in range(len(bulk["premises"]))] + cn_eids
        combined_entropies = [None] * len(bulk["premises"]) + cn_entropies

        # Create datasets
        train_dataset = pilot.NLIDataset(
            premises=combined_premises, hypotheses=combined_hypotheses,
            labels=combined_labels, example_ids=combined_eids,
            entropies=combined_entropies, tokenizer=tokenizer, max_length=128,
        )
        tracking_dataset = pilot.ChaosNLIDataset(
            premises=cn_premises, hypotheses=cn_hypotheses,
            labels=cn_labels,  # Track with ORIGINAL labels (not noisy)
            example_ids=cn_eids,
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

        model = create_lora_model(
            model_name="roberta-base", num_labels=3, rank=4,
            target_modules=["query", "value"],
        )

        tracker = TemporalTracker(loss_threshold=0.693)
        tracker.register_examples(
            example_ids=cn_eids, true_labels=cn_labels,
            annotation_entropies=cn_entropies,
        )

        # Class weights (from combined data including noisy labels)
        all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
        label_counts = torch.bincount(all_labels_t, minlength=3).float()
        class_weights = (1.0 / label_counts.clamp(min=1))
        class_weights = class_weights / class_weights.sum() * 3

        t0 = time.time()
        history = train_with_tracking(
            model=model, train_loader=train_loader,
            tracking_loader=tracking_loader, val_loader=val_loader,
            tracker=tracker, n_epochs=5, learning_rate=2e-5,
            eval_every_n_steps=100, device=device,
            class_weights=class_weights,
        )
        elapsed = time.time() - t0

        # Compute per-group AULC
        clean_aulcs = []
        contested_aulcs = []
        all_aulcs = []

        for i, eid in enumerate(cn_eids):
            record = tracker.records.get(eid)
            if record is None:
                continue
            valid_losses = [l for l in record.losses
                            if not (isinstance(l, float) and np.isnan(l))]
            if len(valid_losses) < 2:
                continue
            aulc_val = float(np.mean(valid_losses))
            all_aulcs.append({"eid": eid, "aulc": aulc_val, "entropy": cn_entropies[i]})

            if i in clean_indices.tolist():
                clean_aulcs.append(aulc_val)
            elif i in contested_indices.tolist():
                contested_aulcs.append(aulc_val)

        # Overall correlation
        aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
        valid_mask = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
        if valid_mask.sum() >= 3:
            rho, p = stats.spearmanr(aulc_arr[valid_mask], aulc_ent[valid_mask])
        else:
            rho, p = 0.0, 1.0

        # Save tracker
        tracker.save(output_dir / f"{run_id}_tracker.json")

        result = {
            "experiment": "noise_injection",
            "condition": condition_name,
            "noise_fraction": noise_frac,
            "seed": seed,
            "n_clean": len(clean_indices),
            "n_flipped": int(len(clean_indices) * noise_frac),
            "n_contested": len(contested_indices),
            "clean_mean_aulc": float(np.mean(clean_aulcs)) if clean_aulcs else None,
            "clean_std_aulc": float(np.std(clean_aulcs)) if clean_aulcs else None,
            "contested_mean_aulc": float(np.mean(contested_aulcs)) if contested_aulcs else None,
            "contested_std_aulc": float(np.std(contested_aulcs)) if contested_aulcs else None,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "final_val_acc": history["val_accuracy"][-1],
            "final_train_loss": history["train_loss"][-1],
            "elapsed_seconds": elapsed,
            "tracking_steps": history["tracking_steps"],
            "noisy_example_ids": noisy_eids,
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        condition_results[condition_name] = result

        print(f"  {condition_name}: clean_AULC={result['clean_mean_aulc']:.4f}, "
              f"contested_AULC={result['contested_mean_aulc']:.4f}, "
              f"rho={rho:+.4f}, val_acc={result['final_val_acc']:.4f}")

        # Hero figure per condition
        plot_hero_figure(
            tracker, history["tracking_steps"],
            figure_dir / f"hero_{run_id}.png",
            title_suffix=f" (noise={condition_name}, seed={seed})",
        )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # ---- Analysis: compare AULC across conditions ----
    print(f"\n{'='*60}")
    print("Noise Injection Analysis")
    print(f"{'='*60}")

    if "control" in condition_results and "moderate_noise" in condition_results:
        ctrl = condition_results["control"]
        mod = condition_results["moderate_noise"]
        high = condition_results.get("high_noise")

        print(f"\n  Clean-example mean AULC by condition:")
        print(f"    Control:       {ctrl['clean_mean_aulc']:.4f} (+/- {ctrl['clean_std_aulc']:.4f})")
        print(f"    Moderate (30%): {mod['clean_mean_aulc']:.4f} (+/- {mod['clean_std_aulc']:.4f})")
        if high:
            print(f"    High (60%):    {high['clean_mean_aulc']:.4f} (+/- {high['clean_std_aulc']:.4f})")

        # Load per-example AULC for paired comparison
        # We need to recompute per-example from trackers for the statistical test
        ctrl_tracker = TemporalTracker.load(output_dir / f"noise_control_s{seed}_tracker.json")
        mod_tracker = TemporalTracker.load(output_dir / f"noise_moderate_noise_s{seed}_tracker.json")

        # Get per-example AULC for clean examples in both conditions
        ctrl_clean_aulcs = {}
        mod_clean_aulcs = {}
        for idx in clean_indices:
            eid = cn_eids[idx]
            for tracker_dict, aulc_dict in [(ctrl_tracker.records, ctrl_clean_aulcs),
                                             (mod_tracker.records, mod_clean_aulcs)]:
                record = tracker_dict.get(eid)
                if record:
                    valid_losses = [l for l in record.losses
                                    if not (isinstance(l, float) and np.isnan(l))]
                    if len(valid_losses) >= 2:
                        aulc_dict[eid] = float(np.mean(valid_losses))

        # Paired comparison on shared examples
        shared_eids = sorted(set(ctrl_clean_aulcs.keys()) & set(mod_clean_aulcs.keys()))
        if len(shared_eids) >= 5:
            ctrl_vals = np.array([ctrl_clean_aulcs[eid] for eid in shared_eids])
            mod_vals = np.array([mod_clean_aulcs[eid] for eid in shared_eids])

            # Wilcoxon signed-rank test: mod > ctrl?
            w_stat, w_p = stats.wilcoxon(mod_vals - ctrl_vals, alternative="greater")
            # Cohen's d
            diff = mod_vals - ctrl_vals
            cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

            print(f"\n  Paired Wilcoxon test (moderate > control), n={len(shared_eids)}:")
            print(f"    W={w_stat:.1f}, p={w_p:.4e}")
            print(f"    Mean diff: {np.mean(diff):+.4f}, Cohen's d: {cohen_d:+.3f}")
        else:
            w_stat, w_p, cohen_d = None, None, None
            print(f"\n  Too few shared examples ({len(shared_eids)}) for paired test")

        # High noise comparison
        high_stats = {}
        if high:
            high_tracker = TemporalTracker.load(output_dir / f"noise_high_noise_s{seed}_tracker.json")
            high_clean_aulcs = {}
            for idx in clean_indices:
                eid = cn_eids[idx]
                record = high_tracker.records.get(eid)
                if record:
                    valid_losses = [l for l in record.losses
                                    if not (isinstance(l, float) and np.isnan(l))]
                    if len(valid_losses) >= 2:
                        high_clean_aulcs[eid] = float(np.mean(valid_losses))

            shared_high = sorted(set(ctrl_clean_aulcs.keys()) & set(high_clean_aulcs.keys()))
            if len(shared_high) >= 5:
                ctrl_h = np.array([ctrl_clean_aulcs[eid] for eid in shared_high])
                high_h = np.array([high_clean_aulcs[eid] for eid in shared_high])
                wh_stat, wh_p = stats.wilcoxon(high_h - ctrl_h, alternative="greater")
                diff_h = high_h - ctrl_h
                cohen_d_h = np.mean(diff_h) / np.std(diff_h, ddof=1) if np.std(diff_h, ddof=1) > 0 else 0.0
                print(f"\n  Paired Wilcoxon test (high > control), n={len(shared_high)}:")
                print(f"    W={wh_stat:.1f}, p={wh_p:.4e}")
                print(f"    Mean diff: {np.mean(diff_h):+.4f}, Cohen's d: {cohen_d_h:+.3f}")
                high_stats = {
                    "wilcoxon_W": float(wh_stat), "wilcoxon_p": float(wh_p),
                    "cohen_d": float(cohen_d_h), "n_paired": len(shared_high),
                }

        # Save summary
        summary = {
            "conditions": condition_results,
            "paired_test_moderate_vs_control": {
                "wilcoxon_W": float(w_stat) if w_stat is not None else None,
                "wilcoxon_p": float(w_p) if w_p is not None else None,
                "cohen_d": float(cohen_d) if cohen_d is not None else None,
                "n_paired": len(shared_eids) if shared_eids else 0,
            },
            "paired_test_high_vs_control": high_stats,
        }
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved: {summary_path}")

        # ---- Plot: mean loss curves for all conditions ----
        _plot_noise_comparison(
            output_dir, figure_dir, cn_eids, cn_entropies,
            clean_indices, contested_indices, seed,
        )


def _plot_noise_comparison(output_dir, figure_dir, cn_eids, cn_entropies,
                           clean_indices, contested_indices, seed):
    """Plot mean loss curves comparing control vs noise-injected conditions."""
    conditions_to_plot = ["control", "moderate_noise", "high_noise"]
    colors = {
        "control_clean": "#2166AC",
        "moderate_clean": "#92C5DE",
        "high_clean": "#F4A582",
        "contested": "#B2182B",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for cond_name in conditions_to_plot:
        tracker_path = output_dir / f"noise_{cond_name}_s{seed}_tracker.json"
        if not tracker_path.exists():
            continue

        tracker = TemporalTracker.load(tracker_path)

        # Get clean-example mean losses
        clean_losses_per_step = []
        n_steps = None
        for idx in clean_indices:
            eid = cn_eids[idx]
            record = tracker.records.get(eid)
            if record and len(record.losses) > 0:
                clean_losses_per_step.append(record.losses)
                if n_steps is None:
                    n_steps = len(record.losses)

        if clean_losses_per_step and n_steps:
            # Pad/truncate to same length
            padded = []
            for losses in clean_losses_per_step:
                if len(losses) >= n_steps:
                    padded.append(losses[:n_steps])
                else:
                    padded.append(losses + [losses[-1]] * (n_steps - len(losses)))
            mean_clean = np.nanmean(padded, axis=0)
            steps = list(range(n_steps))

            label_map = {
                "control": "control-clean",
                "moderate_noise": "30%-noise-clean",
                "high_noise": "60%-noise-clean",
            }
            color_map = {
                "control": colors["control_clean"],
                "moderate_noise": colors["moderate_clean"],
                "high_noise": colors["high_clean"],
            }
            ax.plot(steps, mean_clean, color=color_map[cond_name],
                    linewidth=2, label=label_map[cond_name],
                    alpha=0.9, marker="o" if cond_name == "control" else "s",
                    markersize=3)

        # Also plot contested from control condition for reference
        if cond_name == "control":
            contested_losses_per_step = []
            for idx in contested_indices:
                eid = cn_eids[idx]
                record = tracker.records.get(eid)
                if record and len(record.losses) > 0:
                    contested_losses_per_step.append(record.losses)

            if contested_losses_per_step and n_steps:
                padded_c = []
                for losses in contested_losses_per_step:
                    if len(losses) >= n_steps:
                        padded_c.append(losses[:n_steps])
                    else:
                        padded_c.append(losses + [losses[-1]] * (n_steps - len(losses)))
                mean_contested = np.nanmean(padded_c, axis=0)
                ax.plot(steps, mean_contested, color=colors["contested"],
                        linewidth=2, label="original-contested",
                        alpha=0.9, marker="^", markersize=3)

    ax.set_xlabel("Tracking Step", fontsize=11)
    ax.set_ylabel("Mean Cross-Entropy Loss", fontsize=11)
    ax.set_title("Noise Injection: Clean-Example Learning Dynamics", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    fig_path = figure_dir / "noise_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Robustness experiments: robustness checks and extensions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment selection
    exp = parser.add_argument_group("Experiment selection")
    exp.add_argument("--all", action="store_true",
                     help="Run all experiments.")
    exp.add_argument("--entropy-buckets", action="store_true",
                     help="Exp 1: Entropy-bucket ablation training.")
    exp.add_argument("--gradient-norms", action="store_true",
                     help="Exp 2: Per-example gradient norms by entropy bin.")
    exp.add_argument("--alt-binning", action="store_true",
                     help="Exp 3: Alternative entropy binning (quartile/tercile).")
    exp.add_argument("--bootstrap-ci", action="store_true",
                     help="Exp 4: Bootstrap 95%% CIs on Spearman rho.")
    exp.add_argument("--kendall", action="store_true",
                     help="Exp 5: Kendall tau-b robustness check.")
    exp.add_argument("--gpt2", action="store_true",
                     help="Exp 6: GPT-2 decoder-only model.")
    exp.add_argument("--alphanli", action="store_true",
                     help="Exp 7: ChaosNLI-AlphaNLI third-task experiment.")
    exp.add_argument("--soft-label", action="store_true",
                     help="Exp 8: Soft-label (KL-div) ablation.")
    exp.add_argument("--cartography", action="store_true",
                     help="Exp 9: Dataset Cartography comparison (CPU only).")
    exp.add_argument("--deberta", action="store_true",
                     help="Exp 10: DeBERTa v3-base disentangled attention.")
    exp.add_argument("--noise-injection", action="store_true",
                     help="Exp 11: Synthetic noise injection (causal intervention).")

    # General options
    gen = parser.add_argument_group("General options")
    gen.add_argument("--device", type=str, default=None,
                     help="Device (auto-detected if not set).")
    gen.add_argument("--seed", type=int, default=None,
                     help="Run only this seed (default: all 3).")
    gen.add_argument("--force", action="store_true",
                     help="Re-run experiments even if results exist.")
    gen.add_argument("--dry-run", action="store_true",
                     help="Print what would run without executing.")

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

    args.device = detect_device(args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Robustness Experiments: Robustness Checks and Extensions")
    print("=" * 70)
    print(f"  Device: {args.device}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURE_DIR}")

    # Determine which experiments to run
    run_any = False

    experiments = [
        ("entropy-buckets", args.entropy_buckets or args.all,
         run_entropy_bucket_ablation, "GPU"),
        ("gradient-norms", args.gradient_norms or args.all,
         run_gradient_norms, "GPU"),
        ("alt-binning", args.alt_binning or args.all,
         run_alt_binning, "CPU"),
        ("bootstrap-ci", args.bootstrap_ci or args.all,
         run_bootstrap_ci, "CPU"),
        ("kendall", args.kendall or args.all,
         run_kendall, "CPU"),
        ("gpt2", args.gpt2 or args.all,
         run_gpt2_experiment, "GPU"),
        ("alphanli", args.alphanli or args.all,
         run_alphanli_experiment, "GPU"),
        ("soft-label", args.soft_label or args.all,
         run_soft_label_ablation, "GPU"),
        ("cartography", args.cartography or args.all,
         run_cartography_comparison, "CPU"),
        ("deberta", args.deberta or args.all,
         run_deberta_experiment, "GPU"),
        ("noise-injection", args.noise_injection or args.all,
         run_noise_injection, "GPU"),
    ]

    enabled = [(name, fn, req) for name, enabled, fn, req in experiments if enabled]

    if not enabled:
        print("\n  No experiments selected. Use --all or select specific experiments.")
        print("  Run with --help for details.")
        return

    print(f"\n  Experiments to run: {', '.join(name for name, _, _ in enabled)}")

    if args.dry_run:
        print("\n  DRY RUN -- no experiments will execute.")
        for name, fn, req in enabled:
            print(f"    - {name} ({req})")
        return

    for name, fn, req in enabled:
        exp_t0 = time.time()
        try:
            fn(args)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
        exp_elapsed = time.time() - exp_t0
        print(f"\n  [{name}] completed in {exp_elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"All robustness experiments completed ({elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
