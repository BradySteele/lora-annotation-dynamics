#!/usr/bin/env python3
"""
Decoder-Only Model Experiments: Mid-Scale Decoders on SNLI
==========================================================
Tests whether the annotation-entropy --> learning-dynamics relationship
generalises from encoder-only models (RoBERTa, BERT, DistilBERT, DeBERTa v3)
to mid-scale decoder-only architectures.

A prior GPT-2 (124M) experiment failed because the model was too small to
learn NLI reliably. This script uses larger, instruction-aware decoders
(Qwen2.5-3B, Qwen2.5-7B, optionally LLaMA 3.1 8B) that have the capacity
to handle three-way classification.

Design decisions:
    - AutoModelForSequenceClassification adds a linear classification head
      on top of the decoder, keeping the pipeline identical to the encoder
      experiments.  The model classifies from the *last non-pad token*.
    - Left-padding (padding_side="left") is mandatory so the final token
      is always the last content token, not a pad token.
    - 3B models run in float32 (batch_size=16).
    - 7B+ models run in float16 (batch_size=4, gradient_accumulation_steps=8)
      to fit on an Apple M4 Max with 64-128 GB unified memory.
    - LoRA targets are set per-architecture to the Q/V projections.
    - Input format: the tokenizer's default sentence-pair encoding
      (premise, hypothesis) via tokenizer(premise, hypothesis, ...).

Models:
    - Qwen/Qwen2.5-3B       (not gated, 3B parameters)
    - Qwen/Qwen2.5-7B       (not gated, 7B parameters)
    - meta-llama/Meta-Llama-3.1-8B  (gated, requires HF_TOKEN)

Usage:
    python scripts/16_decoder_experiments.py
    python scripts/16_decoder_experiments.py --models qwen3b qwen7b
    python scripts/16_decoder_experiments.py --models qwen3b --configs r4 --seeds 42
    python scripts/16_decoder_experiments.py --models llama8b --half
    python scripts/16_decoder_experiments.py --skip-existing
    python scripts/16_decoder_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
import traceback
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
ENTROPY_LOW = 0.4
ENTROPY_HIGH = 0.7

OUTPUT_DIR = PROJECT_ROOT / "results" / "tracking" / "decoder_experiments"
FIGURE_DIR = PROJECT_ROOT / "figures" / "decoder_experiments"

# Model registry: maps short name to HuggingFace ID and config
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "qwen1.5b": {
        "hf_name": "Qwen/Qwen2.5-1.5B",
        "lora_targets": ["q_proj", "v_proj"],
        "dtype": torch.float16,
        "batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 4,   # effective BS = 32
        "learning_rate": 2e-5,
        "gated": False,
        "description": "Qwen2.5 1.5B (decoder-only, float16)",
    },
    "qwen3b": {
        "hf_name": "Qwen/Qwen2.5-3B",
        "lora_targets": ["q_proj", "v_proj"],
        "dtype": torch.float16,
        "batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 8,   # effective BS = 32
        "learning_rate": 2e-5,
        "gated": False,
        "description": "Qwen2.5 3B (decoder-only, float16)",
    },
    "llama3.2-1b": {
        "hf_name": "meta-llama/Llama-3.2-1B",
        "lora_targets": ["q_proj", "v_proj"],
        "dtype": torch.float16,
        "batch_size": 8,
        "eval_batch_size": 16,
        "gradient_accumulation_steps": 4,   # effective BS = 32
        "learning_rate": 2e-5,
        "gated": True,
        "description": "LLaMA 3.2 1B (decoder-only, float16, gated)",
    },
    "llama3.2-3b": {
        "hf_name": "meta-llama/Llama-3.2-3B",
        "lora_targets": ["q_proj", "v_proj"],
        "dtype": torch.float16,
        "batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 8,   # effective BS = 32
        "learning_rate": 1e-5,
        "gated": True,
        "description": "LLaMA 3.2 3B (decoder-only, float16, gated)",
    },
}

# Configurations: LoRA ranks to test
LORA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "r4":  {"type": "lora", "rank": 4},
    "r16": {"type": "lora", "rank": 16},
}


# --------------------------------------------------------------------------- #
# Import helpers from existing scripts
# --------------------------------------------------------------------------- #

def _import_pilot():
    """Import 02_pilot_experiment.py to reuse NLIDataset, ChaosNLIDataset."""
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "02_pilot_experiment.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_robustness():
    """Import 10_robustness_experiments.py to reuse data-loading helpers."""
    spec = importlib.util.spec_from_file_location(
        "robustness", str(PROJECT_ROOT / "scripts" / "10_robustness_experiments.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Device detection
# --------------------------------------------------------------------------- #

def detect_device(requested: Optional[str] = None) -> str:
    """Detect best available compute device."""
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Decoder-aware dataset
# --------------------------------------------------------------------------- #

class DecoderNLIDataset(torch.utils.data.Dataset):
    """NLI dataset for decoder-only models with left-padding.

    Decoder-only models with a classification head use the last
    non-padding token for classification.  Left-padding ensures that
    the last position always corresponds to real content, not a pad
    token, which is critical for correct gradient flow.

    The input is formatted as a single string:
        "premise: {premise} hypothesis: {hypothesis}"
    This explicit formatting is more robust than relying on tokenizer
    sentence-pair handling, which varies across decoder tokenisers.
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
        text = f"premise: {self.premises[idx]} hypothesis: {self.hypotheses[idx]}"
        encoding = self.tokenizer(
            text,
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
        if self.entropies[idx] is not None:
            item["entropy"] = torch.tensor(self.entropies[idx], dtype=torch.float32)
        else:
            item["entropy"] = torch.tensor(float("nan"), dtype=torch.float32)
        return item


class DecoderChaosNLIDataset(torch.utils.data.Dataset):
    """ChaosNLI-only dataset for decoder-only tracking passes."""

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
        text = f"premise: {self.premises[idx]} hypothesis: {self.hypotheses[idx]}"
        encoding = self.tokenizer(
            text,
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

def create_decoder_lora_model(
    hf_name: str,
    num_labels: int = 3,
    rank: int = 4,
    lora_targets: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float32,
    gated: bool = False,
) -> nn.Module:
    """Create a decoder-only model with a classification head and LoRA adapters.

    Uses AutoModelForSequenceClassification, which appends a linear
    classification head.  Decoder models classify from the last token,
    so pad_token_id must be set in the config.

    Args:
        hf_name: HuggingFace model identifier.
        num_labels: Number of output classes (3 for NLI).
        rank: LoRA rank r.
        lora_targets: Attention projections to adapt.
        dtype: Model dtype (torch.float32 or torch.float16).
        gated: Whether the model requires HF_TOKEN authentication.

    Returns:
        PEFT-wrapped model ready for training.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    if lora_targets is None:
        lora_targets = ["q_proj", "v_proj"]

    # Build kwargs for from_pretrained
    model_kwargs: Dict[str, Any] = {
        "num_labels": num_labels,
        "dtype": dtype,
    }
    if gated:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                f"Model {hf_name} is gated and requires HF_TOKEN. "
                f"Set the HF_TOKEN environment variable or use a non-gated model."
            )
        model_kwargs["token"] = hf_token

    print(f"  Loading base model: {hf_name} (dtype={dtype})")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, **model_kwargs,
    )

    # Decoder models often lack a pad_token_id; use eos_token_id
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = base_model.config.eos_token_id
        print(f"  Set pad_token_id = eos_token_id = {base_model.config.eos_token_id}")

    lora_alpha = 2 * rank

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=lora_targets,
        bias="none",
        modules_to_save=["score"],  # Classification head (decoder models use "score")
    )

    model = get_peft_model(base_model, lora_config)

    # Upcast all trainable parameters (LoRA adapters + classification head) to
    # float32.  The frozen base model stays in float16 for memory savings, but
    # all parameters that receive gradients run in float32 to avoid nan from
    # float16 overflow in the freshly-initialized classification head and
    # during gradient accumulation.
    if dtype == torch.float16:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={rank}, alpha={lora_alpha}, targets={lora_targets}")
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model


# --------------------------------------------------------------------------- #
# Training loop with gradient accumulation
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
    gradient_accumulation_steps: int = 1,
    use_fp16: bool = False,
) -> Dict[str, Any]:
    """Train with per-example loss tracking and gradient accumulation.

    This extends the standard training loop from the pilot experiment to
    support gradient accumulation (needed for large decoder models with
    small per-device batch sizes) and optional float16 handling.

    The tracking pass is identical to existing experiments: evaluate all
    ChaosNLI examples and record per-example losses in the tracker.

    Args:
        model: PEFT-wrapped model.
        train_loader: Full training DataLoader (SNLI + ChaosNLI).
        tracking_loader: ChaosNLI-only DataLoader for per-example tracking.
        val_loader: Validation DataLoader.
        tracker: TemporalTracker instance.
        n_epochs: Training epochs.
        learning_rate: Peak learning rate.
        eval_every_n_steps: Record per-example losses every N optimizer steps.
        device: Compute device string.
        max_grad_norm: Gradient clipping threshold.
        class_weights: Optional class weights for imbalanced labels.
        gradient_accumulation_steps: Number of forward passes per optimizer step.
        use_fp16: Whether the model is in float16 (affects loss casting).

    Returns:
        History dictionary with training/validation metrics and tracking steps.
    """
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

    total_optimizer_steps = n_epochs * (len(train_loader) // gradient_accumulation_steps)
    warmup_steps = int(0.06 * total_optimizer_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_optimizer_steps - warmup_steps)
        )
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if class_weights is not None:
        class_weights = class_weights.to(device).float()  # Always float32 for CE loss

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_fn_mean = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "tracking_steps": [],
    }

    global_step = 0       # counts forward passes
    optimizer_step = 0    # counts optimizer updates
    tracking_step = 0

    effective_batch_size = train_loader.batch_size * gradient_accumulation_steps
    print(f"  Total optimizer steps: {total_optimizer_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Batch size: {train_loader.batch_size} x {gradient_accumulation_steps} "
          f"(grad accum) = {effective_batch_size} effective")
    print(f"  Tracking every {eval_every_n_steps} optimizer steps")
    print(f"  Training examples: {len(train_loader.dataset)}")
    print(f"  Tracking examples: {len(tracking_loader.dataset)} (ChaosNLI only)")

    # Initial tracking pass (step 0, before any training)
    print("  Recording initial per-example losses (step 0)...")
    _record_tracking_pass(model, tracking_loader, tracker, tracking_step,
                          loss_fn, device, use_fp16)
    history["tracking_steps"].append(0)
    tracking_step += 1

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch + 1}/{n_epochs}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # Always float32 for stable CE loss
            loss = loss_fn_mean(logits, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            global_step += 1

            # Optimizer step after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                pbar.set_postfix(
                    loss=f"{epoch_losses[-1]:.4f}",
                    opt_step=optimizer_step,
                )

                # Periodic tracking pass
                if optimizer_step % eval_every_n_steps == 0:
                    _record_tracking_pass(
                        model, tracking_loader, tracker, tracking_step,
                        loss_fn, device, use_fp16,
                    )
                    history["tracking_steps"].append(optimizer_step)
                    tracking_step += 1

        # Handle leftover gradients at end of epoch
        leftover = len(train_loader) % gradient_accumulation_steps
        if leftover > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1

        # End-of-epoch tracking pass (if not already done at this step)
        if optimizer_step % eval_every_n_steps != 0:
            _record_tracking_pass(
                model, tracking_loader, tracker, tracking_step,
                loss_fn, device, use_fp16,
            )
            history["tracking_steps"].append(optimizer_step)
            tracking_step += 1

        # Epoch-level validation
        train_loss = float(np.mean(epoch_losses))
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn_mean, device, use_fp16)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"  Epoch {epoch + 1}/{n_epochs}: "
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
    use_fp16: bool = False,
) -> None:
    """Record per-example losses for all ChaosNLI tracking examples."""
    model.eval()
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        example_ids = batch["example_id"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        losses = loss_fn(logits, labels)

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
    use_fp16: bool = False,
) -> Tuple[float, float]:
    """Evaluate on validation set, returning (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

def compute_aulc_from_tracker(
    tracker: TemporalTracker,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AULC and entropy arrays from a tracker."""
    aulcs = []
    entropies = []
    for eid, record in tracker.records.items():
        valid_losses = [
            l for l in record.losses
            if not (isinstance(l, float) and np.isnan(l))
        ]
        if len(valid_losses) < 2:
            aulcs.append(np.nan)
        else:
            aulcs.append(float(np.mean(valid_losses)))
        entropies.append(
            record.annotation_entropy
            if record.annotation_entropy is not None
            else np.nan
        )
    return np.array(aulcs), np.array(entropies)


def compute_delta_ell(
    tracker: TemporalTracker,
    low_threshold: float = ENTROPY_LOW,
    high_threshold: float = ENTROPY_HIGH,
) -> Dict[str, Any]:
    """Compute delta-ell (start-to-end loss change) by entropy category.

    Delta-ell measures how much the per-example loss decreased over
    training. Clean examples (low entropy) should show larger negative
    delta-ell (more learning) than contested examples (high entropy).
    """
    delta_ells: Dict[str, List[float]] = {"clean": [], "contested": []}
    for eid, record in tracker.records.items():
        if record.annotation_entropy is None or len(record.losses) < 2:
            continue
        delta = record.losses[-1] - record.losses[0]
        if record.annotation_entropy < low_threshold:
            delta_ells["clean"].append(delta)
        elif record.annotation_entropy >= high_threshold:
            delta_ells["contested"].append(delta)

    return {
        "mean_delta_ell_clean": (
            float(np.mean(delta_ells["clean"])) if delta_ells["clean"] else None
        ),
        "mean_delta_ell_contested": (
            float(np.mean(delta_ells["contested"])) if delta_ells["contested"] else None
        ),
        "n_clean": len(delta_ells["clean"]),
        "n_contested": len(delta_ells["contested"]),
    }


# --------------------------------------------------------------------------- #
# Hero figure
# --------------------------------------------------------------------------- #

def plot_hero_figure(
    tracker: TemporalTracker,
    tracking_steps: List[int],
    output_path: Path,
    title_suffix: str = "",
    loss_threshold: float = 0.693,
) -> None:
    """Plot per-category mean loss curves (same style as existing experiments)."""
    categories: Dict[str, List[str]] = {}
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

    # SEM for confidence bands
    sem_losses: Dict[str, np.ndarray] = {}
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
            n_valid = np.maximum(
                np.sum(~np.isnan(padded), axis=0).astype(float), 1.0
            )
            sem_losses[cat_name] = std / np.sqrt(n_valid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"clean": "#2166AC", "ambiguous": "#F4A582", "contested": "#B2182B"}
    markers = {"clean": "o", "ambiguous": "s", "contested": "^"}

    for cat_name in ["clean", "ambiguous", "contested"]:
        if cat_name not in mean_losses or len(mean_losses[cat_name]) == 0:
            continue
        losses = mean_losses[cat_name]
        n_steps = len(losses)
        steps = (
            tracking_steps[:n_steps]
            if len(tracking_steps) >= n_steps
            else list(range(n_steps))
        )
        n_examples = len(categories.get(cat_name, []))
        color = colors.get(cat_name, "gray")

        ax.plot(
            steps, losses, color=color,
            marker=markers.get(cat_name, "."), markersize=4, linewidth=2,
            label=f"{cat_name} (n={n_examples})", alpha=0.9,
        )

        # 95% CI band
        if cat_name in sem_losses:
            sem = sem_losses[cat_name][:n_steps]
            ci_lower = losses - 1.96 * sem
            ci_upper = losses + 1.96 * sem
            ax.fill_between(steps, ci_lower, ci_upper, color=color, alpha=0.15)

    ax.axhline(
        loss_threshold, color="gray", linestyle="--", linewidth=1.0,
        alpha=0.6, label=f"threshold $\\theta = {loss_threshold:.2f}$",
    )
    ax.set_xlabel("Optimizer Step", fontsize=11)
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
    print(f"  Saved hero figure: {output_path}")


def plot_summary_figure(
    all_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a summary bar chart of AULC-entropy rho across all runs.

    Groups results by model and config, showing mean +/- std across seeds.
    """
    # Group by (model, config)
    groups: Dict[Tuple[str, str], List[float]] = {}
    for r in all_results:
        key = (r["model_short"], r["config"])
        groups.setdefault(key, []).append(r["aulc_rho"])

    if not groups:
        return

    labels = []
    means = []
    stds = []
    for (model, config), rhos in sorted(groups.items()):
        labels.append(f"{model}\n{config}")
        means.append(np.mean(rhos))
        stds.append(np.std(rhos) if len(rhos) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0",
                  edgecolor="black", alpha=0.85, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AULC-Entropy Spearman $\\rho$", fontsize=11)
    ax.set_title("Decoder-Only Models: AULC-Entropy Correlation", fontsize=12)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate bars with mean values
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{m:.3f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary figure: {output_path}")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_chaosnli_data(subset: str, seed: int = 42) -> Dict[str, Any]:
    """Load ChaosNLI data with entropy annotations and train/val split."""
    data = load_chaosnli(subset=subset, data_dir=CHAOSNLI_DATA_DIR)

    entropies = [
        compute_annotation_entropy_from_distribution(dist)
        for dist in data["label_distributions"]
    ]

    cats = categorize_by_entropy(
        np.array(entropies), thresholds=[ENTROPY_LOW, ENTROPY_HIGH]
    )
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
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }


def load_bulk_training_data(
    dataset: str, n_examples: int, seed: int
) -> Dict[str, Any]:
    """Load bulk SNLI training data from HuggingFace."""
    from datasets import load_dataset

    if dataset == "snli":
        ds = load_dataset("stanfordnlp/snli", split="train")
        ds = ds.filter(lambda x: x["label"] != -1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

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
# Single run
# --------------------------------------------------------------------------- #

def run_single_experiment(
    model_key: str,
    config_key: str,
    seed: int,
    device: str,
    chaosnli: Dict[str, Any],
    bulk: Dict[str, Any],
    force: bool = False,
    use_half_override: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Run a single model/config/seed experiment.

    Returns the result dictionary, or None if skipped or failed.
    """
    model_info = MODEL_REGISTRY[model_key]
    config_info = LORA_CONFIGS[config_key]
    hf_name = model_info["hf_name"]
    rank = config_info["rank"]

    run_id = f"{model_key}_snli_{config_key}_s{seed}"
    result_path = OUTPUT_DIR / f"{run_id}.json"

    if result_path.exists() and not force:
        print(f"\n  Skipping {run_id} (exists). Use --force to rerun.")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'=' * 60}")
    print(f"  Running: {run_id}")
    print(f"  Model:   {hf_name} ({model_info['description']})")
    print(f"  Config:  LoRA rank={rank}")
    print(f"  Seed:    {seed}")
    print(f"  Device:  {device}")
    print(f"{'=' * 60}")

    set_seed(seed)

    # Determine dtype
    if use_half_override is not None:
        use_fp16 = use_half_override
    else:
        use_fp16 = model_info["dtype"] == torch.float16

    dtype = torch.float16 if use_fp16 else torch.float32
    batch_size = model_info["batch_size"]
    eval_batch_size = model_info["eval_batch_size"]
    grad_accum = model_info["gradient_accumulation_steps"]
    lr = model_info["learning_rate"]

    print(f"  dtype:   {dtype}")
    print(f"  batch:   {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    print(f"  lr:      {lr}")

    # Tokenizer setup (decoder-specific)
    from transformers import AutoTokenizer

    tokenizer_kwargs: Dict[str, Any] = {}
    if model_info["gated"]:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(hf_name, **tokenizer_kwargs)
    tokenizer.padding_side = "left"  # Critical for decoder classification
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  Set pad_token = eos_token = '{tokenizer.eos_token}'")

    # Prepare data splits
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
    combined_entropies: List[Optional[float]] = (
        [None] * len(bulk["premises"]) + list(cn_entropies)
    )

    # Create datasets (decoder-aware, with left-padding and explicit formatting)
    train_dataset = DecoderNLIDataset(
        premises=combined_premises, hypotheses=combined_hypotheses,
        labels=combined_labels, example_ids=combined_eids,
        entropies=combined_entropies, tokenizer=tokenizer, max_length=128,
    )
    tracking_dataset = DecoderChaosNLIDataset(
        premises=cn_premises, hypotheses=cn_hypotheses,
        labels=cn_labels, example_ids=cn_eids,
        entropies=cn_entropies, tokenizer=tokenizer, max_length=128,
    )
    val_dataset = DecoderChaosNLIDataset(
        premises=val_premises, hypotheses=val_hypotheses,
        labels=val_labels, example_ids=val_eids,
        entropies=val_entropies, tokenizer=tokenizer, max_length=128,
    )

    # MPS-specific DataLoader settings
    use_mps = device == "mps"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0 if use_mps else 2, pin_memory=not use_mps,
        drop_last=False,
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
    try:
        model = create_decoder_lora_model(
            hf_name=hf_name,
            num_labels=3,
            rank=rank,
            lora_targets=model_info["lora_targets"],
            dtype=dtype,
            gated=model_info["gated"],
        )
    except Exception as e:
        print(f"  FAILED to create model: {e}")
        traceback.print_exc()
        return None

    # Initialize tracker
    tracker = TemporalTracker(loss_threshold=0.693)
    tracker.register_examples(
        example_ids=cn_eids,
        true_labels=cn_labels,
        annotation_entropies=cn_entropies,
    )

    # Class weights
    all_labels_t = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_labels_t, minlength=3).float()
    class_weights = 1.0 / label_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * 3

    # Train
    run_t0 = time.time()
    try:
        history = train_with_tracking(
            model=model,
            train_loader=train_loader,
            tracking_loader=tracking_loader,
            val_loader=val_loader,
            tracker=tracker,
            n_epochs=5,
            learning_rate=lr,
            eval_every_n_steps=100,
            device=device,
            class_weights=class_weights,
            gradient_accumulation_steps=grad_accum,
            use_fp16=use_fp16,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS" in str(e):
            print(f"\n  OOM ERROR during training: {e}")
            print(f"  Consider using --half flag or reducing batch size.")
            del model
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            return None
        raise
    elapsed = time.time() - run_t0

    # Compute correlations
    aulc_arr, aulc_ent = compute_aulc_from_tracker(tracker)
    valid = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
    if valid.sum() >= 3:
        rho, p = stats.spearmanr(aulc_arr[valid], aulc_ent[valid])
        tau, p_tau = stats.kendalltau(aulc_arr[valid], aulc_ent[valid])
    else:
        rho, p, tau, p_tau = 0.0, 1.0, 0.0, 1.0

    # Delta-ell
    delta_ell_info = compute_delta_ell(tracker)

    # Save tracker
    tracker.save(OUTPUT_DIR / f"{run_id}_tracker.json")

    # Assemble result
    result = {
        "experiment": "decoder_experiments",
        "model": hf_name,
        "model_short": model_key,
        "model_description": model_info["description"],
        "dataset": "snli",
        "config": config_key,
        "rank": rank,
        "seed": seed,
        "dtype": str(dtype),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "aulc_rho": float(rho),
        "aulc_p": float(p),
        "kendall_tau": float(tau),
        "kendall_p": float(p_tau),
        "mean_delta_ell_clean": delta_ell_info["mean_delta_ell_clean"],
        "mean_delta_ell_contested": delta_ell_info["mean_delta_ell_contested"],
        "n_clean": delta_ell_info["n_clean"],
        "n_contested": delta_ell_info["n_contested"],
        "final_val_acc": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_accuracy_history": history["val_accuracy"],
        "train_loss_history": history["train_loss"],
        "elapsed_seconds": elapsed,
        "tracking_steps": history["tracking_steps"],
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  {run_id} RESULTS:")
    print(f"    AULC-entropy Spearman rho = {rho:+.4f} (p = {p:.2e})")
    print(f"    Kendall tau               = {tau:+.4f} (p = {p_tau:.2e})")
    print(f"    Final val accuracy         = {result['final_val_acc']:.4f}")
    print(f"    delta-ell clean            = {delta_ell_info['mean_delta_ell_clean']}")
    print(f"    delta-ell contested        = {delta_ell_info['mean_delta_ell_contested']}")
    print(f"    Time: {elapsed:.1f}s")

    # Hero figure
    plot_hero_figure(
        tracker, history["tracking_steps"],
        FIGURE_DIR / f"hero_{run_id}.png",
        title_suffix=f" ({model_key}, SNLI, {config_key}, seed={seed})",
    )

    # Cleanup
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decoder-only model experiments for annotation-entropy "
                    "learning dynamics.",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["qwen1.5b", "qwen3b"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Which decoder models to test. Default: qwen1.5b qwen3b",
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["r4", "r16"],
        choices=list(LORA_CONFIGS.keys()),
        help="LoRA configurations to test. Default: r4 r16",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=SEEDS,
        help=f"Random seeds. Default: {SEEDS}",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device (auto-detected if not specified).",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Force float16 for all models (overrides per-model default).",
    )
    parser.add_argument(
        "--no-half", action="store_true",
        help="Force float32 for all models (overrides per-model default). "
             "May OOM on 7B models.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs whose result file already exists (same as default "
             "behavior, but explicit).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun experiments even if results already exist.",
    )
    parser.add_argument(
        "--snli-size", type=int, default=20000,
        help="Number of SNLI training examples to subsample. Default: 20000",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs. Default: 5",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configurations without running experiments.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    t0 = time.time()

    device = detect_device(args.device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine half-precision override
    half_override: Optional[bool] = None
    if args.half:
        half_override = True
    elif args.no_half:
        half_override = False

    # Print experiment plan
    n_runs = len(args.models) * len(args.configs) * len(args.seeds)
    print("=" * 70)
    print("Decoder-Only Model Experiments")
    print("=" * 70)
    print(f"  Models:  {args.models}")
    print(f"  Configs: {args.configs}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Device:  {device}")
    print(f"  SNLI:    {args.snli_size} training examples")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Total runs: {n_runs}")
    if half_override is True:
        print(f"  Override: forcing float16 for all models")
    elif half_override is False:
        print(f"  Override: forcing float32 for all models")
    print()

    # Print model details
    for model_key in args.models:
        info = MODEL_REGISTRY[model_key]
        print(f"  {model_key}: {info['hf_name']}")
        print(f"    {info['description']}")
        print(f"    Default dtype: {info['dtype']}, batch_size: {info['batch_size']}, "
              f"grad_accum: {info['gradient_accumulation_steps']}, lr: {info['learning_rate']}")
        if info["gated"]:
            hf_token = os.environ.get("HF_TOKEN")
            print(f"    GATED: HF_TOKEN {'is set' if hf_token else 'NOT SET (will fail)'}")
        print()

    if args.dry_run:
        print("  DRY RUN: printing planned experiments only.\n")
        for model_key in args.models:
            for config_key in args.configs:
                for seed in args.seeds:
                    run_id = f"{model_key}_snli_{config_key}_s{seed}"
                    result_path = OUTPUT_DIR / f"{run_id}.json"
                    exists = result_path.exists()
                    status = "EXISTS" if exists else "PENDING"
                    print(f"    [{status}] {run_id}")
        print("\n  Remove --dry-run to execute.")
        return

    # Load data once (shared across all runs)
    print("Loading ChaosNLI data...")
    chaosnli = load_chaosnli_data(subset="snli", seed=SEEDS[0])
    print(f"  ChaosNLI: {len(chaosnli['premises'])} examples")

    print("Loading bulk SNLI training data...")
    bulk = load_bulk_training_data(
        dataset="snli", n_examples=args.snli_size, seed=SEEDS[0]
    )
    print(f"  SNLI bulk: {len(bulk['premises'])} examples")

    # Run experiments
    all_results: List[Dict[str, Any]] = []
    n_completed = 0
    n_failed = 0
    n_skipped = 0

    for model_key in args.models:
        for config_key in args.configs:
            for seed in args.seeds:
                result = run_single_experiment(
                    model_key=model_key,
                    config_key=config_key,
                    seed=seed,
                    device=device,
                    chaosnli=chaosnli,
                    bulk=bulk,
                    force=args.force,
                    use_half_override=half_override,
                )

                if result is not None:
                    all_results.append(result)
                    if "aulc_rho" in result:
                        n_completed += 1
                    else:
                        n_skipped += 1
                else:
                    n_failed += 1

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 70}")
    print("Decoder-Only Experiments: Summary")
    print(f"{'=' * 70}")
    print(f"  Completed: {n_completed}, Skipped: {n_skipped}, Failed: {n_failed}")
    print()

    if all_results:
        header = (
            f"{'Model':>10} {'Config':>6} {'Seed':>6} "
            f"{'AULC rho':>10} {'p-value':>12} "
            f"{'Kendall':>8} {'Val Acc':>8} "
            f"{'dl clean':>10} {'dl contest':>10}"
        )
        print(header)
        print("-" * len(header))

        for r in all_results:
            delta_c = (
                f"{r.get('mean_delta_ell_clean', 0):+.4f}"
                if r.get("mean_delta_ell_clean") is not None
                else "N/A"
            )
            delta_t = (
                f"{r.get('mean_delta_ell_contested', 0):+.4f}"
                if r.get("mean_delta_ell_contested") is not None
                else "N/A"
            )
            val_acc = (
                f"{r['final_val_acc']:.4f}"
                if r.get("final_val_acc") is not None
                else "N/A"
            )
            print(
                f"{r.get('model_short', r.get('model', '?')):>10} "
                f"{r.get('config', '?'):>6} "
                f"{r.get('seed', '?'):>6} "
                f"{r.get('aulc_rho', 0):>10.4f} "
                f"{r.get('aulc_p', 1.0):>12.2e} "
                f"{r.get('kendall_tau', 0):>8.4f} "
                f"{val_acc:>8} "
                f"{delta_c:>10} "
                f"{delta_t:>10}"
            )

        # Per-model/config aggregation
        print()
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in all_results:
            key = (r.get("model_short", "?"), r.get("config", "?"))
            groups.setdefault(key, []).append(r)

        for (model_short, config), runs in sorted(groups.items()):
            if len(runs) > 1:
                rhos = [r["aulc_rho"] for r in runs if "aulc_rho" in r]
                taus = [r["kendall_tau"] for r in runs if "kendall_tau" in r]
                accs = [r["final_val_acc"] for r in runs
                        if r.get("final_val_acc") is not None]
                if rhos:
                    print(
                        f"  {model_short} {config}: "
                        f"rho={np.mean(rhos):+.4f} +/- {np.std(rhos):.4f}, "
                        f"tau={np.mean(taus):+.4f} +/- {np.std(taus):.4f}"
                        + (f", val_acc={np.mean(accs):.4f}" if accs else "")
                    )

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved summary to {summary_path}")

    # Summary figure
    if len(all_results) > 0:
        plot_summary_figure(
            all_results,
            FIGURE_DIR / "decoder_summary.png",
        )

    # Comparison with encoder baselines (if available)
    _print_cross_architecture_comparison(all_results)

    elapsed = time.time() - t0
    print(f"\nDecoder experiments complete ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


def _print_cross_architecture_comparison(
    decoder_results: List[Dict[str, Any]],
) -> None:
    """Print a comparison table including encoder baselines if available."""
    # Try to load encoder results for context
    encoder_summary_paths = [
        PROJECT_ROOT / "results" / "tracking" / "expanded" / "summary.json",
        PROJECT_ROOT / "results" / "tracking" / "deberta_extended" / "summary.json",
    ]

    encoder_rhos: Dict[str, List[float]] = {}
    for path in encoder_summary_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for r in data:
                        model = r.get("model", r.get("model_name", "unknown"))
                        rho = r.get("aulc_rho", r.get("spearman_aulc_rho"))
                        if rho is not None:
                            encoder_rhos.setdefault(model, []).append(float(rho))
            except (json.JSONDecodeError, KeyError):
                pass

    if not encoder_rhos and not decoder_results:
        return

    print(f"\n{'=' * 70}")
    print("Cross-Architecture Comparison (Encoder vs Decoder)")
    print(f"{'=' * 70}")

    if encoder_rhos:
        print("\n  Encoder models (from existing experiments):")
        for model, rhos in sorted(encoder_rhos.items()):
            mean_rho = np.mean(rhos)
            if len(rhos) > 1:
                print(f"    {model}: rho={mean_rho:+.4f} +/- {np.std(rhos):.4f} "
                      f"(n={len(rhos)} runs)")
            else:
                print(f"    {model}: rho={mean_rho:+.4f} (n=1)")
    else:
        print("\n  No encoder baseline results found for comparison.")

    if decoder_results:
        print("\n  Decoder models (this experiment):")
        groups: Dict[str, List[float]] = {}
        for r in decoder_results:
            key = r.get("model_short", r.get("model", "?"))
            if "aulc_rho" in r:
                groups.setdefault(key, []).append(r["aulc_rho"])
        for model, rhos in sorted(groups.items()):
            mean_rho = np.mean(rhos)
            if len(rhos) > 1:
                print(f"    {model}: rho={mean_rho:+.4f} +/- {np.std(rhos):.4f} "
                      f"(n={len(rhos)} runs)")
            else:
                print(f"    {model}: rho={mean_rho:+.4f} (n=1)")


if __name__ == "__main__":
    main()
