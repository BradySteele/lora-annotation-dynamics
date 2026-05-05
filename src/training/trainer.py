"""
Temporal Trainer
================
Training loop with per-example loss tracking for the temporal separation study.

This is the central component of Paper 2: "Clean Before Contested: LoRA Temporal
Separation with Annotator Disagreement."  It wraps a standard LoRA fine-tuning
loop with periodic per-example loss recording, feeding loss trajectories into the
TemporalTracker so that post-hoc analysis can measure whether clean examples are
learned before contested ones.

Key design points:
    - Every ``eval_every_n_steps`` training steps, the trainer runs inference
      on the *full* training set with ``reduction='none'`` cross-entropy to
      capture each example's loss at that checkpoint.
    - Losses are keyed by ``example_id`` (an integer present in every batch)
      and recorded via ``tracker.record_epoch_losses()``.
    - End-of-epoch evaluation reports overall accuracy plus per-entropy-category
      accuracy (clean / ambiguous / contested).
    - MPS-compatible: float32, no pin_memory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .temporal_tracker import TemporalTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for the temporal training loop.

    All hyperparameters live here so experiments can be driven entirely by
    config files with no magic numbers in the training code.
    """

    # Optimization
    learning_rate: float = 2e-4
    num_epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # Temporal tracking
    eval_every_n_steps: int = 50
    loss_threshold: float = 0.693  # -log(0.5), learning-time threshold for 3-class

    # Logging / output
    output_dir: str = "./results"
    log_every_n_steps: int = 10


# ---------------------------------------------------------------------------
# Linear warmup scheduler
# ---------------------------------------------------------------------------


def _create_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by linear decay to zero.

    During the first ``num_warmup_steps`` the learning rate increases linearly
    from 0 to the optimizer's base lr.  After warmup it decays linearly back
    to 0 over the remaining steps.

    Args:
        optimizer: The optimizer whose lr groups will be scheduled.
        num_warmup_steps: Steps spent ramping up.
        num_training_steps: Total number of training steps (warmup + decay).

    Returns:
        A LambdaLR scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TemporalTrainer:
    """Training loop with integrated per-example loss tracking.

    This trainer is purpose-built for the temporal separation study.  Beyond
    the standard train/evaluate cycle it periodically snapshots per-example
    cross-entropy losses across the full training set and feeds them into a
    :class:`TemporalTracker` for later learning-order analysis.

    Args:
        model: A PEFT-wrapped (LoRA) sequence classification model.  Must
            accept ``input_ids``, ``attention_mask``, ``labels`` and return an
            object with ``.loss`` and ``.logits``.
        train_dataloader: DataLoader for training.  Each batch must contain
            ``input_ids``, ``attention_mask``, ``labels``, and ``example_ids``.
        val_dataloader: DataLoader for validation.  Same batch keys minus
            ``example_ids`` (optional there).
        tracker: A pre-initialized :class:`TemporalTracker`.  Example
            registration should already be done before this trainer is
            constructed.
        config: :class:`TrainingConfig` controlling hyperparameters.
        device: The torch device to use (``"mps"``, ``"cuda"``, ``"cpu"``).
        entropy_categories: Optional dict mapping category name (e.g.
            ``"clean"``, ``"ambiguous"``, ``"contested"``) to a list of
            ``example_id`` strings.  Used for per-category evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tracker: TemporalTracker,
        config: TrainingConfig,
        device: torch.device,
        entropy_categories: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tracker = tracker
        self.config = config
        self.device = device
        self.entropy_categories = entropy_categories or {}

        # Derived quantities
        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.num_warmup_steps = int(self.total_steps * config.warmup_ratio)

        # Optimizer: AdamW with weight-decay exclusion for bias/LayerNorm
        self.optimizer = self._create_optimizer()

        # Linear warmup + linear decay scheduler
        self.scheduler = _create_linear_warmup_scheduler(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Loss function for training (mean reduction for backprop)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        # Loss function for per-example tracking (no reduction)
        self.criterion_none = nn.CrossEntropyLoss(reduction="none")

        # Training state
        self.global_step: int = 0
        self.checkpoint_counter: int = 0

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _create_optimizer(self) -> AdamW:
        """Create AdamW with proper weight-decay parameter groups.

        Bias and LayerNorm parameters are excluded from weight decay
        following standard practice (Loshchilov & Hutter, 2019).

        Returns:
            Configured AdamW optimizer.
        """
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(param_groups, lr=self.config.learning_rate)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full training loop with temporal tracking.

        At every ``config.eval_every_n_steps`` training steps, computes
        per-example losses on the full training set and records them in
        the tracker.  At the end of every epoch, runs validation evaluation
        (overall and per-category).

        Returns:
            Dictionary containing:
                - ``epoch_metrics``: list of per-epoch metric dicts
                - ``step_losses``: list of (global_step, batch_loss) tuples
                - ``gradient_norms``: list of (global_step, grad_norm) tuples
                - ``checkpoint_steps``: list of global steps where per-example
                  losses were recorded
                - ``final_eval``: final validation metrics
                - ``total_time_seconds``: wall-clock training time
        """
        logger.info("=" * 60)
        logger.info("Starting temporal training")
        logger.info(
            f"  epochs={self.config.num_epochs}, "
            f"steps/epoch={self.steps_per_epoch}, "
            f"total_steps={self.total_steps}"
        )
        logger.info(
            f"  lr={self.config.learning_rate}, "
            f"warmup_steps={self.num_warmup_steps}, "
            f"eval_every={self.config.eval_every_n_steps}"
        )
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"  trainable parameters: {n_trainable:,}")
        logger.info("=" * 60)

        start_time = time.time()
        step_losses: List[Tuple[int, float]] = []
        gradient_norms: List[Tuple[int, float]] = []
        checkpoint_steps: List[int] = []
        epoch_metrics: List[Dict[str, Any]] = []

        # ------------------------------------------------------------------
        # Record initial (pre-training) per-example losses as checkpoint 0
        # ------------------------------------------------------------------
        logger.info("Recording pre-training per-example losses (checkpoint 0)")
        initial_losses = self._compute_per_example_losses()
        example_ids_list = list(initial_losses.keys())
        losses_list = [initial_losses[eid] for eid in example_ids_list]
        self.tracker.record_epoch_losses(
            example_ids=example_ids_list,
            losses=losses_list,
            epoch=self.checkpoint_counter,
        )
        checkpoint_steps.append(self.global_step)
        self.checkpoint_counter += 1

        # ------------------------------------------------------------------
        # Training epochs
        # ------------------------------------------------------------------
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_loss_sum = 0.0
            epoch_grad_norms: List[float] = []
            num_batches = 0

            for batch in self.train_dataloader:
                # Move tensors to device (skip non-tensor fields like example_ids)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Use model's own loss if available; otherwise compute manually
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    loss = self.criterion(outputs.logits, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping and norm logging
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                grad_norm_val = float(grad_norm)
                gradient_norms.append((self.global_step, grad_norm_val))
                epoch_grad_norms.append(grad_norm_val)

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Bookkeeping
                batch_loss = loss.item()
                epoch_loss_sum += batch_loss
                num_batches += 1
                self.global_step += 1
                step_losses.append((self.global_step, batch_loss))

                # Periodic console logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  step {self.global_step}/{self.total_steps} | "
                        f"loss={batch_loss:.4f} | "
                        f"grad_norm={grad_norm_val:.4f} | "
                        f"lr={current_lr:.2e}"
                    )

                # ----------------------------------------------------------
                # Per-example loss checkpoint
                # ----------------------------------------------------------
                if self.global_step % self.config.eval_every_n_steps == 0:
                    logger.info(
                        f"  [checkpoint {self.checkpoint_counter}] "
                        f"Recording per-example losses at step {self.global_step}"
                    )
                    per_example = self._compute_per_example_losses()
                    eid_list = list(per_example.keys())
                    loss_list = [per_example[eid] for eid in eid_list]
                    self.tracker.record_epoch_losses(
                        example_ids=eid_list,
                        losses=loss_list,
                        epoch=self.checkpoint_counter,
                    )
                    checkpoint_steps.append(self.global_step)
                    self.checkpoint_counter += 1
                    # Return to training mode after inference pass
                    self.model.train()

            # End-of-epoch metrics
            avg_epoch_loss = epoch_loss_sum / max(num_batches, 1)
            mean_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0
            epoch_time = time.time() - epoch_start

            # Validation evaluation
            val_metrics = self.evaluate()
            category_metrics = self.evaluate_by_category()

            epoch_info: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "mean_grad_norm": mean_grad_norm,
                "epoch_time_seconds": epoch_time,
                "global_step": self.global_step,
                "val_metrics": val_metrics,
                "category_metrics": category_metrics,
            }
            epoch_metrics.append(epoch_info)

            # Log epoch summary
            logger.info("-" * 60)
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} complete | "
                f"train_loss={avg_epoch_loss:.4f} | "
                f"val_accuracy={val_metrics.get('accuracy', 0.0):.4f} | "
                f"mean_grad_norm={mean_grad_norm:.4f} | "
                f"time={epoch_time:.1f}s"
            )
            for cat_name, cat_metrics in category_metrics.items():
                logger.info(
                    f"  {cat_name}: "
                    f"accuracy={cat_metrics.get('accuracy', 0.0):.4f}, "
                    f"n={cat_metrics.get('count', 0)}"
                )
            logger.info("-" * 60)

        # Final evaluation
        final_eval = self.evaluate()
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.1f}s")
        logger.info(
            f"Final val accuracy: {final_eval.get('accuracy', 0.0):.4f} | "
            f"Checkpoints recorded: {self.checkpoint_counter}"
        )

        return {
            "epoch_metrics": epoch_metrics,
            "step_losses": step_losses,
            "gradient_norms": gradient_norms,
            "checkpoint_steps": checkpoint_steps,
            "final_eval": final_eval,
            "total_time_seconds": total_time,
        }

    # ------------------------------------------------------------------
    # Per-example loss computation
    # ------------------------------------------------------------------

    def _compute_per_example_losses(self) -> Dict[str, float]:
        """Run inference on the full training set and return per-example losses.

        Sets the model to eval mode, iterates through the training dataloader,
        computes unreduced cross-entropy for every example, and maps each
        ``example_id`` to its scalar loss value.

        Returns:
            Dictionary mapping example_id (as string) to its cross-entropy
            loss at the current model state.
        """
        self.model.eval()
        per_example_losses: Dict[str, float] = {}

        with torch.no_grad():
            for batch in self.train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                example_ids = batch["example_id"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Per-example cross-entropy (no reduction)
                losses = self.criterion_none(outputs.logits, labels)

                for i, eid in enumerate(example_ids):
                    per_example_losses[str(eid)] = float(losses[i].item())

        return per_example_losses

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on a dataset and return aggregate metrics.

        Computes accuracy and mean cross-entropy loss over the given
        dataloader (defaults to the validation set).

        Args:
            dataloader: DataLoader to evaluate on.  If ``None``, uses the
                validation dataloader provided at construction.

        Returns:
            Dictionary with keys ``accuracy``, ``loss``, and ``count``
            (number of examples evaluated).
        """
        if dataloader is None:
            dataloader = self.val_dataloader

        self.model.eval()
        total_correct = 0
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = self.criterion(outputs.logits, labels)
                preds = outputs.logits.argmax(dim=-1)

                total_correct += int((preds == labels).sum().item())
                total_loss += loss.item() * labels.size(0)
                total_count += labels.size(0)

        accuracy = total_correct / max(total_count, 1)
        avg_loss = total_loss / max(total_count, 1)

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "count": total_count,
        }

    def evaluate_by_category(self) -> Dict[str, Dict[str, float]]:
        """Evaluate separately for each entropy category (clean/ambiguous/contested).

        Uses ``self.entropy_categories`` to partition validation examples
        by their annotator-entropy bin, then computes per-bin accuracy.
        Requires that the validation dataloader yields ``example_ids``.

        If no entropy categories were provided at construction, returns an
        empty dictionary.

        Returns:
            Dictionary mapping category name to a metrics dict with keys
            ``accuracy``, ``loss``, and ``count``.
        """
        if not self.entropy_categories:
            return {}

        self.model.eval()

        # Build a set for O(1) lookup per category
        category_sets: Dict[str, set] = {
            cat: set(ids) for cat, ids in self.entropy_categories.items()
        }

        # Accumulators per category
        correct: Dict[str, int] = {cat: 0 for cat in category_sets}
        loss_sum: Dict[str, float] = {cat: 0.0 for cat in category_sets}
        count: Dict[str, int] = {cat: 0 for cat in category_sets}

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # example_ids may not be present in every val dataloader
                example_ids = batch.get("example_id")
                if example_ids is None:
                    continue

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                per_ex_loss = self.criterion_none(outputs.logits, labels)
                preds = outputs.logits.argmax(dim=-1)

                for i, eid in enumerate(example_ids):
                    eid_str = str(eid)
                    for cat, id_set in category_sets.items():
                        if eid_str in id_set:
                            is_correct = int(preds[i] == labels[i])
                            correct[cat] += is_correct
                            loss_sum[cat] += float(per_ex_loss[i].item())
                            count[cat] += 1
                            break  # each example belongs to at most one category

        results: Dict[str, Dict[str, float]] = {}
        for cat in category_sets:
            n = count[cat]
            results[cat] = {
                "accuracy": correct[cat] / max(n, 1),
                "loss": loss_sum[cat] / max(n, 1),
                "count": n,
            }

        return results
