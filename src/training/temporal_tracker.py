"""
Temporal Tracker: Per-Example Learning Dynamics
================================================
Tracks per-example loss values at every epoch during LoRA fine-tuning to
measure the temporal ordering of learning.

The central hypothesis of this paper is that LoRA's low-rank constraint
imposes a specific learning order: examples with high annotator consensus
(low H_i) are learned before examples with high annotator disagreement
(high H_i).  This tracker provides the instrumentation to measure that.

Key concepts:
    - Learning time t_i: the first epoch at which example i's loss drops
      below a threshold theta
    - Learning order: the permutation of examples sorted by learning time
    - Temporal separation: the gap in learning times between clean and
      contested examples
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class ExampleRecord:
    """Per-example tracking record across training epochs."""

    example_id: str
    losses: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    predictions: List[int] = field(default_factory=list)
    true_label: Optional[int] = None
    annotation_entropy: Optional[float] = None


class TemporalTracker:
    """Tracks per-example loss trajectories across training epochs.

    This is the core instrumentation class for the paper.  It records
    the loss of every training example at every epoch, enabling post-hoc
    analysis of learning dynamics.

    Attributes:
        records: Dict mapping example_id -> ExampleRecord.
        n_epochs_recorded: Number of epochs for which data has been recorded.
        loss_threshold: Default threshold for determining when an example
            is considered "learned".

    Example usage::

        tracker = TemporalTracker(loss_threshold=0.5)

        for epoch in range(n_epochs):
            for batch in dataloader:
                losses = model(batch)  # per-example losses
                tracker.record_epoch_losses(
                    example_ids=batch["ids"],
                    losses=losses.detach().cpu().numpy(),
                    epoch=epoch,
                )

        # After training: analyze learning order
        order = tracker.get_learning_order()
        # order[0] = (example_id, learning_time) for the first-learned example
    """

    def __init__(
        self,
        loss_threshold: float = 0.5,
        track_gradients: bool = False,
        track_predictions: bool = False,
    ) -> None:
        """Initialize the temporal tracker.

        Args:
            loss_threshold: Default loss value below which an example is
                considered "learned".
            track_gradients: Whether to also track per-example gradient norms.
            track_predictions: Whether to track model predictions per epoch.
        """
        self.loss_threshold = loss_threshold
        self.track_gradients = track_gradients
        self.track_predictions = track_predictions
        self.records: Dict[str, ExampleRecord] = {}
        self.n_epochs_recorded: int = 0
        self._epoch_losses: Dict[int, Dict[str, float]] = {}

    def register_examples(
        self,
        example_ids: Sequence[str],
        true_labels: Optional[Sequence[int]] = None,
        annotation_entropies: Optional[Sequence[float]] = None,
    ) -> None:
        """Pre-register examples with metadata before training begins.

        Args:
            example_ids: Unique identifiers for each example.
            true_labels: Ground-truth (majority-vote) labels.
            annotation_entropies: Pre-computed H_i values.
        """
        for idx, eid in enumerate(example_ids):
            record = ExampleRecord(example_id=eid)
            if true_labels is not None:
                record.true_label = int(true_labels[idx])
            if annotation_entropies is not None:
                record.annotation_entropy = float(annotation_entropies[idx])
            self.records[eid] = record

    def record_epoch_losses(
        self,
        example_ids: Sequence[str],
        losses: Union[Sequence[float], np.ndarray],
        epoch: int,
        gradient_norms: Optional[Union[Sequence[float], np.ndarray]] = None,
        predictions: Optional[Sequence[int]] = None,
    ) -> None:
        """Record per-example losses for one epoch.

        This should be called once per epoch (or accumulated across batches
        within an epoch) with the loss for every training example.

        Args:
            example_ids: Example identifiers matching the batch.
            losses: Per-example loss values.
            epoch: Current epoch number (0-indexed).
            gradient_norms: Optional per-example gradient norms.
            predictions: Optional model predictions.
        """
        losses = np.asarray(losses, dtype=np.float64)

        if epoch not in self._epoch_losses:
            self._epoch_losses[epoch] = {}

        for idx, eid in enumerate(example_ids):
            eid = str(eid)

            # Create record if not pre-registered
            if eid not in self.records:
                self.records[eid] = ExampleRecord(example_id=eid)

            record = self.records[eid]

            # Ensure losses list is long enough (handles out-of-order epochs)
            while len(record.losses) <= epoch:
                record.losses.append(float("nan"))
            record.losses[epoch] = float(losses[idx])

            # Store in epoch-indexed structure for fast epoch-level queries
            self._epoch_losses[epoch][eid] = float(losses[idx])

            if self.track_gradients and gradient_norms is not None:
                while len(record.gradient_norms) <= epoch:
                    record.gradient_norms.append(float("nan"))
                record.gradient_norms[epoch] = float(gradient_norms[idx])

            if self.track_predictions and predictions is not None:
                while len(record.predictions) <= epoch:
                    record.predictions.append(-1)
                record.predictions[epoch] = int(predictions[idx])

        self.n_epochs_recorded = max(self.n_epochs_recorded, epoch + 1)

    def get_learning_time(
        self,
        example_id: str,
        threshold: Optional[float] = None,
    ) -> Optional[int]:
        """Get the epoch at which an example's loss first drops below threshold.

        This is the operational definition of "learning time" t_i in the paper:
        the first epoch t such that L_i(t) < theta.

        Args:
            example_id: The example to query.
            threshold: Loss threshold.  If None, uses self.loss_threshold.

        Returns:
            The epoch number (0-indexed) when the example was first learned,
            or None if the loss never dropped below the threshold.
        """
        if threshold is None:
            threshold = self.loss_threshold

        if example_id not in self.records:
            return None

        losses = self.records[example_id].losses
        for epoch, loss in enumerate(losses):
            if not np.isnan(loss) and loss < threshold:
                return epoch

        return None  # never learned

    def get_learning_order(
        self,
        threshold: Optional[float] = None,
        include_unlearned: bool = True,
    ) -> List[Tuple[str, Optional[int]]]:
        """Get all examples sorted by learning time (earliest first).

        This is the main output for correlation analysis: we compare this
        ordering against the annotation entropy ordering to test the
        temporal separation hypothesis.

        Args:
            threshold: Loss threshold for learning time definition.
            include_unlearned: If True, examples that never reached the
                threshold are included at the end with learning_time=None.

        Returns:
            List of (example_id, learning_time) tuples sorted by learning_time.
            Unlearned examples (learning_time=None) appear at the end.
        """
        if threshold is None:
            threshold = self.loss_threshold

        learned = []
        unlearned = []

        for eid, record in self.records.items():
            t = self.get_learning_time(eid, threshold=threshold)
            if t is not None:
                learned.append((eid, t))
            else:
                unlearned.append((eid, None))

        # Sort learned examples by learning time, then by id for stability
        learned.sort(key=lambda x: (x[1], x[0]))

        if include_unlearned:
            return learned + unlearned
        return learned

    def get_loss_trajectory(self, example_id: str) -> np.ndarray:
        """Get the full loss trajectory for one example.

        Args:
            example_id: The example to query.

        Returns:
            Array of shape (n_epochs,) with the loss at each epoch.
        """
        if example_id not in self.records:
            raise KeyError(f"Example {example_id} not found in tracker.")
        return np.array(self.records[example_id].losses)

    def get_mean_loss_by_category(
        self,
        categories: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """Compute mean loss trajectory per category (e.g., clean/contested).

        Args:
            categories: Dict mapping category name -> list of example_ids.

        Returns:
            Dict mapping category name -> mean loss array of shape (n_epochs,).
        """
        result = {}
        for cat_name, eids in categories.items():
            trajectories = []
            for eid in eids:
                if eid in self.records and len(self.records[eid].losses) > 0:
                    trajectories.append(self.records[eid].losses)
            if trajectories:
                # Pad to same length
                max_len = max(len(t) for t in trajectories)
                padded = np.full((len(trajectories), max_len), np.nan)
                for i, t in enumerate(trajectories):
                    padded[i, : len(t)] = t
                result[cat_name] = np.nanmean(padded, axis=0)
            else:
                result[cat_name] = np.array([])
        return result

    def get_learning_time_statistics(
        self,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute summary statistics of learning times.

        Returns:
            Dict with keys: mean, median, std, min, max, n_learned, n_unlearned.
        """
        if threshold is None:
            threshold = self.loss_threshold

        times = []
        n_unlearned = 0
        for eid in self.records:
            t = self.get_learning_time(eid, threshold=threshold)
            if t is not None:
                times.append(t)
            else:
                n_unlearned += 1

        if not times:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": None,
                "max": None,
                "n_learned": 0,
                "n_unlearned": n_unlearned,
            }

        times_arr = np.array(times, dtype=np.float64)
        return {
            "mean": float(np.mean(times_arr)),
            "median": float(np.median(times_arr)),
            "std": float(np.std(times_arr)),
            "min": int(np.min(times_arr)),
            "max": int(np.max(times_arr)),
            "n_learned": len(times),
            "n_unlearned": n_unlearned,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save tracker state to JSON for reproducibility.

        Args:
            path: File path to save to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "loss_threshold": self.loss_threshold,
            "n_epochs_recorded": self.n_epochs_recorded,
            "records": {
                eid: {
                    "losses": rec.losses,
                    "gradient_norms": rec.gradient_norms,
                    "predictions": rec.predictions,
                    "true_label": rec.true_label,
                    "annotation_entropy": rec.annotation_entropy,
                }
                for eid, rec in self.records.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemporalTracker":
        """Load tracker state from JSON.

        Args:
            path: File path to load from.

        Returns:
            Reconstructed TemporalTracker instance.
        """
        with open(path, "r") as f:
            data = json.load(f)

        tracker = cls(loss_threshold=data["loss_threshold"])
        tracker.n_epochs_recorded = data["n_epochs_recorded"]

        for eid, rec_data in data["records"].items():
            record = ExampleRecord(
                example_id=eid,
                losses=rec_data["losses"],
                gradient_norms=rec_data.get("gradient_norms", []),
                predictions=rec_data.get("predictions", []),
                true_label=rec_data.get("true_label"),
                annotation_entropy=rec_data.get("annotation_entropy"),
            )
            tracker.records[eid] = record

        return tracker
