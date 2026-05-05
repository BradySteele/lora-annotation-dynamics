"""
Learning Order Analysis
=======================
Analyzes the relationship between annotation entropy and the order
in which examples are learned during LoRA fine-tuning.

Provides functions to:
    - Extract per-example learning times from a TemporalTracker
    - Measure learning order consistency across seeds (Kendall's W)
    - Compute stratified loss curves by entropy category
    - Analyze how Spearman(learning_time, entropy) varies with LoRA rank
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from src.training.temporal_tracker import TemporalTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning time extraction
# ---------------------------------------------------------------------------


def compute_learning_times(
    tracker: TemporalTracker,
    threshold: Optional[float] = None,
) -> Dict[str, Optional[int]]:
    """Extract learning time for each example from a TemporalTracker.

    The learning time t_i of example i is defined as the first checkpoint
    (epoch index) at which the example's loss drops below the given
    threshold.  If the example never reaches the threshold, its learning
    time is None.

    Args:
        tracker: A TemporalTracker that has recorded per-example losses
            across training epochs.
        threshold: Loss value below which an example is considered
            "learned".  If None, the tracker's default loss_threshold
            is used.

    Returns:
        Dict mapping example_id (str) to learning_time (int checkpoint
        index, or None if the example was never learned).
    """
    if threshold is None:
        threshold = tracker.loss_threshold

    learning_times: Dict[str, Optional[int]] = {}
    for eid in tracker.records:
        learning_times[eid] = tracker.get_learning_time(eid, threshold=threshold)

    n_learned = sum(1 for t in learning_times.values() if t is not None)
    logger.info(
        "Computed learning times: %d/%d examples learned (threshold=%.4f)",
        n_learned,
        len(learning_times),
        threshold,
    )

    return learning_times


# ---------------------------------------------------------------------------
# Learning order consistency across seeds (Kendall's W)
# ---------------------------------------------------------------------------


def compute_learning_order_consistency(
    trackers_by_seed: Dict[int, TemporalTracker],
    threshold: Optional[float] = None,
) -> float:
    """Measure consistency of learning order across random seeds.

    Uses Kendall's W (coefficient of concordance) to quantify whether
    the same examples are consistently learned first across seeds.
    This is a key reproducibility check: if the learning order is
    determined by the data (annotation entropy) rather than random
    initialization, W should be high.

    Kendall's W is defined as:

        W = 12 * S / (m^2 * (n^3 - n))

    where:
        - m = number of rankers (seeds)
        - n = number of items (examples)
        - S = sum of squared deviations of the rank sums from the mean
              rank sum

    W ranges from 0 (no agreement) to 1 (perfect agreement).

    Only examples that are learned in ALL seeds are included in the
    computation.  Unlearned examples are excluded because they do not
    have a meaningful rank.

    Args:
        trackers_by_seed: Dict mapping seed (int) to TemporalTracker.
            Each tracker must cover the same set of examples.
        threshold: Loss threshold for learning time definition.

    Returns:
        Kendall's W in [0, 1].  W = 1 means perfect agreement across
        seeds about the learning order.

    Raises:
        ValueError: If fewer than 2 seeds are provided.
    """
    seeds = sorted(trackers_by_seed.keys())
    m = len(seeds)

    if m < 2:
        raise ValueError(
            f"Need at least 2 seeds for concordance analysis, got {m}."
        )

    # Extract learning times per seed
    times_per_seed: Dict[int, Dict[str, Optional[int]]] = {}
    for seed in seeds:
        times_per_seed[seed] = compute_learning_times(
            trackers_by_seed[seed], threshold=threshold
        )

    # Find examples that are learned in ALL seeds
    all_example_ids = set(times_per_seed[seeds[0]].keys())
    for seed in seeds[1:]:
        all_example_ids &= set(times_per_seed[seed].keys())

    common_learned = [
        eid
        for eid in sorted(all_example_ids)
        if all(
            times_per_seed[seed][eid] is not None for seed in seeds
        )
    ]

    n = len(common_learned)
    if n < 2:
        logger.warning(
            "Only %d examples learned across all %d seeds; returning W=0.",
            n,
            m,
        )
        return 0.0

    # Build the rank matrix: shape (m, n)
    # Each row is the rank ordering from one seed
    rank_matrix = np.zeros((m, n), dtype=np.float64)
    for i, seed in enumerate(seeds):
        times = np.array(
            [times_per_seed[seed][eid] for eid in common_learned],
            dtype=np.float64,
        )
        # Use average ranks to handle ties (examples learned at the same epoch)
        rank_matrix[i] = stats.rankdata(times, method="average")

    # Kendall's W computation
    rank_sums = rank_matrix.sum(axis=0)  # shape (n,)
    mean_rank_sum = rank_sums.mean()
    s_squared = np.sum((rank_sums - mean_rank_sum) ** 2)

    # W = 12 * S / (m^2 * (n^3 - n))
    w = 12.0 * s_squared / (m**2 * (n**3 - n))

    logger.info(
        "Kendall's W = %.4f (m=%d seeds, n=%d examples)", w, m, n
    )
    return float(w)


# ---------------------------------------------------------------------------
# Stratified learning curves by entropy category
# ---------------------------------------------------------------------------


def stratified_learning_curves(
    tracker: TemporalTracker,
    entropy_categories: Dict[str, List[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute mean loss trajectory and standard error per entropy category.

    This produces the data for the paper's hero figure (Figure 1): loss
    curves stratified by clean / ambiguous / contested.  The hypothesis
    predicts that clean examples have a loss curve that drops early,
    while contested examples drop later.

    Args:
        tracker: A TemporalTracker with recorded per-example losses.
        entropy_categories: Dict mapping category name (e.g., "clean",
            "ambiguous", "contested") to a list of example_ids in that
            category.

    Returns:
        Dict mapping category_name to a sub-dict with:
            "mean": np.ndarray of shape (n_epochs,) -- mean loss per epoch
            "sem": np.ndarray of shape (n_epochs,) -- standard error of mean
            "std": np.ndarray of shape (n_epochs,) -- standard deviation
            "n": int -- number of examples in the category
            "trajectories": np.ndarray of shape (n_examples, n_epochs) --
                individual loss trajectories (NaN-padded)
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}

    for cat_name, eids in entropy_categories.items():
        trajectories = []
        for eid in eids:
            if eid in tracker.records and len(tracker.records[eid].losses) > 0:
                trajectories.append(tracker.records[eid].losses)

        if not trajectories:
            logger.warning(
                "Category '%s' has no examples with recorded losses.", cat_name
            )
            result[cat_name] = {
                "mean": np.array([]),
                "sem": np.array([]),
                "std": np.array([]),
                "n": 0,
                "trajectories": np.array([]),
            }
            continue

        # Pad trajectories to the same length
        max_len = max(len(t) for t in trajectories)
        padded = np.full((len(trajectories), max_len), np.nan)
        for i, t in enumerate(trajectories):
            padded[i, : len(t)] = t

        mean_loss = np.nanmean(padded, axis=0)
        std_loss = np.nanstd(padded, axis=0, ddof=1)
        # Count non-NaN entries per epoch for proper SEM
        n_valid = np.sum(~np.isnan(padded), axis=0).astype(np.float64)
        n_valid = np.maximum(n_valid, 1.0)  # avoid division by zero
        sem_loss = std_loss / np.sqrt(n_valid)

        result[cat_name] = {
            "mean": mean_loss,
            "sem": sem_loss,
            "std": std_loss,
            "n": len(trajectories),
            "trajectories": padded,
        }

        logger.info(
            "Category '%s': %d examples, %d epochs",
            cat_name,
            len(trajectories),
            max_len,
        )

    return result


# ---------------------------------------------------------------------------
# Rank modulation: how Spearman(learning_time, entropy) varies with rank
# ---------------------------------------------------------------------------


def rank_modulation_summary(
    trackers_by_rank: Dict[int, Dict[int, TemporalTracker]],
    threshold: Optional[float] = None,
) -> Dict[int, Dict[str, Any]]:
    """Compute Spearman(learning_time, entropy) per LoRA rank across seeds.

    The theory predicts that lower LoRA rank produces a stronger
    positive correlation between learning time and annotation entropy
    (because lambda(r) is larger at low rank).  This function computes
    the correlation at each rank, aggregated across seeds, providing
    the data for Figure 2.

    Args:
        trackers_by_rank: Nested dict mapping rank (int) to a dict
            of seed (int) -> TemporalTracker.  Each tracker must have
            annotation_entropy stored in its ExampleRecords.
        threshold: Loss threshold for learning time definition.

    Returns:
        Dict mapping rank (int) to a sub-dict with:
            "mean_spearman": float -- mean Spearman rho across seeds
            "std_spearman": float -- standard deviation of rho
            "spearman_values": List[float] -- rho per seed
            "p_values": List[float] -- p-value per seed
            "n_seeds": int -- number of seeds
    """
    results: Dict[int, Dict[str, Any]] = {}

    for rank in sorted(trackers_by_rank.keys()):
        seed_trackers = trackers_by_rank[rank]
        rho_values: List[float] = []
        p_values: List[float] = []

        for seed in sorted(seed_trackers.keys()):
            tracker = seed_trackers[seed]
            learning_times = compute_learning_times(tracker, threshold=threshold)

            # Collect (learning_time, entropy) pairs for learned examples
            times_list: List[float] = []
            entropy_list: List[float] = []

            for eid, t in learning_times.items():
                if t is None:
                    continue
                record = tracker.records[eid]
                if record.annotation_entropy is None:
                    continue
                times_list.append(float(t))
                entropy_list.append(record.annotation_entropy)

            if len(times_list) < 3:
                logger.warning(
                    "Rank %d, seed %d: only %d learned examples with entropy; "
                    "skipping.",
                    rank,
                    seed,
                    len(times_list),
                )
                continue

            rho, p = stats.spearmanr(entropy_list, times_list)
            rho_values.append(float(rho))
            p_values.append(float(p))

        if not rho_values:
            results[rank] = {
                "mean_spearman": float("nan"),
                "std_spearman": float("nan"),
                "spearman_values": [],
                "p_values": [],
                "n_seeds": 0,
            }
        else:
            results[rank] = {
                "mean_spearman": float(np.mean(rho_values)),
                "std_spearman": float(np.std(rho_values, ddof=1))
                if len(rho_values) > 1
                else 0.0,
                "spearman_values": rho_values,
                "p_values": p_values,
                "n_seeds": len(rho_values),
            }

        logger.info(
            "Rank %d: Spearman rho = %.4f +/- %.4f (%d seeds)",
            rank,
            results[rank]["mean_spearman"],
            results[rank]["std_spearman"],
            results[rank]["n_seeds"],
        )

    return results
