"""
Annotation Entropy Computation
==============================
Computes per-example annotation entropy H_i from multi-annotator labels.

Annotation entropy measures the degree of annotator disagreement on a single
example. For K label classes and N annotators, the empirical label distribution
is p_k = (count of annotators choosing class k) / N, and the Shannon entropy is:

    H_i = -sum_{k=1}^{K} p_k * log(p_k)

where we use the convention 0 * log(0) = 0.

Lower H_i indicates strong consensus (clean examples); higher H_i indicates
genuine ambiguity or contested labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------------


def compute_annotation_entropy(
    labels_per_annotator: Union[Sequence[int], np.ndarray],
    n_classes: Optional[int] = None,
) -> float:
    """Compute Shannon entropy of the annotator label distribution for one example.

    Given a collection of labels (one per annotator), compute the empirical
    label distribution and return its Shannon entropy in nats (natural log).

    Args:
        labels_per_annotator: Array-like of integer class labels, one per
            annotator.  Shape (N,) where N is the number of annotators.
        n_classes: Total number of possible label classes.  If None, inferred
            as max(labels) + 1.

    Returns:
        H_i: Shannon entropy of the empirical label distribution (in nats).
            Ranges from 0 (perfect agreement) to log(K) (uniform distribution
            over K classes).

    Example:
        >>> compute_annotation_entropy([0, 0, 0, 0, 0])  # perfect agreement
        0.0
        >>> round(compute_annotation_entropy([0, 1, 2]), 4)  # uniform over 3
        1.0986
    """
    labels = np.asarray(labels_per_annotator, dtype=np.int64)
    if labels.size == 0:
        return 0.0

    if n_classes is None:
        n_classes = int(labels.max()) + 1

    # Empirical label distribution
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    probs = counts / counts.sum()

    # Shannon entropy: -sum p_k log(p_k), with 0*log(0) = 0
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
    return float(entropy)


def compute_annotation_entropy_batch(
    labels_matrix: Union[List[List[int]], np.ndarray],
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute annotation entropy for a batch of examples.

    Args:
        labels_matrix: 2D array-like of shape (n_examples, n_annotators).
            Entry [i, j] is the label assigned by annotator j to example i.
            Use -1 or NaN to indicate missing annotations.
        n_classes: Total number of possible label classes.

    Returns:
        entropies: Array of shape (n_examples,) containing H_i for each example.
    """
    labels_matrix = np.asarray(labels_matrix)
    n_examples = labels_matrix.shape[0]
    entropies = np.zeros(n_examples, dtype=np.float64)

    for i in range(n_examples):
        row = labels_matrix[i]
        # Filter out missing annotations (encoded as -1)
        valid = row[row >= 0]
        entropies[i] = compute_annotation_entropy(valid, n_classes=n_classes)

    return entropies


def compute_annotation_entropy_from_distribution(
    label_distribution: Union[Sequence[float], np.ndarray],
) -> float:
    """Compute entropy directly from a label probability distribution.

    Useful when the dataset already provides aggregated distributions
    (e.g., ChaosNLI provides counts rather than raw per-annotator labels).

    Args:
        label_distribution: Array-like of shape (K,) giving the probability
            (or count) for each of K classes.

    Returns:
        H_i: Shannon entropy in nats.
    """
    dist = np.asarray(label_distribution, dtype=np.float64)
    if dist.sum() == 0:
        return 0.0

    # Normalize to probabilities if counts are given
    probs = dist / dist.sum()
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
    return float(entropy)


# ---------------------------------------------------------------------------
# Entropy categorization
# ---------------------------------------------------------------------------


@dataclass
class EntropyCategorization:
    """Result of categorizing examples by annotation entropy."""

    categories: np.ndarray  # integer category per example: 0=clean, 1=ambiguous, 2=contested
    category_names: List[str]
    thresholds: List[float]
    counts: Dict[str, int]
    mean_entropy_per_category: Dict[str, float]


def categorize_by_entropy(
    entropies: Union[Sequence[float], np.ndarray],
    thresholds: Optional[List[float]] = None,
) -> EntropyCategorization:
    """Categorize examples into clean / ambiguous / contested by entropy.

    The default thresholds follow the paper (ACL SRW 2026) for NLI with
    3 classes (max entropy = log(3) ~ 1.099 nats):
        - H_i < 0.4:       "clean"      -- strong annotator consensus
        - 0.4 <= H_i < 0.7: "ambiguous"  -- moderate disagreement
        - H_i >= 0.7:       "contested"  -- near-uniform, genuine ambiguity

    Args:
        entropies: Array of per-example entropy values, shape (n_examples,).
        thresholds: List of boundary values defining categories.  Length T
            produces T+1 categories.  Default: [0.4, 0.7] (paper setting).

    Returns:
        EntropyCategorization with integer category assignments and metadata.
    """
    if thresholds is None:
        thresholds = [0.4, 0.7]

    entropies = np.asarray(entropies, dtype=np.float64)
    n_categories = len(thresholds) + 1

    # Default category names for 2 thresholds
    if n_categories == 3:
        category_names = ["clean", "ambiguous", "contested"]
    else:
        category_names = [f"category_{i}" for i in range(n_categories)]

    # Assign categories via digitize: bin index for each entropy value
    categories = np.digitize(entropies, thresholds).astype(np.int64)

    # Compute statistics per category
    counts = {}
    mean_entropy = {}
    for cat_idx, name in enumerate(category_names):
        mask = categories == cat_idx
        counts[name] = int(mask.sum())
        mean_entropy[name] = float(entropies[mask].mean()) if mask.any() else 0.0

    return EntropyCategorization(
        categories=categories,
        category_names=category_names,
        thresholds=thresholds,
        counts=counts,
        mean_entropy_per_category=mean_entropy,
    )


# ---------------------------------------------------------------------------
# Utility: normalized entropy (0-1 scale)
# ---------------------------------------------------------------------------


def normalized_entropy(
    labels_per_annotator: Union[Sequence[int], np.ndarray],
    n_classes: Optional[int] = None,
) -> float:
    """Compute entropy normalized to [0, 1] by dividing by log(K).

    Args:
        labels_per_annotator: Array-like of integer class labels.
        n_classes: Total number of possible label classes.

    Returns:
        H_i / log(K), in [0, 1].  Returns 0 if K <= 1.
    """
    labels = np.asarray(labels_per_annotator, dtype=np.int64)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    if n_classes <= 1:
        return 0.0

    h = compute_annotation_entropy(labels, n_classes=n_classes)
    return float(h / np.log(n_classes))
