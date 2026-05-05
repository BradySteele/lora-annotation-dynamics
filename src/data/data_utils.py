"""
Data Utilities
==============
Shared utility functions for the data loading pipeline, including:

- Difficulty proxy computation (sentence length, word frequency, token count)
- Tracked DataLoader creation that preserves example_ids for per-example
  loss tracking throughout training
- Stratified train/val splitting that maintains entropy category proportions

These utilities ensure that the temporal learning dynamics analysis can
correctly associate per-example metrics with their annotation entropy.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Difficulty proxy computation
# ---------------------------------------------------------------------------

# Zipf's law approximation: common English words by decreasing frequency.
# We use word length as a simple proxy for log-frequency (longer words tend
# to be rarer). This avoids loading an external frequency list while still
# providing a useful signal for difficulty analysis.
#
# A more sophisticated version could use the SUBTLEX-US frequency norms
# or unigram counts from Google Ngrams.


def compute_difficulty_proxies(
    texts: List[str],
    tokenizer: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Compute surface-level difficulty proxies for a list of texts.

    These proxies capture text complexity features that may correlate with
    (but should be distinguished from) annotator disagreement. They serve
    as controls in the temporal separation analysis.

    Computed proxies:
        - sentence_length: number of whitespace-delimited words
        - mean_word_length: average character length of words (proxy for
          word rarity via Zipf's law)
        - log_word_rarity: average estimated log-rarity of words, using
          word length as a simple frequency proxy. Longer words tend to be
          rarer in natural language.
        - num_tokens: number of subword tokens from the tokenizer (if provided)

    Args:
        texts: List of input text strings.
        tokenizer: Optional HuggingFace tokenizer. If provided, computes
            num_tokens. Otherwise num_tokens is set to sentence_length.

    Returns:
        Dictionary mapping proxy name to np.ndarray of shape (n_texts,).
    """
    n_texts = len(texts)

    sentence_lengths = np.zeros(n_texts, dtype=np.float64)
    mean_word_lengths = np.zeros(n_texts, dtype=np.float64)
    log_word_rarities = np.zeros(n_texts, dtype=np.float64)
    num_tokens = np.zeros(n_texts, dtype=np.int64)

    for i, text in enumerate(texts):
        words = text.split()
        n_words = len(words)
        sentence_lengths[i] = n_words

        if n_words > 0:
            word_lens = [len(w) for w in words]
            mean_word_lengths[i] = np.mean(word_lens)

            # Log-rarity proxy: log(word_length + 1) as a rough estimate
            # of word rarity. The +1 avoids log(0) for empty strings.
            # Calibrated so common short words (3-4 chars) get low rarity
            # and long specialized words (10+ chars) get high rarity.
            log_word_rarities[i] = np.mean(
                [math.log(max(wl, 1) + 1) for wl in word_lens]
            )
        else:
            mean_word_lengths[i] = 0.0
            log_word_rarities[i] = 0.0

    # Compute subword token counts if tokenizer is available
    if tokenizer is not None:
        try:
            # Batch tokenize for efficiency (without padding)
            encodings = tokenizer(
                texts,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            for i, input_ids in enumerate(encodings["input_ids"]):
                num_tokens[i] = len(input_ids)
        except Exception as e:
            logger.warning(
                f"Tokenizer failed for token count computation: {e}. "
                f"Falling back to word count."
            )
            num_tokens = sentence_lengths.astype(np.int64)
    else:
        # Approximate token count as word count when no tokenizer available
        num_tokens = sentence_lengths.astype(np.int64)

    return {
        "sentence_length": sentence_lengths,
        "mean_word_length": mean_word_lengths,
        "log_word_rarity": log_word_rarities,
        "num_tokens": num_tokens,
    }


# ---------------------------------------------------------------------------
# Tracked DataLoader
# ---------------------------------------------------------------------------


def _tracked_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate function that preserves example_id and metadata tensors.

    Standard PyTorch default_collate does not handle our custom fields
    well. This function stacks all tensor fields and preserves the
    example_id field which is critical for per-example loss tracking.

    Args:
        batch: List of sample dicts from the dataset __getitem__.

    Returns:
        Batched dictionary with all fields stacked into tensors.
    """
    import torch

    if len(batch) == 0:
        return {}

    result: Dict[str, Any] = {}
    for key in batch[0].keys():
        items = [item[key] for item in batch]

        if isinstance(items[0], torch.Tensor):
            result[key] = torch.stack(items)
        elif isinstance(items[0], (int, float, np.integer, np.floating)):
            result[key] = torch.tensor(items)
        elif isinstance(items[0], np.ndarray):
            result[key] = torch.from_numpy(np.stack(items))
        else:
            # Keep as list (e.g., string IDs)
            result[key] = items

    return result


def create_tracked_dataloader(
    dataset: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> Any:
    """Create a DataLoader that preserves example_ids for per-example tracking.

    The returned DataLoader uses a custom collate function that includes
    'example_id' (and other metadata like 'annotation_entropy' and
    'entropy_category') in every batch dictionary. This is essential for
    recording per-example losses during training to analyze temporal
    learning dynamics.

    Args:
        dataset: A PyTorch Dataset that returns dicts with 'example_id' key.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the data each epoch. Set True for training,
            False for validation/analysis.
        num_workers: Number of data loading workers. Use 0 for MPS compatibility.
        pin_memory: Whether to pin memory. Use False for MPS compatibility.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        torch.utils.data.DataLoader with tracked collation.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_tracked_collate_fn,
    )


# ---------------------------------------------------------------------------
# Stratified train/val split
# ---------------------------------------------------------------------------


def stratified_train_val_split(
    dataset: Any,
    val_fraction: float = 0.2,
    entropy_categories: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[Any, Any]:
    """Split a dataset into train and val sets, stratified by entropy category.

    Ensures that each entropy category (clean/ambiguous/contested) is
    proportionally represented in both train and val sets. This is critical
    for the temporal separation analysis: we need sufficient examples from
    each category in the training set to observe learning dynamics, AND
    in the val set to evaluate generalization per category.

    Works with both PyTorch Dataset objects (ChaosNLIDataset, GoEmotionsDataset)
    and raw dictionaries from load_chaosnli_with_entropy() / load_goemotions().

    Args:
        dataset: Either a PyTorch Dataset with per-example entropy_category
            accessible via indexing, or a dict with "entropy_categories" key.
        val_fraction: Fraction of data to use for validation (0 to 1).
        entropy_categories: Optional array of integer category codes (one per
            example). If None, extracted from dataset.
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_dataset, val_dataset). The type depends on the input:
            - If input is a dict, returns (dict, dict) with sliced arrays/lists.
            - If input is a Dataset, returns SubsetDatasets wrapping the original.
    """
    rng = np.random.RandomState(seed)

    # Extract entropy categories if not provided
    if entropy_categories is None:
        if isinstance(dataset, dict):
            entropy_categories = np.asarray(dataset["entropy_categories"])
        elif hasattr(dataset, "entropy_categories"):
            entropy_categories = np.asarray(dataset.entropy_categories)
        else:
            # No category info available: fall back to simple random split
            logger.warning(
                "No entropy_categories found. "
                "Falling back to simple random split."
            )
            n = _get_dataset_length(dataset)
            indices = rng.permutation(n)
            n_val = max(1, int(n * val_fraction))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            return _create_subsets(dataset, train_indices, val_indices)

    n = len(entropy_categories)
    unique_categories = np.unique(entropy_categories)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for cat in unique_categories:
        cat_indices = np.where(entropy_categories == cat)[0]
        rng.shuffle(cat_indices)

        n_val_cat = max(1, int(len(cat_indices) * val_fraction))
        val_indices.extend(cat_indices[:n_val_cat].tolist())
        train_indices.extend(cat_indices[n_val_cat:].tolist())

    # Shuffle within splits to avoid ordering artifacts
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_indices_arr = np.array(train_indices, dtype=np.int64)
    val_indices_arr = np.array(val_indices, dtype=np.int64)

    logger.info(
        f"Stratified split: {len(train_indices)} train, "
        f"{len(val_indices)} val (val_fraction={val_fraction})"
    )

    # Log per-category distribution in each split
    for cat in unique_categories:
        n_train_cat = np.sum(entropy_categories[train_indices_arr] == cat)
        n_val_cat = np.sum(entropy_categories[val_indices_arr] == cat)
        logger.info(
            f"  Category {cat}: train={n_train_cat}, val={n_val_cat}"
        )

    return _create_subsets(dataset, train_indices_arr, val_indices_arr)


def _get_dataset_length(dataset: Any) -> int:
    """Get the number of examples in a dataset (dict or Dataset object)."""
    if isinstance(dataset, dict):
        # Find the first list/array value and use its length
        for key, val in dataset.items():
            if isinstance(val, (list, np.ndarray)):
                return len(val)
        raise ValueError("Cannot determine dataset length from dict.")
    elif hasattr(dataset, "__len__"):
        return len(dataset)
    else:
        raise ValueError(f"Cannot determine length for dataset type: {type(dataset)}")


def _create_subsets(
    dataset: Any,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[Any, Any]:
    """Create train/val subsets from a dataset and index arrays.

    Handles both dict-based datasets and PyTorch Dataset objects.

    Args:
        dataset: The full dataset.
        train_indices: Indices for the training subset.
        val_indices: Indices for the validation subset.

    Returns:
        Tuple of (train_subset, val_subset).
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)

    if isinstance(dataset, dict):
        return (
            _slice_dict(dataset, train_indices),
            _slice_dict(dataset, val_indices),
        )
    else:
        # PyTorch Dataset: create Subset wrappers
        from torch.utils.data import Subset

        return (
            Subset(dataset, train_indices.tolist()),
            Subset(dataset, val_indices.tolist()),
        )


def _slice_dict(
    data: Dict[str, Any],
    indices: np.ndarray,
) -> Dict[str, Any]:
    """Slice all list/array values in a dictionary by the given indices.

    Non-sliceable values (scalars, objects) are copied as-is.

    Args:
        data: Dictionary with list/array values.
        indices: Integer indices to select.

    Returns:
        New dictionary with sliced values.
    """
    result: Dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            result[key] = val[indices]
        elif isinstance(val, list):
            result[key] = [val[i] for i in indices]
        else:
            # Scalar or non-indexable: copy as-is
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Dataset info and validation
# ---------------------------------------------------------------------------


def validate_tracked_dataset(dataset: Any) -> bool:
    """Validate that a dataset has the required fields for tracked training.

    Checks that the dataset __getitem__ returns a dict with required keys:
    'input_ids', 'attention_mask', 'labels', 'example_id'.

    Args:
        dataset: A PyTorch Dataset to validate.

    Returns:
        True if valid, False otherwise.
    """
    required_keys = {"input_ids", "attention_mask", "labels", "example_id"}

    try:
        sample = dataset[0]
        if not isinstance(sample, dict):
            logger.error(
                f"Dataset __getitem__ returns {type(sample)}, expected dict."
            )
            return False

        missing = required_keys - set(sample.keys())
        if missing:
            logger.error(f"Dataset missing required keys: {missing}")
            return False

        return True
    except Exception as e:
        logger.error(f"Failed to validate dataset: {e}")
        return False


def dataset_summary(dataset: Any) -> Dict[str, Any]:
    """Generate a summary of a tracked dataset.

    Args:
        dataset: A dataset (PyTorch Dataset or dict).

    Returns:
        Dictionary with summary statistics.
    """
    summary: Dict[str, Any] = {}

    if isinstance(dataset, dict):
        n_examples = _get_dataset_length(dataset)
        summary["n_examples"] = n_examples
        summary["keys"] = list(dataset.keys())

        if "entropies" in dataset:
            ent = np.asarray(dataset["entropies"])
            summary["entropy_mean"] = float(np.mean(ent))
            summary["entropy_std"] = float(np.std(ent))
            summary["entropy_min"] = float(np.min(ent))
            summary["entropy_max"] = float(np.max(ent))

        if "entropy_categories" in dataset:
            cats = np.asarray(dataset["entropy_categories"])
            unique, counts = np.unique(cats, return_counts=True)
            summary["category_distribution"] = {
                int(u): int(c) for u, c in zip(unique, counts)
            }

        if "majority_labels" in dataset:
            labels = np.asarray(dataset["majority_labels"])
            unique, counts = np.unique(labels, return_counts=True)
            summary["label_distribution"] = {
                int(u): int(c) for u, c in zip(unique, counts)
            }
        elif "labels" in dataset:
            labels = np.asarray(dataset["labels"])
            unique, counts = np.unique(labels, return_counts=True)
            summary["label_distribution"] = {
                int(u): int(c) for u, c in zip(unique, counts)
            }
    elif hasattr(dataset, "__len__"):
        summary["n_examples"] = len(dataset)
        if hasattr(dataset, "annotation_entropy"):
            ent = np.asarray(dataset.annotation_entropy)
            summary["entropy_mean"] = float(np.mean(ent))
            summary["entropy_std"] = float(np.std(ent))
    else:
        summary["type"] = str(type(dataset))

    return summary
