"""
GoEmotions Data Loader
======================
Loads the GoEmotions dataset (Demszky et al., 2020) for annotator disagreement
analysis in the emotion classification domain.

GoEmotions provides 58K Reddit comments annotated with 27 emotion categories
plus neutral. In the simplified version, each example has a text and a list of
label indices. For disagreement analysis, we compute pseudo-entropy from the
multi-label distribution: examples with multiple conflicting labels assigned
by different annotators have higher entropy.

This serves as a secondary dataset to validate that temporal separation
findings from ChaosNLI generalize beyond NLI.

Reference:
    Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., &
    Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.annotation_entropy import (
    categorize_by_entropy,
    compute_annotation_entropy_from_distribution,
)

logger = logging.getLogger(__name__)

# GoEmotions emotion categories (28 total: 27 emotions + neutral)
GOEMOTIONS_LABELS: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

N_GOEMOTIONS_CLASSES: int = len(GOEMOTIONS_LABELS)  # 28


def _compute_pseudo_entropy_from_multilabel(
    label_indices: List[int],
    n_classes: int = N_GOEMOTIONS_CLASSES,
) -> float:
    """Compute pseudo-entropy from a multi-label annotation.

    Since GoEmotions simplified format provides the union of annotator labels
    (not per-annotator breakdowns), we approximate disagreement by treating
    the number of assigned labels as a signal of annotation ambiguity.

    The pseudo-entropy is computed by creating a uniform distribution over
    the assigned labels and computing Shannon entropy of that distribution.
    This means:
        - 1 label  -> H = 0 (clear consensus)
        - 2 labels -> H = log(2) ~ 0.693 (moderate disagreement)
        - 3 labels -> H = log(3) ~ 1.099 (high disagreement)
        - k labels -> H = log(k)

    This is a proxy: the true per-annotator entropy would require the raw
    annotation files.

    Args:
        label_indices: List of integer label indices assigned to this example.
        n_classes: Total number of label classes (28 for GoEmotions).

    Returns:
        Pseudo-entropy value in nats.
    """
    if len(label_indices) == 0:
        return 0.0
    if len(label_indices) == 1:
        return 0.0

    # Create a distribution: uniform over assigned labels
    dist = np.zeros(n_classes, dtype=np.float64)
    for idx in label_indices:
        if 0 <= idx < n_classes:
            dist[idx] = 1.0

    return compute_annotation_entropy_from_distribution(dist)


def load_goemotions(
    config: str = "simplified",
    max_examples: Optional[int] = None,
    max_seq_length: int = 128,
    entropy_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Load GoEmotions dataset with pseudo-entropy for disagreement analysis.

    Loads from HuggingFace datasets ("google-research-datasets/go_emotions"),
    computes pseudo-entropy from the multi-label annotations, and returns
    a structured dictionary suitable for creating PyTorch datasets.

    For training, we use the FIRST label in each example's label list as the
    primary (majority-vote equivalent) label for cross-entropy training.

    Args:
        config: HuggingFace dataset config. Default "simplified" which has
            columns: text (str), labels (List[int]).
        max_examples: Maximum number of examples to load (for debugging).
        max_seq_length: Maximum sequence length for tokenization reference.
        entropy_thresholds: Thresholds for entropy categorization.
            Default [0.4, 0.7] matching ChaosNLI (paper setting).

    Returns:
        Dictionary with keys:
            - "texts": List[str] of input texts
            - "labels": np.ndarray of shape (n_examples,) -- primary label
            - "label_lists": List[List[int]] -- all labels per example
            - "n_labels_per_example": np.ndarray of shape (n_examples,)
            - "example_ids": List[str] of example identifiers
            - "entropies": np.ndarray of shape (n_examples,)
            - "entropy_categories": np.ndarray of shape (n_examples,)
            - "category_names": List[str]
            - "categorization": EntropyCategorization object
            - "num_classes": int (28)
            - "label_names": List[str]

    Raises:
        RuntimeError: If the dataset cannot be loaded from HuggingFace.
    """
    from datasets import load_dataset

    # Load from HuggingFace
    logger.info(
        f"Loading GoEmotions ({config}) from HuggingFace datasets..."
    )
    try:
        dataset = load_dataset(
            "google-research-datasets/go_emotions",
            config,
            trust_remote_code=True,
        )
    except Exception as e1:
        # Fallback path
        try:
            dataset = load_dataset(
                "go_emotions",
                config,
                trust_remote_code=True,
            )
        except Exception as e2:
            raise RuntimeError(
                f"Could not load GoEmotions from HuggingFace: "
                f"First attempt: {e1}, Second attempt: {e2}"
            ) from e2

    # Merge train/validation/test into a single pool for our analysis
    # (we do our own stratified split later)
    all_splits = []
    for split_name in ["train", "validation", "test"]:
        if split_name in dataset:
            all_splits.append(dataset[split_name])

    if not all_splits:
        raise RuntimeError(
            f"GoEmotions dataset has no recognized splits. "
            f"Available: {list(dataset.keys())}"
        )

    # Concatenate splits
    from datasets import concatenate_datasets
    full_dataset = concatenate_datasets(all_splits)
    logger.info(f"GoEmotions full dataset: {len(full_dataset)} examples")

    if max_examples is not None:
        full_dataset = full_dataset.select(
            range(min(len(full_dataset), max_examples))
        )

    # Extract fields
    texts: List[str] = []
    label_lists: List[List[int]] = []
    primary_labels: List[int] = []
    example_ids: List[str] = []
    entropies_list: List[float] = []

    columns = full_dataset.column_names

    for i, row in enumerate(full_dataset):
        # Text
        text = row.get("text", row.get("sentence", ""))
        texts.append(str(text))

        # Labels -- may be a list of ints or a single int
        raw_labels = row.get("labels", row.get("label", []))
        if isinstance(raw_labels, int):
            raw_labels = [raw_labels]
        elif isinstance(raw_labels, np.ndarray):
            raw_labels = raw_labels.tolist()

        label_lists.append(raw_labels)

        # Primary label: first label in the list (most common / first annotated)
        primary_label = raw_labels[0] if len(raw_labels) > 0 else 0
        primary_labels.append(primary_label)

        # Example ID
        example_id = row.get("id", f"ge_{i}")
        example_ids.append(str(example_id))

        # Compute pseudo-entropy from multi-label distribution
        entropy = _compute_pseudo_entropy_from_multilabel(raw_labels)
        entropies_list.append(entropy)

    # Convert to arrays
    labels_arr = np.array(primary_labels, dtype=np.int64)
    entropies_arr = np.array(entropies_list, dtype=np.float64)
    n_labels_per_example = np.array(
        [len(ll) for ll in label_lists], dtype=np.int64
    )

    # Categorize by entropy
    categorization = categorize_by_entropy(
        entropies_arr, thresholds=entropy_thresholds
    )

    logger.info(
        f"Loaded GoEmotions: {len(texts)} examples, "
        f"entropy category counts: {categorization.counts}"
    )

    return {
        "texts": texts,
        "labels": labels_arr,
        "label_lists": label_lists,
        "n_labels_per_example": n_labels_per_example,
        "example_ids": example_ids,
        "entropies": entropies_arr,
        "entropy_categories": categorization.categories,
        "category_names": categorization.category_names,
        "categorization": categorization,
        "num_classes": N_GOEMOTIONS_CLASSES,
        "label_names": GOEMOTIONS_LABELS,
    }


def create_goemotions_torch_dataset(
    data: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 128,
) -> "GoEmotionsDataset":
    """Create a PyTorch Dataset from loaded GoEmotions data.

    Tokenizes texts and bundles with primary labels, example IDs, and
    pseudo-entropy values for per-example tracking during training.

    Args:
        data: Dictionary returned by load_goemotions().
        tokenizer: HuggingFace tokenizer for the model.
        max_seq_length: Maximum sequence length for tokenization.

    Returns:
        GoEmotionsDataset instance ready for DataLoader.
    """
    import torch

    # Tokenize all texts
    encodings = tokenizer(
        data["texts"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    labels = torch.tensor(data["labels"], dtype=torch.long)
    example_ids = torch.arange(len(labels), dtype=torch.long)
    entropies = torch.tensor(data["entropies"], dtype=torch.float32)
    entropy_categories = torch.tensor(
        data["entropy_categories"], dtype=torch.long
    )

    return GoEmotionsDataset(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=labels,
        example_ids=example_ids,
        annotation_entropy=entropies,
        entropy_categories=entropy_categories,
    )


class GoEmotionsDataset:
    """PyTorch-compatible dataset for GoEmotions with per-example tracking.

    Each item yields a dictionary with:
        - input_ids: tokenized input, shape (max_seq_length,)
        - attention_mask: attention mask, shape (max_seq_length,)
        - labels: primary label, scalar
        - example_id: unique integer ID for tracking, scalar
        - annotation_entropy: pseudo-entropy value, scalar
        - entropy_category: integer category (0=clean, 1=ambiguous, 2=contested)
    """

    def __init__(
        self,
        input_ids: Any,
        attention_mask: Any,
        labels: Any,
        example_ids: Any,
        annotation_entropy: Any,
        entropy_categories: Any,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.example_ids = example_ids
        self.annotation_entropy = annotation_entropy
        self.entropy_categories = entropy_categories

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "example_id": self.example_ids[idx],
            "annotation_entropy": self.annotation_entropy[idx],
            "entropy_category": self.entropy_categories[idx],
        }


def create_synthetic_goemotions(
    n_examples: int = 500,
    n_classes: int = 28,
    max_labels_per_example: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create synthetic GoEmotions-like data for development and testing.

    Args:
        n_examples: Total number of examples to generate.
        n_classes: Number of emotion label classes.
        max_labels_per_example: Maximum number of labels per example.
        seed: Random seed.

    Returns:
        Dictionary matching the load_goemotions() return format.
    """
    rng = np.random.RandomState(seed)

    texts: List[str] = []
    label_lists: List[List[int]] = []
    primary_labels: List[int] = []
    entropies_list: List[float] = []

    for i in range(n_examples):
        # Generate synthetic text
        texts.append(f"synthetic_text_{i}")

        # Random number of labels (1 to max_labels_per_example)
        n_labels = rng.randint(1, max_labels_per_example + 1)
        labels = sorted(rng.choice(n_classes, size=n_labels, replace=False).tolist())
        label_lists.append(labels)
        primary_labels.append(labels[0])

        # Pseudo-entropy
        entropy = _compute_pseudo_entropy_from_multilabel(labels, n_classes)
        entropies_list.append(entropy)

    labels_arr = np.array(primary_labels, dtype=np.int64)
    entropies_arr = np.array(entropies_list, dtype=np.float64)
    n_labels_per_example = np.array(
        [len(ll) for ll in label_lists], dtype=np.int64
    )

    categorization = categorize_by_entropy(entropies_arr)

    return {
        "texts": texts,
        "labels": labels_arr,
        "label_lists": label_lists,
        "n_labels_per_example": n_labels_per_example,
        "example_ids": [f"synth_ge_{i}" for i in range(n_examples)],
        "entropies": entropies_arr,
        "entropy_categories": categorization.categories,
        "category_names": categorization.category_names,
        "categorization": categorization,
        "num_classes": n_classes,
        "label_names": GOEMOTIONS_LABELS[:n_classes],
    }
