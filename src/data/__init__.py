"""
Data Loading Pipeline
=====================
Data loading, preprocessing, and per-example tracking for the
LoRA temporal separation study.

Primary dataset: ChaosNLI (100 annotators per NLI example)
Secondary dataset: GoEmotions (multi-label emotion classification)

All dataloaders preserve example_ids in each batch for per-example
loss tracking throughout training.
"""

# Annotation entropy computation
from src.data.annotation_entropy import (
    EntropyCategorization,
    categorize_by_entropy,
    compute_annotation_entropy,
    compute_annotation_entropy_batch,
    compute_annotation_entropy_from_distribution,
    normalized_entropy,
)

# ChaosNLI (primary dataset)
from src.data.chaosnli import (
    ChaosNLIDataset,
    create_chaosnli_torch_dataset,
    create_synthetic_chaosnli,
    load_chaosnli,
    load_chaosnli_with_entropy,
)

# GoEmotions (secondary dataset)
from src.data.goemotions import (
    GoEmotionsDataset,
    create_goemotions_torch_dataset,
    create_synthetic_goemotions,
    load_goemotions,
)

# Data utilities
from src.data.data_utils import (
    compute_difficulty_proxies,
    create_tracked_dataloader,
    dataset_summary,
    stratified_train_val_split,
    validate_tracked_dataset,
)

__all__ = [
    # Entropy
    "compute_annotation_entropy",
    "compute_annotation_entropy_batch",
    "compute_annotation_entropy_from_distribution",
    "categorize_by_entropy",
    "normalized_entropy",
    "EntropyCategorization",
    # ChaosNLI
    "load_chaosnli",
    "load_chaosnli_with_entropy",
    "create_chaosnli_torch_dataset",
    "create_synthetic_chaosnli",
    "ChaosNLIDataset",
    # GoEmotions
    "load_goemotions",
    "create_goemotions_torch_dataset",
    "create_synthetic_goemotions",
    "GoEmotionsDataset",
    # Utils
    "compute_difficulty_proxies",
    "create_tracked_dataloader",
    "stratified_train_val_split",
    "validate_tracked_dataset",
    "dataset_summary",
]
