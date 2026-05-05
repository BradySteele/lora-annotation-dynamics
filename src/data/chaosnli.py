"""
ChaosNLI Data Loader
====================
Loads the ChaosNLI dataset (Nie et al., 2020), which provides 100 annotator
labels per example for subsets of SNLI, MultiNLI, and AbductiveNLI.

ChaosNLI is the ideal testbed for this paper because it has:
  - Fine-grained annotation disagreement (100 annotators per example)
  - Known examples with genuine ambiguity vs. clear consensus
  - Established baselines for annotator agreement studies

The dataset is used as BOTH training and analysis data (~4700 examples total).
We create an 80/20 train/val split stratified by entropy category to ensure
each split has representative examples from clean, ambiguous, and contested.

Reference:
    Nie, Y., Zhou, X., & Bansal, M. (2020). What Can We Learn from Collective
    Human Opinions on Natural Language Inference Data? EMNLP.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.data.annotation_entropy import (
    categorize_by_entropy,
    compute_annotation_entropy_from_distribution,
)

logger = logging.getLogger(__name__)

# NLI label mapping used by ChaosNLI
NLI_LABEL_MAP: Dict[str, int] = {
    "entailment": 0,
    "e": 0,
    "neutral": 1,
    "n": 1,
    "contradiction": 2,
    "c": 2,
}

# ChaosNLI JSONL filenames by subset (used for both local lookup and download)
CHAOSNLI_FILENAMES: Dict[str, str] = {
    "snli": "chaosNLI_snli.jsonl",
    "mnli": "chaosNLI_mnli_m.jsonl",
    "mnli_mm": "chaosNLI_mnli_mm.jsonl",
    "alphanli": "chaosNLI_alphanli.jsonl",
}

# Multiple GitHub raw URL patterns to try, in order.  The repository may use
# "main" or "master" as the default branch, and the data directory structure
# may vary across forks or releases.
_GITHUB_URL_TEMPLATES: List[str] = [
    "https://raw.githubusercontent.com/easonnie/ChaosNLI/main/data/chaosNLI_v1.0/{filename}",
    "https://raw.githubusercontent.com/easonnie/ChaosNLI/master/data/chaosNLI_v1.0/{filename}",
    "https://raw.githubusercontent.com/easonnie/ChaosNLI/main/chaosNLI_v1.0/{filename}",
    "https://raw.githubusercontent.com/easonnie/ChaosNLI/master/chaosNLI_v1.0/{filename}",
    "https://raw.githubusercontent.com/easonnie/ChaosNLI/main/data/{filename}",
    "https://raw.githubusercontent.com/easonnie/ChaosNLI/master/data/{filename}",
]

# Number of NLI classes
N_NLI_CLASSES: int = 3


def _download_chaosnli_jsonl(
    subset: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Download ChaosNLI JSONL file from GitHub if not already cached.

    Tries multiple GitHub raw URL patterns in sequence because the
    repository may use different branch names or directory layouts.

    Args:
        subset: One of "snli", "mnli", "mnli_mm", "alphanli".
        cache_dir: Directory to cache the downloaded file. Defaults to
            ~/.cache/chaosnli/.

    Returns:
        Local path to the downloaded JSONL file.

    Raises:
        RuntimeError: If all download URLs fail, with a clear message
            listing what was tried and how to work around the issue.
    """
    if subset not in CHAOSNLI_FILENAMES:
        raise ValueError(
            f"Unknown ChaosNLI subset: {subset}. "
            f"Available: {list(CHAOSNLI_FILENAMES.keys())}"
        )

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "chaosnli")
    os.makedirs(cache_dir, exist_ok=True)

    filename = CHAOSNLI_FILENAMES[subset]
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        # Validate that the cached file is not empty or corrupt
        file_size = os.path.getsize(local_path)
        if file_size > 100:
            logger.info(f"Using cached ChaosNLI {subset} from {local_path} ({file_size:,} bytes)")
            return local_path
        else:
            logger.warning(
                f"Cached file {local_path} is suspiciously small ({file_size} bytes), re-downloading."
            )
            os.remove(local_path)

    # Try each URL pattern in order
    tried_urls: List[str] = []
    errors: List[str] = []
    for template in _GITHUB_URL_TEMPLATES:
        url = template.format(filename=filename)
        tried_urls.append(url)
        logger.info(f"Trying to download ChaosNLI {subset} from {url}")
        try:
            urllib.request.urlretrieve(url, local_path)
            file_size = os.path.getsize(local_path)
            if file_size < 100:
                os.remove(local_path)
                errors.append(f"  {url} -> downloaded but file too small ({file_size} bytes)")
                continue
            logger.info(f"Downloaded to {local_path} ({file_size:,} bytes)")
            return local_path
        except Exception as e:
            errors.append(f"  {url} -> {e}")
            # Clean up any partial download
            if os.path.exists(local_path):
                os.remove(local_path)
            continue

    # All URLs failed -- raise a clear, actionable error
    error_details = "\n".join(errors)
    raise RuntimeError(
        f"\n{'=' * 70}\n"
        f"ERROR: Failed to download ChaosNLI '{subset}' data.\n\n"
        f"Tried {len(tried_urls)} URLs, all failed:\n{error_details}\n\n"
        f"To fix this, either:\n"
        f"  1. Clone the repo and pass the data directory:\n"
        f"     git clone https://github.com/easonnie/ChaosNLI\n"
        f"     python scripts/01_prepare_data.py --data-dir /path/to/ChaosNLI/data/chaosNLI_v1.0\n\n"
        f"  2. Manually download the JSONL files from:\n"
        f"     https://github.com/easonnie/ChaosNLI\n"
        f"     and place them in: {cache_dir}/\n"
        f"{'=' * 70}"
    )


def _parse_chaosnli_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Parse a ChaosNLI JSONL file into a list of example dicts.

    Each line in the JSONL file is a JSON object with fields:
      - uid: unique example identifier
      - label_counter: dict mapping label string -> count from 100 annotators
      - old_label: original gold label
      - example: dict with premise and hypothesis (key names vary by subset)

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of parsed example dictionaries.
    """
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def _extract_label_distribution(label_counter: Dict[str, int]) -> np.ndarray:
    """Convert a ChaosNLI label_counter dict to a fixed-size count array.

    Args:
        label_counter: Dict mapping label strings (e.g., "entailment",
            "neutral", "contradiction" or "e", "n", "c") to annotator counts.

    Returns:
        Array of shape (3,) with counts for [entailment, neutral, contradiction].
    """
    dist = np.zeros(N_NLI_CLASSES, dtype=np.int64)
    for label_str, count in label_counter.items():
        label_str_lower = label_str.lower().strip()
        if label_str_lower in NLI_LABEL_MAP:
            label_idx = NLI_LABEL_MAP[label_str_lower]
            dist[label_idx] += count
        else:
            logger.warning(f"Unknown label string in label_counter: '{label_str}'")
    return dist


def _extract_premise_hypothesis(
    example_data: Dict[str, Any],
    raw_example: Dict[str, Any],
) -> Tuple[str, str]:
    """Extract premise and hypothesis from a ChaosNLI example.

    ChaosNLI stores the original NLI example in the "example" field, but
    the key names vary: SNLI uses "premise"/"hypothesis", while MNLI may
    use "sentence1"/"sentence2". We try multiple key patterns.

    Additionally, for some versions the premise/hypothesis may be at the
    top level of raw_example rather than inside a nested "example" dict.

    Args:
        example_data: The "example" sub-dict (may be a dict or a string).
        raw_example: The full raw example dict (fallback for top-level keys).

    Returns:
        Tuple of (premise, hypothesis) strings.
    """
    # If example_data is a string (some versions store serialized JSON), parse it
    if isinstance(example_data, str):
        try:
            example_data = json.loads(example_data)
        except (json.JSONDecodeError, TypeError):
            # Could not parse; treat the string as premise, empty hypothesis
            return example_data, ""

    # Try various key patterns for premise
    premise_keys = ["premise", "sentence1", "sent1", "Premise"]
    hypothesis_keys = ["hypothesis", "sentence2", "sent2", "Hypothesis"]

    premise = ""
    hypothesis = ""

    # Check inside the example_data dict first
    if isinstance(example_data, dict):
        for key in premise_keys:
            if key in example_data:
                premise = str(example_data[key])
                break
        for key in hypothesis_keys:
            if key in example_data:
                hypothesis = str(example_data[key])
                break

    # Fallback: check top-level raw_example
    if not premise:
        for key in premise_keys:
            if key in raw_example:
                premise = str(raw_example[key])
                break
    if not hypothesis:
        for key in hypothesis_keys:
            if key in raw_example:
                hypothesis = str(raw_example[key])
                break

    return premise, hypothesis


def load_chaosnli(
    subset: str = "snli",
    data_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Load ChaosNLI dataset with per-annotator label distributions.

    Loads from a local JSONL file (if data_dir is provided) or downloads
    from the official ChaosNLI GitHub repository. Returns real text and
    real annotation distributions -- never synthetic data.

    Args:
        subset: Which NLI subset to use. One of "snli", "mnli", "mnli_mm",
            "alphanli".
        data_dir: Path to a directory containing ChaosNLI JSONL files. If None,
            downloads from GitHub and caches in ~/.cache/chaosnli/.
        max_examples: Maximum number of examples to load (for debugging).

    Returns:
        Dictionary with keys:
            - "premises": List[str] of premise sentences
            - "hypotheses": List[str] of hypothesis sentences
            - "label_distributions": np.ndarray of shape (n_examples, 3)
                giving the count of annotators choosing each class
                (entailment=0, neutral=1, contradiction=2)
            - "majority_labels": np.ndarray of shape (n_examples,)
                giving the majority-vote label
            - "example_ids": List[str] of original example identifiers
            - "n_annotators": int, number of annotators per example (100)

    Raises:
        FileNotFoundError: If data_dir is given but the JSONL file is not found.
        RuntimeError: If download from GitHub fails.
        ValueError: If the loaded data appears invalid (no examples, empty text).
    """
    # Step 1: Locate or download the JSONL file
    if data_dir is not None:
        # Look for the file in the provided directory
        filename = CHAOSNLI_FILENAMES.get(subset)
        if filename is None:
            raise ValueError(
                f"Unknown ChaosNLI subset: {subset}. "
                f"Available: {list(CHAOSNLI_FILENAMES.keys())}"
            )
        candidates = [
            os.path.join(data_dir, filename),
            os.path.join(data_dir, f"chaosNLI_v1.0", filename),
            # Also try subset-named files for "mnli" -> "chaosNLI_mnli.jsonl"
            os.path.join(data_dir, f"chaosNLI_{subset}.jsonl"),
            os.path.join(data_dir, f"chaosNLI_v1.0", f"chaosNLI_{subset}.jsonl"),
        ]
        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            norm = os.path.normpath(c)
            if norm not in seen:
                seen.add(norm)
                unique_candidates.append(c)

        jsonl_path = None
        for candidate in unique_candidates:
            if os.path.exists(candidate):
                jsonl_path = candidate
                break
        if jsonl_path is None:
            raise FileNotFoundError(
                f"Could not find ChaosNLI {subset} JSONL file in {data_dir}. "
                f"Looked for: {unique_candidates}"
            )
    else:
        # Download from GitHub (no HuggingFace -- ChaosNLI is not on HF Hub)
        jsonl_path = _download_chaosnli_jsonl(subset)

    # Step 2: Parse the JSONL file
    raw_examples = _parse_chaosnli_jsonl(jsonl_path)
    logger.info(f"Parsed {len(raw_examples)} examples from {jsonl_path}")

    if max_examples is not None:
        raw_examples = raw_examples[:max_examples]

    # Step 3: Extract structured fields
    premises: List[str] = []
    hypotheses: List[str] = []
    label_distributions: List[np.ndarray] = []
    example_ids: List[str] = []

    for raw in raw_examples:
        # Extract UID
        uid = raw.get("uid", raw.get("id", f"unknown_{len(example_ids)}"))
        example_ids.append(str(uid))

        # Extract label distribution from the 100-annotator label_counter
        label_counter = raw.get("label_counter", raw.get("label_dist", {}))
        dist = _extract_label_distribution(label_counter)
        label_distributions.append(dist)

        # Extract premise and hypothesis
        example_data = raw.get("example", {})
        premise, hypothesis = _extract_premise_hypothesis(example_data, raw)
        premises.append(premise)
        hypotheses.append(hypothesis)

    label_distributions_arr = np.stack(label_distributions, axis=0)
    majority_labels = label_distributions_arr.argmax(axis=1)

    logger.info(
        f"Loaded ChaosNLI/{subset}: {len(premises)} examples, "
        f"label distribution shape: {label_distributions_arr.shape}"
    )

    # Step 4: Validate the loaded data
    if len(premises) == 0:
        raise ValueError(
            f"ChaosNLI {subset}: parsed 0 examples from {jsonl_path}. "
            f"The file may be corrupt or empty."
        )

    # Check that we got real text, not empty strings
    non_empty_premises = sum(1 for p in premises if p.strip())
    non_empty_hypotheses = sum(1 for h in hypotheses if h.strip())
    if non_empty_premises < len(premises) * 0.5:
        logger.warning(
            f"ChaosNLI {subset}: only {non_empty_premises}/{len(premises)} "
            f"premises are non-empty. Check the JSONL field parsing."
        )
    if non_empty_hypotheses < len(hypotheses) * 0.5:
        logger.warning(
            f"ChaosNLI {subset}: only {non_empty_hypotheses}/{len(hypotheses)} "
            f"hypotheses are non-empty. Check the JSONL field parsing."
        )

    # Check that label distributions have non-zero sums (100 annotators)
    row_sums = label_distributions_arr.sum(axis=1)
    zero_rows = (row_sums == 0).sum()
    if zero_rows > 0:
        logger.warning(
            f"ChaosNLI {subset}: {zero_rows} examples have all-zero label "
            f"distributions. These may have unparseable label_counter fields."
        )

    return {
        "premises": premises,
        "hypotheses": hypotheses,
        "label_distributions": label_distributions_arr,
        "majority_labels": majority_labels,
        "example_ids": example_ids,
        "n_annotators": 100,
    }


def load_chaosnli_with_entropy(
    subset: str = "snli",
    data_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
    entropy_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Load ChaosNLI and augment with per-example entropy and categories.

    Convenience wrapper that loads the data and computes annotation entropy
    for each example using the 100-annotator label distribution, then
    categorizes into clean/ambiguous/contested.

    Args:
        subset: Which NLI subset ("snli", "mnli", "alphanli").
        data_dir: Path to manually downloaded ChaosNLI data.
        max_examples: Maximum examples to load.
        entropy_thresholds: Thresholds for categorization. Default [0.4, 0.7]
            (paper setting, ACL SRW 2026).

    Returns:
        Dictionary with all keys from load_chaosnli() plus:
            - "entropies": np.ndarray of shape (n_examples,), H_i per example
            - "entropy_categories": np.ndarray of shape (n_examples,), int codes
                (0=clean, 1=ambiguous, 2=contested)
            - "category_names": List[str], ["clean", "ambiguous", "contested"]
            - "categorization": EntropyCategorization object with full details
    """
    data = load_chaosnli(subset=subset, data_dir=data_dir, max_examples=max_examples)

    # Compute per-example entropy from label distributions
    n_examples = data["label_distributions"].shape[0]
    entropies = np.zeros(n_examples, dtype=np.float64)
    for i in range(n_examples):
        entropies[i] = compute_annotation_entropy_from_distribution(
            data["label_distributions"][i]
        )

    # Categorize by entropy
    categorization = categorize_by_entropy(
        entropies, thresholds=entropy_thresholds
    )

    data["entropies"] = entropies
    data["entropy_categories"] = categorization.categories
    data["category_names"] = categorization.category_names
    data["categorization"] = categorization

    logger.info(
        f"Entropy stats: {categorization.counts}, "
        f"mean per category: {categorization.mean_entropy_per_category}"
    )

    return data


def create_chaosnli_torch_dataset(
    data: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 128,
) -> "ChaosNLIDataset":
    """Create a PyTorch Dataset from loaded ChaosNLI data.

    Tokenizes premise-hypothesis pairs and bundles with majority-vote labels,
    example IDs, and annotation entropy values for per-example tracking
    during training.

    Args:
        data: Dictionary returned by load_chaosnli_with_entropy().
        tokenizer: HuggingFace tokenizer for the model.
        max_seq_length: Maximum sequence length for tokenization.

    Returns:
        ChaosNLIDataset instance ready for DataLoader.
    """
    import torch
    from torch.utils.data import Dataset

    # Tokenize all premise-hypothesis pairs
    encodings = tokenizer(
        data["premises"],
        data["hypotheses"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # Prepare labels (majority vote) and metadata
    labels = torch.tensor(data["majority_labels"], dtype=torch.long)

    # Assign sequential integer IDs for tracking (independent of string UIDs)
    example_ids = torch.arange(len(labels), dtype=torch.long)

    # Entropy values as float tensor
    if "entropies" in data:
        entropies = torch.tensor(data["entropies"], dtype=torch.float32)
    else:
        entropies = torch.zeros(len(labels), dtype=torch.float32)

    # Entropy categories as int tensor
    if "entropy_categories" in data:
        entropy_categories = torch.tensor(
            data["entropy_categories"], dtype=torch.long
        )
    else:
        entropy_categories = torch.zeros(len(labels), dtype=torch.long)

    return ChaosNLIDataset(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        labels=labels,
        example_ids=example_ids,
        annotation_entropy=entropies,
        entropy_categories=entropy_categories,
        string_ids=data.get("example_ids", []),
    )


class ChaosNLIDataset:
    """PyTorch-compatible dataset for ChaosNLI with per-example tracking.

    Each item yields a dictionary with:
        - input_ids: tokenized input, shape (max_seq_length,)
        - attention_mask: attention mask, shape (max_seq_length,)
        - labels: majority-vote label, scalar
        - example_id: unique integer ID for tracking, scalar
        - annotation_entropy: H_i value, scalar
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
        string_ids: Optional[List[str]] = None,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.example_ids = example_ids
        self.annotation_entropy = annotation_entropy
        self.entropy_categories = entropy_categories
        self.string_ids = string_ids or []

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


def create_synthetic_chaosnli(
    n_examples: int = 1000,
    n_annotators: int = 100,
    n_classes: int = 3,
    clean_fraction: float = 0.4,
    ambiguous_fraction: float = 0.3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Create synthetic ChaosNLI-like data for development and testing.

    Generates synthetic annotator label distributions with controlled
    entropy levels for the three categories (clean/ambiguous/contested).

    Args:
        n_examples: Total number of examples to generate.
        n_annotators: Number of simulated annotators per example.
        n_classes: Number of label classes.
        clean_fraction: Fraction of examples with low entropy.
        ambiguous_fraction: Fraction with medium entropy.
        seed: Random seed.

    Returns:
        Dictionary matching the load_chaosnli() return format, with
        synthetic text fields ("premise_0", "hypothesis_0", etc.).
    """
    rng = np.random.RandomState(seed)

    n_clean = int(n_examples * clean_fraction)
    n_ambiguous = int(n_examples * ambiguous_fraction)
    n_contested = n_examples - n_clean - n_ambiguous

    label_distributions = np.zeros((n_examples, n_classes), dtype=np.int64)

    # Clean examples: one class gets ~90% of votes
    for i in range(n_clean):
        dominant = rng.randint(n_classes)
        alpha = np.ones(n_classes) * 0.5
        alpha[dominant] = 20.0
        probs = rng.dirichlet(alpha)
        label_distributions[i] = rng.multinomial(n_annotators, probs)

    # Ambiguous examples: two classes split ~60/30
    for i in range(n_clean, n_clean + n_ambiguous):
        classes = rng.choice(n_classes, size=2, replace=False)
        alpha = np.ones(n_classes) * 0.5
        alpha[classes[0]] = 8.0
        alpha[classes[1]] = 4.0
        probs = rng.dirichlet(alpha)
        label_distributions[i] = rng.multinomial(n_annotators, probs)

    # Contested examples: roughly uniform
    for i in range(n_clean + n_ambiguous, n_examples):
        alpha = np.ones(n_classes) * 5.0
        probs = rng.dirichlet(alpha)
        label_distributions[i] = rng.multinomial(n_annotators, probs)

    majority_labels = label_distributions.argmax(axis=1)

    return {
        "premises": [f"premise_{i}" for i in range(n_examples)],
        "hypotheses": [f"hypothesis_{i}" for i in range(n_examples)],
        "label_distributions": label_distributions,
        "majority_labels": majority_labels,
        "example_ids": [f"synth_{i}" for i in range(n_examples)],
        "n_annotators": n_annotators,
    }
