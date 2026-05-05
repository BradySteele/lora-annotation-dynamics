"""
LoRA Model Wrapper
==================
Wraps a HuggingFace model with PEFT LoRA configuration for the
annotation-dynamics temporal separation study (Paper 2).

Key design choice: alpha = 2 * rank so that the effective LoRA scaling
factor (alpha / r) is always 2.0 regardless of rank.  This is critical
for the rank-modulation analysis -- a fixed alpha would make the scaling
factor vary across ranks, confounding the comparison.

MPS compatibility notes:
  - float32 only (no half-precision on Apple Silicon MPS)
  - No pin_memory in DataLoaders when using MPS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device / dtype utilities
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Auto-detect the best available device.

    Priority: CUDA > MPS > CPU.

    Returns:
        torch.device for the selected accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(device: torch.device) -> torch.dtype:
    """Return the appropriate floating-point dtype for *device*.

    Args:
        device: The target device.

    Returns:
        - float32 for MPS or CPU (MPS does not support half-precision).
        - bfloat16 for CUDA if supported, otherwise float16.
    """
    if device.type == "mps":
        return torch.float32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # CPU -- stay in float32 for numerical stability
    return torch.float32


# ---------------------------------------------------------------------------
# LoRA model wrapper
# ---------------------------------------------------------------------------


class LoRAModelWrapper:
    """Lifecycle manager for a LoRA-adapted sequence-classification model.

    Attributes:
        model_name:      HuggingFace model identifier (e.g. ``roberta-base``).
        num_labels:      Number of output classes.
        rank:            LoRA rank *r*.
        alpha:           LoRA scaling numerator, always ``2 * rank``.
        target_modules:  Which attention projections receive LoRA adapters.
        device:          Torch device the model lives on.
        dropout:         Dropout probability inside LoRA layers.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        rank: int,
        target_modules: List[str],
        device: torch.device,
        dropout: float = 0.05,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.rank = rank
        # alpha = 2 * rank  =>  scaling = alpha / r = 2.0 for every rank
        self.alpha = 2 * rank
        self.target_modules = list(target_modules)
        self.device = device
        self.dropout = dropout

        self._peft_model: Optional[PeftModel] = None
        self._is_initialized: bool = False

    # ----- initialisation ---------------------------------------------------

    def initialize(self) -> "LoRAModelWrapper":
        """Create the base model, apply LoRA, and move to device.

        Returns:
            *self* for method chaining (``wrapper.initialize().model``).
        """
        self._peft_model = create_lora_model(
            model_name=self.model_name,
            num_labels=self.num_labels,
            rank=self.rank,
            target_modules=self.target_modules,
            device=self.device,
            dropout=self.dropout,
        )
        self._is_initialized = True

        trainable_params = self.get_trainable_params()
        total = sum(p.numel() for p in self._peft_model.parameters())
        trainable = sum(p.numel() for _, p in trainable_params)
        pct = 100.0 * trainable / total if total > 0 else 0.0
        logger.info(
            "LoRA model initialized: %s | rank=%d | alpha=%d | "
            "trainable=%s / %s (%.2f%%)",
            self.model_name,
            self.rank,
            self.alpha,
            f"{trainable:,}",
            f"{total:,}",
            pct,
        )
        return self

    # ----- properties -------------------------------------------------------

    @property
    def model(self) -> PeftModel:
        """Access the underlying PEFT model.

        Raises:
            RuntimeError: If ``initialize()`` has not been called.
        """
        if not self._is_initialized or self._peft_model is None:
            raise RuntimeError(
                "Model not initialized. Call initialize() first."
            )
        return self._peft_model

    # ----- parameter introspection ------------------------------------------

    def get_trainable_params(self) -> List[Tuple[str, nn.Parameter]]:
        """Return a list of ``(name, param)`` for trainable parameters only.

        This includes LoRA A/B matrices and any other parameters whose
        ``requires_grad`` flag is ``True`` (e.g. the classification head).
        """
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]

    def get_lora_scaling(self) -> float:
        """Return the effective LoRA scaling factor alpha / r.

        Because we set ``alpha = 2 * rank``, this always returns 2.0.
        """
        return self.alpha / self.rank

    # ----- persistence ------------------------------------------------------

    def save(self, save_path: Union[str, Path]) -> None:
        """Save the LoRA adapter weights via PEFT's ``save_pretrained``.

        Args:
            save_path: Directory that will contain the adapter files.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        logger.info("Adapter saved to %s", save_path)

    def load(self, load_path: Union[str, Path]) -> None:
        """Load LoRA adapter weights from a previously saved directory.

        If the model has not been initialized yet, ``initialize()`` is
        called first so that there is a base model to attach the adapter to.

        Args:
            load_path: Path to the adapter directory.
        """
        if not self._is_initialized:
            self.initialize()
        # Re-load from the base model stored inside the PEFT wrapper
        base_model = self.model.get_base_model()
        self._peft_model = PeftModel.from_pretrained(
            base_model, str(load_path)
        )
        self._peft_model.to(self.device)
        logger.info("Adapter loaded from %s", load_path)


# ---------------------------------------------------------------------------
# Standalone factory function
# ---------------------------------------------------------------------------


def create_lora_model(
    model_name: str = "roberta-base",
    num_labels: int = 3,
    rank: int = 8,
    target_modules: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    dropout: float = 0.05,
) -> PeftModel:
    """Create a PEFT LoRA-adapted model for sequence classification.

    The LoRA scaling parameter ``alpha`` is set to ``2 * rank`` so that
    the effective scaling factor is a constant 2.0 across all rank
    settings.  This is a deliberate design choice for the temporal
    separation study -- see module docstring.

    Args:
        model_name:      HuggingFace model identifier.
        num_labels:      Number of output labels (3 for ChaosNLI, variable
                         for GoEmotions).
        rank:            LoRA rank *r*.
        target_modules:  Attention projections that receive LoRA adapters.
                         Defaults to ``["query", "value"]`` for RoBERTa.
        device:          Target device.  Auto-detected if ``None``.
        dropout:         Dropout probability for LoRA layers.

    Returns:
        A ``PeftModel`` wrapping the base transformer, with LoRA applied
        to the specified target modules, moved to *device*.
    """
    if target_modules is None:
        target_modules = ["query", "value"]
    if device is None:
        device = get_device()

    # alpha = 2r  =>  scaling = alpha/r = 2.0 for every rank
    alpha = 2 * rank

    # Load base transformer for sequence classification
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )

    # Build the PEFT LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier"],  # Keep classification head trainable
    )

    # Wrap and move to device
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(device)

    logger.info(
        "Created LoRA model: %s | labels=%d | rank=%d | alpha=%d | "
        "target=%s | device=%s",
        model_name,
        num_labels,
        rank,
        alpha,
        target_modules,
        device,
    )

    return peft_model


# ---------------------------------------------------------------------------
# Standalone parameter helpers (preserved from original stub)
# ---------------------------------------------------------------------------


def get_lora_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the LoRA parameters from a PEFT model.

    Args:
        model: PEFT-wrapped model.

    Returns:
        Dict mapping parameter name to tensor for LoRA parameters only.
    """
    return {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }


def count_lora_params(model: nn.Module) -> int:
    """Count the number of trainable LoRA parameters.

    Args:
        model: PEFT-wrapped model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
