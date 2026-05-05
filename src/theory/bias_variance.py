"""
Bias-Variance Decomposition with Annotation Noise
==================================================
Decomposes the expected error of a LoRA-adapted model into bias, variance,
and a novel interaction term C(r, H) that captures the coupling between
LoRA rank and annotation entropy.

Standard bias-variance decomposition:
    E[error] = bias^2 + variance + noise

Our decomposition (Theorem 2 in the paper):
    E[error] = B(r)^2 + V(r, n) + C(r, H) + epsilon_Bayes

where:
    - B(r)^2: Approximation bias from the rank-r constraint.  Decreases
      monotonically with r.
    - V(r, n): Estimation variance.  Depends on rank r and sample size n.
      For LoRA, V = O(r * d / n) where d is the model dimension.
    - C(r, H): THE NOVEL TERM.  Interaction between rank constraint and
      annotation entropy.  Captures how LoRA's limited capacity interacts
      with label noise from annotator disagreement.
    - epsilon_Bayes: Irreducible Bayes error (independent of model).

The key insight is that C(r, H) is not simply the product of the rank-
dependent bias and the annotation noise.  It has a specific functional
form that depends on how the label noise aligns with LoRA's subspace:

    C(r, H) = (1 - r/d) * H_bar * sigma^2_grad

where H_bar is the mean annotation entropy and sigma^2_grad captures
the variance of gradient directions due to label noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class BiasVarianceDecomposition:
    """Complete bias-variance-interaction decomposition."""

    bias_squared: float
    variance: float
    interaction_C: float  # C(r, H) - the novel term
    bayes_error: float
    total_error: float
    rank: int
    n_samples: int
    mean_entropy: float

    def __repr__(self) -> str:
        return (
            f"BV(bias^2={self.bias_squared:.4f}, var={self.variance:.4f}, "
            f"C={self.interaction_C:.4f}, bayes={self.bayes_error:.4f}, "
            f"total={self.total_error:.4f}, r={self.rank})"
        )


# ---------------------------------------------------------------------------
# Bias term: B(r)^2
# ---------------------------------------------------------------------------


def approximation_bias_squared(
    rank: int,
    d_model: int,
    singular_values: Optional[np.ndarray] = None,
    spectral_decay_rate: float = 1.0,
) -> float:
    """Compute the squared approximation bias from rank-r truncation.

    The bias arises because LoRA can only represent weight updates in a
    rank-r subspace.  If the optimal weight update Delta_W* has singular
    values sigma_1 >= sigma_2 >= ... >= sigma_d, the bias from rank-r
    approximation is:

        B(r)^2 = sum_{j=r+1}^{d} sigma_j^2

    which is the tail energy of the SVD.

    If actual singular values are not available, we model them as
    sigma_j = j^(-alpha) (power-law decay) and compute analytically.

    Args:
        rank: LoRA rank r.
        d_model: Model hidden dimension d.
        singular_values: Optional array of singular values.  If provided,
            compute exact tail energy.
        spectral_decay_rate: Power-law exponent alpha for the modeled
            singular value distribution sigma_j = j^(-alpha).

    Returns:
        B(r)^2: the squared approximation bias.
    """
    if singular_values is not None:
        sv = np.sort(singular_values)[::-1]  # descending
        if rank >= len(sv):
            return 0.0
        return float(np.sum(sv[rank:] ** 2))

    # Model: sigma_j = j^(-alpha)
    # B(r)^2 = sum_{j=r+1}^{d} j^(-2*alpha)
    alpha = spectral_decay_rate
    indices = np.arange(rank + 1, d_model + 1, dtype=np.float64)
    if len(indices) == 0:
        return 0.0
    return float(np.sum(indices ** (-2 * alpha)))


# ---------------------------------------------------------------------------
# Variance term: V(r, n)
# ---------------------------------------------------------------------------


def estimation_variance(
    rank: int,
    d_model: int,
    n_samples: int,
    sigma_noise: float = 1.0,
) -> float:
    """Compute the estimation variance for rank-r LoRA.

    The variance arises from finite training data.  For LoRA with rank r,
    the number of trainable parameters is approximately 2 * r * d
    (from matrices A in R^{r x d} and B in R^{d x r}).

    Following standard learning theory:

        V(r, n) = sigma^2 * (2 * r * d) / n

    where sigma^2 is the noise variance in the labels.

    Args:
        rank: LoRA rank r.
        d_model: Model hidden dimension d.
        n_samples: Number of training examples n.
        sigma_noise: Label noise standard deviation.

    Returns:
        V(r, n): the estimation variance.
    """
    n_params = 2 * rank * d_model  # LoRA parameter count
    return sigma_noise**2 * n_params / n_samples


# ---------------------------------------------------------------------------
# Novel interaction term: C(r, H)
# ---------------------------------------------------------------------------


def interaction_term(
    rank: int,
    d_model: int,
    mean_entropy: float,
    gradient_variance: float = 1.0,
) -> float:
    """Compute the interaction term C(r, H) between rank and annotation entropy.

    THIS IS THE NOVEL THEORETICAL CONTRIBUTION.

    C(r, H) captures the additional error from the interaction between
    LoRA's rank constraint and annotation noise.  It is NOT simply the
    product of bias and noise -- it has a specific geometric structure.

    Derivation sketch (full proof in paper Appendix A):
        1. Label noise from annotator disagreement causes gradient noise
           with variance proportional to H_i.
        2. LoRA's rank-r constraint projects gradients onto a subspace,
           which interacts with the noise structure.
        3. The interaction term measures how much of the noise-induced
           gradient variance falls outside LoRA's subspace.

    Functional form:

        C(r, H) = (1 - r/d) * H_bar * sigma^2_grad

    where:
        - (1 - r/d) is the "missed fraction" of gradient space
        - H_bar is the mean annotation entropy across training examples
        - sigma^2_grad is the variance of gradient directions due to noise

    Note: At full rank (r = d), C = 0 because LoRA can capture all
    gradient directions.  At rank 1, nearly all noise-gradient interaction
    is missed.

    Args:
        rank: LoRA rank r.
        d_model: Model hidden dimension d.
        mean_entropy: H_bar, mean annotation entropy across training data.
        gradient_variance: sigma^2_grad, estimated variance of gradient
            directions due to label noise.

    Returns:
        C(r, H): the interaction term (non-negative).
    """
    if rank >= d_model:
        return 0.0

    missed_fraction = 1.0 - rank / d_model
    return missed_fraction * mean_entropy * gradient_variance


# ---------------------------------------------------------------------------
# Full decomposition
# ---------------------------------------------------------------------------


def decompose_error(
    rank: int,
    n_samples: int,
    entropy: float,
    bayes_error: float,
    d_model: int = 768,
    sigma_noise: float = 1.0,
    gradient_variance: float = 1.0,
    spectral_decay_rate: float = 1.0,
    singular_values: Optional[np.ndarray] = None,
) -> BiasVarianceDecomposition:
    """Full bias-variance-interaction decomposition of expected error.

    Combines all three terms plus Bayes error:

        E[error] = B(r)^2 + V(r, n) + C(r, H) + epsilon_Bayes

    This is the function to call to generate Table 2 and Figure 3 in
    the paper.

    Args:
        rank: LoRA rank r.
        n_samples: Number of training examples.
        entropy: Mean annotation entropy H_bar.
        bayes_error: Irreducible Bayes error epsilon_Bayes.
        d_model: Model hidden dimension.
        sigma_noise: Label noise standard deviation.
        gradient_variance: Variance of gradient directions from noise.
        spectral_decay_rate: Power-law exponent for modeled singular values.
        singular_values: Optional actual singular values (overrides model).

    Returns:
        BiasVarianceDecomposition with all components.
    """
    b2 = approximation_bias_squared(
        rank=rank,
        d_model=d_model,
        singular_values=singular_values,
        spectral_decay_rate=spectral_decay_rate,
    )

    v = estimation_variance(
        rank=rank,
        d_model=d_model,
        n_samples=n_samples,
        sigma_noise=sigma_noise,
    )

    c = interaction_term(
        rank=rank,
        d_model=d_model,
        mean_entropy=entropy,
        gradient_variance=gradient_variance,
    )

    total = b2 + v + c + bayes_error

    return BiasVarianceDecomposition(
        bias_squared=b2,
        variance=v,
        interaction_C=c,
        bayes_error=bayes_error,
        total_error=total,
        rank=rank,
        n_samples=n_samples,
        mean_entropy=entropy,
    )


def decompose_across_ranks(
    ranks: Sequence[int],
    n_samples: int,
    entropy: float,
    bayes_error: float,
    d_model: int = 768,
    sigma_noise: float = 1.0,
    gradient_variance: float = 1.0,
    spectral_decay_rate: float = 1.0,
) -> list[BiasVarianceDecomposition]:
    """Compute decomposition across a range of LoRA ranks.

    This generates the data for the paper's main theoretical figure
    showing how each error component changes with rank, revealing
    the optimal rank r* that minimizes total error.

    Args:
        ranks: Sequence of LoRA ranks to evaluate.
        n_samples: Number of training examples.
        entropy: Mean annotation entropy.
        bayes_error: Irreducible error.
        d_model: Model hidden dimension.
        sigma_noise: Label noise standard deviation.
        gradient_variance: Gradient direction variance from noise.
        spectral_decay_rate: Spectral decay parameter.

    Returns:
        List of BiasVarianceDecomposition, one per rank.
    """
    return [
        decompose_error(
            rank=r,
            n_samples=n_samples,
            entropy=entropy,
            bayes_error=bayes_error,
            d_model=d_model,
            sigma_noise=sigma_noise,
            gradient_variance=gradient_variance,
            spectral_decay_rate=spectral_decay_rate,
        )
        for r in ranks
    ]


def optimal_rank(
    ranks: Sequence[int],
    n_samples: int,
    entropy: float,
    bayes_error: float,
    d_model: int = 768,
    sigma_noise: float = 1.0,
    gradient_variance: float = 1.0,
    spectral_decay_rate: float = 1.0,
) -> Tuple[int, BiasVarianceDecomposition]:
    """Find the rank that minimizes total predicted error.

    The interaction term C(r, H) shifts the optimal rank compared to
    standard bias-variance tradeoff: higher annotation entropy pushes
    the optimum toward higher rank (need more capacity to handle noise).

    Args:
        ranks: Candidate ranks to search over.
        (remaining args as in decompose_error)

    Returns:
        (optimal_r, decomposition_at_optimal_r).
    """
    decompositions = decompose_across_ranks(
        ranks=ranks,
        n_samples=n_samples,
        entropy=entropy,
        bayes_error=bayes_error,
        d_model=d_model,
        sigma_noise=sigma_noise,
        gradient_variance=gradient_variance,
        spectral_decay_rate=spectral_decay_rate,
    )

    best_idx = int(np.argmin([d.total_error for d in decompositions]))
    best_rank = list(ranks)[best_idx]
    return best_rank, decompositions[best_idx]
