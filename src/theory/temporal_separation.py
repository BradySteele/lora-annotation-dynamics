"""
Temporal Separation Theory
==========================
Core theoretical contribution of the paper: predicting learning order
from annotation entropy under LoRA's rank constraint.

Central claim (Theorem 1):
    For a LoRA adapter of rank r fine-tuning a pre-trained model on data
    with per-example annotation entropy H_i, the expected learning time
    of example i is:

        E[t_i] proportional to (1 / ||P_r grad_i||) * (1 + lambda * H_i)

    where:
        - P_r is the rank-r projection operator (determined by LoRA's
          low-rank parameterization B @ A)
        - grad_i is the gradient of the loss on example i at initialization
        - H_i is the annotation entropy of example i
        - lambda >= 0 is a coupling constant that depends on rank r

    Intuition: LoRA can only move in a rank-r subspace.  Clean examples
    (low H_i) have gradients that align better with the dominant singular
    directions, so they are learned first.  Contested examples (high H_i)
    have gradient directions that are more "spread out" across singular
    directions, requiring more training to capture.

    The lambda parameter captures how strongly the rank constraint couples
    with annotation noise.  At full rank (r = d), lambda -> 0 and all
    examples are learned at similar rates.  At low rank, lambda is large
    and the temporal separation between clean and contested is pronounced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core prediction: learning time from gradient norm and entropy
# ---------------------------------------------------------------------------


def predict_learning_time(
    gradient_norm: float,
    entropy: float,
    rank: int,
    lambda_param: float,
    learning_rate: float = 1.0,
) -> float:
    """Predict the learning time of an example from theory.

    Implements the main theoretical prediction:

        t_i = (1 / (eta * ||P_r grad_i||)) * (1 + lambda * H_i)

    where eta is the learning rate.

    The term (1 / ||P_r grad_i||) captures how well the example's gradient
    aligns with LoRA's rank-r subspace.  The term (1 + lambda * H_i)
    captures the slowdown from annotation noise: contested examples have
    "noisier" gradient directions that require more updates.

    Args:
        gradient_norm: ||P_r grad_i||, the norm of the example's gradient
            projected onto LoRA's rank-r subspace.  Must be > 0.
        entropy: H_i, the annotation entropy of the example (in nats).
        rank: The LoRA rank r (used for computing lambda if not provided
            directly, but here lambda_param is given).
        lambda_param: The coupling constant lambda(r) >= 0.
        learning_rate: The optimizer learning rate eta.

    Returns:
        Predicted learning time t_i (in units of epochs or gradient steps,
        depending on the normalization convention).

    Raises:
        ValueError: If gradient_norm <= 0.
    """
    if gradient_norm <= 0:
        raise ValueError(
            f"gradient_norm must be positive, got {gradient_norm}. "
            "This typically means the example's gradient is orthogonal "
            "to LoRA's subspace."
        )

    alignment_factor = 1.0 / (learning_rate * gradient_norm)
    noise_factor = 1.0 + lambda_param * entropy
    return alignment_factor * noise_factor


def predict_learning_times_batch(
    gradient_norms: Union[Sequence[float], np.ndarray],
    entropies: Union[Sequence[float], np.ndarray],
    rank: int,
    lambda_param: float,
    learning_rate: float = 1.0,
) -> np.ndarray:
    """Vectorized batch prediction of learning times.

    Args:
        gradient_norms: Array of shape (n,) with ||P_r grad_i|| per example.
        entropies: Array of shape (n,) with H_i per example.
        rank: LoRA rank.
        lambda_param: Coupling constant.
        learning_rate: Optimizer learning rate.

    Returns:
        Array of shape (n,) with predicted learning times.
    """
    g = np.asarray(gradient_norms, dtype=np.float64)
    h = np.asarray(entropies, dtype=np.float64)

    # Handle zero gradient norms by assigning infinity (never learned)
    with np.errstate(divide="ignore"):
        alignment = np.where(g > 0, 1.0 / (learning_rate * g), np.inf)

    noise = 1.0 + lambda_param * h
    return alignment * noise


# ---------------------------------------------------------------------------
# Lambda estimation: coupling between rank and annotation noise
# ---------------------------------------------------------------------------


def estimate_lambda(
    rank: int,
    d_model: int,
    spectral_decay_rate: float = 1.0,
) -> float:
    """Estimate the coupling constant lambda(r) from model properties.

    The coupling constant captures how strongly the rank constraint
    amplifies the effect of annotation noise on learning time.

    Theoretical form (derived in Section 3.2 of the paper):

        lambda(r) = (d / r - 1) * gamma

    where:
        - d is the model dimension (hidden size)
        - r is the LoRA rank
        - gamma is a spectral decay parameter that depends on how quickly
          the singular values of the weight update matrix decay

    At full rank (r = d), lambda = 0: no coupling between rank and noise.
    At very low rank (r << d), lambda ~ d*gamma/r: strong coupling.

    Args:
        rank: LoRA rank r.
        d_model: Model hidden dimension d.
        spectral_decay_rate: Spectral decay parameter gamma > 0.

    Returns:
        Estimated lambda(r).
    """
    if rank >= d_model:
        return 0.0
    return (d_model / rank - 1.0) * spectral_decay_rate


def fit_lambda_from_data(
    observed_learning_times: np.ndarray,
    gradient_norms: np.ndarray,
    entropies: np.ndarray,
    learning_rate: float = 1.0,
) -> Tuple[float, float]:
    """Fit lambda from observed learning times using least-squares.

    Given observed t_i, ||P_r grad_i||, and H_i, find lambda that
    minimizes the squared error between predicted and observed learning
    times:

        min_lambda  sum_i (t_i - t_i_hat(lambda))^2

    Since t_i_hat = (1/(eta*g_i)) * (1 + lambda*H_i), this is linear
    in lambda and can be solved in closed form.

    Args:
        observed_learning_times: Array of observed t_i.
        gradient_norms: Array of ||P_r grad_i||.
        entropies: Array of H_i.
        learning_rate: The learning rate used in training.

    Returns:
        (lambda_hat, r_squared): Fitted lambda and R^2 goodness of fit.
    """
    g = np.asarray(gradient_norms, dtype=np.float64)
    h = np.asarray(entropies, dtype=np.float64)
    t = np.asarray(observed_learning_times, dtype=np.float64)

    # Filter out examples with zero gradient norm or inf learning time
    valid = np.isfinite(t) & (g > 0)
    g, h, t = g[valid], h[valid], t[valid]

    if len(t) < 2:
        return 0.0, 0.0

    # Rewrite: t_i * eta * g_i = 1 + lambda * H_i
    # => y_i = 1 + lambda * x_i
    # where y_i = t_i * eta * g_i and x_i = H_i
    y = t * learning_rate * g
    x = h

    # OLS: y = beta_0 + beta_1 * x
    # beta_0 should be ~1, beta_1 = lambda
    X = np.column_stack([np.ones_like(x), x])
    beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)

    lambda_hat = max(0.0, beta[1])  # lambda should be non-negative

    # R^2
    ss_res = np.sum((y - X @ beta) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return lambda_hat, r_squared


# ---------------------------------------------------------------------------
# Temporal separation gap: statistical test of the main hypothesis
# ---------------------------------------------------------------------------


@dataclass
class SeparationTestResult:
    """Result of testing for temporal separation between clean and contested."""

    mean_clean_time: float
    mean_contested_time: float
    separation_gap: float  # mean_contested - mean_clean
    effect_size: float  # Cohen's d
    t_statistic: float
    p_value: float
    n_clean: int
    n_contested: int
    significant: bool  # at alpha=0.05

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "n.s."
        return (
            f"SeparationTest(gap={self.separation_gap:.3f}, "
            f"d={self.effect_size:.3f}, p={self.p_value:.4f} {sig}, "
            f"n_clean={self.n_clean}, n_contested={self.n_contested})"
        )


def compute_separation_gap(
    clean_times: Union[Sequence[float], np.ndarray],
    contested_times: Union[Sequence[float], np.ndarray],
    alpha: float = 0.05,
) -> SeparationTestResult:
    """Test whether contested examples are learned significantly later than clean ones.

    Performs a one-sided Welch's t-test:
        H0: mean(t_contested) <= mean(t_clean)
        H1: mean(t_contested) >  mean(t_clean)

    Also computes Cohen's d effect size.

    This is the primary statistical test for the paper's central claim.

    Args:
        clean_times: Learning times of clean examples (low H_i).
        contested_times: Learning times of contested examples (high H_i).
        alpha: Significance level.

    Returns:
        SeparationTestResult with test statistics and effect size.
    """
    clean = np.asarray(clean_times, dtype=np.float64)
    contested = np.asarray(contested_times, dtype=np.float64)

    # Remove NaN / inf (unlearned examples)
    clean = clean[np.isfinite(clean)]
    contested = contested[np.isfinite(contested)]

    if len(clean) < 2 or len(contested) < 2:
        return SeparationTestResult(
            mean_clean_time=float(np.mean(clean)) if len(clean) > 0 else float("nan"),
            mean_contested_time=float(np.mean(contested)) if len(contested) > 0 else float("nan"),
            separation_gap=float("nan"),
            effect_size=float("nan"),
            t_statistic=float("nan"),
            p_value=1.0,
            n_clean=len(clean),
            n_contested=len(contested),
            significant=False,
        )

    mean_clean = float(np.mean(clean))
    mean_contested = float(np.mean(contested))
    gap = mean_contested - mean_clean

    # Cohen's d with pooled standard deviation
    s_clean = np.std(clean, ddof=1)
    s_contested = np.std(contested, ddof=1)
    pooled_std = np.sqrt(
        ((len(clean) - 1) * s_clean**2 + (len(contested) - 1) * s_contested**2)
        / (len(clean) + len(contested) - 2)
    )
    cohens_d = gap / pooled_std if pooled_std > 0 else 0.0

    # One-sided Welch's t-test
    t_stat, p_two_sided = stats.ttest_ind(contested, clean, equal_var=False)
    # Convert to one-sided: we test contested > clean
    p_one_sided = p_two_sided / 2.0 if t_stat > 0 else 1.0 - p_two_sided / 2.0

    return SeparationTestResult(
        mean_clean_time=mean_clean,
        mean_contested_time=mean_contested,
        separation_gap=gap,
        effect_size=cohens_d,
        t_statistic=float(t_stat),
        p_value=float(p_one_sided),
        n_clean=len(clean),
        n_contested=len(contested),
        significant=p_one_sided < alpha,
    )


# ---------------------------------------------------------------------------
# Rank-dependent separation: how gap changes with LoRA rank
# ---------------------------------------------------------------------------


def separation_gap_vs_rank(
    ranks: Sequence[int],
    clean_times_per_rank: Dict[int, np.ndarray],
    contested_times_per_rank: Dict[int, np.ndarray],
) -> List[Tuple[int, SeparationTestResult]]:
    """Compute temporal separation gap across multiple LoRA ranks.

    The theory predicts that the gap decreases as rank increases
    (because lambda(r) decreases), eventually vanishing at full rank.

    Args:
        ranks: List of LoRA ranks to analyze.
        clean_times_per_rank: Dict mapping rank -> array of clean learning times.
        contested_times_per_rank: Dict mapping rank -> array of contested times.

    Returns:
        List of (rank, SeparationTestResult) tuples, sorted by rank.
    """
    from typing import Dict

    results = []
    for r in sorted(ranks):
        clean = clean_times_per_rank.get(r, np.array([]))
        contested = contested_times_per_rank.get(r, np.array([]))
        result = compute_separation_gap(clean, contested)
        results.append((r, result))
    return results


# ---------------------------------------------------------------------------
# Correlation analysis: entropy vs. learning time
# ---------------------------------------------------------------------------


def entropy_learning_time_correlation(
    entropies: np.ndarray,
    learning_times: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute rank correlation between annotation entropy and learning time.

    Positive correlation supports the hypothesis: higher entropy => later
    learning.

    Args:
        entropies: Per-example annotation entropy H_i.
        learning_times: Per-example learning time t_i.
        method: "spearman" or "kendall".

    Returns:
        (correlation, p_value).
    """
    # Filter out non-finite values
    valid = np.isfinite(entropies) & np.isfinite(learning_times)
    h = entropies[valid]
    t = learning_times[valid]

    if len(h) < 3:
        return 0.0, 1.0

    if method == "spearman":
        corr, p = stats.spearmanr(h, t)
    elif method == "kendall":
        corr, p = stats.kendalltau(h, t)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'kendall'.")

    return float(corr), float(p)
