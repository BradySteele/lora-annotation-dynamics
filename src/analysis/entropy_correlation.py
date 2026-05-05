"""
Entropy-Learning Time Correlation Analysis
==========================================
Statistical analysis of the correlation between annotation entropy
and per-example learning dynamics.

Provides:
    - Spearman / Kendall-tau-b correlations with proper handling of
      unlearned examples
    - Partial correlation controlling for difficulty proxies
    - Bootstrap confidence intervals for Spearman rho
    - Hierarchical regression testing the incremental contribution
      of annotation entropy beyond standard difficulty proxies
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core correlation analysis
# ---------------------------------------------------------------------------


def correlation_analysis(
    learning_times: np.ndarray,
    entropies: np.ndarray,
    method: str = "spearman",
) -> Dict[str, Any]:
    """Compute rank correlation between learning times and annotation entropies.

    A positive correlation supports the paper's central hypothesis: higher
    annotation entropy (more annotator disagreement) leads to later learning
    under LoRA fine-tuning.

    Both Spearman rho and Kendall tau-b are computed regardless of the
    ``method`` argument, but the primary metric (used for reporting) is
    determined by ``method``.

    Non-finite values (NaN, Inf) in either array are excluded pairwise
    before correlation computation.

    Args:
        learning_times: Array of per-example learning times t_i.  Use
            np.inf or np.nan for examples that were never learned.
        entropies: Array of per-example annotation entropies H_i.
        method: Primary correlation method for reporting.  One of
            "spearman" or "kendall".  Both are always computed.

    Returns:
        Dict with keys:
            "rho": Spearman rank correlation coefficient.
            "p_value": Two-sided p-value for the Spearman test.
            "tau": Kendall tau-b correlation coefficient.
            "tau_p_value": Two-sided p-value for the Kendall test.
            "n": Number of valid (finite) data points used.
            "primary_method": The method name used as primary metric.

    Raises:
        ValueError: If arrays have different lengths or method is unknown.
    """
    learning_times = np.asarray(learning_times, dtype=np.float64)
    entropies = np.asarray(entropies, dtype=np.float64)

    if len(learning_times) != len(entropies):
        raise ValueError(
            f"Array lengths must match: got {len(learning_times)} learning_times "
            f"and {len(entropies)} entropies."
        )

    if method not in ("spearman", "kendall"):
        raise ValueError(
            f"Unknown method '{method}'. Use 'spearman' or 'kendall'."
        )

    # Filter to finite entries only
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    lt = learning_times[valid]
    ent = entropies[valid]
    n = len(lt)

    if n < 3:
        logger.warning(
            "Only %d valid data points; returning zero correlations.", n
        )
        return {
            "rho": 0.0,
            "p_value": 1.0,
            "tau": 0.0,
            "tau_p_value": 1.0,
            "n": n,
            "primary_method": method,
        }

    rho, rho_p = stats.spearmanr(lt, ent)
    tau, tau_p = stats.kendalltau(lt, ent)

    result = {
        "rho": float(rho),
        "p_value": float(rho_p),
        "tau": float(tau),
        "tau_p_value": float(tau_p),
        "n": n,
        "primary_method": method,
    }

    logger.info(
        "Correlation analysis (n=%d): Spearman rho=%.4f (p=%.2e), "
        "Kendall tau=%.4f (p=%.2e)",
        n,
        rho,
        rho_p,
        tau,
        tau_p,
    )

    return result


# ---------------------------------------------------------------------------
# Partial correlation controlling for difficulty proxies
# ---------------------------------------------------------------------------


def partial_correlation(
    learning_times: np.ndarray,
    entropies: np.ndarray,
    difficulty_proxies: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Partial Spearman correlation between learning times and entropy,
    controlling for difficulty proxies.

    This tests whether the entropy-learning time relationship holds
    after accounting for confounds like sentence length, word frequency,
    or label imbalance.  The method is residualization:

        1. Rank-transform learning_times and entropies.
        2. For each variable, regress (OLS) on the rank-transformed
           difficulty proxies and take the residuals.
        3. Compute Pearson correlation of the residuals (which equals
           partial Spearman correlation).

    Args:
        learning_times: Array of shape (n,) with per-example learning times.
        entropies: Array of shape (n,) with per-example annotation entropies.
        difficulty_proxies: Dict mapping proxy name (e.g., "sentence_length",
            "word_frequency") to arrays of shape (n,).  All proxies are used
            simultaneously as control variables.

    Returns:
        Dict with keys:
            "partial_rho": Partial Spearman correlation.
            "p_value": Two-sided p-value (using the t-distribution with
                n - k - 2 degrees of freedom, where k is the number of
                control variables).
            "n": Number of valid data points.
            "n_controls": Number of control variables.
            "control_names": List of control variable names.

    Raises:
        ValueError: If array shapes are incompatible.
    """
    learning_times = np.asarray(learning_times, dtype=np.float64)
    entropies = np.asarray(entropies, dtype=np.float64)
    n = len(learning_times)

    if len(entropies) != n:
        raise ValueError("learning_times and entropies must have the same length.")

    # Stack proxies into a matrix
    proxy_names = sorted(difficulty_proxies.keys())
    proxy_arrays = []
    for name in proxy_names:
        arr = np.asarray(difficulty_proxies[name], dtype=np.float64)
        if len(arr) != n:
            raise ValueError(
                f"Proxy '{name}' has length {len(arr)} but expected {n}."
            )
        proxy_arrays.append(arr)

    # Filter to rows where all values are finite
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    for arr in proxy_arrays:
        valid &= np.isfinite(arr)

    lt = learning_times[valid]
    ent = entropies[valid]
    proxies = np.column_stack([arr[valid] for arr in proxy_arrays]) if proxy_arrays else np.empty((valid.sum(), 0))
    n_valid = len(lt)
    k = len(proxy_names)

    if n_valid < k + 3:
        logger.warning(
            "Insufficient data for partial correlation: n=%d, k=%d controls.",
            n_valid,
            k,
        )
        return {
            "partial_rho": 0.0,
            "p_value": 1.0,
            "n": n_valid,
            "n_controls": k,
            "control_names": proxy_names,
        }

    # Rank-transform for Spearman partial correlation
    lt_ranks = stats.rankdata(lt)
    ent_ranks = stats.rankdata(ent)

    if k == 0:
        # No controls: just compute Spearman directly
        rho, p = stats.pearsonr(lt_ranks, ent_ranks)
        return {
            "partial_rho": float(rho),
            "p_value": float(p),
            "n": n_valid,
            "n_controls": 0,
            "control_names": [],
        }

    # Rank-transform proxies as well
    proxy_ranks = np.column_stack(
        [stats.rankdata(proxies[:, j]) for j in range(k)]
    )

    # Residualize: regress lt_ranks and ent_ranks on proxy_ranks
    # Design matrix with intercept
    Z = np.column_stack([np.ones(n_valid), proxy_ranks])

    lt_resid = _ols_residuals(Z, lt_ranks)
    ent_resid = _ols_residuals(Z, ent_ranks)

    # Partial correlation = Pearson(residuals)
    rho, _ = stats.pearsonr(lt_resid, ent_resid)

    # p-value via t-distribution with df = n - k - 2
    df = n_valid - k - 2
    if df > 0 and abs(rho) < 1.0:
        t_stat = rho * np.sqrt(df / (1.0 - rho**2))
        p_value = 2.0 * stats.t.sf(abs(t_stat), df)
    else:
        p_value = 0.0 if abs(rho) == 1.0 else 1.0

    logger.info(
        "Partial Spearman rho = %.4f (p=%.2e), controlling for %s",
        rho,
        p_value,
        proxy_names,
    )

    return {
        "partial_rho": float(rho),
        "p_value": float(p_value),
        "n": n_valid,
        "n_controls": k,
        "control_names": proxy_names,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for Spearman correlation
# ---------------------------------------------------------------------------


def bootstrap_correlation_ci(
    learning_times: np.ndarray,
    entropies: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap confidence interval for Spearman correlation.

    Uses the percentile method: resample (learning_times, entropies) pairs
    with replacement, compute Spearman rho for each bootstrap sample, and
    report the percentile CI.

    Args:
        learning_times: Array of per-example learning times.
        entropies: Array of per-example annotation entropies.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            "rho": Point estimate of Spearman rho.
            "ci_lower": Lower bound of the confidence interval.
            "ci_upper": Upper bound of the confidence interval.
            "ci_level": The confidence level used.
            "bootstrap_rhos": Array of shape (n_bootstrap,) with bootstrap
                rho values (for diagnostic plotting).
            "n": Number of valid data points.
            "n_bootstrap": Number of bootstrap samples.
    """
    learning_times = np.asarray(learning_times, dtype=np.float64)
    entropies = np.asarray(entropies, dtype=np.float64)

    # Filter to finite entries
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    lt = learning_times[valid]
    ent = entropies[valid]
    n = len(lt)

    if n < 3:
        logger.warning(
            "Only %d valid data points; cannot bootstrap.", n
        )
        return {
            "rho": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_level": ci,
            "bootstrap_rhos": np.array([]),
            "n": n,
            "n_bootstrap": 0,
        }

    # Point estimate
    rho_point, _ = stats.spearmanr(lt, ent)

    # Bootstrap
    rng = np.random.RandomState(seed)
    bootstrap_rhos = np.zeros(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Guard against constant arrays in bootstrap sample
        if np.all(lt[idx] == lt[idx][0]) or np.all(ent[idx] == ent[idx][0]):
            bootstrap_rhos[b] = 0.0
        else:
            r, _ = stats.spearmanr(lt[idx], ent[idx])
            bootstrap_rhos[b] = r

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(bootstrap_rhos, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2)))

    logger.info(
        "Bootstrap CI (%.0f%%): rho=%.4f [%.4f, %.4f] (%d resamples)",
        ci * 100,
        rho_point,
        ci_lower,
        ci_upper,
        n_bootstrap,
    )

    return {
        "rho": float(rho_point),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "bootstrap_rhos": bootstrap_rhos,
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Hierarchical regression: incremental R^2 of entropy beyond difficulty
# ---------------------------------------------------------------------------


def hierarchical_regression(
    learning_times: np.ndarray,
    entropies: np.ndarray,
    difficulty_proxies: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Two-step hierarchical regression testing entropy's incremental contribution.

    Step 1: Regress learning_time on difficulty proxies only.
        learning_time = beta_0 + beta_1*proxy_1 + ... + beta_k*proxy_k + eps
        => R^2_difficulty

    Step 2: Add annotation entropy to the model.
        learning_time = beta_0 + beta_1*proxy_1 + ... + beta_k*proxy_k
                        + beta_{k+1}*entropy + eps
        => R^2_full

    The incremental R^2 = R^2_full - R^2_difficulty quantifies how much
    additional variance in learning time is explained by annotation entropy
    after controlling for standard difficulty measures.

    An F-test for the incremental contribution:

        F = [(RSS_1 - RSS_2) / (df_1 - df_2)] / [RSS_2 / df_2]

    tests whether the entropy coefficient is significantly different from
    zero after controlling for difficulty.

    Args:
        learning_times: Array of shape (n,) with learning times.
        entropies: Array of shape (n,) with annotation entropies.
        difficulty_proxies: Dict mapping proxy name to array of shape (n,).

    Returns:
        Dict with keys:
            "r2_difficulty": R^2 from Step 1 (difficulty only).
            "r2_full": R^2 from Step 2 (difficulty + entropy).
            "r2_incremental": R^2_full - R^2_difficulty.
            "f_statistic": F-test statistic for the incremental R^2.
            "f_p_value": p-value for the F-test.
            "entropy_coefficient": Fitted coefficient for entropy in Step 2.
            "entropy_std_error": Standard error of the entropy coefficient.
            "entropy_t_stat": t-statistic for the entropy coefficient.
            "entropy_p_value": p-value for the entropy coefficient.
            "n": Number of valid data points.
            "n_proxies": Number of difficulty proxies.
            "proxy_names": List of proxy names.

    Raises:
        ValueError: If array shapes are incompatible.
    """
    learning_times = np.asarray(learning_times, dtype=np.float64)
    entropies = np.asarray(entropies, dtype=np.float64)
    n = len(learning_times)

    proxy_names = sorted(difficulty_proxies.keys())
    proxy_arrays = []
    for name in proxy_names:
        arr = np.asarray(difficulty_proxies[name], dtype=np.float64)
        if len(arr) != n:
            raise ValueError(
                f"Proxy '{name}' has length {len(arr)} but expected {n}."
            )
        proxy_arrays.append(arr)

    # Filter to rows where all values are finite
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    for arr in proxy_arrays:
        valid &= np.isfinite(arr)

    y = learning_times[valid]
    ent = entropies[valid]
    n_valid = len(y)
    k = len(proxy_names)

    if n_valid < k + 3:
        logger.warning(
            "Insufficient data for hierarchical regression: n=%d, k=%d.",
            n_valid,
            k,
        )
        return {
            "r2_difficulty": 0.0,
            "r2_full": 0.0,
            "r2_incremental": 0.0,
            "f_statistic": 0.0,
            "f_p_value": 1.0,
            "entropy_coefficient": 0.0,
            "entropy_std_error": float("inf"),
            "entropy_t_stat": 0.0,
            "entropy_p_value": 1.0,
            "n": n_valid,
            "n_proxies": k,
            "proxy_names": proxy_names,
        }

    # Build design matrices
    if proxy_arrays:
        X_proxies = np.column_stack([arr[valid] for arr in proxy_arrays])
    else:
        X_proxies = np.empty((n_valid, 0))

    # Step 1: difficulty only
    X1 = np.column_stack([np.ones(n_valid), X_proxies]) if k > 0 else np.ones((n_valid, 1))
    fit1 = _ols_fit_internal(X1, y)

    # Step 2: difficulty + entropy
    X2 = np.column_stack([X1, ent])
    fit2 = _ols_fit_internal(X2, y)

    r2_difficulty = fit1["r_squared"]
    r2_full = fit2["r_squared"]
    r2_incremental = r2_full - r2_difficulty

    # F-test for incremental contribution
    rss1 = fit1["rss"]
    rss2 = fit2["rss"]
    df1 = n_valid - X1.shape[1]  # residual df for Step 1
    df2 = n_valid - X2.shape[1]  # residual df for Step 2
    df_diff = df1 - df2  # should be 1

    if rss2 > 1e-15 and df2 > 0 and df_diff > 0:
        f_stat = ((rss1 - rss2) / df_diff) / (rss2 / df2)
        f_p_value = float(1.0 - stats.f.cdf(max(f_stat, 0.0), df_diff, df2))
    elif rss2 < 1e-15:
        f_stat = float("inf")
        f_p_value = 0.0
    else:
        f_stat = 0.0
        f_p_value = 1.0

    # Entropy coefficient from Step 2 (last coefficient)
    entropy_coef = float(fit2["params"][-1])
    entropy_se = float(fit2["bse"][-1]) if fit2["bse"][-1] > 0 else float("inf")
    if entropy_se < float("inf") and entropy_se > 1e-15:
        entropy_t = entropy_coef / entropy_se
        entropy_p = float(2.0 * stats.t.sf(abs(entropy_t), df2))
    else:
        entropy_t = 0.0
        entropy_p = 1.0

    logger.info(
        "Hierarchical regression: R2_diff=%.4f, R2_full=%.4f, "
        "delta_R2=%.4f, F=%.2f (p=%.2e)",
        r2_difficulty,
        r2_full,
        r2_incremental,
        f_stat,
        f_p_value,
    )

    return {
        "r2_difficulty": float(r2_difficulty),
        "r2_full": float(r2_full),
        "r2_incremental": float(r2_incremental),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_p_value),
        "entropy_coefficient": entropy_coef,
        "entropy_std_error": entropy_se,
        "entropy_t_stat": float(entropy_t),
        "entropy_p_value": float(entropy_p),
        "n": n_valid,
        "n_proxies": k,
        "proxy_names": proxy_names,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ols_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute OLS residuals: y - X @ (X'X)^{-1} X'y.

    Args:
        X: Design matrix of shape (n, k).
        y: Response vector of shape (n,).

    Returns:
        Residual vector of shape (n,).
    """
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _ols_fit_internal(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Internal OLS fit returning coefficients, R^2, RSS, and standard errors.

    Args:
        X: Design matrix of shape (n, k), must include intercept column.
        y: Response vector of shape (n,).

    Returns:
        Dict with params, r_squared, rss, bse (standard errors).
    """
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))

    r_squared = 1.0 - rss / tss if tss > 1e-15 else 1.0

    # Standard errors
    df_resid = n - k
    if df_resid > 0 and rss > 0:
        mse = rss / df_resid
        try:
            cov_beta = mse * np.linalg.inv(X.T @ X)
            bse = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
        except np.linalg.LinAlgError:
            cov_beta = mse * np.linalg.pinv(X.T @ X)
            bse = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    else:
        bse = np.zeros(k)

    return {
        "params": beta,
        "r_squared": r_squared,
        "rss": rss,
        "bse": bse,
    }
