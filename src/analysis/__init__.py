"""
Analysis Module
===============
Statistical analysis and publication-quality visualization for the
LoRA annotation dynamics paper.

Submodules:
    learning_order: Learning time extraction, consistency analysis,
        stratified curves, rank modulation summary.
    entropy_correlation: Spearman/Kendall correlation, partial correlation,
        bootstrap CIs, hierarchical regression.
    visualization: ACL SRW-formatted figures (Okabe-Ito palette).
"""

from src.analysis.entropy_correlation import (
    bootstrap_correlation_ci,
    correlation_analysis,
    hierarchical_regression,
    partial_correlation,
)
from src.analysis.learning_order import (
    compute_learning_order_consistency,
    compute_learning_times,
    rank_modulation_summary,
    stratified_learning_curves,
)
from src.analysis.visualization import (
    CATEGORY_COLORS,
    PALETTE,
    plot_entropy_distribution,
    plot_entropy_vs_learning_time_scatter,
    plot_example_trajectories,
    plot_learning_curves_by_entropy,
    plot_spearman_vs_rank,
    set_acl_style,
)

__all__ = [
    # learning_order
    "compute_learning_times",
    "compute_learning_order_consistency",
    "stratified_learning_curves",
    "rank_modulation_summary",
    # entropy_correlation
    "correlation_analysis",
    "partial_correlation",
    "bootstrap_correlation_ci",
    "hierarchical_regression",
    # visualization
    "set_acl_style",
    "plot_learning_curves_by_entropy",
    "plot_example_trajectories",
    "plot_spearman_vs_rank",
    "plot_entropy_vs_learning_time_scatter",
    "plot_entropy_distribution",
    "PALETTE",
    "CATEGORY_COLORS",
]
