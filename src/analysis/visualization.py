"""
Publication-quality visualization for the LoRA annotation dynamics paper.

Generates figures formatted for ACL SRW (Student Research Workshop) submission:
- Single-column figures: 3.25 inches wide
- Double-column figures: 6.75 inches wide
- Serif font family (Times New Roman / DejaVu Serif)
- 9pt labels, 8pt tick labels
- 300 DPI for raster outputs
- Vector PDF for camera-ready submission

Figure inventory for the paper:
- Figure 1 (HERO): Learning curves stratified by entropy category
    (clean/ambiguous/contested), showing temporal separation
- Figure 1 (ALT): Individual example loss trajectories colored by entropy
- Figure 2: Spearman rho vs. LoRA rank (rank modulation effect)
- Figure 3: Scatter of annotation entropy vs. learning time
- Diagnostic: Annotation entropy distribution histogram

Color scheme uses the Okabe-Ito colorblind-friendly palette.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-friendly palette
# ---------------------------------------------------------------------------
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Entropy category colors (consistent across all figures)
CATEGORY_COLORS = {
    "clean": PALETTE["blue"],
    "ambiguous": PALETTE["orange"],
    "contested": PALETTE["red"],
}


# ---------------------------------------------------------------------------
# ACL style configuration
# ---------------------------------------------------------------------------


def set_acl_style() -> None:
    """Configure matplotlib for ACL publication formatting.

    Sets global rcParams to produce figures that match ACL SRW style
    requirements:
    - Serif font family for text consistency with the paper body
    - 9pt font for axis labels, 8pt for tick labels
    - Minimal chart junk (no top/right spines by default)
    - PDF-compatible output settings

    Call this once at the start of figure generation.  Individual functions
    in this module call it internally, so explicit invocation is optional.
    """
    plt.rcParams.update({
        # Font settings
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # Figure settings
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Axes settings
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,

        # Tick settings
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Line and marker settings
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Legend settings
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.4,

        # PDF backend settings for vector output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    logger.debug("Applied ACL publication style to matplotlib rcParams.")


# ---------------------------------------------------------------------------
# Figure 1 (HERO): Learning curves by entropy category
# ---------------------------------------------------------------------------


def plot_learning_curves_by_entropy(
    loss_trajectories_by_category: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    rank: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Figure 1 (HERO FIGURE): Mean loss trajectories per entropy category.

    Plots mean loss over training epochs for clean, ambiguous, and contested
    examples.  The central prediction of the paper is visible here: clean
    examples (low annotation entropy) have loss curves that drop early,
    while contested examples (high entropy) drop later.

    Each category curve includes a shaded 95% CI band computed as
    mean +/- 1.96 * SEM, where SEM is the standard error of the mean
    across examples in that category.

    Args:
        loss_trajectories_by_category: Dict mapping category name (e.g.,
            "clean", "ambiguous", "contested") to a sub-dict with keys:
                "mean": np.ndarray of shape (n_epochs,) -- mean loss
                "sem": np.ndarray of shape (n_epochs,) -- standard error
                "n": int -- number of examples in the category
            This is the output format of
            learning_order.stratified_learning_curves().
        save_path: If provided, save figure to this path (without extension).
            Both .pdf and .png versions are saved.
        rank: Optional LoRA rank for annotation in the figure title.
        figsize: Figure dimensions (width, height) in inches.
            Default is (3.25, 2.6) for single-column ACL.

    Returns:
        matplotlib Figure object.
    """
    set_acl_style()

    if figsize is None:
        figsize = (3.25, 2.6)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot categories in a fixed order for visual consistency
    category_order = ["clean", "ambiguous", "contested"]
    plotted_categories = [
        c for c in category_order
        if c in loss_trajectories_by_category
    ]
    # Add any categories not in the standard order
    for c in loss_trajectories_by_category:
        if c not in plotted_categories:
            plotted_categories.append(c)

    for cat_name in plotted_categories:
        data = loss_trajectories_by_category[cat_name]
        mean = data["mean"]
        sem = data["sem"]
        n_examples = data.get("n", 0)

        if len(mean) == 0:
            continue

        epochs = np.arange(len(mean))
        color = CATEGORY_COLORS.get(cat_name, PALETTE["black"])

        # Label includes example count for informativeness
        label = f"{cat_name.capitalize()} (n={n_examples})"

        ax.plot(
            epochs,
            mean,
            color=color,
            linewidth=1.5,
            label=label,
            zorder=3,
        )

        # 95% CI band: mean +/- 1.96 * SEM
        ci_lower = mean - 1.96 * sem
        ci_upper = mean + 1.96 * sem
        ax.fill_between(
            epochs,
            ci_lower,
            ci_upper,
            color=color,
            alpha=0.15,
            zorder=2,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean cross-entropy loss")
    ax.legend(loc="upper right")

    # Ensure x-axis starts at 0
    ax.set_xlim(left=0)

    if rank is not None:
        ax.set_title(f"LoRA rank = {rank}")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Alternative Figure 1: Individual example trajectories
# ---------------------------------------------------------------------------


def plot_example_trajectories(
    individual_trajectories: Dict[str, np.ndarray],
    entropies: Dict[str, float],
    save_path: Optional[str] = None,
    n_examples: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Alternative Figure 1: Individual loss curves colored by entropy.

    Shows loss trajectories for individual examples spanning the full
    entropy spectrum.  Examples are selected at the 10th, 20th, ...,
    90th, and 100th percentiles of annotation entropy to show the full
    range of learning dynamics.

    Each curve is colored by its annotation entropy using a continuous
    colormap (coolwarm), providing a gradient view of the temporal
    separation phenomenon.

    Args:
        individual_trajectories: Dict mapping example_id to np.ndarray
            of shape (n_epochs,) with loss values.
        entropies: Dict mapping example_id to annotation entropy H_i.
        save_path: If provided, save figure to this path (without extension).
        n_examples: Number of example trajectories to plot.
            Examples are selected at equally-spaced entropy percentiles.
        figsize: Figure dimensions.  Default (3.25, 2.6).

    Returns:
        matplotlib Figure object.
    """
    set_acl_style()

    if figsize is None:
        figsize = (3.25, 2.6)

    # Find common examples that have both trajectories and entropies
    common_ids = sorted(
        set(individual_trajectories.keys()) & set(entropies.keys())
    )
    if len(common_ids) == 0:
        logger.warning("No common example IDs between trajectories and entropies.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    # Sort by entropy and select at percentile intervals
    sorted_by_entropy = sorted(common_ids, key=lambda eid: entropies[eid])
    n_total = len(sorted_by_entropy)
    n_select = min(n_examples, n_total)

    # Select indices at evenly-spaced percentiles
    percentile_indices = np.linspace(0, n_total - 1, n_select, dtype=int)
    selected_ids = [sorted_by_entropy[i] for i in percentile_indices]

    # Gather entropy values for colormap normalization
    all_entropies = np.array([entropies[eid] for eid in common_ids])
    vmin, vmax = float(all_entropies.min()), float(all_entropies.max())

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.coolwarm

    # Normalize entropy to [0, 1] for colormap
    if vmax > vmin:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    for eid in selected_ids:
        traj = individual_trajectories[eid]
        h = entropies[eid]
        color = cmap(norm(h))
        epochs = np.arange(len(traj))
        ax.plot(
            epochs,
            traj,
            color=color,
            linewidth=1.0,
            alpha=0.8,
            zorder=3,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xlim(left=0)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Annotation entropy $H_i$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Spearman rho vs. LoRA rank
# ---------------------------------------------------------------------------


def plot_spearman_vs_rank(
    rank_results: Dict[int, Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Figure 2: Spearman correlation vs. LoRA rank (rank modulation).

    Line plot with error bars showing how the Spearman correlation
    between annotation entropy and learning time changes with LoRA rank.
    The theory predicts that lower rank produces a stronger positive
    correlation (larger rho) because lambda(r) is larger.

    X-axis is on a log2 scale to match the geometric progression of
    typical LoRA ranks {1, 2, 4, 8, 16, 32}.

    Error bars represent +/- 1 standard deviation across seeds.

    Args:
        rank_results: Dict mapping rank (int) to sub-dict with keys:
            "mean_spearman": float -- mean Spearman rho across seeds
            "std_spearman": float -- std of rho across seeds
            "n_seeds": int -- number of seeds
            This is the output format of
            learning_order.rank_modulation_summary().
        save_path: If provided, save figure to this path (without extension).
        figsize: Figure dimensions.  Default (3.25, 2.6).

    Returns:
        matplotlib Figure object.
    """
    set_acl_style()

    if figsize is None:
        figsize = (3.25, 2.6)

    fig, ax = plt.subplots(figsize=figsize)

    ranks = sorted(rank_results.keys())
    means = [rank_results[r]["mean_spearman"] for r in ranks]
    stds = [rank_results[r]["std_spearman"] for r in ranks]
    n_seeds_list = [rank_results[r].get("n_seeds", 0) for r in ranks]

    # Filter out ranks with no valid data
    valid = [(r, m, s, ns) for r, m, s, ns in zip(ranks, means, stds, n_seeds_list)
             if ns > 0 and np.isfinite(m)]

    if not valid:
        logger.warning("No valid rank results to plot.")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    ranks_v, means_v, stds_v, _ = zip(*valid)
    ranks_v = np.array(ranks_v)
    means_v = np.array(means_v)
    stds_v = np.array(stds_v)

    # Line plot with error bars
    ax.errorbar(
        ranks_v,
        means_v,
        yerr=stds_v,
        fmt="o-",
        color=PALETTE["blue"],
        ecolor=PALETTE["blue"],
        elinewidth=0.8,
        capsize=3,
        capthick=0.8,
        markersize=5,
        linewidth=1.5,
        zorder=3,
    )

    # Individual seed values as faded points (if available)
    for r_val in ranks_v:
        r_int = int(r_val)
        spearman_values = rank_results[r_int].get("spearman_values", [])
        if len(spearman_values) > 1:
            jitter = np.random.RandomState(42).uniform(
                -0.03 * r_val, 0.03 * r_val, len(spearman_values)
            )
            ax.scatter(
                np.full(len(spearman_values), r_val) + jitter,
                spearman_values,
                color=PALETTE["sky"],
                s=12,
                alpha=0.5,
                zorder=2,
                edgecolors="none",
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel(r"Spearman $\rho$(learning time, entropy)")

    # Set x-ticks to the actual rank values
    ax.set_xticks(ranks_v)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Add a horizontal reference line at rho = 0
    ax.axhline(y=0, color=PALETTE["black"], linewidth=0.5, linestyle="--",
               alpha=0.4, zorder=1)

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 3: Entropy vs. learning time scatter
# ---------------------------------------------------------------------------


def plot_entropy_vs_learning_time_scatter(
    learning_times: np.ndarray,
    entropies: np.ndarray,
    save_path: Optional[str] = None,
    rank: Optional[int] = None,
    show_regression: bool = True,
    show_marginals: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Figure 3: Scatter plot of annotation entropy vs. learning time.

    X-axis is annotation entropy H_i, Y-axis is learning time (checkpoint
    index).  Includes a linear regression line with 95% CI band, and
    optional marginal histograms showing the distributions of both
    variables.

    Non-finite values (NaN, Inf) are excluded from the plot.

    Args:
        learning_times: Array of per-example learning times.
        entropies: Array of per-example annotation entropies.
        save_path: If provided, save figure to this path (without extension).
        rank: Optional LoRA rank for annotation in the figure title.
        show_regression: Whether to overlay a linear regression line
            with 95% confidence band.
        show_marginals: Whether to add marginal histograms on the top
            and right sides.
        figsize: Figure dimensions.  Default (3.25, 3.25) for square.

    Returns:
        matplotlib Figure object.
    """
    set_acl_style()

    if figsize is None:
        figsize = (3.25, 3.25) if show_marginals else (3.25, 2.6)

    learning_times = np.asarray(learning_times, dtype=np.float64)
    entropies = np.asarray(entropies, dtype=np.float64)

    # Filter to finite entries
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    lt = learning_times[valid]
    ent = entropies[valid]

    if len(lt) == 0:
        logger.warning("No valid data points for scatter plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    if show_marginals:
        # Create figure with marginal histograms using GridSpec
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            4, 4,
            hspace=0.05,
            wspace=0.05,
        )
        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    else:
        fig, ax_main = plt.subplots(figsize=figsize)
        ax_top = None
        ax_right = None

    # Main scatter
    ax_main.scatter(
        ent,
        lt,
        c=PALETTE["blue"],
        s=8,
        alpha=0.4,
        edgecolors="none",
        zorder=3,
    )

    # Regression line with CI
    if show_regression and len(lt) >= 4:
        _add_regression_line(ax_main, ent, lt)

    ax_main.set_xlabel("Annotation entropy $H_i$")
    ax_main.set_ylabel("Learning time (epoch)")

    if rank is not None:
        title = f"LoRA rank = {rank}"
        if show_marginals:
            ax_top.set_title(title)
        else:
            ax_main.set_title(title)

    # Marginal histograms
    if show_marginals and ax_top is not None and ax_right is not None:
        ax_top.hist(
            ent,
            bins=30,
            color=PALETTE["sky"],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.7,
        )
        ax_top.tick_params(labelbottom=False)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        ax_right.hist(
            lt,
            bins=30,
            orientation="horizontal",
            color=PALETTE["sky"],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.7,
        )
        ax_right.tick_params(labelleft=False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Diagnostic: Annotation entropy distribution
# ---------------------------------------------------------------------------


def plot_entropy_distribution(
    entropies: np.ndarray,
    save_path: Optional[str] = None,
    categories: Optional[Dict[str, np.ndarray]] = None,
    thresholds: Optional[List[float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Diagnostic figure: histogram of annotation entropies.

    Essential for the Phase 0 validation gate -- verifies that the dataset
    has sufficient entropy variation for meaningful temporal separation
    analysis.

    If categories are provided, the histogram is colored by category.
    If thresholds are provided instead, vertical lines are drawn at the
    threshold boundaries with category labels.

    Args:
        entropies: Array of shape (n_examples,) with annotation entropies.
        save_path: If provided, save figure to this path (without extension).
        categories: Optional dict mapping category name to boolean array
            of shape (n_examples,) indicating membership.  If provided,
            bars are stacked/colored by category.
        thresholds: Optional list of threshold values (e.g., [0.4, 0.7])
            used for categorization.  Vertical lines are drawn.
        figsize: Figure dimensions.  Default (3.25, 2.2).

    Returns:
        matplotlib Figure object.
    """
    set_acl_style()

    if figsize is None:
        figsize = (3.25, 2.2)

    entropies = np.asarray(entropies, dtype=np.float64)
    entropies = entropies[np.isfinite(entropies)]

    fig, ax = plt.subplots(figsize=figsize)

    if categories is not None:
        # Stacked histogram by category
        category_order = ["clean", "ambiguous", "contested"]
        ordered_cats = [c for c in category_order if c in categories]
        for c in categories:
            if c not in ordered_cats:
                ordered_cats.append(c)

        hist_data = []
        hist_labels = []
        hist_colors = []
        for cat_name in ordered_cats:
            mask = np.asarray(categories[cat_name], dtype=bool)
            if mask.any():
                hist_data.append(entropies[mask])
                hist_labels.append(f"{cat_name.capitalize()} (n={mask.sum()})")
                hist_colors.append(CATEGORY_COLORS.get(cat_name, PALETTE["black"]))

        if hist_data:
            ax.hist(
                hist_data,
                bins=30,
                stacked=True,
                color=hist_colors,
                label=hist_labels,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.85,
            )
            ax.legend(loc="upper right")
    else:
        ax.hist(
            entropies,
            bins=30,
            color=PALETTE["blue"],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.85,
        )

    # Draw threshold lines
    if thresholds is not None:
        for thresh in thresholds:
            ax.axvline(
                x=thresh,
                color=PALETTE["black"],
                linewidth=0.8,
                linestyle="--",
                alpha=0.6,
                zorder=4,
            )

    ax.set_xlabel("Annotation entropy $H_i$ (nats)")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0)

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _add_regression_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    ci_alpha: float = 0.12,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> None:
    """Add a linear regression line with 95% bootstrap CI band.

    Args:
        ax: Matplotlib axes to draw on.
        x: Predictor values.
        y: Response values.
        ci_alpha: Transparency for the confidence band fill.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.
    """
    # Fit linear regression
    coeffs = np.polyfit(x, y, 1)
    x_smooth = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_smooth)

    ax.plot(
        x_smooth,
        y_fit,
        color=PALETTE["red"],
        linewidth=1.5,
        linestyle="-",
        zorder=4,
    )

    # Bootstrap confidence band
    if len(x) >= 10:
        rng = np.random.RandomState(seed)
        n = len(x)
        predictions = np.zeros((n_bootstrap, len(x_smooth)))

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            coeffs_b = np.polyfit(x[idx], y[idx], 1)
            predictions[b] = np.polyval(coeffs_b, x_smooth)

        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)

        ax.fill_between(
            x_smooth,
            ci_lower,
            ci_upper,
            alpha=ci_alpha,
            color=PALETTE["red"],
            zorder=2,
        )

    # Annotate with Spearman rho
    from scipy import stats
    rho, p = stats.spearmanr(x, y)
    sig_str = ""
    if p < 0.001:
        sig_str = "***"
    elif p < 0.01:
        sig_str = "**"
    elif p < 0.05:
        sig_str = "*"
    annotation = f"$\\rho$ = {rho:.3f}{sig_str}"
    ax.annotate(
        annotation,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="none", alpha=0.8),
    )


def _save_figure(
    fig: plt.Figure,
    save_path: str,
) -> None:
    """Save a figure in both PDF and PNG formats.

    Creates parent directories if they do not exist.  PDF is the primary
    output for paper submission; PNG is a convenience for quick previewing.

    Args:
        fig: matplotlib Figure to save.
        save_path: Base path without extension.  ".pdf" and ".png" are appended.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")

    fig.savefig(str(pdf_path), format="pdf")
    fig.savefig(str(png_path), format="png")

    logger.info("Saved figure to %s and %s", pdf_path, png_path)
