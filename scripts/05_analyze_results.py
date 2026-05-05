#!/usr/bin/env python3
"""
Phase 4: Full Analysis
======================
Load all tracker files from results/tracking/ and compute the full
analysis suite for the paper:

1. Spearman(learning_time, entropy) per rank per seed -- Table 1
2. Partial Spearman controlling for difficulty proxies
3. Learning order consistency across seeds (Kendall's W) per rank
4. Separation gap statistics (Welch's t-test, Cohen's d) per rank
5. Hierarchical regression: difficulty-only R^2 vs difficulty+entropy R^2

Usage:
    python scripts/05_analyze_results.py
    python scripts/05_analyze_results.py --tracking-dir results/tracking/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.theory.temporal_separation import (
    compute_separation_gap,
    entropy_learning_time_correlation,
)
from src.training.temporal_tracker import TemporalTracker


# --------------------------------------------------------------------------- #
# Tracker loading utilities
# --------------------------------------------------------------------------- #

def discover_trackers(tracking_dir: Path) -> Dict[Tuple[int, int], Path]:
    """Discover all tracker files and parse rank/seed from filenames.

    Supports naming conventions: full_rX_sY.json, sweep_rX_sY.json,
    pilot_rX_sY.json.

    Returns:
        Dictionary mapping (rank, seed) -> tracker file path.
    """
    import re

    trackers = {}
    pattern = re.compile(r"(?:full|sweep|pilot)_r(\d+)_s(\d+)\.json")

    for path in sorted(tracking_dir.glob("*.json")):
        match = pattern.match(path.name)
        if match:
            rank = int(match.group(1))
            seed = int(match.group(2))
            # Prefer full > sweep > pilot if duplicates exist
            key = (rank, seed)
            if key not in trackers or path.name.startswith("full"):
                trackers[key] = path

    return trackers


def load_tracker_with_times(
    path: Path,
    threshold: float = 0.693,
) -> Tuple[TemporalTracker, np.ndarray, np.ndarray, np.ndarray]:
    """Load a tracker and extract learning times and entropies.

    Returns:
        (tracker, example_ids, learning_times, entropies).
        learning_times is np.inf for unlearned examples.
    """
    tracker = TemporalTracker.load(path)

    ids = []
    times = []
    entropies = []

    for eid, record in tracker.records.items():
        t = tracker.get_learning_time(eid, threshold=threshold)
        ids.append(eid)
        times.append(float(t) if t is not None else np.inf)
        entropies.append(
            record.annotation_entropy if record.annotation_entropy is not None else np.nan
        )

    return (
        tracker,
        np.array(ids),
        np.array(times, dtype=np.float64),
        np.array(entropies, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# Analysis 1: Spearman correlation per rank per seed (Table 1)
# --------------------------------------------------------------------------- #

def analysis_spearman_table(
    tracker_map: Dict[Tuple[int, int], Path],
    threshold: float = 0.693,
) -> Dict[str, Any]:
    """Compute Spearman(learning_time, entropy) for every (rank, seed).

    Returns:
        Dictionary with per-run results and per-rank aggregated statistics.
    """
    results = []
    per_rank = {}

    for (rank, seed), path in sorted(tracker_map.items()):
        _, _, times, entropies = load_tracker_with_times(path, threshold)

        valid = np.isfinite(times) & np.isfinite(entropies)
        if valid.sum() < 3:
            rho, p_val = 0.0, 1.0
        else:
            rho, p_val = stats.spearmanr(times[valid], entropies[valid])
            rho, p_val = float(rho), float(p_val)

        n_learned = int(np.isfinite(times).sum())

        results.append({
            "rank": rank,
            "seed": seed,
            "spearman_rho": rho,
            "spearman_p": p_val,
            "n_learned": n_learned,
            "n_total": len(times),
        })

        if rank not in per_rank:
            per_rank[rank] = []
        per_rank[rank].append(rho)

    # Aggregate per rank
    aggregated = {}
    for rank, rhos in sorted(per_rank.items()):
        rhos_arr = np.array(rhos)
        aggregated[rank] = {
            "mean": float(np.mean(rhos_arr)),
            "std": float(np.std(rhos_arr, ddof=1)) if len(rhos_arr) > 1 else 0.0,
            "median": float(np.median(rhos_arr)),
            "n_seeds": len(rhos_arr),
            "all_rhos": [float(r) for r in rhos_arr],
        }

    return {"per_run": results, "per_rank": aggregated}


# --------------------------------------------------------------------------- #
# Analysis 2: Partial Spearman controlling for difficulty
# --------------------------------------------------------------------------- #

def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[float, float]:
    """Compute partial Spearman correlation between x and y controlling for z.

    Uses the standard formula for partial correlation:
        r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    where r_xy, r_xz, r_yz are Spearman correlations.

    Returns:
        (partial_rho, approximate_p_value).
    """
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    n = len(x)

    if n < 5:
        return 0.0, 1.0

    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)

    denom = np.sqrt(max(1e-10, (1 - r_xz**2) * (1 - r_yz**2)))
    partial_r = (r_xy - r_xz * r_yz) / denom

    # Approximate p-value using the t-distribution
    t_stat = partial_r * np.sqrt((n - 3) / max(1e-10, 1 - partial_r**2))
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 3)

    return float(partial_r), float(p_value)


def analysis_partial_correlations(
    tracker_map: Dict[Tuple[int, int], Path],
    data_path: Optional[Path] = None,
    threshold: float = 0.693,
) -> Dict[str, Any]:
    """Partial Spearman controlling for difficulty proxies.

    We want to show that entropy predicts learning time even after
    controlling for sentence length and other difficulty proxies.
    """
    # Load difficulty proxies
    proxy_path = data_path or (PROJECT_ROOT / "results" / "data" / "processed_chaosnli.json")
    if not proxy_path.exists():
        return {"status": "skipped", "reason": f"Processed data not found at {proxy_path}"}

    with open(proxy_path, "r") as f:
        full_data = json.load(f)

    proxies = full_data.get("difficulty_proxies", {})
    all_example_ids = full_data.get("example_ids", [])
    all_entropies = full_data.get("entropies", [])

    if not proxies:
        return {"status": "skipped", "reason": "No difficulty proxies in data"}

    # Build lookup from example_id to proxy values
    id_to_idx = {eid: i for i, eid in enumerate(all_example_ids)}

    results = []

    # Use a representative subset: pick one seed per rank
    representative = {}
    for (rank, seed), path in sorted(tracker_map.items()):
        if rank not in representative:
            representative[rank] = (seed, path)

    for rank, (seed, path) in sorted(representative.items()):
        tracker, ids_arr, times, entropies = load_tracker_with_times(path, threshold)

        for proxy_name, proxy_values in proxies.items():
            proxy_arr = np.array(proxy_values, dtype=np.float64)

            # Align proxy values with tracker examples
            aligned_proxy = np.full(len(ids_arr), np.nan)
            for i, eid in enumerate(ids_arr):
                if eid in id_to_idx:
                    aligned_proxy[i] = proxy_arr[id_to_idx[eid]]

            partial_rho, partial_p = partial_spearman(
                entropies, times, aligned_proxy,
            )

            # Also compute raw Spearman for comparison
            valid = np.isfinite(times) & np.isfinite(entropies)
            raw_rho, raw_p = stats.spearmanr(entropies[valid], times[valid]) if valid.sum() >= 3 else (0.0, 1.0)

            results.append({
                "rank": rank,
                "seed": seed,
                "proxy": proxy_name,
                "raw_spearman": float(raw_rho),
                "raw_p": float(raw_p),
                "partial_spearman": partial_rho,
                "partial_p": partial_p,
                "delta_rho": partial_rho - float(raw_rho),
            })

    return {"status": "completed", "results": results}


# --------------------------------------------------------------------------- #
# Analysis 3: Kendall's W (learning order consistency across seeds)
# --------------------------------------------------------------------------- #

def compute_kendalls_w(rankings: np.ndarray) -> float:
    """Compute Kendall's coefficient of concordance (W).

    Measures the agreement between multiple rankings of the same items.
    W = 1 means perfect agreement; W = 0 means no agreement.

    Args:
        rankings: Array of shape (n_raters, n_items) where each row
            contains the rank of each item according to that rater.

    Returns:
        Kendall's W in [0, 1].
    """
    m, n = rankings.shape  # m = number of raters, n = number of items
    if m < 2 or n < 2:
        return 0.0

    # Column sums of ranks
    R_j = rankings.sum(axis=0)
    R_bar = R_j.mean()

    # Sum of squared deviations
    S = np.sum((R_j - R_bar) ** 2)

    # Kendall's W
    W = 12.0 * S / (m**2 * (n**3 - n))
    return float(np.clip(W, 0.0, 1.0))


def analysis_learning_order_consistency(
    tracker_map: Dict[Tuple[int, int], Path],
    threshold: float = 0.693,
) -> Dict[str, Any]:
    """Compute Kendall's W for learning order consistency across seeds.

    For each rank, we compare the learning order (permutation of examples
    sorted by learning time) across all seeds. High W means the same
    examples are consistently learned first regardless of seed.
    """
    # Group trackers by rank
    by_rank: Dict[int, List[Tuple[int, Path]]] = {}
    for (rank, seed), path in tracker_map.items():
        if rank not in by_rank:
            by_rank[rank] = []
        by_rank[rank].append((seed, path))

    results = {}

    for rank in sorted(by_rank.keys()):
        seed_paths = sorted(by_rank[rank], key=lambda x: x[0])
        if len(seed_paths) < 2:
            results[rank] = {"W": None, "n_seeds": len(seed_paths), "reason": "need >= 2 seeds"}
            continue

        # Load learning times for each seed
        seed_times = {}
        for seed, path in seed_paths:
            _, ids_arr, times, _ = load_tracker_with_times(path, threshold)
            seed_times[seed] = {eid: t for eid, t in zip(ids_arr, times)}

        # Find common examples across all seeds
        common_ids = set(seed_times[seed_paths[0][0]].keys())
        for seed, _ in seed_paths[1:]:
            common_ids &= set(seed_times[seed].keys())

        common_ids = sorted(common_ids)
        if len(common_ids) < 10:
            results[rank] = {
                "W": None, "n_seeds": len(seed_paths), "n_common": len(common_ids),
                "reason": "too few common examples",
            }
            continue

        # Build rankings matrix: (n_seeds, n_examples)
        # For each seed, rank examples by learning time
        n_seeds = len(seed_paths)
        n_items = len(common_ids)
        rankings = np.zeros((n_seeds, n_items))

        for s_idx, (seed, _) in enumerate(seed_paths):
            times_for_seed = np.array([seed_times[seed][eid] for eid in common_ids])
            # Replace inf with a very large number for ranking
            times_for_seed = np.where(np.isinf(times_for_seed), 1e6, times_for_seed)
            rankings[s_idx] = stats.rankdata(times_for_seed)

        W = compute_kendalls_w(rankings)

        # Significance test: chi-squared approximation
        chi2 = n_seeds * (n_items - 1) * W
        df = n_items - 1
        p_value = float(1.0 - stats.chi2.cdf(chi2, df))

        results[rank] = {
            "W": W,
            "chi2": float(chi2),
            "df": df,
            "p_value": p_value,
            "n_seeds": n_seeds,
            "n_common_examples": n_items,
        }

    return results


# --------------------------------------------------------------------------- #
# Analysis 4: Separation gap statistics
# --------------------------------------------------------------------------- #

def analysis_separation_gaps(
    tracker_map: Dict[Tuple[int, int], Path],
    threshold: float = 0.693,
    entropy_thresholds: List[float] = [0.4, 0.7],
) -> Dict[str, Any]:
    """Compute separation gap (Welch's t-test, Cohen's d) per rank.

    Tests whether contested examples (H >= 0.7) are learned significantly
    later than clean examples (H < 0.4) under the paper's binning.
    """
    results = {}

    by_rank: Dict[int, List[Tuple[int, Path]]] = {}
    for (rank, seed), path in tracker_map.items():
        if rank not in by_rank:
            by_rank[rank] = []
        by_rank[rank].append((seed, path))

    for rank in sorted(by_rank.keys()):
        rank_gaps = []

        for seed, path in sorted(by_rank[rank]):
            _, ids_arr, times, entropies = load_tracker_with_times(path, threshold)

            # Categorize
            clean_mask = entropies < entropy_thresholds[0]
            contested_mask = entropies >= entropy_thresholds[1]

            clean_times = times[clean_mask & np.isfinite(times)]
            contested_times = times[contested_mask & np.isfinite(times)]

            sep_result = compute_separation_gap(clean_times, contested_times)

            rank_gaps.append({
                "seed": seed,
                "gap": sep_result.separation_gap,
                "cohens_d": sep_result.effect_size,
                "t_stat": sep_result.t_statistic,
                "p_value": sep_result.p_value,
                "n_clean": sep_result.n_clean,
                "n_contested": sep_result.n_contested,
                "significant": sep_result.significant,
            })

        # Aggregate across seeds
        gaps = [r["gap"] for r in rank_gaps if np.isfinite(r["gap"])]
        ds = [r["cohens_d"] for r in rank_gaps if np.isfinite(r["cohens_d"])]

        results[rank] = {
            "per_seed": rank_gaps,
            "mean_gap": float(np.mean(gaps)) if gaps else None,
            "std_gap": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None,
            "mean_cohens_d": float(np.mean(ds)) if ds else None,
            "std_cohens_d": float(np.std(ds, ddof=1)) if len(ds) > 1 else None,
            "n_significant": sum(1 for r in rank_gaps if r["significant"]),
            "n_total": len(rank_gaps),
        }

    return results


# --------------------------------------------------------------------------- #
# Analysis 5: Hierarchical regression
# --------------------------------------------------------------------------- #

def analysis_hierarchical_regression(
    tracker_map: Dict[Tuple[int, int], Path],
    data_path: Optional[Path] = None,
    threshold: float = 0.693,
) -> Dict[str, Any]:
    """Hierarchical regression: difficulty-only R^2 vs difficulty+entropy R^2.

    Step 1: Regress learning_time ~ difficulty_proxies -> R^2_diff
    Step 2: Regress learning_time ~ difficulty_proxies + entropy -> R^2_full
    Delta R^2 = R^2_full - R^2_diff measures the unique variance explained
    by entropy above and beyond difficulty.
    """
    proxy_path = data_path or (PROJECT_ROOT / "results" / "data" / "processed_chaosnli.json")
    if not proxy_path.exists():
        return {"status": "skipped", "reason": f"Processed data not found at {proxy_path}"}

    with open(proxy_path, "r") as f:
        full_data = json.load(f)

    proxies = full_data.get("difficulty_proxies", {})
    all_example_ids = full_data.get("example_ids", [])

    if not proxies:
        return {"status": "skipped", "reason": "No difficulty proxies in data"}

    id_to_idx = {eid: i for i, eid in enumerate(all_example_ids)}

    # Use one seed per rank (representative)
    representative = {}
    for (rank, seed), path in sorted(tracker_map.items()):
        if rank not in representative:
            representative[rank] = (seed, path)

    results = {}

    for rank, (seed, path) in sorted(representative.items()):
        _, ids_arr, times, entropies = load_tracker_with_times(path, threshold)

        # Build design matrices
        valid_mask = np.isfinite(times) & np.isfinite(entropies)

        proxy_names = sorted(proxies.keys())
        n = len(ids_arr)
        X_diff = np.full((n, len(proxy_names)), np.nan)
        for p_idx, pname in enumerate(proxy_names):
            proxy_arr = np.array(proxies[pname], dtype=np.float64)
            for i, eid in enumerate(ids_arr):
                if eid in id_to_idx:
                    X_diff[i, p_idx] = proxy_arr[id_to_idx[eid]]

        proxy_valid = np.all(np.isfinite(X_diff), axis=1)
        mask = valid_mask & proxy_valid
        if mask.sum() < 10:
            results[rank] = {"status": "insufficient_data", "n_valid": int(mask.sum())}
            continue

        y = times[mask]
        X_d = X_diff[mask]
        X_e = entropies[mask].reshape(-1, 1)

        # Step 1: difficulty only
        X_d_with_intercept = np.column_stack([np.ones(len(y)), X_d])
        try:
            beta_d, res_d, _, _ = np.linalg.lstsq(X_d_with_intercept, y, rcond=None)
            y_hat_d = X_d_with_intercept @ beta_d
            ss_res_d = np.sum((y - y_hat_d) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_diff = 1.0 - ss_res_d / ss_tot if ss_tot > 0 else 0.0
        except np.linalg.LinAlgError:
            r2_diff = 0.0

        # Step 2: difficulty + entropy
        X_full = np.column_stack([np.ones(len(y)), X_d, X_e])
        try:
            beta_f, res_f, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
            y_hat_f = X_full @ beta_f
            ss_res_f = np.sum((y - y_hat_f) ** 2)
            r2_full = 1.0 - ss_res_f / ss_tot if ss_tot > 0 else 0.0
        except np.linalg.LinAlgError:
            r2_full = r2_diff

        delta_r2 = r2_full - r2_diff

        # F-test for delta R^2 significance
        n_obs = len(y)
        df1 = 1  # entropy adds 1 predictor
        df2 = n_obs - X_full.shape[1]
        if df2 > 0 and (1 - r2_full) > 0:
            f_stat = (delta_r2 / df1) / ((1 - r2_full) / df2)
            f_p = float(1.0 - stats.f.cdf(f_stat, df1, df2))
        else:
            f_stat = 0.0
            f_p = 1.0

        results[rank] = {
            "r2_difficulty_only": float(r2_diff),
            "r2_difficulty_plus_entropy": float(r2_full),
            "delta_r2": float(delta_r2),
            "f_statistic": float(f_stat),
            "f_p_value": f_p,
            "n_observations": n_obs,
            "seed": seed,
            "proxy_names": proxy_names,
            "status": "completed",
        }

    return {"status": "completed", "per_rank": results}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Full analysis of learning dynamics."
    )
    parser.add_argument(
        "--tracking-dir", type=str, default=None,
        help="Directory containing tracker JSON files.",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to processed data JSON (for difficulty proxies).",
    )
    parser.add_argument(
        "--loss-threshold", type=float, default=0.693,
        help="Loss threshold for learning time computation.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for analysis results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    tracking_dir = Path(args.tracking_dir) if args.tracking_dir else PROJECT_ROOT / "results" / "tracking"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "analysis"
    data_path = Path(args.data_path) if args.data_path else None
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 4: Full Analysis")
    print("=" * 70)
    print(f"  Tracking dir: {tracking_dir}")
    print(f"  Output dir:   {output_dir}")
    print()

    # Discover tracker files
    tracker_map = discover_trackers(tracking_dir)
    ranks = sorted(set(r for r, _ in tracker_map.keys()))
    seeds = sorted(set(s for _, s in tracker_map.keys()))

    print(f"  Found {len(tracker_map)} tracker files")
    print(f"  Ranks: {ranks}")
    print(f"  Seeds: {seeds}")

    if len(tracker_map) == 0:
        print("\n  ERROR: No tracker files found. Run Phases 1-3 first.")
        return

    # ------------------------------------------------------------------ #
    # Analysis 1: Spearman correlation table
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Analysis 1: Spearman(learning_time, entropy) -- Table 1")
    print(f"{'=' * 60}")

    spearman_results = analysis_spearman_table(tracker_map, args.loss_threshold)

    print(f"\n{'Rank':>6} {'mean rho':>10} {'std':>8} {'n_seeds':>8}")
    print("-" * 40)
    for rank, agg in sorted(spearman_results["per_rank"].items(), key=lambda x: int(x[0])):
        print(f"{rank:>6} {agg['mean']:>10.4f} {agg['std']:>8.4f} {agg['n_seeds']:>8}")

    # ------------------------------------------------------------------ #
    # Analysis 2: Partial correlations
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Analysis 2: Partial Spearman (controlling for difficulty)")
    print(f"{'=' * 60}")

    partial_results = analysis_partial_correlations(
        tracker_map, data_path, args.loss_threshold,
    )

    if partial_results.get("status") == "completed":
        print(f"\n{'Rank':>6} {'Proxy':>25} {'Raw rho':>10} {'Partial rho':>12} {'Delta':>8}")
        print("-" * 65)
        for r in partial_results["results"]:
            print(f"{r['rank']:>6} {r['proxy']:>25} {r['raw_spearman']:>10.4f} "
                  f"{r['partial_spearman']:>12.4f} {r['delta_rho']:>8.4f}")
    else:
        print(f"  {partial_results.get('reason', 'Unknown reason')}")

    # ------------------------------------------------------------------ #
    # Analysis 3: Kendall's W
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Analysis 3: Learning Order Consistency (Kendall's W)")
    print(f"{'=' * 60}")

    consistency_results = analysis_learning_order_consistency(
        tracker_map, args.loss_threshold,
    )

    print(f"\n{'Rank':>6} {'W':>8} {'chi2':>10} {'p-value':>12} {'n_seeds':>8}")
    print("-" * 50)
    for rank, res in sorted(consistency_results.items()):
        if res.get("W") is not None:
            print(f"{rank:>6} {res['W']:>8.4f} {res['chi2']:>10.2f} "
                  f"{res['p_value']:>12.2e} {res['n_seeds']:>8}")
        else:
            print(f"{rank:>6} {'N/A':>8} {res.get('reason', ''):>30}")

    # ------------------------------------------------------------------ #
    # Analysis 4: Separation gap statistics
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Analysis 4: Temporal Separation Gap (Welch's t-test)")
    print(f"{'=' * 60}")

    gap_results = analysis_separation_gaps(tracker_map, args.loss_threshold)

    print(f"\n{'Rank':>6} {'mean gap':>10} {'mean d':>10} {'sig/total':>10}")
    print("-" * 40)
    for rank, res in sorted(gap_results.items()):
        gap_str = f"{res['mean_gap']:.3f}" if res['mean_gap'] is not None else "N/A"
        d_str = f"{res['mean_cohens_d']:.3f}" if res['mean_cohens_d'] is not None else "N/A"
        print(f"{rank:>6} {gap_str:>10} {d_str:>10} {res['n_significant']}/{res['n_total']:>5}")

    # ------------------------------------------------------------------ #
    # Analysis 5: Hierarchical regression
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print("Analysis 5: Hierarchical Regression (Delta R^2)")
    print(f"{'=' * 60}")

    regression_results = analysis_hierarchical_regression(
        tracker_map, data_path, args.loss_threshold,
    )

    if regression_results.get("status") == "completed":
        print(f"\n{'Rank':>6} {'R2_diff':>10} {'R2_full':>10} {'Delta R2':>10} {'F-test p':>12}")
        print("-" * 55)
        for rank, res in sorted(regression_results.get("per_rank", {}).items(), key=lambda x: int(x[0])):
            if res.get("status") == "completed":
                print(f"{rank:>6} {res['r2_difficulty_only']:>10.4f} "
                      f"{res['r2_difficulty_plus_entropy']:>10.4f} "
                      f"{res['delta_r2']:>10.4f} {res['f_p_value']:>12.2e}")
            else:
                print(f"{rank:>6} {'N/A':>10} (insufficient data)")
    else:
        print(f"  {regression_results.get('reason', 'Unknown reason')}")

    # ------------------------------------------------------------------ #
    # Save all results
    # ------------------------------------------------------------------ #
    full_analysis = {
        "spearman_table": spearman_results,
        "partial_correlations": partial_results,
        "learning_order_consistency": {str(k): v for k, v in consistency_results.items()},
        "separation_gaps": {str(k): v for k, v in gap_results.items()},
        "hierarchical_regression": regression_results,
        "metadata": {
            "tracking_dir": str(tracking_dir),
            "loss_threshold": args.loss_threshold,
            "n_trackers": len(tracker_map),
            "ranks": ranks,
            "seeds": seeds,
        },
    }

    analysis_path = output_dir / "full_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(full_analysis, f, indent=2, default=str)
    print(f"\nSaved full analysis to {analysis_path}")

    # Save a concise summary for quick reference
    summary = {
        "spearman_per_rank": spearman_results["per_rank"],
        "separation_gap_per_rank": {
            str(k): {"mean_gap": v["mean_gap"], "mean_d": v["mean_cohens_d"]}
            for k, v in gap_results.items()
        },
        "consistency_per_rank": {
            str(k): v.get("W") for k, v in consistency_results.items()
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {summary_path}")

    elapsed = time.time() - t0
    print(f"\nPhase 4 complete ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
