#!/usr/bin/env python3
"""
Per-Label-Class AULC-Entropy Decomposition
===========================================
Analyzes existing tracking data to decompose the AULC-entropy correlation
by gold label class (entailment=0, neutral=1, contradiction=2).

This is a pure analysis script -- NO training required.

Usage:
    python scripts/14_per_label_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.temporal_tracker import TemporalTracker

TRACKING_DIR = PROJECT_ROOT / "results" / "tracking"
EXPANDED_DIR = TRACKING_DIR / "expanded"
OUTPUT_DIR = TRACKING_DIR / "per_label_analysis"

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def compute_aulc_by_label(tracker: TemporalTracker) -> Dict[str, Dict]:
    """Compute AULC-entropy correlation broken down by gold label."""
    label_groups = {}
    for eid, record in tracker.records.items():
        if record.annotation_entropy is None:
            continue
        valid_losses = [l for l in record.losses
                        if not (isinstance(l, float) and np.isnan(l))]
        if len(valid_losses) < 2:
            continue
        aulc = float(np.mean(valid_losses))
        label = record.true_label
        if label not in label_groups:
            label_groups[label] = {"aulcs": [], "entropies": []}
        label_groups[label]["aulcs"].append(aulc)
        label_groups[label]["entropies"].append(record.annotation_entropy)

    results = {}
    for label, data in sorted(label_groups.items()):
        aulcs = np.array(data["aulcs"])
        entropies = np.array(data["entropies"])
        valid = np.isfinite(aulcs) & np.isfinite(entropies)
        if valid.sum() >= 3:
            rho, p = stats.spearmanr(aulcs[valid], entropies[valid])
        else:
            rho, p = 0.0, 1.0
        label_name = LABEL_NAMES.get(label, f"label_{label}")
        results[label_name] = {
            "label_id": int(label),
            "n_examples": int(valid.sum()),
            "spearman_rho": float(rho),
            "p_value": float(p),
            "mean_aulc": float(np.mean(aulcs[valid])),
            "mean_entropy": float(np.mean(entropies[valid])),
        }

    # Also compute overall for reference
    all_aulcs = []
    all_entropies = []
    for data in label_groups.values():
        all_aulcs.extend(data["aulcs"])
        all_entropies.extend(data["entropies"])
    all_aulcs = np.array(all_aulcs)
    all_entropies = np.array(all_entropies)
    valid = np.isfinite(all_aulcs) & np.isfinite(all_entropies)
    if valid.sum() >= 3:
        rho_all, p_all = stats.spearmanr(all_aulcs[valid], all_entropies[valid])
    else:
        rho_all, p_all = 0.0, 1.0
    results["overall"] = {
        "n_examples": int(valid.sum()),
        "spearman_rho": float(rho_all),
        "p_value": float(p_all),
    }

    return results


def find_all_trackers() -> List[Dict]:
    """Find all existing tracker files across the project."""
    trackers = []

    # Main tracking directory (RoBERTa-SNLI sweep + pilot)
    for pattern in ["sweep_r*_s*.json", "pilot_r*_s*.json", "fullft_s*.json"]:
        for p in sorted(TRACKING_DIR.glob(pattern)):
            if "results" in p.name or "summary" in p.name:
                continue
            parts = p.stem.split("_")
            model = "roberta-base"
            dataset = "snli"
            # Parse config and seed from filename
            config = "unknown"
            seed = "unknown"
            for part in parts:
                if part.startswith("r") and part[1:].isdigit():
                    config = part
                elif part.startswith("s") and part[1:].isdigit():
                    seed = int(part[1:])
                elif part == "fullft":
                    config = "fullft"
            trackers.append({
                "path": p,
                "model": model,
                "dataset": dataset,
                "config": config,
                "seed": seed,
            })

    # Expanded directory (multi-model)
    for p in sorted(EXPANDED_DIR.glob("*_tracker.json")):
        parts = p.stem.replace("_tracker", "").split("_")
        # Pattern: model_dataset_config_seed
        if len(parts) >= 4:
            model = parts[0]
            if model == "roberta-base" or model == "bert-base-uncased" or model == "distilbert-base-uncased":
                pass
            elif parts[0] == "roberta":
                model = "-".join(parts[:2]) if parts[1] == "base" else parts[0]
                parts = parts[2:] if parts[1] == "base" else parts[1:]
            # Simpler: just use the filename
            trackers.append({
                "path": p,
                "model": "_".join(parts[:-2]) if len(parts) > 2 else parts[0],
                "dataset": parts[-3] if len(parts) >= 3 else "unknown",
                "config": parts[-2] if len(parts) >= 2 else "unknown",
                "seed": int(parts[-1][1:]) if parts[-1].startswith("s") else "unknown",
                "filename": p.stem,
            })

    return trackers


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Per-Label-Class AULC-Entropy Decomposition")
    print("=" * 70)

    all_results = []

    # Process main sweep trackers (RoBERTa-SNLI, well-structured)
    print("\n--- RoBERTa-SNLI Main Sweep ---")
    for pattern in ["sweep_r*_s*.json", "fullft_s*.json"]:
        for p in sorted(TRACKING_DIR.glob(pattern)):
            if "results" in p.name or "summary" in p.name:
                continue

            print(f"\n  Analyzing: {p.name}")
            try:
                tracker = TemporalTracker.load(p)
                results = compute_aulc_by_label(tracker)

                run_info = {
                    "file": p.name,
                    "model": "roberta-base",
                    "dataset": "snli",
                    "per_label": results,
                }
                all_results.append(run_info)

                # Print per-label breakdown
                print(f"    {'Label':>15} {'n':>5} {'ρ':>8} {'p':>12}")
                print(f"    {'-'*42}")
                for label_name in ["entailment", "neutral", "contradiction", "overall"]:
                    if label_name in results:
                        r = results[label_name]
                        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
                        print(f"    {label_name:>15} {r['n_examples']:>5} {r['spearman_rho']:>8.4f} {r['p_value']:>12.2e} {sig}")
            except Exception as e:
                print(f"    Error: {e}")

    # Process expanded trackers
    print("\n--- Expanded Experiments ---")
    for p in sorted(EXPANDED_DIR.glob("*_tracker.json")):
        print(f"\n  Analyzing: {p.name}")
        try:
            tracker = TemporalTracker.load(p)
            results = compute_aulc_by_label(tracker)

            run_info = {
                "file": p.name,
                "per_label": results,
            }
            all_results.append(run_info)

            print(f"    {'Label':>15} {'n':>5} {'ρ':>8} {'p':>12}")
            print(f"    {'-'*42}")
            for label_name in ["entailment", "neutral", "contradiction", "overall"]:
                if label_name in results:
                    r = results[label_name]
                    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
                    print(f"    {label_name:>15} {r['n_examples']:>5} {r['spearman_rho']:>8.4f} {r['p_value']:>12.2e} {sig}")
        except Exception as e:
            print(f"    Error: {e}")

    # Save all results
    output_path = OUTPUT_DIR / "per_label_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved results to {output_path}")

    # Summary: is the correlation label-independent?
    print(f"\n{'='*70}")
    print("Summary: Is the entropy-AULC correlation label-independent?")
    print(f"{'='*70}")

    label_rhos = {label: [] for label in ["entailment", "neutral", "contradiction"]}
    for run in all_results:
        for label in label_rhos:
            if label in run["per_label"]:
                r = run["per_label"][label]
                if r["n_examples"] >= 10:
                    label_rhos[label].append(r["spearman_rho"])

    for label, rhos in label_rhos.items():
        if rhos:
            print(f"  {label:>15}: mean ρ = {np.mean(rhos):+.4f} ± {np.std(rhos):.4f} "
                  f"(n_runs={len(rhos)}, range=[{min(rhos):.4f}, {max(rhos):.4f}])")


if __name__ == "__main__":
    main()
