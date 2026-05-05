#!/bin/bash
# =============================================================================
# Master Runner: All New Experiments for ACL SRW 2026 Revision
# =============================================================================
#
# Runs all new experiments and analyses added for the paper revision.
# Expected total compute: ~8-12 hours on Apple M4 Max (or ~6-8 hours on A100).
#
# Usage:
#   bash scripts/run_new_experiments.sh            # Run everything
#   bash scripts/run_new_experiments.sh --dry-run   # Print commands only
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

run_cmd() {
    local desc="$1"
    local cmd="$2"
    echo ""
    echo "======================================================================"
    echo "  $desc"
    echo "  Command: $cmd"
    echo "  Started: $(date)"
    echo "======================================================================"
    if [ "$DRY_RUN" = false ]; then
        eval "$cmd"
        echo "  Completed: $(date)"
    else
        echo "  [SKIPPED - dry run]"
    fi
}

echo "======================================================================"
echo "  ACL SRW 2026 Revision: New Experiments"
echo "  Project: $PROJECT_ROOT"
echo "  Started: $(date)"
echo "======================================================================"

# ------------------------------------------------------------------ #
# 1. BERT-base Rank Sweep (~60 min on A100)
# ------------------------------------------------------------------ #
run_cmd \
    "1/6: BERT-base rank sweep on SNLI (r=1,2,4,8,16,32, seed 42)" \
    "python scripts/11_bert_rank_sweep.py --ranks 1 2 4 8 16 32 --seed 42"

# ------------------------------------------------------------------ #
# 2. DeBERTa v3 Extended (~3 hours on A100)
#    - LoRA r=16 x 3 seeds
#    - Full FT x 3 seeds
# ------------------------------------------------------------------ #
run_cmd \
    "2/6: DeBERTa v3 extended (r=16 + Full FT, 3 seeds each)" \
    "python scripts/12_deberta_extended.py --configs r16 fullft --seeds 42 123 456"

# ------------------------------------------------------------------ #
# 3. Noise Injection Multi-Seed (~40 min on A100)
#    - Seeds 123 and 456 (seed 42 already done)
#    - 3 conditions x 2 seeds = 6 runs
# ------------------------------------------------------------------ #
run_cmd \
    "3/6: Noise injection on seeds 123, 456" \
    "python scripts/13_noise_injection_multiseed.py --seeds 123 456"

# ------------------------------------------------------------------ #
# 4. Adapter Baseline (~30 min on A100)
#    - RoBERTa-base on SNLI, 3 seeds
# ------------------------------------------------------------------ #
run_cmd \
    "4/6: Adapter baseline (RoBERTa-SNLI, 3 seeds)" \
    "python scripts/15_adapter_baseline.py --seeds 42 123 456"

# ------------------------------------------------------------------ #
# 5. Per-Label Analysis (CPU only, ~2 min)
#    - Run AFTER all training to include new results
# ------------------------------------------------------------------ #
run_cmd \
    "5/6: Per-label-class AULC-entropy decomposition (analysis only)" \
    "python scripts/14_per_label_analysis.py"

# ------------------------------------------------------------------ #
# 6. Decoder-Only Model Experiments (~3-4 hours on M4 Max)
#    - Qwen2.5-1.5B and Qwen2.5-3B with LoRA r=4 and r=16
#    - 3 seeds each = 12 runs total
#    - All float16 to fit in 36GB unified memory
# ------------------------------------------------------------------ #
run_cmd \
    "6/6: Decoder-only models (Qwen2.5-1.5B + 3B, LoRA r=4/r=16, 3 seeds)" \
    "python scripts/16_decoder_experiments.py --models qwen1.5b qwen3b --configs r4 r16 --seeds 42 123 456"

echo ""
echo "======================================================================"
echo "  All experiments complete!"
echo "  Finished: $(date)"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Review results in results/tracking/{bert_rank_sweep,deberta_extended,adapter_baseline}/"
echo "  2. Review noise injection multi-seed in results/tracking/robustness_experiments/noise_injection/"
echo "  3. Review per-label analysis in results/tracking/per_label_analysis/"
echo "  4. Review decoder experiments in results/tracking/decoder_experiments/"
echo "  5. Update paper tables and text with new results"
echo "  6. Regenerate figures with colorblind-friendly palettes (see figures/ACCESSIBILITY_TODO.txt)"
echo "  7. Rebuild paper PDF: cd paper && latexmk -pdf main.tex"
