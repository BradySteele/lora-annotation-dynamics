#!/bin/bash
# ==============================================================================
# LoRA Temporal Separation -- Full Experiment Pipeline
#
# Runs all phases of the experiment sequentially with gate checks:
#   Phase 0: Data preparation and validation
#   Phase 1: Pilot experiment (rank=4, seed=42)
#   Phase 2: Rank sweep (6 ranks, seed=42)
#   Phase 3: Full experiment (6 ranks x 5 seeds)
#   Phase 4: Full analysis
#   Phase 5: Generate publication figures
#
# Prerequisites:
#   pip install -e .  (or pip install -r requirements.txt)
#   Ensure roberta-base is downloaded or internet is available
#
# Usage:
#   bash scripts/run_all.sh              # Run full pipeline
#   bash scripts/run_all.sh --phase 2    # Start from Phase 2
#   bash scripts/run_all.sh --dry-run    # Print commands without executing
#   bash scripts/run_all.sh --synthetic  # Use synthetic data throughout
# ==============================================================================

set -euo pipefail

# Project root is the parent directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #
START_PHASE=0
DRY_RUN=false
SYNTHETIC=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            START_PHASE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --synthetic)
            SYNTHETIC="--synthetic"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_phase() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}${BOLD}  Phase $1: $2${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

log_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

log_info() {
    echo -e "${CYAN}[INFO] $1${NC}"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $*"
        return 0
    else
        echo -e "${CYAN}[RUN]${NC} $*"
        "$@"
        return $?
    fi
}

format_time() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local remaining_seconds=$((seconds % 60))
    if [ "$minutes" -gt 0 ]; then
        echo "${minutes}m ${remaining_seconds}s"
    else
        echo "${seconds}s"
    fi
}

# ---------------------------------------------------------------------------- #
# Verify environment
# ---------------------------------------------------------------------------- #
echo -e "${BOLD}LoRA Temporal Separation -- Full Experiment Pipeline${NC}"
echo "Project root:   $PROJECT_ROOT"
echo "Start phase:    $START_PHASE"
echo "Dry run:        $DRY_RUN"
echo "Synthetic data: ${SYNTHETIC:-no}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please install Python >= 3.9."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo "Python: $PYTHON_VERSION"

# Check key packages
if ! python -c "import torch" 2>/dev/null; then
    log_warning "PyTorch not found. Run: pip install -r requirements.txt"
fi

if ! python -c "import transformers" 2>/dev/null; then
    log_warning "transformers not found. Run: pip install -r requirements.txt"
fi

if ! python -c "import peft" 2>/dev/null; then
    log_warning "peft not found. Run: pip install -r requirements.txt"
fi

# Create output directories
mkdir -p results/data results/tracking results/analysis results/logs
mkdir -p figures

echo ""

# Track phase results
declare -a PHASE_NAMES
declare -a PHASE_TIMES
declare -a PHASE_STATUSES

TOTAL_START=$(date +%s)

# ---------------------------------------------------------------------------- #
# Phase 0: Data Preparation
# ---------------------------------------------------------------------------- #
if [ "$START_PHASE" -le 0 ]; then
    log_phase 0 "Data Preparation and Validation"
    PHASE_START=$(date +%s)

    if run_cmd python scripts/01_prepare_data.py $SYNTHETIC $EXTRA_ARGS 2>&1 | tee results/logs/phase0.log; then
        PHASE_END=$(date +%s)
        PHASE_ELAPSED=$((PHASE_END - PHASE_START))
        log_success "Phase 0 complete ($(format_time $PHASE_ELAPSED))"

        # Check gate
        if grep -q "PHASE 0 GATE WARNING" results/logs/phase0.log 2>/dev/null; then
            log_warning "Phase 0 gate warning detected. Review entropy distribution."
        fi

        PHASE_NAMES+=("Phase 0: Data Prep")
        PHASE_TIMES+=("$(format_time $PHASE_ELAPSED)")
        PHASE_STATUSES+=("PASS")
    else
        log_error "Phase 0 failed. Check results/logs/phase0.log"
        PHASE_NAMES+=("Phase 0: Data Prep")
        PHASE_TIMES+=("N/A")
        PHASE_STATUSES+=("FAIL")
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Phase 1: Pilot Experiment
# ---------------------------------------------------------------------------- #
if [ "$START_PHASE" -le 1 ]; then
    log_phase 1 "Pilot Experiment (rank=4, seed=42)"
    PHASE_START=$(date +%s)

    if run_cmd python scripts/02_pilot_experiment.py $SYNTHETIC $EXTRA_ARGS 2>&1 | tee results/logs/phase1.log; then
        PHASE_END=$(date +%s)
        PHASE_ELAPSED=$((PHASE_END - PHASE_START))

        # Check gate
        if grep -q "PHASE 1 GATE PASSED" results/logs/phase1.log 2>/dev/null; then
            log_success "Phase 1 GATE PASSED ($(format_time $PHASE_ELAPSED))"
            PHASE_STATUSES+=("PASS")
        elif grep -q "PHASE 1 GATE FAILED" results/logs/phase1.log 2>/dev/null; then
            log_error "Phase 1 GATE FAILED. Review pilot results."
            log_warning "The temporal separation hypothesis may not hold."
            log_warning "Review figures/hero_loss_curves_r4_s42.png"
            PHASE_STATUSES+=("GATE FAIL")
            echo ""
            echo "Continue anyway? The rank sweep may reveal rank-dependent effects."
            echo "To continue: bash scripts/run_all.sh --phase 2 $SYNTHETIC"
            exit 1
        else
            log_warning "Phase 1 gate status unclear. Check log."
            PHASE_STATUSES+=("UNCLEAR")
        fi

        PHASE_NAMES+=("Phase 1: Pilot")
        PHASE_TIMES+=("$(format_time $PHASE_ELAPSED)")
    else
        log_error "Phase 1 failed. Check results/logs/phase1.log"
        PHASE_NAMES+=("Phase 1: Pilot")
        PHASE_TIMES+=("N/A")
        PHASE_STATUSES+=("FAIL")
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Phase 2: Rank Sweep
# ---------------------------------------------------------------------------- #
if [ "$START_PHASE" -le 2 ]; then
    log_phase 2 "Rank Sweep (6 ranks, seed=42)"
    PHASE_START=$(date +%s)

    if run_cmd python scripts/03_rank_sweep.py $SYNTHETIC $EXTRA_ARGS 2>&1 | tee results/logs/phase2.log; then
        PHASE_END=$(date +%s)
        PHASE_ELAPSED=$((PHASE_END - PHASE_START))
        log_success "Phase 2 complete ($(format_time $PHASE_ELAPSED))"

        PHASE_NAMES+=("Phase 2: Rank Sweep")
        PHASE_TIMES+=("$(format_time $PHASE_ELAPSED)")

        if grep -q "PHASE 2 GATE WARNING" results/logs/phase2.log 2>/dev/null; then
            log_warning "Rank trend is not clearly non-increasing. Review before Phase 3."
            PHASE_STATUSES+=("WARN")
        else
            PHASE_STATUSES+=("PASS")
        fi
    else
        log_error "Phase 2 failed. Check results/logs/phase2.log"
        PHASE_NAMES+=("Phase 2: Rank Sweep")
        PHASE_TIMES+=("N/A")
        PHASE_STATUSES+=("FAIL")
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Phase 3: Full Experiment
# ---------------------------------------------------------------------------- #
if [ "$START_PHASE" -le 3 ]; then
    log_phase 3 "Full Experiment (6 ranks x 5 seeds)"
    PHASE_START=$(date +%s)

    if run_cmd python scripts/04_full_sweep.py $SYNTHETIC $EXTRA_ARGS 2>&1 | tee results/logs/phase3.log; then
        PHASE_END=$(date +%s)
        PHASE_ELAPSED=$((PHASE_END - PHASE_START))
        log_success "Phase 3 complete ($(format_time $PHASE_ELAPSED))"

        PHASE_NAMES+=("Phase 3: Full Sweep")
        PHASE_TIMES+=("$(format_time $PHASE_ELAPSED)")
        PHASE_STATUSES+=("PASS")
    else
        log_error "Phase 3 failed. Check results/logs/phase3.log"
        PHASE_NAMES+=("Phase 3: Full Sweep")
        PHASE_TIMES+=("N/A")
        PHASE_STATUSES+=("FAIL")
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Phase 4: Full Analysis
# ---------------------------------------------------------------------------- #
if [ "$START_PHASE" -le 4 ]; then
    log_phase 4 "Full Analysis"
    PHASE_START=$(date +%s)

    if run_cmd python scripts/05_analyze_results.py $EXTRA_ARGS 2>&1 | tee results/logs/phase4.log; then
        PHASE_END=$(date +%s)
        PHASE_ELAPSED=$((PHASE_END - PHASE_START))
        log_success "Phase 4 complete ($(format_time $PHASE_ELAPSED))"

        PHASE_NAMES+=("Phase 4: Analysis")
        PHASE_TIMES+=("$(format_time $PHASE_ELAPSED)")
        PHASE_STATUSES+=("PASS")
    else
        log_error "Phase 4 failed. Check results/logs/phase4.log"
        PHASE_NAMES+=("Phase 4: Analysis")
        PHASE_TIMES+=("N/A")
        PHASE_STATUSES+=("FAIL")
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Phase 5: Generate Figures
# ---------------------------------------------------------------------------- #
if [ "$START_PHASE" -le 5 ]; then
    log_phase 5 "Generate Publication Figures"
    PHASE_START=$(date +%s)

    if run_cmd python scripts/06_generate_figures.py $EXTRA_ARGS 2>&1 | tee results/logs/phase5.log; then
        PHASE_END=$(date +%s)
        PHASE_ELAPSED=$((PHASE_END - PHASE_START))
        log_success "Phase 5 complete ($(format_time $PHASE_ELAPSED))"

        PHASE_NAMES+=("Phase 5: Figures")
        PHASE_TIMES+=("$(format_time $PHASE_ELAPSED)")
        PHASE_STATUSES+=("PASS")
    else
        log_error "Phase 5 failed. Check results/logs/phase5.log"
        PHASE_NAMES+=("Phase 5: Figures")
        PHASE_TIMES+=("N/A")
        PHASE_STATUSES+=("FAIL")
        exit 1
    fi
fi

# ---------------------------------------------------------------------------- #
# Summary
# ---------------------------------------------------------------------------- #
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}${BOLD}  Pipeline Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

printf "%-30s %-15s %-10s\n" "Phase" "Time" "Status"
printf "%-30s %-15s %-10s\n" "-----" "----" "------"

for i in "${!PHASE_NAMES[@]}"; do
    STATUS="${PHASE_STATUSES[$i]}"
    if [ "$STATUS" = "PASS" ]; then
        COLOR="$GREEN"
    elif [ "$STATUS" = "FAIL" ] || [ "$STATUS" = "GATE FAIL" ]; then
        COLOR="$RED"
    else
        COLOR="$YELLOW"
    fi
    printf "%-30s %-15s ${COLOR}%-10s${NC}\n" "${PHASE_NAMES[$i]}" "${PHASE_TIMES[$i]}" "$STATUS"
done

echo ""
echo "Total time: $(format_time $TOTAL_ELAPSED)"
echo ""
echo "Output locations:"
echo "  Data:       results/data/"
echo "  Tracking:   results/tracking/"
echo "  Analysis:   results/analysis/"
echo "  Figures:    figures/"
echo "  Logs:       results/logs/"
echo ""

# List generated figures
if [ -d "figures" ] && [ "$(ls -A figures 2>/dev/null)" ]; then
    echo "Generated figures:"
    ls -la figures/fig*.pdf 2>/dev/null || true
    ls -la figures/fig*.png 2>/dev/null || true
    ls -la figures/hero*.png 2>/dev/null || true
    ls -la figures/entropy*.png 2>/dev/null || true
fi

echo ""

# Final status
ALL_PASS=true
for status in "${PHASE_STATUSES[@]}"; do
    if [ "$status" != "PASS" ]; then
        ALL_PASS=false
        break
    fi
done

if [ "$ALL_PASS" = true ]; then
    log_success "All phases completed successfully."
else
    log_warning "Some phases had warnings or failures. Review logs."
fi
