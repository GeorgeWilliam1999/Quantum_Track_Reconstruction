#!/bin/bash
# ── Parameter optimisation sweep submission ────────────────────
# Generates parameter files for the k / theta_d / physics sweep,
# creates log directories, and submits all jobs to HTCondor.
#
# Usage:
#   ./submit.sh [--dry-run]
set -e

BASE_DIR="/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/EpsilonStudies"
RESULTS_DIR="$BASE_DIR/results"
CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Q_env"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN — no files will be written ==="
fi

echo "=========================================="
echo " Parameter Optimisation Submission"
echo " Base dir:    $BASE_DIR"
echo " Results dir: $RESULTS_DIR"
echo " Date:        $(date)"
echo "=========================================="

# Step 1: Generate parameter files
echo ""
echo "Step 1: Generating parameter files..."
"$CONDA" run -n "$ENV_NAME" python "$BASE_DIR/scripts/gen_params_opt.py" \
    --outdir "$RESULTS_DIR" $DRY_RUN

if [[ -n "$DRY_RUN" ]]; then
    echo ""
    echo "Dry run complete. No jobs submitted."
    exit 0
fi

# Step 2: Create log directory
echo ""
echo "Step 2: Creating log directory..."
mkdir -p "$RESULTS_DIR/logs"

# Step 3: Submit to Condor
echo ""
echo "Step 3: Submitting to Condor..."
condor_submit "$BASE_DIR/condor/scan.sub" \
    resultsdir="$RESULTS_DIR" \
    batches_file="$RESULTS_DIR/batches.txt"

echo ""
echo "=========================================="
echo " Submission complete!"
echo " Monitor with: condor_q"
echo " Aggregate with: python $BASE_DIR/scripts/aggregate.py --results-dir $RESULTS_DIR"
echo "=========================================="
