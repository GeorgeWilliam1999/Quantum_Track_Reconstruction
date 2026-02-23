#!/usr/bin/env bash
# =============================================================================
# Condor worker wrapper – activates conda and runs run_batch.py
#
# Arguments (passed by scan.sub):
#   $1 = batch ID
#   $2 = job tag  (Cluster.Process)
#   $3 = results directory
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------
# Configuration – adjust paths to your Nikhef setup
# ----------------------------------------------------------------------
CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Q_env"

BASE_DIR="/data/bfys/gscriven/velo_scan"
PYTHON_SCRIPT="$BASE_DIR/scripts/run_batch.py"

# ----------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------
BATCH="${1:?Missing batch ID}"
JOBTAG="${2:?Missing job tag}"
RUNS_DIR="${3:?Missing results directory}"

echo "========================================"
echo "  Velo Scan – Batch Worker"
echo "========================================"
echo "Batch ID : $BATCH"
echo "Job Tag  : $JOBTAG"
echo "Runs Dir : $RUNS_DIR"
echo "Host     : $(hostname)"
echo "Date     : $(date)"
echo "========================================"

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
if [ ! -x "$CONDA" ]; then
    echo "[ERROR] Conda not found at $CONDA" >&2
    exit 1
fi

echo "Activating conda env: $ENV_NAME"

# Add the lhcb_velo_toy package (and any local code) to PYTHONPATH
export PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}"
export MPLBACKEND=Agg

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
echo ""
echo "[INFO] Starting batch $BATCH ..."

"$CONDA" run -n "$ENV_NAME" python "$PYTHON_SCRIPT" \
    --batch "$BATCH" \
    --runs-dir "$RUNS_DIR" \
    --params "$RUNS_DIR/params.csv"

echo ""
echo "[INFO] Batch $BATCH finished."
echo "========================================"
