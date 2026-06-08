#!/usr/bin/env bash
# ── Condor worker wrapper for large-event reconstruction ─────────
# Activates the Q_env conda environment and runs the large-event worker.
#
# Arguments:
#   $1  Path to the job parameter JSON file
#   $2  Path to the results directory
set -euo pipefail

CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Q_env"

BASE_DIR="/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Verify_new_results"
WORKER_SCRIPT="$BASE_DIR/scripts/run_worker_large.py"

PARAMS_JSON="${1:?Missing parameter JSON path}"
RESULTS_DIR="${2:?Missing results directory}"

# Extract job ID from filename
JOB_NAME=$(basename "$PARAMS_JSON" .json)
JOB_OUTDIR="$RESULTS_DIR/$JOB_NAME"

echo "=========================================="
echo " Large-Event Worker: $JOB_NAME"
echo " Params: $PARAMS_JSON"
echo " Output: $JOB_OUTDIR"
echo " Host:   $(hostname)"
echo " Date:   $(date)"
echo "=========================================="

# Set matplotlib backend to non-interactive
export MPLBACKEND=Agg

# Numba threading: use all available CPUs
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"

# Run the worker
"$CONDA" run -n "$ENV_NAME" python "$WORKER_SCRIPT" \
    --params-json "$PARAMS_JSON" \
    --outdir "$JOB_OUTDIR"

echo "Worker $JOB_NAME finished at $(date)"
