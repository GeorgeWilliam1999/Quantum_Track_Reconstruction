#!/usr/bin/env bash
# ── Condor worker wrapper for segment-grass sweep ────────────────
# Activates Q_env and runs the worker script.
#
# Arguments:
#   $1  Path to job parameter JSON
#   $2  Path to results directory
set -euo pipefail

CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Q_env"

BASE_DIR="/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Segment_Grass"
WORKER_SCRIPT="$BASE_DIR/scripts/run_worker.py"

PARAMS_JSON="${1:?Missing parameter JSON path}"
RESULTS_DIR="${2:?Missing results directory}"

JOB_NAME=$(basename "$PARAMS_JSON" .json)
JOB_OUTDIR="$RESULTS_DIR/$JOB_NAME"

echo "=========================================="
echo " Segment-Grass Worker: $JOB_NAME"
echo " Params: $PARAMS_JSON"
echo " Output: $JOB_OUTDIR"
echo " Host:   $(hostname)"
echo " Date:   $(date)"
echo "=========================================="

export MPLBACKEND=Agg

"$CONDA" run -n "$ENV_NAME" python "$WORKER_SCRIPT" \
    --params-json "$PARAMS_JSON" \
    --outdir "$JOB_OUTDIR"

echo "Worker $JOB_NAME finished at $(date)"
