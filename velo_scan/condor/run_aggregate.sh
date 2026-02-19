#!/usr/bin/env bash
# =============================================================================
# Condor wrapper – run the aggregation script
#
# Arguments:
#   $1 = results directory (e.g. /data/bfys/gscriven/velo_scan/results)
# =============================================================================
set -euo pipefail

CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Velo_Toy_Tester"

BASE_DIR="/data/bfys/gscriven/velo_scan"
AGG_SCRIPT="$BASE_DIR/scripts/aggregate.py"

RUNS_DIR="${1:?Missing results directory}"

echo "========================================"
echo "  Velo Scan – Aggregation"
echo "========================================"
echo "Runs Dir : $RUNS_DIR"
echo "Host     : $(hostname)"
echo "Date     : $(date)"
echo "========================================"

export PYTHONPATH="$BASE_DIR/scripts:${PYTHONPATH:-}"
export MPLBACKEND=Agg

"$CONDA" run --no-banner -n "$ENV_NAME" python "$AGG_SCRIPT" \
    --runs-dir "$RUNS_DIR"

echo ""
echo "[OK] Aggregation complete."
echo "========================================"
