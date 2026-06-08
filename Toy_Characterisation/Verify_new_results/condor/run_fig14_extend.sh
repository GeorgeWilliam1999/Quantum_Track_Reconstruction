#!/usr/bin/env bash
# Condor wrapper for fig14 extension worker.
#   Args: $1 = n_tracks, $2 = rep, $3 = outdir
set -euo pipefail
CONDA="/data/bfys/gscriven/conda/bin/conda"
ENV_NAME="Q_env"
BASE="/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Verify_new_results"
SCRIPT="$BASE/scripts/run_fig14_extend.py"
N="${1:?n}"
REP="${2:?rep}"
OUTDIR="${3:?outdir}"

echo "=========================================="
echo " fig14 extend: n=$N rep=$REP"
echo " out: $OUTDIR"
echo " host: $(hostname)  date: $(date)"
echo "=========================================="
export MPLBACKEND=Agg
"$CONDA" run -n "$ENV_NAME" python "$SCRIPT" --n "$N" --rep "$REP" --outdir "$OUTDIR"
echo "done at $(date)"
