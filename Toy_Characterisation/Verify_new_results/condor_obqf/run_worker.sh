#!/bin/bash
# ============================================================
# Condor worker wrapper for §18b/c/e 1BQF jobs.
# Arguments: n_trk rep shots device resultsdir drop_rate gamma readout
# ============================================================
set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "usage: $0 n_trk rep shots device resultsdir [drop_rate] [gamma] [readout]" >&2
    exit 2
fi

N_TRK="$1"
REP="$2"
SHOTS="$3"
DEVICE="$4"
RESULTSDIR="$5"
DROP_RATE="${6:-0.0}"
GAMMA="${7:-3.0}"
READOUT="${8:-sampling}"

# Matplotlib headless
export MPLBACKEND=Agg

# Toy package (editable install)
export PYTHONPATH="/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src:${PYTHONPATH:-}"

# Path to the worker script — resolve relative to this wrapper
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Conda env
CONDA="/data/bfys/gscriven/conda/bin/conda"

mkdir -p "${RESULTSDIR}"

echo "[worker] host=$(hostname) n_trk=${N_TRK} rep=${REP} shots=${SHOTS} device=${DEVICE} drop_rate=${DROP_RATE} gamma=${GAMMA} readout=${READOUT}"
echo "[worker] resultsdir=${RESULTSDIR}"
echo "[worker] pwd=$(pwd)"

exec "${CONDA}" run --no-capture-output -n Q_env python \
    "${SCRIPT_DIR}/run_event.py" \
    --n-trk "${N_TRK}" \
    --rep   "${REP}" \
    --shots "${SHOTS}" \
    --device "${DEVICE}" \
    --drop-rate "${DROP_RATE}" \
    --gamma "${GAMMA}" \
    --readout "${READOUT}" \
    --outdir "${RESULTSDIR}"