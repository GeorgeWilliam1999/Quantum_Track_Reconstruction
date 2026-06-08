#!/bin/bash
# Wrapper for rethreshold_event.py jobs on condor.
# Arguments: n_trk rep readout tau_rel srcdir outdir [drop_rate]
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "usage: $0 n_trk rep readout tau_rel srcdir outdir [drop_rate]" >&2
    exit 2
fi
N_TRK="$1"; REP="$2"; READOUT="$3"; TAU_REL="$4"
SRCDIR="$5"; OUTDIR="$6"; DROP_RATE="${7:-0.01}"

export MPLBACKEND=Agg
export PYTHONPATH="/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src:${PYTHONPATH:-}"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONDA="/data/bfys/gscriven/conda/bin/conda"

mkdir -p "${OUTDIR}"

echo "[worker] host=$(hostname) n_trk=${N_TRK} rep=${REP} readout=${READOUT} tau_rel=${TAU_REL}"
echo "[worker] srcdir=${SRCDIR} outdir=${OUTDIR}"

exec "${CONDA}" run --no-capture-output -n Q_env python \
    "${SCRIPT_DIR}/rethreshold_event.py" \
    --n-trk "${N_TRK}" \
    --rep "${REP}" \
    --readout "${READOUT}" \
    --tau-rel "${TAU_REL}" \
    --srcdir "${SRCDIR}" \
    --outdir "${OUTDIR}" \
    --drop-rate "${DROP_RATE}"
