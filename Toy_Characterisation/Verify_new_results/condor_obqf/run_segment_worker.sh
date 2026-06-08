#!/bin/bash
# ============================================================
# Segment-level statevector OneBitHHL worker wrapper.
# Arguments:  n_trk  rep  gamma  delta  epsilon  threshold  outdir  [save_vectors]
# ============================================================
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "usage: $0 n_trk rep gamma delta eps tau outdir [save_vectors]" >&2
    exit 2
fi

N_TRK="$1"
REP="$2"
GAMMA="$3"
DELTA="$4"
EPS="$5"
TAU="$6"
OUTDIR="$7"
SAVE_VEC="${8:-0}"

# Headless matplotlib (just in case any submodule imports it)
export MPLBACKEND=Agg

# Toy package (editable install)
export PYTHONPATH="/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src:${PYTHONPATH:-}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONDA="/data/bfys/gscriven/conda/bin/conda"

mkdir -p "${OUTDIR}"

echo "[worker] host=$(hostname) n_trk=${N_TRK} rep=${REP} gamma=${GAMMA}"
echo "[worker] outdir=${OUTDIR}  save_vec=${SAVE_VEC}"

ARGS=(
    --n-trk "${N_TRK}"
    --rep "${REP}"
    --gamma "${GAMMA}"
    --delta "${DELTA}"
    --epsilon "${EPS}"
    --threshold "${TAU}"
    --outdir "${OUTDIR}"
)
if [[ "${SAVE_VEC}" == "1" ]]; then
    ARGS+=(--save-vectors)
fi

exec "${CONDA}" run --no-capture-output -n Q_env python \
    "${SCRIPT_DIR}/run_segment_event.py" "${ARGS[@]}"
