#!/bin/bash
# ============================================================
# Universal worker wrapper for the four segment-level
# Toy_Characterisation experiments.
#
# Positional args (mirroring the queue line):
#   1  n_trk
#   2  rep
#   3  sigma_res         (mm)
#   4  sigma_scatt       (rad)
#   5  phi_max           (rad)
#   6  hit_ineff         (probability in [0,1])
#   7  convolution       (0=step / 1=erf)
#   8  erf_sigma         (theta_d for the ERF Hamiltonian)
#   9  epsilon
#  10  gamma
#  11  delta
#  12  tau_default       (relative threshold for default metric reporting)
#  13  run_quantum       (0/1)
#  14  tag               (filename prefix and bookkeeping label)
#  15  outdir            (absolute path)
# ============================================================
set -euo pipefail

if [[ $# -lt 15 ]]; then
    echo "usage: $0 n_trk rep sigma_res sigma_scatt phi_max hit_ineff convolution erf_sigma epsilon gamma delta tau_default run_quantum tag outdir" >&2
    exit 2
fi

N_TRK="$1"; REP="$2"; SIGMA_RES="$3"; SIGMA_SCATT="$4"
PHI_MAX="$5"; HIT_INEFF="$6"; CONV="$7"; ERFSIG="$8"
EPSILON="$9"; GAMMA="${10}"; DELTA="${11}"; TAUD="${12}"
RUNQ="${13}"; TAG="${14}"; OUTDIR="${15}"; DEVICE="${16:-CPU}"

export MPLBACKEND=Agg
# editable install path (in case PYTHONPATH is wiped)
export PYTHONPATH="/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src:${PYTHONPATH:-}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"

mkdir -p "${OUTDIR}"
echo "[wrapper] host=$(hostname) tag=${TAG} n_trk=${N_TRK} rep=${REP}"
echo "[wrapper] outdir=${OUTDIR}"

exec "${PYBIN}" "${SCRIPT_DIR}/run_worker.py" \
    --n-trk "${N_TRK}" --rep "${REP}" \
    --sigma-res "${SIGMA_RES}" --sigma-scatt "${SIGMA_SCATT}" \
    --phi-max "${PHI_MAX}" --hit-ineff "${HIT_INEFF}" \
    --convolution "${CONV}" --erf-sigma "${ERFSIG}" \
    --epsilon "${EPSILON}" --gamma "${GAMMA}" --delta "${DELTA}" \
    --tau-default "${TAUD}" --run-quantum "${RUNQ}" \
    --device "${DEVICE}" \
    --tag "${TAG}" --outdir "${OUTDIR}"
