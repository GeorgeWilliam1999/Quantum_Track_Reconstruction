#!/bin/bash
# ============================================================
# Condor wrapper for the qtrk manifest-driven worker.
# Argument: 1 = absolute path to a shard CSV.
# ============================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <shard.csv>" >&2
    exit 2
fi
SHARD="$1"

export MPLBACKEND=Agg
export QTRK_STORE="${QTRK_STORE:-/data/bfys/gscriven/qtrk_store}"
export PYTHONPATH="/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared:/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src:${PYTHONPATH:-}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"

echo "[worker] host=$(hostname) shard=${SHARD} QTRK_STORE=${QTRK_STORE}"
exec "${PYBIN}" "${SCRIPT_DIR}/run_shard.py" "${SHARD}"
