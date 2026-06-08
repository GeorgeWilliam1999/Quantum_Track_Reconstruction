#!/bin/bash
# ============================================================
# Submit the Epsilon_study_2 top-up jobs:
#   1. T=200 CPU statevector fill-in (sr=0.05 missing reps)
#   2. T=400 / 700 / 1000 GPU quantum jobs (AerSimulator CUDA)
#
# Usage:
#   ./submit_topup.sh             # regenerate params then submit
#   ./submit_topup.sh --resubmit  # skip param regen
# ============================================================
set -euo pipefail

HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SHARED="${HERE}/../_shared"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"
PARAMS_DIR="${HERE}/params"
LOG_DIR="${HERE}/logs"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" != "--resubmit" ]]; then
    echo "[params] regenerating..."
    "${PYBIN}" "${HERE}/gen_params_topup.py"
fi

# T=200 CPU statevector top-up (submit_base.sub — 15 columns, no device)
csv200="${PARAMS_DIR}/eps2_topup_mem32.csv"
if [[ -s "${csv200}" ]]; then
    echo "[submit] T=200 CPU statevector  rows=$(wc -l < "${csv200}")  mem=32 GB"
    condor_submit \
        -append "PARAMS_CSV=${csv200}" \
        -append "LOGDIR=${LOG_DIR}" \
        -append "MEM_GB=32" \
        "${SHARED}/submit_base.sub"
else
    echo "[skip] ${csv200} is empty — all T=200 reps already on disk"
fi

# T=400/700/1000 GPU quantum jobs (submit_gpu.sub — 16 columns, device=GPU)
# 128 GB: sr=0.05 events produce dense Hamiltonians that exceed 64 GB
csv_gpu="${PARAMS_DIR}/eps2_topup_gpu_mem128.csv"
if [[ -s "${csv_gpu}" ]]; then
    echo "[submit] T=400/700/1000 GPU quantum  rows=$(wc -l < "${csv_gpu}")  mem=128 GB"
    condor_submit \
        -append "PARAMS_CSV=${csv_gpu}" \
        -append "LOGDIR=${LOG_DIR}" \
        -append "MEM_GB=128" \
        "${HERE}/submit_gpu.sub"
else
    echo "[skip] ${csv_gpu} is empty — all GPU quantum jobs already on disk"
fi

echo "[submit] done. Watch with: condor_q -nobatch"
