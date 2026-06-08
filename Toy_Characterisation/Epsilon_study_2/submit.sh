#!/bin/bash
# ============================================================
# Submit all memory tiers for Epsilon_study_2.
#
# Usage:
#   ./submit.sh           # regenerates params and submits everything
#   ./submit.sh --resubmit  # skip param regen
# ============================================================
set -euo pipefail

HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="$( cd -- "${HERE}/../.." &> /dev/null && pwd )"
SHARED="${ROOT}/Toy_Characterisation/_shared"

PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"

PARAMS_DIR="${HERE}/params"
LOG_DIR="${HERE}/logs"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" != "--resubmit" ]]; then
    "${PYBIN}" "${HERE}/gen_params.py"
fi

shopt -s nullglob
for csv in "${PARAMS_DIR}"/eps2_mem*.csv; do
    base=$(basename "${csv}" .csv)
    mem=$(echo "${base}" | sed 's/.*mem//')
    echo "[submit] ${csv}  mem=${mem} GB  rows=$(wc -l <"${csv}")"
    condor_submit \
        -append "PARAMS_CSV=${csv}" \
        -append "LOGDIR=${LOG_DIR}" \
        -append "MEM_GB=${mem}" \
        "${SHARED}/submit_base.sub"
done

echo "[submit] all tiers queued. Watch with: condor_q -nobatch"
