#!/bin/bash
# ============================================================
# Submit the §14e calculated-epsilon mirror to HTCondor.
#
# Usage:
#   ./submit_seg14e.sh             # regenerate params + submit
#   ./submit_seg14e.sh --resubmit  # skip param regen, just resubmit
# ============================================================
set -euo pipefail

HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="$( cd -- "${HERE}/../.." &> /dev/null && pwd )"
SHARED="${ROOT}/Toy_Characterisation/_shared"

PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"

PARAMS_DIR="${HERE}/params/seg14e_calc_eps"
LOG_DIR="${HERE}/logs/seg14e_calc_eps"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" != "--resubmit" ]]; then
    "${PYBIN}" "${HERE}/gen_params_seg14e.py"
fi

shopt -s nullglob
csvs=( "${PARAMS_DIR}"/seg14e_*_mem*.csv )
if [[ ${#csvs[@]} -eq 0 ]]; then
    echo "[submit_seg14e] no CSVs found in ${PARAMS_DIR}" >&2
    exit 1
fi

for csv in "${csvs[@]}"; do
    base=$(basename "${csv}" .csv)
    mem=$(echo "${base}" | sed 's/.*mem//')
    echo "[submit_seg14e] ${csv}  mem=${mem} GB  rows=$(wc -l <"${csv}")"
    condor_submit \
        -append "PARAMS_CSV=${csv}" \
        -append "LOGDIR=${LOG_DIR}" \
        -append "MEM_GB=${mem}" \
        "${SHARED}/submit_base.sub"
done

echo "[submit_seg14e] all tiers queued. Watch with: condor_q -nobatch"
