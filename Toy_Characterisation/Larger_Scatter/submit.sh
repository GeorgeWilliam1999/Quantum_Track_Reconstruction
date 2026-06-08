#!/bin/bash
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SHARED="$( cd -- "${HERE}/../_shared" &> /dev/null && pwd )"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"
mkdir -p "${HERE}/logs"
[[ "${1:-}" == "--resubmit" ]] || "${PYBIN}" "${HERE}/gen_params.py"
shopt -s nullglob
for csv in "${HERE}/params"/ls_mem*.csv; do
    mem=$(basename "${csv}" .csv | sed 's/.*mem//')
    echo "[submit] ${csv}  mem=${mem} GB  rows=$(wc -l <"${csv}")"
    condor_submit -append "PARAMS_CSV=${csv}" \
                  -append "LOGDIR=${HERE}/logs" \
                  -append "MEM_GB=${mem}" \
                  "${SHARED}/submit_base.sub"
done
