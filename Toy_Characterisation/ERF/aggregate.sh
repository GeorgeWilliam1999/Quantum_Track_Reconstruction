#!/bin/bash
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SHARED="$( cd -- "${HERE}/../_shared" &> /dev/null && pwd )"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"
"${PYBIN}" "${SHARED}/aggregate.py" \
    --indir "${HERE}/results" --outprefix "${HERE}/results/erf" \
    --group-keys "erf_sigma,sigma_scatt,sigma_res"
