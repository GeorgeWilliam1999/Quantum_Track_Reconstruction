#!/bin/bash
# Run the universal aggregator over Epsilon_study_2 results.
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="$( cd -- "${HERE}/../.." &> /dev/null && pwd )"
SHARED="${ROOT}/Toy_Characterisation/_shared"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"

"${PYBIN}" "${SHARED}/aggregate.py" \
    --indir "${HERE}/results" \
    --outprefix "${HERE}/results/eps2" \
    --group-keys "sigma_res,sigma_scatt"
