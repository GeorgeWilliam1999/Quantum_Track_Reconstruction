#!/bin/bash
set -euo pipefail
HERE="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SHARED="$( cd -- "${HERE}/../_shared" &> /dev/null && pwd )"
PYBIN="/data/bfys/gscriven/conda/envs/Q_env/bin/python"
"${PYBIN}" "${SHARED}/aggregate.py" \
    --indir "${HERE}/results" --outprefix "${HERE}/results/ls" \
    --group-keys "sigma_scatt,hit_ineff"
