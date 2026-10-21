#!/bin/bash
# Part A driver: 13 cases in parallel, two threads each.
cd "$(dirname "$0")"
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh && conda activate Q_env
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cat <<'LIST' | xargs -P 13 -L 1 python3 04_refit_heldout.py
5um 200 base
5um 200 fork
5um 200 occupancy
5um 200 both
5um 1000 base
5um 1000 fork
15um 200 base
15um 200 fork
15um 200 occupancy
15um 200 both
15um 400 base
15um 400 fork
15um 400 occupancy
LIST
echo ALL_DONE
