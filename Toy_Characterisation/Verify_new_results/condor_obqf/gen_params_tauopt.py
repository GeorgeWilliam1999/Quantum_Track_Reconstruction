#!/usr/bin/env python3
"""Generate parameter CSV for rethreshold Condor jobs.

One row per (readout, n_trk, rep) combination, with the F1-optimal τ_rel
from best_tau_rel_f1.csv plugged in per T.

Columns: n_trk, rep, readout, tau_rel, srcdir, outdir
"""
from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / 'Toy_Characterisation' / 'Verify_new_results' / 'outputs' / 'quantum_segment_analysis'

# Per-T F1-optimal τ_rel from best_tau_rel_f1.csv
TAU_BY_T = {
    2: 0.05, 5: 0.05, 10: 0.05, 20: 0.05, 50: 0.05,
    100: 0.05, 200: 0.05, 500: 0.40, 1000: 0.30,
}

# Rep grid matches the original seg14e_T1000 generation
N_TRKS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
REPS_BY_T = {
    2: 30, 5: 30, 10: 30, 20: 30, 50: 20,
    100: 20, 200: 10, 500: 5, 1000: 3,
}

ROWS = []
for readout in ('statevector', 'sampling'):
    src = BASE / f'seg14e_T1000_{readout}'  / 'pickles'
    out = BASE / f'seg14e_T1000_{readout}_tauopt' / 'pickles'
    for T in N_TRKS:
        tau = TAU_BY_T[T]
        for rep in range(REPS_BY_T[T]):
            ROWS.append((T, rep, readout, tau, str(src), str(out)))

csv_path = HERE / 'params_seg14e_tauopt.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    for row in ROWS:
        w.writerow(row)

print(f'wrote {csv_path} ({len(ROWS)} rows)')
for readout in ('statevector', 'sampling'):
    n = sum(1 for r in ROWS if r[2] == readout)
    print(f'  {readout}: {n} rows')
