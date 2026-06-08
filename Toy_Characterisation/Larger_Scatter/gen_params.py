#!/usr/bin/env python3
"""Params for Larger_Scatter (Task 3): scatter x hit-inefficiency."""
from __future__ import annotations
import csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

TRACK_GRID = [10, 20, 50, 100, 200, 400, 700, 1000]
def reps_for_T(T):
    return 10 if T <= 50 else 5 if T <= 200 else 3

def run_quantum_for_T(T):
    # OneBQF statevector cost is O(A_nnz * 2^n_sys): ~22 min at T=100, ~6 h at
    # T=200, intractable above. Classical SimpleHamiltonianFast is ~10 s at any T.
    return 1 if T <= 200 else 0

SCATTER_GRID = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
INEFF_GRID = [0.0, 0.01, 0.02, 0.05, 0.10]

PHI_MAX = 0.2; SIGMA_RES = 0.0
CONVOLUTION = 0; ERF_SIGMA = 1e-4
GAMMA = 3.0; DELTA = 1.0
TAU_DEFAULT = 0.35

def mem_tier_gb(T):
    if T <= 150:  return 16
    if T <= 400:  return 32
    if T <= 700:  return 64
    if T <= 1000: return 128
    return 192

def emit(outdir_results, csv_dir):
    outdir_results.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ss in SCATTER_GRID:
        eps = compute_epsilon(SIGMA_RES, ss)
        for he in INEFF_GRID:
            gridtag = f"ss{ss:g}_he{he:g}"
            cell = outdir_results / gridtag
            cell.mkdir(parents=True, exist_ok=True)
            for T in TRACK_GRID:
                for rep in range(reps_for_T(T)):
                    rows.append(dict(
                        n_trk=T, rep=rep,
                        sigma_res=SIGMA_RES, sigma_scatt=ss,
                        phi_max=PHI_MAX, hit_ineff=he,
                        convolution=CONVOLUTION, erf_sigma=ERF_SIGMA,
                        epsilon=eps, gamma=GAMMA, delta=DELTA,
                        tau_default=TAU_DEFAULT, run_quantum=run_quantum_for_T(T),
                        tag=f"ls_{gridtag}", outdir=str(cell),
                    ))
    fields = ['n_trk','rep','sigma_res','sigma_scatt','phi_max','hit_ineff',
              'convolution','erf_sigma','epsilon','gamma','delta',
              'tau_default','run_quantum','tag','outdir']
    tiers = {}
    for r in rows:
        tiers.setdefault(mem_tier_gb(int(r['n_trk'])), []).append(r)
    for mem, group in sorted(tiers.items()):
        p = csv_dir / f"ls_mem{mem}.csv"
        with p.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            for r in group: w.writerow(r)
        print(f"[gen_params] mem={mem} GB -> {len(group):5d} rows -> {p}")
    print(f"[gen_params] total {len(rows)} rows")

if __name__ == '__main__':
    emit(HERE/'results', HERE/'params')
