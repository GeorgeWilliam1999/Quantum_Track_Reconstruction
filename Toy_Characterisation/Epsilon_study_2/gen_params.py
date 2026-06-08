#!/usr/bin/env python3
"""
Generate the params CSV for Epsilon_study_2.

Two sections in one CSV:

1. ``validation``: sigma_res=0, sigma_scatt=1e-4 — must match
   Verify_new_results within +/- 1 stderr per T.
2. ``sensitivity``: full grid over (sigma_res, sigma_scatt, T, rep).

Columns match _shared/submit_base.sub:

    n_trk, rep, sigma_res, sigma_scatt, phi_max, hit_ineff,
    convolution, erf_sigma, epsilon, gamma, delta, tau_default,
    run_quantum, tag, outdir
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # Quantum_Track_Reconstruction
sys.path.insert(0, str(ROOT / "Toy_Characterisation" / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402


# ------------------------------------------------------------------
TRACK_GRID = [10, 20, 50, 100, 200, 400, 700, 1000]


def reps_for_T(T: int) -> int:
    if T <= 50:
        return 10
    if T <= 200:
        return 5
    return 3


PHI_MAX = 0.2
HIT_INEFF = 0.0
CONVOLUTION = 0
ERF_SIGMA = 1e-4   # ignored when convolution=0 but kept for argparse
GAMMA = 3.0
DELTA = 1.0
TAU_DEFAULT = 0.35
RUN_QUANTUM = 1


VALIDATION_POINT = dict(sigma_res=0.0, sigma_scatt=1e-4)

SENSITIVITY_GRID = [
    (sr, ss)
    for sr in (0.0, 0.01, 0.02, 0.05)
    for ss in (1e-4, 3e-4, 5e-4)
]


def mem_tier_gb(T: int) -> int:
    """Memory request tier (GB) by track count."""
    if T < 150:
        return 16
    if T < 500:
        return 32
    if T < 1500:
        return 64
    return 128


def emit(outdir_results: Path, csv_dir: Path) -> dict:
    outdir_results.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1) validation
    sr, ss = VALIDATION_POINT['sigma_res'], VALIDATION_POINT['sigma_scatt']
    eps = compute_epsilon(sr, ss)
    val_dir = outdir_results / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    for T in TRACK_GRID:
        for rep in range(reps_for_T(T)):
            rows.append(dict(
                n_trk=T, rep=rep,
                sigma_res=sr, sigma_scatt=ss,
                phi_max=PHI_MAX, hit_ineff=HIT_INEFF,
                convolution=CONVOLUTION, erf_sigma=ERF_SIGMA,
                epsilon=eps, gamma=GAMMA, delta=DELTA,
                tau_default=TAU_DEFAULT, run_quantum=RUN_QUANTUM,
                tag="eps2val", outdir=str(val_dir),
            ))

    # 2) sensitivity
    for sr, ss in SENSITIVITY_GRID:
        eps = compute_epsilon(sr, ss)
        gridtag = f"sr{sr:g}_ss{ss:g}"
        cell = outdir_results / "sensitivity" / gridtag
        cell.mkdir(parents=True, exist_ok=True)
        for T in TRACK_GRID:
            for rep in range(reps_for_T(T)):
                rows.append(dict(
                    n_trk=T, rep=rep,
                    sigma_res=sr, sigma_scatt=ss,
                    phi_max=PHI_MAX, hit_ineff=HIT_INEFF,
                    convolution=CONVOLUTION, erf_sigma=ERF_SIGMA,
                    epsilon=eps, gamma=GAMMA, delta=DELTA,
                    tau_default=TAU_DEFAULT, run_quantum=RUN_QUANTUM,
                    tag=f"eps2_{gridtag}", outdir=str(cell),
                ))

    fields = [
        'n_trk', 'rep', 'sigma_res', 'sigma_scatt',
        'phi_max', 'hit_ineff', 'convolution', 'erf_sigma',
        'epsilon', 'gamma', 'delta', 'tau_default',
        'run_quantum', 'tag', 'outdir',
    ]

    # Split rows by memory tier so each condor submit is uniform.
    tiers: dict[int, list] = {}
    for r in rows:
        tiers.setdefault(mem_tier_gb(int(r['n_trk'])), []).append(r)

    written = {}
    for mem, group in sorted(tiers.items()):
        csv_path = csv_dir / f"eps2_mem{mem}.csv"
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            # NOTE: HTCondor queue-from-CSV treats every line as data;
            # do NOT write a header.
            for r in group:
                w.writerow(r)
        written[mem] = (csv_path, len(group))
        print(f"[gen_params] mem={mem} GB -> {len(group):5d} rows -> {csv_path}")
    total = sum(v[1] for v in written.values())
    print(f"[gen_params] total {total} rows")
    print(f"[gen_params] validation eps = {compute_epsilon(0.0,1e-4):.6e}")
    return written


if __name__ == '__main__':
    csv_dir = HERE / "params"
    outdir = HERE / "results"
    emit(outdir, csv_dir)
