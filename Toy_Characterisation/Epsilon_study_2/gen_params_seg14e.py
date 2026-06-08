#!/usr/bin/env python3
"""
Generate Condor params CSV(s) for the §14e mirror at the calculated epsilon.

Mirrors the operating point used in §14e of
``Verify_new_results/segment_level_analysis.ipynb`` and
``Quantum_segment_level_analysis.ipynb`` (cell §7d), but with epsilon
supplied by ``lhcb_velo_toy.analysis.compute_epsilon`` instead of the
hand-tuned 2 mrad scalar.

Frozen knobs (first pass)::

    sigma_res    = 5e-3 mm        (5 micron)
    sigma_scatt  = 1e-4 rad       (0.1 mrad)
    hit_ineff    = 0.01           (1 % hit drop)
    phi_max      = 0.2
    gamma, delta = 3, 1
    tau_default  = 0.35           (worker uses tau * max(sol); the
                                   notebook recomputes metrics at the
                                   §14e *absolute* tau = 0.35 from the
                                   saved sol_C / sol_Q vectors)
    convolution  = 0 (step)
    erf_sigma    = 1e-4           (unused when convolution=0)

Two row groups are emitted to the same CSV (the worker doesn't care):

* ``classical`` -- T = [2, ..., 1000], ``run_quantum = 0``.
* ``quantum``   -- T = [2, 5, 10, 20, 50, 100, 200], ``run_quantum = 1``.

Rows are split into per-memory-tier CSVs so each ``condor_submit`` is
uniform (matches the existing ``gen_params.py`` convention in this dir).
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


# ---- §14e operating point ----------------------------------------------------
SIGMA_RES   = 5e-3
SIGMA_SCATT = 1e-4
HIT_INEFF   = 0.01
PHI_MAX     = 0.2
GAMMA       = 3.0
DELTA       = 1.0
TAU_DEFAULT = 0.35
CONVOLUTION = 0
ERF_SIGMA   = 1e-4


TRACK_GRID_CLASSICAL = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 75, 100,
                        150, 200, 300, 500, 750, 1000]
TRACK_GRID_QUANTUM   = [2, 5, 10, 20, 50, 100, 200]


def reps_for_T(T: int) -> int:
    if T <= 20:
        return 30
    if T <= 100:
        return 20
    if T <= 300:
        return 10
    if T <= 750:
        return 5
    return 3


def mem_tier_gb(T: int, run_quantum: bool) -> int:
    """Memory tier per (T, solver).

    Statevector OneBitHHL needs more headroom than the classical solver
    at the same T because qiskit holds the full 2^(n_sys+2) amplitudes
    plus a deep circuit description.
    """
    if run_quantum:
        if T <= 20:
            return 16
        if T <= 100:
            return 32
        return 64  # T in {200}
    # classical-only
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

    eps = compute_epsilon(SIGMA_RES, SIGMA_SCATT)
    print(f"[gen] calculated epsilon = {eps:.6e} rad  ({eps*1e3:.4f} mrad)")

    rows = []
    out_C = outdir_results / "classical"
    out_C.mkdir(parents=True, exist_ok=True)
    for T in TRACK_GRID_CLASSICAL:
        for rep in range(reps_for_T(T)):
            rows.append(dict(
                n_trk=T, rep=rep,
                sigma_res=SIGMA_RES, sigma_scatt=SIGMA_SCATT,
                phi_max=PHI_MAX, hit_ineff=HIT_INEFF,
                convolution=CONVOLUTION, erf_sigma=ERF_SIGMA,
                epsilon=eps, gamma=GAMMA, delta=DELTA,
                tau_default=TAU_DEFAULT, run_quantum=0,
                tag="seg14e_C", outdir=str(out_C),
            ))

    out_Q = outdir_results / "quantum"
    out_Q.mkdir(parents=True, exist_ok=True)
    for T in TRACK_GRID_QUANTUM:
        for rep in range(reps_for_T(T)):
            rows.append(dict(
                n_trk=T, rep=rep,
                sigma_res=SIGMA_RES, sigma_scatt=SIGMA_SCATT,
                phi_max=PHI_MAX, hit_ineff=HIT_INEFF,
                convolution=CONVOLUTION, erf_sigma=ERF_SIGMA,
                epsilon=eps, gamma=GAMMA, delta=DELTA,
                tau_default=TAU_DEFAULT, run_quantum=1,
                tag="seg14e_Q", outdir=str(out_Q),
            ))

    fields = [
        'n_trk', 'rep', 'sigma_res', 'sigma_scatt',
        'phi_max', 'hit_ineff', 'convolution', 'erf_sigma',
        'epsilon', 'gamma', 'delta', 'tau_default',
        'run_quantum', 'tag', 'outdir',
    ]

    # Split by (mem_tier, run_quantum) so each submit is uniform.
    tiers: dict[tuple[int, int], list] = {}
    for r in rows:
        key = (mem_tier_gb(int(r['n_trk']), bool(int(r['run_quantum']))),
               int(r['run_quantum']))
        tiers.setdefault(key, []).append(r)

    written = {}
    for (mem, runq), group in sorted(tiers.items()):
        kind = 'Q' if runq else 'C'
        csv_path = csv_dir / f"seg14e_{kind}_mem{mem}.csv"
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            # HTCondor queue-from-CSV: no header line.
            for r in group:
                w.writerow(r)
        written[(mem, runq)] = (csv_path, len(group))
        print(f"[gen] kind={kind} mem={mem:3d} GB -> {len(group):4d} rows -> {csv_path.name}")

    print(f"[gen] total {sum(v[1] for v in written.values())} rows")
    return written


if __name__ == '__main__':
    emit(HERE / "results" / "seg14e_calc_eps",
         HERE / "params" / "seg14e_calc_eps")
