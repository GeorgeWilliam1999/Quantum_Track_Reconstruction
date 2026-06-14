#!/usr/bin/env python3
"""Top up Verify_new_results 1BQF (quantum) reps at T=700/1000 to match qsvt.

Background: the Verify_new_results 3-solver benchmark (classical / 1BQF / qsvt)
was capped at **1 rep** of 1BQF at T=700/1000 because the legacy Aer statevector
path OOM'd / was too slow there.  With the matrix-free 1BQF engine (default in
``helpers.solve_quantum_statevector``) those solves are seconds each, so we top
the 1BQF reps up to mirror the existing **qsvt** coverage exactly:

  - gamma=3 fixed-eps ('set', eps=2 mrad): reps 0..4   (qsvt has 5)
  - gamma=1,2 fixed-eps ('set'):           reps 0..2   (qsvt has 3)
  - both hit_ineff in {0.0, 0.01}, T in {700, 1000}

Each new 1BQF row reuses the SAME (event_key, ham_key) as the qsvt row at that
rep, so the 1BQF and qsvt points pair on the identical event/matrix A.  The
solve is matrix-free statevector (device label 'CPU', readout 'statevector').

This script is key-safe and idempotent:
  - it ONLY adds new sol_keys (quantum mirrors of qsvt rep>=1 rows); it never
    rebuilds the manifest (which would drop the campaign-registered qsvt rows);
  - run_shard-style: ensure_event(params) -> build_hamiltonian -> solve_quantum
    -> save_solution at the precomputed sol_key (skips any already on disk);
  - it appends the new rows to manifest/solutions.csv (with a .bak) so
    build_metrics.py will pick them up.

Run from the repo root with QTRK_STORE set and the qtrk_pipeline + lhcb_velo_toy
on PYTHONPATH (see DATA_GENERATION_GUIDE.md S5).  Idempotent: safe to re-run.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SHARED = Path(__file__).resolve().parents[1] / "_shared"
for p in (str(_SHARED), "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

import qtrk_pipeline as qp  # noqa: E402
from helpers import segment_truth_mask  # noqa: E402

HIGH_T = (700, 1000)


def build_new_rows(sol: pd.DataFrame) -> pd.DataFrame:
    """Mirror every qsvt row at T in HIGH_T, rep>=1, as a 1BQF (quantum) row."""
    src = sol[(sol.solver == "qsvt")
              & (sol.n_trk.isin(HIGH_T))
              & (sol.rep >= 1)].copy()
    existing = set(sol.sol_key)
    rows = []
    for r in src.itertuples(index=False):
        skey = qp.sol_key(r.event_key, r.ham_key, "quantum", "CPU", "statevector")
        if skey in existing:
            continue
        d = {c: getattr(r, c) for c in sol.columns}
        d["sol_key"] = skey
        d["solver"] = "quantum"
        d["device"] = "CPU"
        d["readout"] = "statevector"
        d["shots"] = 0
        rows.append(d)
    new = pd.DataFrame(rows, columns=list(sol.columns))
    return new.drop_duplicates("sol_key").reset_index(drop=True)


def solve_rows(new: pd.DataFrame) -> int:
    done = skipped = 0
    for r in new.itertuples(index=False):
        if qp.solution_exists(r.sol_key):
            skipped += 1
            continue
        ev, ekey = qp.ensure_event(
            n_trk=int(r.n_trk), rep=int(r.rep),
            sigma_scatt=float(r.sigma_scatt), sigma_res=float(r.sigma_res),
            phi_max=float(r.phi_max), hit_ineff=float(r.hit_ineff),
            ghost_rate=float(r.ghost_rate),
        )
        assert ekey == r.event_key, f"event_key mismatch {ekey} != {r.event_key}"
        ham = qp.build_hamiltonian(
            ev, epsilon=float(r.epsilon), kernel=str(r.kernel),
            erf_sigma=float(r.erf_sigma) if str(r.kernel) == "erf" else 1e-4,
            gamma=float(r.gamma), delta=float(r.delta),
        )
        truth = segment_truth_mask(ham)
        n_true = int(np.asarray(truth, bool).sum())
        t0 = time.time()
        qd = qp.solve_quantum(ham, device="CPU", readout="statevector", shots=0)
        qp.save_solution(
            r.sol_key, qd["sol"], event_key_=r.event_key, ham_key_=r.ham_key,
            solver="quantum", n_seg=int(ham.n_segments), n_true=n_true,
            A_nnz=int(ham.A.nnz), epsilon=float(r.epsilon),
            eps_provenance=str(r.eps_provenance), kernel=str(r.kernel),
            erf_sigma=float(r.erf_sigma), gamma=float(r.gamma), delta=float(r.delta),
            device="CPU", readout="statevector", shots=0,
            n_trk=int(r.n_trk), rep=int(r.rep), study=str(r.studies),
            P_anc=qd["P_anc"], n_qubits=qd["n_qubits"], t_solve=qd["t_solve"],
        )
        done += 1
        print(f"[done] T={r.n_trk} rep={r.rep} g={r.gamma} hi={r.hit_ineff} "
              f"P={qd['P_anc']:.3e} t={time.time()-t0:.1f}s", flush=True)
    return done, skipped


def main() -> int:
    solcsv = qp.manifest_dir() / "solutions.csv"
    sol = pd.read_csv(solcsv)
    new = build_new_rows(sol)
    print(f"[topup] {len(new)} new 1BQF rows to add (T in {HIGH_T}, rep>=1)", flush=True)
    if len(new) == 0:
        print("[topup] nothing to do")
        return 0
    print(new.groupby(["n_trk", "gamma", "hit_ineff"]).rep.agg(
        n="count", reps=lambda s: sorted(s.unique())).to_string(), flush=True)

    done, skipped = solve_rows(new)
    print(f"[topup] solved={done} skipped(on-disk)={skipped}", flush=True)

    # append the new rows to the manifest (idempotent: drop any sol_key already there)
    sol_now = pd.read_csv(solcsv)
    add = new[~new.sol_key.isin(set(sol_now.sol_key))]
    if len(add):
        shutil.copy(solcsv, str(solcsv) + ".bak")
        pd.concat([sol_now, add], ignore_index=True).to_csv(solcsv, index=False)
        print(f"[topup] appended {len(add)} rows to {solcsv} (.bak written)", flush=True)
    else:
        print("[topup] all rows already in solutions.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
