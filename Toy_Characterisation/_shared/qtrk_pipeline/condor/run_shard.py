#!/usr/bin/env python3
"""
Manifest-driven Condor worker: process one shard of the qtrk manifest.

A shard is a CSV slice of either ``manifest/events.csv`` (event-gen mode) or
``manifest/solutions.csv`` (solve mode); the mode is detected from the columns.
Both modes are fully idempotent (skip work already on disk), so jobs are safe to
re-run, and event generation is race-safe (atomic writes, content-addressed).

Usage:  run_shard.py <shard.csv>
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# qtrk_pipeline (+ _shared/helpers) and lhcb_velo_toy come in via PYTHONPATH set
# by worker.sh; add the package parent defensively.
_PKG = Path(__file__).resolve().parents[1]
_SHARED = _PKG.parent
for p in (str(_SHARED), "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

import qtrk_pipeline as qp  # noqa: E402


def process_events(df: pd.DataFrame) -> tuple[int, int]:
    done = skipped = 0
    for r in df.itertuples(index=False):
        path = qp.event_path(r.event_key)
        if path.exists():
            skipped += 1
            continue
        qp.ensure_event(
            n_trk=int(r.n_trk), rep=int(r.rep),
            sigma_scatt=float(r.sigma_scatt), sigma_res=float(r.sigma_res),
            phi_max=float(r.phi_max), hit_ineff=float(r.hit_ineff),
            ghost_rate=float(getattr(r, "ghost_rate", 0.0)),
        )
        done += 1
    return done, skipped


def _solve_one(r) -> None:
    from helpers import segment_truth_mask

    ev, _ = qp.ensure_event(
        n_trk=int(r.n_trk), rep=int(r.rep),
        sigma_scatt=float(r.sigma_scatt), sigma_res=float(r.sigma_res),
        phi_max=float(r.phi_max), hit_ineff=float(r.hit_ineff),
        ghost_rate=float(getattr(r, "ghost_rate", 0.0)),
    )
    ham = qp.build_hamiltonian(
        ev, epsilon=float(r.epsilon), kernel=str(r.kernel),
        erf_sigma=float(r.erf_sigma) if str(r.kernel) == "erf" else 1e-4,
        gamma=float(r.gamma), delta=float(r.delta),
    )
    truth = segment_truth_mask(ham)
    n_true = int(np.asarray(truth, bool).sum())
    A_nnz = int(ham.A.nnz)

    scal = dict(
        A_nnz=A_nnz, epsilon=float(r.epsilon),
        eps_provenance=str(r.eps_provenance), kernel=str(r.kernel),
        erf_sigma=float(r.erf_sigma), gamma=float(r.gamma), delta=float(r.delta),
        device=str(r.device), readout=str(r.readout), shots=int(r.shots),
        n_trk=int(r.n_trk), rep=int(r.rep), study=str(getattr(r, "studies", "")),
    )
    if str(r.solver) == "classical":
        sol, t = qp.solve_classical(ham)
        scal["t_solve"] = t
    elif str(r.solver) == "qsvt":
        lo, hi = qp.spectral_bounds_for(ham)
        s = float(r.gamma) + float(r.delta)
        dom = None
        if lo < s - 3.8 or hi > s + 3.8:  # rare: widen the comb design window
            half = max(s - lo, hi - s)
            dom = (s - half, s + half)
        res = qp.solve_qsvt(ham, degree=40, spectral_bounds=(lo, hi), domain=dom)
        sol = res["sol"]
        scal.update(P_anc=res["P_anc"], n_sys=res["n_sys"], n_qubits=res["n_qubits"],
                    degree=res["degree"], filter_design=res["filter_design"],
                    t_solve=res["t_solve"])
    else:
        qd = qp.solve_quantum(ham, device=str(r.device), readout=str(r.readout),
                              shots=int(r.shots))
        sol = qd["sol"]
        scal.update(P_anc=qd["P_anc"], n_qubits=qd["n_qubits"], t_solve=qd["t_solve"])

    qp.save_solution(
        r.sol_key, sol, event_key_=r.event_key, ham_key_=r.ham_key,
        solver=str(r.solver), n_seg=int(ham.n_segments), n_true=n_true, **scal,
    )


def process_solves(df: pd.DataFrame) -> tuple[int, int, int]:
    done = skipped = failed = 0
    for r in df.itertuples(index=False):
        if qp.solution_exists(r.sol_key):
            skipped += 1
            continue
        try:
            _solve_one(r)
            done += 1
        except Exception as exc:  # one bad point must not kill the shard
            failed += 1
            print(f"[FAIL] sol_key={r.sol_key} n_trk={r.n_trk} solver={r.solver}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    return done, skipped, failed


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_shard.py <shard.csv>", file=sys.stderr)
        return 2
    shard = Path(sys.argv[1])
    df = pd.read_csv(shard)
    t0 = time.time()
    print(f"[shard] {shard.name}: {len(df)} rows  store={qp.store_root()}", flush=True)

    if "sol_key" in df.columns:
        done, skipped, failed = process_solves(df)
        print(f"[shard] solves done={done} skipped={skipped} failed={failed} "
              f"t={time.time()-t0:.1f}s", flush=True)
        return 1 if failed and not done else 0
    else:
        done, skipped = process_events(df)
        print(f"[shard] events done={done} skipped={skipped} "
              f"t={time.time()-t0:.1f}s", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
