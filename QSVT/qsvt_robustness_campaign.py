"""qsvt_robustness_campaign.py — WP1 of the paper plan: comb-QSVT noise robustness.

One-axis-at-a-time noise sweeps around the standard fixed-eps configuration
(sigma_scatt=1e-4, sigma_res=0, hit_ineff=0), gamma=3, eps=2 mrad, step kernel:

    scatter sweep : sigma_scatt in {1e-4, 3e-4, 5e-4, 8e-4}
    resolution    : sigma_res   in {0, 0.005, 0.01}
    hit drop      : hit_ineff   in {0, 0.01, 0.02, 0.05}

at T in {100, 400}, reps 0-2, solving BOTH classical (MINRES partner for the
signal rescale) and qsvt (gamma-aware line-comb, qp.solve_qsvt). Everything goes
through the common qtrk_pipeline functions: ensure_event (events generated into
the store on demand), build_hamiltonian, sol_key / save_solution (dedup against
any existing solutions for the same event/ham), rows registered in
manifest/solutions.csv under study='QSVT_robustness', and the metrics view
rebuilt for that study and merged into manifest/metrics.csv.

Usage:  Q_env python qsvt_robustness_campaign.py [--dry-run] [--overwrite]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import os
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402

STUDY = "QSVT_robustness"
EPS = 0.002
GAMMA, DELTA = 3.0, 1.0
DEGREE = 40
PAD = 0.10

BASE = dict(sigma_scatt=1e-4, sigma_res=0.0, hit_ineff=0.0)
CONFIGS = [dict(BASE)]
for ss in (3e-4, 5e-4, 8e-4):
    CONFIGS.append(dict(BASE, sigma_scatt=ss))
for sr in (0.005, 0.01):
    CONFIGS.append(dict(BASE, sigma_res=sr))
for hi in (0.01, 0.02, 0.05):
    CONFIGS.append(dict(BASE, hit_ineff=hi))
TS = (100, 400)
REPS = (0, 1, 2)


def spectral_bounds_for(ham) -> tuple[float, float]:
    s = GAMMA + DELTA
    n = ham.A.shape[0]
    C = (s * sp.identity(n, format="csr") - ham.A.tocsr())
    try:
        rho = float(eigsh(C, k=1, which="LA", return_eigenvectors=False,
                          maxiter=3000, tol=1e-4)[0])
    except Exception:
        rho = float(np.asarray(abs(C).sum(axis=1)).ravel().max())
    return s - rho - PAD, s + rho + PAD


def main(dry_run=False, overwrite=False) -> int:
    md = qp.manifest_dir()
    sols = pd.read_csv(md / "solutions.csv")
    s = GAMMA + DELTA
    new_rows, done, total, t00 = [], 0, len(CONFIGS) * len(TS) * len(REPS), time.time()
    for cfg in CONFIGS:
        for T in TS:
            for rep in REPS:
                ev, ekey = qp.ensure_event(n_trk=T, rep=rep, phi_max=0.2, **cfg)
                ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step",
                                           gamma=GAMMA, delta=DELTA)
                hkey = qp.ham_key(epsilon=EPS, kernel="step", gamma=GAMMA,
                                  delta=DELTA, eps_provenance="set")
                truth = np.asarray(qp.truth_from_event(ev), bool)
                common = dict(event_key=ekey, ham_key=hkey, study=STUDY,
                              studies=STUDY, n_trk=T, rep=rep,
                              sigma_scatt=cfg["sigma_scatt"], sigma_res=cfg["sigma_res"],
                              phi_max=0.2, hit_ineff=cfg["hit_ineff"], ghost_rate=0.0,
                              epsilon=EPS, eps_provenance="set", kernel="step",
                              erf_sigma=0.0, gamma=GAMMA, delta=DELTA, shots=0)
                # ---- classical partner --------------------------------------
                kC = qp.sol_key(ekey, hkey, "classical")
                new_rows.append(dict(common, sol_key=kC, solver="classical",
                                     device="na", readout="na"))
                if not dry_run and (overwrite or not qp.solution_exists(kC)):
                    solC, tC = qp.solve_classical(ham)
                    qp.save_solution(kC, solC, event_key_=ekey, ham_key_=hkey,
                                     solver="classical", n_seg=int(ham.n_segments),
                                     n_true=int(truth.sum()), overwrite=overwrite,
                                     t_solve=tC, A_nnz=int(ham.A.nnz),
                                     epsilon=EPS, kernel="step", gamma=GAMMA, delta=DELTA)
                # ---- qsvt ----------------------------------------------------
                kQ = qp.sol_key(ekey, hkey, "qsvt", readout="statevector")
                new_rows.append(dict(common, sol_key=kQ, solver="qsvt",
                                     device="na", readout="statevector"))
                if not dry_run and (overwrite or not qp.solution_exists(kQ)):
                    lo, hi = spectral_bounds_for(ham)
                    dom = None
                    if lo < s - 3.8 or hi > s + 3.8:
                        half = max(s - lo, hi - s) + 0.5
                        dom = (s - half, s + half)
                    res = qp.solve_qsvt(ham, degree=DEGREE,
                                        spectral_bounds=(lo, hi), domain=dom)
                    qp.save_solution(kQ, res["sol"], event_key_=ekey, ham_key_=hkey,
                                     solver="qsvt", n_seg=int(ham.n_segments),
                                     n_true=int(truth.sum()), overwrite=overwrite,
                                     P_anc=res["P_anc"], n_sys=res["n_sys"],
                                     n_qubits=res["n_qubits"], degree=res["degree"],
                                     filter_design=res["filter_design"],
                                     t_solve=res["t_solve"], A_nnz=int(ham.A.nnz),
                                     epsilon=EPS, kernel="step", gamma=GAMMA, delta=DELTA)
                done += 1
                print(f"  [{done:3d}/{total}] ss={cfg['sigma_scatt']:.0e} "
                      f"sr={cfg['sigma_res']:.3f} drop={cfg['hit_ineff']:.2f} "
                      f"T={T:4d} rep={rep}  n={ham.n_segments}", flush=True)

    if not dry_run and new_rows:
        nr = pd.DataFrame(new_rows)
        merged = pd.concat([sols, nr], ignore_index=True)
        merged = merged.drop_duplicates(subset="sol_key", keep="first")
        shutil.copy(md / "solutions.csv", md / "solutions.csv.bak")
        merged.to_csv(md / "solutions.csv", index=False)
        print(f"[robustness] solutions.csv: {len(sols)} -> {len(merged)} rows")
        from qtrk_pipeline import build_metrics as bm
        df_new = bm.build(STUDY)
        mpath = md / "metrics.csv"
        old = pd.read_csv(mpath)
        out = pd.concat([old[old.study != STUDY], df_new], ignore_index=True)
        shutil.copy(mpath, md / "metrics.csv.bak")
        out.to_csv(mpath, index=False)
        print(f"[robustness] metrics.csv: {len(old)} -> {len(out)} rows; "
              f"{STUDY} by solver: {df_new.groupby('solver').size().to_dict()}")
    print(f"[robustness] total {time.time()-t00:.0f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(a.dry_run, a.overwrite))
