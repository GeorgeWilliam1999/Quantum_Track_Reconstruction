"""qsvt_codesign_campaign.py — WP7 of the paper plan: the (beta, eps_B) scan of the
operator-response co-design  A' = A0 + beta * B_{eps_B}.

The deep dive showed the eps-windowed fork recovers contaminated clusters but is
blind to isolated floor bridges whose competitors lie OUTSIDE the fork window.
Here the fork window eps_B is decoupled from the acceptance epsilon and scanned:

    acceptance eps = 4 mrad (the bridge-floor-rich configuration), gamma = 3,
    T = 400, reps 0-2:
        beta in {0.25, 0.5, 1.0} x eps_B in {4, 12, 30} mrad   (+ beta=0 baseline)
    acceptance eps = 2 mrad (production config) safety point:
        beta = 0.5, eps_B = 30 mrad

Both solvers (classical MINRES partner + comb-QSVT via qp.solve_qsvt on the
modified system) are stored through the common qtrk_pipeline functions; the fork
provenance lives in the ham_key (fork_beta / fork_eps, normalised out at beta=0)
and in the manifest columns of study='QSVT_codesign'.

Usage:  Q_env python qsvt_codesign_campaign.py [--dry-run] [--overwrite]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import os
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402
import bif                  # noqa: E402  (fork_graph_eps)

STUDY = "QSVT_codesign"
GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
DEGREE = 40
T = 400
REPS = (0, 1, 2)

# (acceptance_eps, fork_beta, fork_eps)
CONFIGS = [(0.004, 0.0, None)]
for b in (0.25, 0.5, 1.0):
    for fe in (0.004, 0.012, 0.03):
        CONFIGS.append((0.004, b, fe))
CONFIGS.append((0.002, 0.5, 0.03))     # production-acceptance safety point


def bounds_for(A) -> tuple[float, float]:
    hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                     maxiter=3000, tol=1e-4)[0])
    lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                     maxiter=3000, tol=1e-4)[0])
    return lo - 0.05, hi + 0.05


def main(dry_run=False, overwrite=False) -> int:
    md = qp.manifest_dir()
    sols = pd.read_csv(md / "solutions.csv")
    new_rows, done, total, t00 = [], 0, len(CONFIGS) * len(REPS), time.time()
    for (eps_acc, beta, fork_eps) in CONFIGS:
        for rep in REPS:
            ev, ekey = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=1e-4,
                                       sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)
            ham0 = qp.build_hamiltonian(ev, epsilon=eps_acc, kernel="step",
                                        gamma=GAMMA, delta=DELTA)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            if beta:
                B = bif.fork_graph_eps(ham0._segment_to_hit_ids,
                                       ham0._segment_vectors, fork_eps)
                A = (ham0.A.tocsr() + beta * B).tocsr()
                nnzB = int(B.nnz)
            else:
                A = ham0.A.tocsr()
                nnzB = 0
            ham = SimpleNamespace(A=A, b=np.asarray(ham0.b, float).ravel(),
                                  n_segments=int(ham0.n_segments),
                                  gamma=GAMMA, delta=DELTA)
            hkey = qp.ham_key(epsilon=eps_acc, kernel="step", gamma=GAMMA,
                              delta=DELTA, eps_provenance="set",
                              fork_beta=beta, fork_eps=fork_eps)
            common = dict(event_key=ekey, ham_key=hkey, study=STUDY,
                          studies=STUDY, n_trk=T, rep=rep, sigma_scatt=1e-4,
                          sigma_res=0.0, phi_max=0.2, hit_ineff=0.0,
                          ghost_rate=0.0, epsilon=eps_acc, eps_provenance="set",
                          kernel="step", erf_sigma=0.0, gamma=GAMMA,
                          delta=DELTA, fork_beta=beta,
                          fork_eps=(fork_eps if fork_eps is not None else 0.0),
                          shots=0)
            # classical partner on A'
            kC = qp.sol_key(ekey, hkey, "classical")
            new_rows.append(dict(common, sol_key=kC, solver="classical",
                                 device="na", readout="na"))
            if not dry_run and (overwrite or not qp.solution_exists(kC)):
                solC, tC = qp.solve_classical(ham)
                qp.save_solution(kC, solC, event_key_=ekey, ham_key_=hkey,
                                 solver="classical", n_seg=ham.n_segments,
                                 n_true=int(truth.sum()), overwrite=overwrite,
                                 t_solve=tC, A_nnz=int(A.nnz),
                                 epsilon=eps_acc, kernel="step", gamma=GAMMA,
                                 delta=DELTA, fork_beta=beta,
                                 fork_eps=(fork_eps or 0.0), B_nnz=nnzB)
            # comb-QSVT on A'
            kQ = qp.sol_key(ekey, hkey, "qsvt", readout="statevector")
            new_rows.append(dict(common, sol_key=kQ, solver="qsvt",
                                 device="na", readout="statevector"))
            if not dry_run and (overwrite or not qp.solution_exists(kQ)):
                lo, hi = bounds_for(A)
                # margin must exceed the QSVT class's internal 2%-of-span pad
                half = max(S - lo, hi - S) + 0.05 * (hi - lo) + 0.2
                dom = (S - half, S + half) if half > 3.8 else None
                # comb degree scales with the domain (resolution d ~ 2W/h);
                # the degree cost of wide fork windows IS part of the result
                deg = int(min(127, max(DEGREE, np.ceil(2 * half / 0.30))))
                res = qp.solve_qsvt(ham, degree=deg,
                                    spectral_bounds=(lo, hi), domain=dom)
                qp.save_solution(kQ, res["sol"], event_key_=ekey, ham_key_=hkey,
                                 solver="qsvt", n_seg=ham.n_segments,
                                 n_true=int(truth.sum()), overwrite=overwrite,
                                 P_anc=res["P_anc"], n_sys=res["n_sys"],
                                 n_qubits=res["n_qubits"], degree=res["degree"],
                                 filter_design=res["filter_design"],
                                 t_solve=res["t_solve"], A_nnz=int(A.nnz),
                                 epsilon=eps_acc, kernel="step", gamma=GAMMA,
                                 delta=DELTA, fork_beta=beta,
                                 fork_eps=(fork_eps or 0.0), B_nnz=nnzB)
            done += 1
            print(f"  [{done:2d}/{total}] eps_acc={eps_acc*1e3:.0f}mrad "
                  f"beta={beta:4.2f} eps_B={(fork_eps or 0)*1e3:4.0f}mrad rep={rep} "
                  f" nnz(B)={nnzB}", flush=True)

    if not dry_run and new_rows:
        nr = pd.DataFrame(new_rows)
        merged = pd.concat([sols, nr], ignore_index=True)
        merged = merged.drop_duplicates(subset="sol_key", keep="first")
        shutil.copy(md / "solutions.csv", md / "solutions.csv.bak")
        merged.to_csv(md / "solutions.csv", index=False)
        print(f"[codesign] solutions.csv: {len(sols)} -> {len(merged)} rows")
        from qtrk_pipeline import build_metrics as bm
        df_new = bm.build(STUDY)
        mpath = md / "metrics.csv"
        old = pd.read_csv(mpath)
        out = pd.concat([old[old.study != STUDY], df_new], ignore_index=True)
        shutil.copy(mpath, md / "metrics.csv.bak")
        out.to_csv(mpath, index=False)
        print(f"[codesign] metrics.csv: {len(old)} -> {len(out)} rows; "
              f"{STUDY} by solver: {df_new.groupby('solver').size().to_dict()}")
    print(f"[codesign] total {time.time()-t00:.0f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    raise SystemExit(main(a.dry_run, a.overwrite))
