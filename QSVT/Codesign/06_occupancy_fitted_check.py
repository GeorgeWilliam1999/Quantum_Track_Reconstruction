#!/usr/bin/env python3
"""Does the occupancy verdict survive a FITTED response? (George's review, 2026-07-19)

Paper review question (item 12 on the draft): the fork no-go dissolved once the
response was refitted to the modified spectrum — is the occupancy no-go the same
artefact?  The occupancy studies measured the notch (1BQF) and the classical
solver on A_occ, but never gave A_occ the fitted-polynomial treatment that
redeemed the fork.  This script closes that gap.

Method: the occupancy coupling 2α·B_all merges the compatibility graph into a
giant component, so the exact per-component eigendecomposition used by
04_fit_comb_to_measured_spectrum.py is unavailable.  But a degree-d fit does not
need the spectrum: with moment vectors m_k = T_k(X) b (X the affinely rescaled
Hamiltonian, one sparse matvec pair per k), any degree-d response gives
amplitudes x(c) = Σ_k c_k m_k, LINEAR in the coefficients, so the fit
    min_c  Σ_false x_s(c)²  +  μ Σ_true (x_s(c) − x_s^cls)²
is a ridge-regularised least-squares in d+1 dimensions, solved exactly.  The
same coefficient-space fit is run on the BASE system so the comparison is
machinery-identical: only the Hamiltonian differs.

Configs: heavy T=200, reps 0-2, alpha in {0.05, 0.3} (0.3 = the classical
benchmark value of the occupancy studies), degrees 40..960.

Outputs: outputs/fork_noisy/occupancy_fitted_check.{csv,log}
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp  # noqa: E402
import dp_terms  # noqa: E402
from dp_matrix_characterisation import NOISES  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "fork_noisy"
OUT.mkdir(parents=True, exist_ok=True)
GAMMA, DELTA = 3.0, 1.0
T = 200
REPS = (0, 1, 2)
ALPHAS = (0.05, 0.3)
DEGS = (40, 80, 160, 240, 480, 960)
EFF_TARGETS = (0.99, 0.985, 0.98, 0.97, 0.95)
MUS = np.geomspace(1e-4, 1e3, 15)
MU_SELECT_EFF = 0.985

LOG = open(OUT / "occupancy_fitted_check.log", "w")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


def far_at_eff(x, truth, targets=EFF_TARGETS):
    x = np.abs(np.asarray(x).ravel())
    truth = np.asarray(truth, bool)
    ts = np.sort(x[truth])[::-1]
    nt = len(ts)
    out = {}
    for tgt in targets:
        k = min(int(np.ceil(tgt * nt)), nt)
        tau = ts[k - 1] * (1 - 1e-12) if ts[k - 1] > 0 else 0.0
        act = x > tau
        e = int((act & truth).sum()) / nt
        f = int((act & ~truth).sum()) / max(int(act.sum()), 1)
        out[tgt] = (e, f, float(tau))
    return out


def cheb_moments(A, b, lo, hi, dmax):
    """m_k = T_k(X) b for k=0..dmax with X = (A - c0)/c1 mapped onto [-1,1]."""
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    n = A.shape[0]
    M = np.empty((n, dmax + 1))
    M[:, 0] = b
    M[:, 1] = (A @ b - c0 * b) / c1
    for k in range(2, dmax + 1):
        M[:, k] = 2.0 * ((A @ M[:, k - 1]) - c0 * M[:, k - 1]) / c1 - M[:, k - 2]
    return M, (c0, c1)


def fit_degree(M, truth, x_cls, deg):
    """Exact coefficient-space fit at degree deg; mu selected at eff 0.985."""
    Md = M[:, :deg + 1]
    MT, MF = Md[truth], Md[~truth]
    GF = MF.T @ MF
    GT = MT.T @ MT
    bT = MT.T @ x_cls[truth]
    reg = 1e-10 * np.trace(GF) / (deg + 1) * np.eye(deg + 1)
    best = None
    for mu in MUS:
        c = np.linalg.solve(GF + mu * GT + reg, mu * bT)
        x = Md @ c
        fe = far_at_eff(x, truth, (MU_SELECT_EFF,))[MU_SELECT_EFF]
        if best is None or fe[1] < best[0]:
            best = (fe[1], mu, x)
    return best  # (far@985, mu, amplitudes)


def run_system(tag, A, b, truth, rep):
    tt = time.time()
    n = A.shape[0]
    lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                     tol=1e-7, maxiter=5000)[0])
    hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                     tol=1e-7, maxiter=5000)[0])
    xm, info = minres(A.tocsc(), b, rtol=1e-10, maxiter=20000)
    x_cls = np.abs(np.asarray(xm).ravel())
    M, dom = cheb_moments(A.tocsr(), b, lo, hi, max(DEGS))
    row = dict(system=tag, rep=rep, lam_min=lo, lam_max=hi, span=hi - lo,
               n=n, n_true=int(truth.sum()))
    fe = far_at_eff(x_cls, truth)
    for tgt, (e_, f_, _) in fe.items():
        row[f"far_cls_e{int(tgt*1000)}"] = f_
    for deg in DEGS:
        f985, mu, x = fit_degree(M, truth, x_cls, deg)
        fe = far_at_eff(x, truth)
        for tgt, (e_, f_, _) in fe.items():
            row[f"far_fit{deg}_e{int(tgt*1000)}"] = f_
        row[f"mu_fit{deg}"] = mu
    best985 = min(row[f"far_fit{d}_e985"] for d in DEGS)
    say(f"[{tag} rep{rep}] span {hi-lo:7.1f} | cls@985 "
        f"{row['far_cls_e985']:.3f} | fit@985 by degree "
        + " ".join(f"d{d}:{row[f'far_fit{d}_e985']:.3f}" for d in DEGS)
        + f" | best {best985:.3f} [{time.time()-tt:.0f}s]")
    return row


def main():
    t00 = time.time()
    nz = NOISES["heavy"]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    say(f"[setup] heavy formula eps={eps*1e3:.3f} mrad, T={T}, "
        f"alphas={ALPHAS}, degs={DEGS}")
    rows = []
    for rep in REPS:
        ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
        ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                   gamma=GAMMA, delta=DELTA)
        truth = np.asarray(qp.truth_from_event(ev), bool)
        A0, b0, _, _ = dp_terms.dp_system(ham, gamma=GAMMA, delta=DELTA)
        rows.append(run_system("base", A0, b0, truth, rep))
        for alpha in ALPHAS:
            Ao, bo, _, info = dp_terms.dp_system(ham, alpha=alpha,
                                                 gamma=GAMMA, delta=DELTA)
            rows.append(run_system(f"occ_a{alpha}", Ao, bo, truth, rep))
        pd.DataFrame(rows).to_csv(OUT / "occupancy_fitted_check.csv",
                                  index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "occupancy_fitted_check.csv", index=False)
    say("\n=== VERDICT (rep-means, matched efficiency) ===")
    agg = df.groupby("system").mean(numeric_only=True)
    for eff in (985, 980, 970):
        line = f"eff 0.{eff}: "
        for sysname in agg.index:
            best = min(agg.loc[sysname, f"far_fit{d}_e{eff}"] for d in DEGS)
            line += (f"{sysname}: cls {agg.loc[sysname, f'far_cls_e{eff}']:.3f} "
                     f"fit(best) {best:.3f} · ")
        say(line)
    say(f"[done] {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
