#!/usr/bin/env python3
"""Internal check (not for paper 1's text): does the QSVT recover its efficiency at a low
false rate when the Denby-Peterson terms are added to the Hamiltonian?

Operators, exactly as the chapter's efficiency-frontier campaign builds them
(Efficiency_Frontier/06_degree_scan.py, dp_terms.dp_system):
    base          A = sI - C
    fork          A + beta B_fork(eps_B = eps),               beta  = 0.5
    occupancy     A + 4 alpha I + 2 alpha B_all, b + 4 alpha,  alpha = 0.05
    both          both terms
The QSVT is the production comb (degree 40, four passes, half-width 0.18) designed at
s' = s + 4 alpha, on the exact spectral bounds of the modified operator, padded as in the
campaign. No coefficients are refit. The classical solver runs on the same modified
operator (MINRES), so the comparison is fair on every operator.

Readings: classical at 99 % efficiency (false rate); QSVT at false rate <= 1 % and <= 5 %
(efficiency), plus both solvers' false rate at 97 % and 99 % efficiency.

Checks: on T = 50 rep 0 of each configuration and operator, the Chebyshev recursion is
compared with the polynomial applied through a dense eigendecomposition, and MINRES with
a dense solve; the run stops if either differs by more than 1e-8.

Occupancy is run at T <= 400 only: B_all couples every same-role pair at a hit, which grows
as T^2 per hit (the chapter ran it at T = 200).
Output: outputs_terms/terms_check.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp                                          # noqa: E402
import dp_terms                                                     # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon                  # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs_terms")
os.makedirs(OUT, exist_ok=True)
GAMMA, DELTA = 3.0, 1.0
CONFIGS = [("5 um, 0.15 mrad", 0.005, 1.5e-4), ("15 um, 0.45 mrad", 0.015, 4.5e-4)]
OPS = [("base", 0.0, 0.0), ("fork", 0.5, 0.0), ("occupancy", 0.0, 0.05), ("both", 0.5, 0.05)]
TS = (200, 400, 1000)
REPS = (0, 1)


def system(T, rep, sr, ss, beta, alpha):
    eps = float(compute_epsilon(sr, ss))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=ss, sigma_res=sr, phi_max=0.2, hit_ineff=0.01)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=GAMMA, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, info = dp_terms.dp_system(ham, beta=beta, eps_B=eps if beta else None, alpha=alpha,
                                       gamma=GAMMA, delta=DELTA)
    return A.tocsr(), np.asarray(b, float), truth, info


def cheb_apply(poly, A, b):
    lo, hi = poly.domain
    sc, sh = 2.0 / (hi - lo), (lo + hi) / (hi - lo)
    c = poly.coef
    t_prev, y = b, c[0] * b
    t_cur = sc * (A @ b) - sh * b
    y = y + c[1] * t_cur
    for k in range(2, len(c)):
        t_next = 2.0 * (sc * (A @ t_cur) - sh * t_cur) - t_prev
        t_prev, t_cur = t_cur, t_next
        y = y + c[k] * t_cur
    return np.abs(y)


def comb_for(A, alpha, dense_eigs=None):
    if dense_eigs is not None:
        lo, hi = float(dense_eigs.min()), float(dense_eigs.max())
    else:
        lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-7, maxiter=5000)[0])
        hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-7, maxiter=5000)[0])
    pad = 0.05 * (hi - lo) + 0.2
    s_prime = GAMMA + DELTA + 4.0 * alpha
    return design_line_comb_inverse(degree=40, s=s_prime, hw=0.18, domain=(lo - pad, hi + pad)), lo, hi


def curve(x, t):
    o = np.argsort(-x, kind="stable")
    xs, ts = x[o], t[o]
    tp, fp = np.cumsum(ts), np.cumsum(~ts)
    last = np.r_[xs[1:] != xs[:-1], True]
    return tp[last] / t.sum(), fp[last] / (tp[last] + fp[last])


def readings(x, t):
    eff, far = curve(x, t)
    r = {}
    for e in (0.97, 0.99):
        k = min(np.searchsorted(eff, e), len(eff) - 1)
        r[f"far@{e}"] = far[k]
    for fm in (0.01, 0.05):
        ok = far <= fm
        r[f"eff@far{fm}"] = eff[ok].max() if ok.any() else 0.0
    return r


def small_check():
    for cname, sr, ss in CONFIGS:
        for oname, beta, alpha in OPS:
            A, b, truth, _ = system(50, 0, sr, ss, beta, alpha)
            Ad = A.toarray()
            assert np.allclose(np.diag(Ad), GAMMA + DELTA + 4 * alpha), "diagonal not constant"
            w, V = np.linalg.eigh(Ad)
            comb, _, _ = comb_for(A, alpha, dense_eigs=w)
            x_dense = np.abs(V @ (comb(w) * (V.T @ b)))
            x_rec = cheb_apply(comb, A, b)
            xm, _ = minres(A.tocsc(), b, rtol=1e-13, maxiter=100000)
            xs = np.linalg.solve(Ad, b)
            d1, d2 = np.max(np.abs(x_dense - x_rec)), np.max(np.abs(xm - xs))
            print(f"[check T50] {cname} {oname}: recursion vs dense {d1:.1e}, minres vs dense {d2:.1e}",
                  flush=True)
            if d1 > 1e-8 or d2 > 1e-8:
                sys.exit(f"check failed: {cname} {oname}")


def main():
    small_check()
    rows = []
    for cname, sr, ss in CONFIGS:
        for T in TS:
            for oname, beta, alpha in OPS:
                if alpha and T > 400:
                    continue
                for rep in REPS:
                    t0 = time.time()
                    A, b, truth, info = system(T, rep, sr, ss, beta, alpha)
                    comb, lo, hi = comb_for(A, alpha)
                    xm, info_m = minres(A.tocsc(), b, rtol=1e-10, maxiter=50000)
                    xq = cheb_apply(comb, A, b)
                    r = dict(config=cname, T=T, operator=oname, rep=rep, beta=beta, alpha=alpha,
                             nnz=int(A.nnz), lam_min=lo, lam_max=hi, minres_info=int(info_m))
                    r.update({f"classical {k}": v for k, v in readings(np.abs(xm), truth).items()})
                    r.update({f"QSVT {k}": v for k, v in readings(xq, truth).items()})
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(os.path.join(OUT, "terms_check.csv"), index=False)
                    print(f"{cname} T{T} {oname} rep{rep} nnz {A.nnz:,} | classical far@.99 "
                          f"{r['classical far@0.99']:.3f} eff@1% {r['classical eff@far0.01']:.3f} | QSVT "
                          f"eff@1% {r['QSVT eff@far0.01']:.3f} eff@5% {r['QSVT eff@far0.05']:.3f} "
                          f"far@.97 {r['QSVT far@0.97']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
