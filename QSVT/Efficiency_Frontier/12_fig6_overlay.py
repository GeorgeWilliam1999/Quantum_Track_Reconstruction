#!/usr/bin/env python3
"""Stage 7 — the modified-operator series AT THE FIGURE-6 CONFIGURATION.

George (2026-09-01): "plot the segment efficiency and false acceptance
rate for the QSVT versions with different Hamiltonians WITH the curves in
figure 6" — i.e. the paper's clean-benchmark campaign: FIXED epsilon =
2 mrad (not the formula), clean noise (sigma_scatt=1e-4, sigma_res=0,
NO hit drop), gamma=3, delta=1, step kernel, n_trk up to 1000, 3 reps.

This driver computes, on the SAME stored events as that campaign
(qtrk_pipeline.ensure_event, clean config) with the Hamiltonian built at
the SAME fixed epsilon=0.002:
    operators: base A / +occupancy(0.05) / +fork(0.5) / +both
    families:  fitted_moment d=40 (refit per operator per rep),
               comb_prod d=40 (production line comb on the operator's
               domain), matched 1BQF cosine (degree-60 Chebyshev),
               classical MINRES reference (fixed tau + frontier columns)
so every series can be drawn at ONE stated working point (the matched
99%-efficiency point) alongside figure 6's own store curves.  The base
comb_prod row doubles as the anchor against the store campaign's qsvt
wp99 columns.

Machinery = 03_highT_frontier.py verbatim (matrix-free occupancy matvec
via the segment->hit incidence: A_occ = A + 2a(HsHs^T + HeHe^T), float32
moment matrix, resume-aware CSV).

Outputs: outputs/12_fig6_overlay.csv (appended per system; resumes)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.polynomial import chebyshev as npcheb
from scipy.sparse.linalg import LinearOperator, eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                   # noqa: E402
import dp_terms                                              # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

GAMMA_D, DELTA = 3.0, 1.0
EPS_FIXED = 0.002                       # figure 6's acceptance — NOT formula
CLEAN = dict(sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)
TS = (10, 50, 100, 200, 400, 700, 1000)
REPS = (0, 1, 2)                        # the campaign standard
SETOUTS = [("base", {}), ("occ_a0.05", dict(occ_alpha=0.05)),
           ("fork_b0.5", dict(fork_beta=0.5)),
           ("occ0.05_fork0.5", dict(occ_alpha=0.05, fork_beta=0.5))]
DMAX = 40
MUS = np.geomspace(3e-3, 30.0, 13)
EFF_TARGETS = (0.97, 0.98, 0.99, 0.995, 0.999)
FAR_TARGETS = (0.001, 0.01, 0.10)

OUTNAME = os.environ.get("EF_OUT", "12_fig6_overlay.csv")
LOG = open(OUT / (OUTNAME.replace(".csv", "") + ".log"), "a")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


def frontier_metrics(x, truth, x_cls):
    x = np.abs(np.asarray(x).ravel())
    truth = np.asarray(truth, bool)
    n_true = int(truth.sum())
    ts = np.sort(x[truth])[::-1]
    out = {}
    for e in EFF_TARGETS:
        k = min(n_true, max(1, int(np.ceil(e * n_true))))
        tau = ts[k - 1]
        act = x >= tau
        na = int(act.sum())
        tag = f"e{int(round(e * 1000)):03d}"
        out[f"far_{tag}"] = int((act & ~truth).sum()) / na if na else np.nan
        out[f"eff_{tag}"] = float((act & truth).sum()) / n_true
    taus = ts[np.arange(1, n_true + 1) - 1]
    xs, xt = np.sort(x), np.sort(x[truth])
    nact = len(x) - np.searchsorted(xs, taus, side="left")
    ntru = n_true - np.searchsorted(xt, taus, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    eff = ntru / n_true
    for ft in FAR_TARGETS:
        okm = far <= ft
        out[f"eff_f{int(round(ft * 1000)):03d}"] = (
            float(eff[okm].max()) if okm.any() else 0.0)
    return out


def cheb_moments(A, b, lo, hi, dmax, n):
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    M = np.empty((n, dmax + 1), dtype=np.float32)
    t_prev = np.asarray(b, dtype=np.float64)
    t_cur = (A @ t_prev - c0 * t_prev) / c1
    M[:, 0], M[:, 1] = t_prev, t_cur
    for k in range(2, dmax + 1):
        t_next = 2.0 * ((A @ t_cur) - c0 * t_cur) / c1 - t_prev
        t_prev, t_cur = t_cur, t_next
        M[:, k] = t_cur
    return M


def _gram64(X):
    d = X.shape[1]
    G = np.zeros((d, d))
    for i in range(0, X.shape[0], 500_000):
        c = X[i:i + 500_000].astype(np.float64)
        G += c.T @ c
    return G


def fit_degree(M, truth, x_cls, deg):
    Md = M[:, :deg + 1]
    MT, MF = Md[truth], Md[~truth]
    GF, GT = _gram64(MF), _gram64(MT)
    bT = MT.astype(np.float64).T @ x_cls[truth]
    reg = 1e-10 * np.trace(GF) / (deg + 1) * np.eye(deg + 1)
    best = None
    for mu in MUS:
        c = np.linalg.solve(GF + mu * GT + reg, mu * bT)
        x = np.abs(Md @ c.astype(np.float32)).astype(np.float64)
        fm = frontier_metrics(x, truth, x_cls)
        key = -fm["eff_f010"]
        if best is None or key < best[0]:
            best = (key, mu, x)
    return best[2], float(best[1])


def cheb_apply(poly, A, b):
    lo, hi = poly.domain
    sc, sh = 2.0 / (hi - lo), (lo + hi) / (hi - lo)
    c = poly.coef
    t_prev = b
    y = c[0] * t_prev
    if len(c) > 1:
        t_cur = sc * (A @ b) - sh * b
        y = y + c[1] * t_cur
        for k in range(2, len(c)):
            t_next = 2.0 * (sc * (A @ t_cur) - sh * t_cur) - t_prev
            t_prev, t_cur = t_cur, t_next
            y = y + c[k] * t_cur
    return np.abs(y)


def main():
    csv_path = OUT / OUTNAME
    rows, done = [], set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = set(map(tuple, prev[["setout", "T", "rep"]]
                       .drop_duplicates().itertuples(index=False)))
        say(f"resume: {len(rows)} rows, {len(done)} cells done")
    for T in TS:
        for tag, kn in SETOUTS:
            alpha = float(kn.get("occ_alpha", 0.0))
            beta = float(kn.get("fork_beta", 0.0))
            s_prime = GAMMA_D + DELTA + 4.0 * alpha
            for rep in REPS:
                if (tag, T, rep) in done:
                    continue
                t0 = time.time()
                ev, _ = qp.ensure_event(n_trk=T, rep=rep, **CLEAN)
                ham = qp.build_hamiltonian(ev, epsilon=EPS_FIXED,
                                           kernel="step", gamma=GAMMA_D,
                                           delta=DELTA)
                truth = np.asarray(qp.truth_from_event(ev), bool)
                A0, b, tau_abs, _ = dp_terms.dp_system(
                    ham, beta=beta, eps_B=(EPS_FIXED if beta else None),
                    alpha=0.0, gamma=GAMMA_D, delta=DELTA)
                A0 = A0.tocsr()
                n = A0.shape[0]
                if alpha > 0.0:
                    sh_ids = np.asarray(ham._segment_to_hit_ids)
                    nh = int(sh_ids.max()) + 1
                    Hs = sp.coo_matrix((np.ones(n), (np.arange(n),
                                        sh_ids[:, 0])), shape=(n, nh)).tocsr()
                    He = sp.coo_matrix((np.ones(n), (np.arange(n),
                                        sh_ids[:, 1])), shape=(n, nh)).tocsr()
                    HsT, HeT = Hs.T.tocsr(), He.T.tocsr()

                    def _mv(v, _A=A0, _a=alpha):
                        v = np.asarray(v, dtype=np.float64).ravel()
                        return (_A @ v + 2.0 * _a * (Hs @ (HsT @ v)
                                                     + He @ (HeT @ v)))
                    A = LinearOperator((n, n), matvec=_mv, rmatvec=_mv,
                                       dtype=np.float64)
                    b = b + 4.0 * alpha
                    tau_abs = (DELTA + 4 * alpha) / s_prime + 0.10
                else:
                    A = A0
                lo = float(eigsh(A, k=1, which="SA",
                                 return_eigenvectors=False,
                                 tol=1e-6, maxiter=5000)[0])
                hi = float(eigsh(A, k=1, which="LA",
                                 return_eigenvectors=False,
                                 tol=1e-6, maxiter=5000)[0])
                xm, _ = minres(A, b, rtol=1e-10, maxiter=40000)
                x_cls = np.abs(np.asarray(xm).ravel())

                def record(family, degree, x, extra=""):
                    fm = frontier_metrics(x, truth, x_cls)
                    rows.append(dict(setout=tag, regime="clean_eps2mrad",
                                     T=T, rep=rep, family=family,
                                     degree=degree, gamma=GAMMA_D,
                                     occ_alpha=alpha, fork_beta=beta,
                                     eps=EPS_FIXED, n_seg=n,
                                     n_true=int(truth.sum()),
                                     lam_min=lo, lam_max=hi, **fm,
                                     notes=extra))

                act = x_cls > tau_abs
                record("classical_invA", np.nan, x_cls,
                       extra=f"fixed-tau {tau_abs:.3f}")
                rows[-1]["eff_fixed_tau"] = float(
                    (act & truth).sum()) / max(int(truth.sum()), 1)
                rows[-1]["far_fixed_tau"] = float(
                    (act & ~truth).sum()) / max(int(act.sum()), 1)

                M = cheb_moments(A, b, lo, hi, DMAX, n)
                x, mu = fit_degree(M, truth, x_cls, DMAX)
                record("fitted_moment", DMAX, x, extra=f"mu={mu:.3g}")
                del M
                pad = 0.05 * (hi - lo) + 0.2
                dom = (lo - pad, hi + pad)
                record("comb_prod", 40,
                       cheb_apply(design_line_comb_inverse(
                           degree=40, s=s_prime, domain=dom), A, b))
                gr = np.linspace(dom[0], dom[1], 4000)
                u = (2 * gr - (dom[0] + dom[1])) / (dom[1] - dom[0])
                cf = npcheb.chebfit(
                    u, np.abs(np.cos(np.pi * gr / (2 * s_prime))), 60)
                record("onebqf_cos", 1,
                       cheb_apply(npcheb.Chebyshev(cf, domain=list(dom)),
                                  A, b))
                bf = [r for r in rows if r["rep"] == rep
                      and r["setout"] == tag and r["T"] == T
                      and r["family"] == "fitted_moment"]
                say(f"[{tag} T{T} rep{rep}] n={n:,} span {hi-lo:.1f} | "
                    f"fitted eff@far1% {bf[-1]['eff_f010']:.4f} "
                    f"far@eff0.99 {bf[-1]['far_e990']:.4f} "
                    f"({time.time()-t0:.0f}s)")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
    say("done.")


if __name__ == "__main__":
    main()
