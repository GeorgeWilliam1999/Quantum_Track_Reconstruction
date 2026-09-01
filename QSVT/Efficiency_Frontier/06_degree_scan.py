#!/usr/bin/env python3
"""Stage 5 — the polynomial-degree scan on the modified Hamiltonians.

George (2026-09-01): two degree figures for the paper + write-up:
  (i)  degree vs segment efficiency (@ far <= 1%) and degree vs false rate
       (@ matched eff 0.99) per operator;
  (ii) the MARGINAL improvement per degree step (absolute change from one
       degree to the next) — expected sigmoidal.

Operators (the paper's five series, minus 1BQF which is degree-1 by
construction): base A, A + occupancy (alpha=0.05), A + bifurcation
(beta=0.5), A + both.  Regimes: clean / moderate / heavy at T=200,
reps 0-9.  Families: fitted_moment (refit per operator, per rep) and
comb_prod (production line-comb rebuilt on the operator's domain) over a
fine degree ladder; classical + matched 1BQF cosine as references.

Machinery identical to 04_heavy_frontier.py (matrix-free Chebyshev
moments; ridge fit selected by eff @ far <= 1%; matched-eff frontier
columns).  NEW: the winning fitted Chebyshev coefficient vector is saved
per (operator, rep, degree) so the response p(lambda) can be DRAWN
against the operator's populated spectrum (the fig-3 analogues).

Outputs: outputs/06_degree_scan.csv (appended after every system; resumes)
         outputs/cache/06_coef_{regime}_{setout}_rep{r}.npz
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
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                   # noqa: E402
import dp_terms                                              # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon           # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

T = 200
REPS = tuple(range(10))
GAMMA_D, DELTA = 3.0, 1.0
REGIMES = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01),
}
SETOUTS = [("base", {}),
           ("occ_a0.05", dict(occ_alpha=0.05)),
           ("fork_b0.5", dict(fork_beta=0.5)),
           ("occ0.05_fork0.5", dict(occ_alpha=0.05, fork_beta=0.5))]
FIT_DEGS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64,
            80, 84, 96, 120, 160)
COMB_DEGS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 120)
DMAX = max(FIT_DEGS)
MUS = np.geomspace(3e-3, 30.0, 13)
EFF_TARGETS = (0.97, 0.98, 0.99, 0.995)
FAR_TARGETS = (0.001, 0.01, 0.10)

OUTNAME = os.environ.get("EF_OUT", "06_degree_scan.csv")
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
    nx, nc = np.linalg.norm(x), np.linalg.norm(x_cls)
    out["cos_C"] = float(x @ np.abs(x_cls) / (nx * nc)) if nx * nc > 0 else np.nan
    return out


def cheb_moments(A, b, lo, hi, dmax):
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    M = np.empty((A.shape[0], dmax + 1))
    M[:, 0] = b
    M[:, 1] = (A @ b - c0 * b) / c1
    for k in range(2, dmax + 1):
        M[:, k] = 2.0 * ((A @ M[:, k - 1]) - c0 * M[:, k - 1]) / c1 - M[:, k - 2]
    return M


def fit_degree(M, truth, x_cls, deg):
    """Ridge fit in Chebyshev-moment space; returns (x, mu, coef)."""
    Md = M[:, :deg + 1]
    MT, MF = Md[truth], Md[~truth]
    GF, GT = MF.T @ MF, MT.T @ MT
    bT = MT.T @ x_cls[truth]
    reg = 1e-10 * np.trace(GF) / (deg + 1) * np.eye(deg + 1)
    best = None
    for mu in MUS:
        c = np.linalg.solve(GF + mu * GT + reg, mu * bT)
        x = np.abs(Md @ c)
        key = -frontier_metrics(x, truth, x_cls)["eff_f010"]
        if best is None or key < best[0]:
            best = (key, mu, x, c)
    return best[2], float(best[1]), best[3]


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
    tags = set(sys.argv[1:])
    setouts = [s for s in SETOUTS if not tags or s[0] in tags]
    regimes = (os.environ["EF_REGIMES"].split(",")
               if "EF_REGIMES" in os.environ else list(REGIMES))
    reps = tuple(int(r) for r in os.environ["EF_REPS"].split(",")) \
        if "EF_REPS" in os.environ else REPS

    done = set()
    if (OUT / OUTNAME).exists():
        prev = pd.read_csv(OUT / OUTNAME)
        done = set(zip(prev.regime, prev.setout, prev.rep))
        rows = prev.to_dict("records")
        say(f"[resume] {len(prev)} rows, {len(done)} systems already done")
    else:
        rows = []

    for regime in regimes:
        noise = REGIMES[regime]
        eps = float(compute_epsilon(noise["sigma_res"], noise["sigma_scatt"]))
        for tag, kn in setouts:
            alpha = float(kn.get("occ_alpha", 0.0))
            beta = float(kn.get("fork_beta", 0.0))
            kernel = kn.get("kernel", "step")
            s_prime = GAMMA_D + DELTA + 4.0 * alpha
            for rep in reps:
                if (regime, tag, rep) in done:
                    continue
                t0 = time.time()
                ev, _ = qp.ensure_event(n_trk=T, rep=rep, **noise)
                kw = dict(epsilon=eps, kernel=kernel, gamma=GAMMA_D,
                          delta=DELTA)
                if kernel == "erf":
                    kw["erf_sigma"] = eps / 3.0
                ham = qp.build_hamiltonian(ev, **kw)
                truth = np.asarray(qp.truth_from_event(ev), bool)
                A, b, tau_abs, _ = dp_terms.dp_system(
                    ham, beta=beta, eps_B=eps if beta else None, alpha=alpha,
                    gamma=GAMMA_D, delta=DELTA)
                A = A.tocsr()
                n = A.shape[0]
                lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                                 tol=1e-7, maxiter=5000)[0])
                hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                 tol=1e-7, maxiter=5000)[0])
                xm, _ = minres(A.tocsc(), b, rtol=1e-10, maxiter=20000)
                x_cls = np.abs(np.asarray(xm).ravel())
                coefs = {"lo": lo, "hi": hi, "eps": eps,
                         "truth": truth, "sol_C": x_cls.astype(np.float32)}

                def record(family, degree, x, extra=""):
                    fm = frontier_metrics(x, truth, x_cls)
                    rows.append(dict(setout=tag, regime=regime, T=T, rep=rep,
                                     family=family, degree=degree,
                                     gamma=GAMMA_D, occ_alpha=alpha,
                                     fork_beta=beta, kernel=kernel, eps=eps,
                                     n_seg=n, n_true=int(truth.sum()),
                                     lam_min=lo, lam_max=hi, **fm,
                                     notes=extra))

                record("classical_invA", np.nan, x_cls,
                       extra=f"reference only (tau_abs {tau_abs:.3f})")
                M = cheb_moments(A, b, lo, hi, DMAX)
                for d in FIT_DEGS:
                    x, mu, c = fit_degree(M, truth, x_cls, d)
                    record("fitted_moment", d, x, extra=f"mu={mu:.3g}")
                    coefs[f"fitted_c_d{d}"] = c
                del M
                pad = 0.05 * (hi - lo) + 0.2
                dom = (lo - pad, hi + pad)
                for d in COMB_DEGS:
                    record("comb_prod", d,
                           cheb_apply(design_line_comb_inverse(
                               degree=d, s=s_prime, domain=dom), A, b))
                gr = np.linspace(dom[0], dom[1], 4000)
                u = (2 * gr - (dom[0] + dom[1])) / (dom[1] - dom[0])
                cf = npcheb.chebfit(
                    u, np.abs(np.cos(np.pi * gr / (2 * s_prime))), 60)
                p1b = npcheb.Chebyshev(cf, domain=list(dom))
                record("onebqf_cos", 1, cheb_apply(p1b, A, b))

                np.savez_compressed(
                    CACHE / f"06_coef_{regime}_{tag}_rep{rep}.npz", **coefs)
                bf = [r for r in rows if r["rep"] == rep
                      and r["setout"] == tag and r["regime"] == regime
                      and r["family"] == "fitted_moment"]
                say(f"[{regime} {tag} rep{rep}] n={n:,} span {hi-lo:.1f} | "
                    f"best fitted eff@far1% "
                    f"{max(r['eff_f010'] for r in bf):.4f} "
                    f"({time.time() - t0:.0f}s)")
                pd.DataFrame(rows).to_csv(OUT / OUTNAME, index=False)

    say("done.")


if __name__ == "__main__":
    main()
