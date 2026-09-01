#!/usr/bin/env python3
"""Stage 5b — matched 1BQF cosine reference at high T, all regimes, base A.

The fig-6-style metrics-vs-T figure needs a 1BQF series computed with the
SAME events and machinery as the QSVT set-outs (the 2026-08-26 baseline
correction: never compare against store rows at a different T or pooled
set-outs).  The store carries real 1BQF amplitudes only for clean/moderate
and not at heavy; this driver computes the matched reference
f(lambda) = |cos(pi lambda / 2 s)| as a degree-60 Chebyshev on base A for
T in {400, 700, 1000} x {clean, moderate, heavy}, reps (5, 3, 3) — the same
grid as outputs/03_highT_frontier.csv, so rows merge cleanly.

Families: classical_invA (reference), onebqf_cos, comb_prod d=40 (sanity
cross-check against the 03 rows — must reproduce).

Outputs: outputs/06b_highT_onebqf.csv (appended after every system; resumes)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
GAMMA_D, DELTA = 3.0, 1.0
REGIMES = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01),
}
TS = (400, 700, 1000)
REPS_BY_T = {400: (0, 1, 2, 3, 4), 700: (0, 1, 2), 1000: (0, 1, 2)}
EFF_TARGETS = (0.97, 0.98, 0.99, 0.995, 0.999)
FAR_TARGETS = (0.001, 0.01, 0.10)

OUTNAME = os.environ.get("EF_OUT", "06b_highT_onebqf.csv")
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
    s_prime = GAMMA_D + DELTA
    done = set()
    if (OUT / OUTNAME).exists():
        prev = pd.read_csv(OUT / OUTNAME)
        done = set(zip(prev.regime, prev.T_, prev.rep)) \
            if "T_" in prev.columns else set(zip(prev.regime, prev["T"], prev.rep))
        rows = prev.to_dict("records")
        say(f"[resume] {len(prev)} rows already done")
    else:
        rows = []

    for regime, noise in REGIMES.items():
        eps = float(compute_epsilon(noise["sigma_res"], noise["sigma_scatt"]))
        for T in TS:
            for rep in REPS_BY_T[T]:
                if (regime, T, rep) in done:
                    continue
                t0 = time.time()
                ev, _ = qp.ensure_event(n_trk=T, rep=rep, **noise)
                ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                           gamma=GAMMA_D, delta=DELTA)
                truth = np.asarray(qp.truth_from_event(ev), bool)
                A, b, tau_abs, _ = dp_terms.dp_system(
                    ham, beta=0.0, eps_B=None, alpha=0.0,
                    gamma=GAMMA_D, delta=DELTA)
                A = A.tocsr()
                n = A.shape[0]
                lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                                 tol=1e-7, maxiter=5000)[0])
                hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                 tol=1e-7, maxiter=5000)[0])
                xm, _ = minres(A.tocsc(), b, rtol=1e-10, maxiter=20000)
                x_cls = np.abs(np.asarray(xm).ravel())

                def record(family, degree, x, extra=""):
                    fm = frontier_metrics(x, truth, x_cls)
                    rows.append(dict(setout="base", regime=regime, T=T,
                                     rep=rep, family=family, degree=degree,
                                     gamma=GAMMA_D, eps=eps, n_seg=n,
                                     n_true=int(truth.sum()), lam_min=lo,
                                     lam_max=hi, **fm, notes=extra))

                record("classical_invA", np.nan, x_cls,
                       extra=f"reference only (tau_abs {tau_abs:.3f})")
                pad = 0.05 * (hi - lo) + 0.2
                dom = (lo - pad, hi + pad)
                gr = np.linspace(dom[0], dom[1], 4000)
                u = (2 * gr - (dom[0] + dom[1])) / (dom[1] - dom[0])
                cf = npcheb.chebfit(
                    u, np.abs(np.cos(np.pi * gr / (2 * s_prime))), 60)
                p1b = npcheb.Chebyshev(cf, domain=list(dom))
                record("onebqf_cos", 1, cheb_apply(p1b, A, b))
                record("comb_prod", 40,
                       cheb_apply(design_line_comb_inverse(
                           degree=40, s=s_prime, domain=dom), A, b))

                r1b = rows[-2]
                say(f"[{regime} T{T} rep{rep}] n={n:,} | 1BQF "
                    f"eff@far1% {r1b['eff_f010']:.4f} "
                    f"far@eff0.99 {r1b['far_e990']:.4f} "
                    f"({time.time() - t0:.0f}s)")
                pd.DataFrame(rows).to_csv(OUT / OUTNAME, index=False)

    say("done.")


if __name__ == "__main__":
    main()
