#!/usr/bin/env python3
"""Internal check (not paper 1 text): does a polynomial refit to the modified Hamiltonian
reclaim efficiency, on events it has never seen?

One (configuration, T, operator) per call:
    python 04_refit_heldout.py <config> <T> <operator>
configurations: 5um | 15um; operators: base | fork | occupancy | both
(fork beta = 0.5 with eps_B = eps; occupancy alpha = 0.05; built by dp_terms.dp_system, as in
the chapter's efficiency-frontier campaign).

Design
  * Ten events (reps 0-9): training = reps 0-4, test = reps 5-9. Nothing is chosen on test events.
  * One shared polynomial domain per (configuration, T, operator), fixed before the run from the
    spectral bounds measured in outputs_terms/terms_check.csv, widened by 10 % of the span + 0.3.
    Every event's own bounds (eigsh) must lie inside it, else the run stops.
  * Refit: the chapter's ridge fit in Chebyshev-moment space (Efficiency_Frontier/06_degree_scan.py,
    fit_degree): minimise |M_F c|^2 + mu |M_T c - x_cls,T|^2, with Gram matrices pooled over the
    training events; mu from the chapter's grid, chosen by the mean training efficiency at a false
    rate <= 1 %. Degrees 40 and 80 are both reported (not selected).
  * Compared on the same test events: the refit polynomial, the unfitted production comb
    (degree 40, s' = s + 4 alpha, each event's own padded bounds, as in the paper), and the
    classical solver on the same operator.
  * In-sample numbers for the refit (training events) are written too, so overfitting shows.
Check: on a T = 50 event of the same configuration and operator, the moment matrix applied to
random coefficients is compared with p(A)b from a dense eigendecomposition on the shared domain
rule; stop if the difference exceeds 1e-8.
Output: outputs_refit/<tag>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
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
OUT = os.path.join(HERE, "outputs_refit")
os.makedirs(OUT, exist_ok=True)
CONFIGS = {"5 um, 0.15 mrad": (0.005, 1.5e-4), "15 um, 0.45 mrad": (0.015, 4.5e-4)}
OPS = {"base": (0.0, 0.0), "fork": (0.5, 0.0), "occupancy": (0.0, 0.05), "both": (0.5, 0.05)}
GAMMA, DELTA = 3.0, 1.0
MUS = np.geomspace(3e-3, 30.0, 13)
DEGS = (40, 80)
TRAIN, TEST = range(0, 5), range(5, 10)


def system(T, rep, sr, ss, beta, alpha):
    eps = float(compute_epsilon(sr, ss))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=ss, sigma_res=sr, phi_max=0.2, hit_ineff=0.01)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=GAMMA, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, _ = dp_terms.dp_system(ham, beta=beta, eps_B=eps if beta else None, alpha=alpha,
                                    gamma=GAMMA, delta=DELTA)
    return A.tocsr(), np.asarray(b, float), truth


def moments(A, b, dom, dmax):
    c0, c1 = 0.5 * (dom[1] + dom[0]), 0.5 * (dom[1] - dom[0])
    M = np.empty((A.shape[0], dmax + 1))
    M[:, 0] = b
    M[:, 1] = (A @ b - c0 * b) / c1
    for k in range(2, dmax + 1):
        M[:, k] = 2.0 * ((A @ M[:, k - 1]) - c0 * M[:, k - 1]) / c1 - M[:, k - 2]
    return M


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


def readings(x, t):
    o = np.argsort(-x, kind="stable")
    xs, ts = x[o], t[o]
    tp, fp = np.cumsum(ts), np.cumsum(~ts)
    last = np.r_[xs[1:] != xs[:-1], True]
    eff, far = tp[last] / t.sum(), fp[last] / (tp[last] + fp[last])
    r = {}
    for e in (0.97, 0.99):
        r[f"far@{e}"] = far[min(np.searchsorted(eff, e), len(eff) - 1)]
    for fm in (0.01, 0.05):
        ok = far <= fm
        r[f"eff@far{fm}"] = eff[ok].max() if ok.any() else 0.0
    return r


def main():
    key, T, op = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    cname = {"5um": "5 um, 0.15 mrad", "15um": "15 um, 0.45 mrad"}.get(key, key)
    sr, ss = CONFIGS[cname]
    beta, alpha = OPS[op]
    tag = f"{cname.split(',')[0].replace(' ', '')}_T{T}_{op}"
    log = open(os.path.join(OUT, f"{tag}.log"), "a")

    def say(*a):
        print(*a, flush=True)
        print(*a, file=log, flush=True)

    tb = pd.read_csv(os.path.join(HERE, "outputs_terms", "terms_check.csv"))
    tb = tb[(tb.config == cname) & (tb["T"] == T) & (tb.operator == op)]
    assert len(tb), "no measured bounds for this case"
    lo, hi = float(tb.lam_min.min()), float(tb.lam_max.max())
    pad = 0.10 * (hi - lo) + 0.3
    dom = (lo - pad, hi + pad)
    say(f"=== {tag} shared domain {dom[0]:.3f} .. {dom[1]:.3f} ({time.strftime('%H:%M')})")

    # check on a small event: moments x random coefficients == dense p(A) b on the same domain rule
    As, bs, _ = system(50, 0, sr, ss, beta, alpha)
    w, V = np.linalg.eigh(As.toarray())
    dsm = (w.min() - 0.3, w.max() + 0.3)
    rng = np.random.default_rng(0)
    c = rng.normal(size=81) / np.arange(1, 82)
    Ms = moments(As, bs, dsm, 80)
    u = (w - 0.5 * (dsm[0] + dsm[1])) / (0.5 * (dsm[1] - dsm[0]))
    pw = np.polynomial.chebyshev.chebval(u, c)
    diff = float(np.max(np.abs(Ms @ c - V @ (pw * (V.T @ bs)))))
    say(f"[check T50] moments vs dense: {diff:.1e}")
    if diff > 1e-8:
        sys.exit("check failed")

    data = {}
    for rep in list(TRAIN) + list(TEST):
        t0 = time.time()
        A, b, truth = system(T, rep, sr, ss, beta, alpha)
        elo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False, tol=1e-7, maxiter=5000)[0])
        ehi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-7, maxiter=5000)[0])
        if not (dom[0] < elo and ehi < dom[1]):
            say(f"STOP: rep {rep} spectrum {elo:.3f}..{ehi:.3f} outside shared domain")
            sys.exit(2)
        xm, _ = minres(A.tocsc(), b, rtol=1e-10, maxiter=50000)
        x_cls = np.abs(np.asarray(xm).ravel())
        epad = 0.05 * (ehi - elo) + 0.2
        comb = design_line_comb_inverse(degree=40, s=GAMMA + DELTA + 4 * alpha, hw=0.18,
                                        domain=(elo - epad, ehi + epad))
        x_comb = cheb_apply(comb, A, b)
        M = moments(A, b, dom, max(DEGS))
        data[rep] = dict(truth=truth, x_cls=x_cls, x_comb=x_comb, M=M.astype(np.float32),
                         G={d: (M[~truth, :d + 1].T @ M[~truth, :d + 1], M[truth, :d + 1].T @ M[truth, :d + 1],
                                M[truth, :d + 1].T @ x_cls[truth]) for d in DEGS})
        del A, M
        say(f"rep {rep}: n={len(truth):,} spectrum {elo:.3f}..{ehi:.3f} ({time.time() - t0:.0f}s)")

    rows = []
    for d in DEGS:
        GF = sum(data[r]["G"][d][0] for r in TRAIN)
        GT = sum(data[r]["G"][d][1] for r in TRAIN)
        bT = sum(data[r]["G"][d][2] for r in TRAIN)
        reg = 1e-10 * np.trace(GF) / (d + 1) * np.eye(d + 1)
        best = None
        for mu in MUS:
            cf = np.linalg.solve(GF + mu * GT + reg, mu * bT)
            score = np.mean([readings(np.abs(data[r]["M"][:, :d + 1].astype(float) @ cf),
                                      data[r]["truth"])["eff@far0.01"] for r in TRAIN])
            if best is None or score > best[0]:
                best = (score, mu, cf)
        _, mu, cf = best
        for split, reps in (("train", TRAIN), ("test", TEST)):
            for r in reps:
                x = np.abs(data[r]["M"][:, :d + 1].astype(float) @ cf)
                rows.append(dict(config=cname, T=T, operator=op, split=split, rep=r,
                                 method=f"refit degree {d}", mu=mu, **readings(x, data[r]["truth"])))
    for r in TEST:
        rows.append(dict(config=cname, T=T, operator=op, split="test", rep=r, method="comb degree 40 (unfitted)",
                         **readings(data[r]["x_comb"], data[r]["truth"])))
        rows.append(dict(config=cname, T=T, operator=op, split="test", rep=r, method="classical",
                         **readings(data[r]["x_cls"], data[r]["truth"])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    say(df.groupby(["split", "method"])[["eff@far0.01", "eff@far0.05", "far@0.99"]].median().round(3).to_string())
    say("done.")


if __name__ == "__main__":
    main()
