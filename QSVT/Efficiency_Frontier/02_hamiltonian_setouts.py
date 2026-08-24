#!/usr/bin/env python3
"""Stage 2 — Hamiltonian set-out axis on the carried responses.

One knob at a time from the baseline set-out (step kernel, formula eps at scale
3.0, gamma=3, no occupancy, no fork), moderate regime, T=200, reps 0-9; fork
beta=0.5 runs on clean only (the heavy no-go stands, PLAN 3B). The H4
combination (occupancy + normalized +-1/2 comb) falls out of the occupancy
set-outs, where the normalized rows ARE the combo.

Machinery (uniform across knobs, Codesign/06's moment method): the occupancy
coupling merges the graph into a giant component, so per-component eigh is
unavailable — instead Chebyshev moment vectors m_k = T_k(X) b' (matrix-free)
make any degree-d response's amplitudes LINEAR in its coefficients:
  * fitted responses = ridge fit in coefficient space (mu selected by
    eff @ far <= 1%, this experiment's axis — 06 used far@0.985);
  * fixed designs (line comb at the set-out's diagonal s' = gamma+delta+4alpha,
    normalized +-1/2 comb on the discriminant of the MODIFIED off-diagonal)
    are applied by the same Chebyshev recursion on their own domain.
The normalized walk is the interesting occupancy partner: normalization maps
any span back into [-1,1], absorbing the occupancy span explosion — whether
the true lines survive the reweighting is exactly what this measures.
gamma does not enter the discriminant (C unchanged), so normalized rows are
skipped on the gamma knob (they would duplicate the base rows identically).

Frontier columns as Stage 1 (matched-eff far, eff@far targets, cos to the
set-out's own classical solve). No fixed-tau columns for filters here: the
gamma/occupancy knobs move the attractor-aware tau, so only the classical
fixed-tau reference is recorded (tau from dp_terms.dp_system).

Outputs: outputs/02_setout_frontier.csv
         outputs/cache/02_amps_{setout}_rep{r}.npz
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
DELTA = 1.0
REGIMES = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
}

# (setout tag, regime, dict of knobs) — one knob at a time from baseline
SETOUTS = [
    ("base",     "moderate", {}),
    ("occ_a0.05", "moderate", dict(occ_alpha=0.05)),
    ("occ_a0.10", "moderate", dict(occ_alpha=0.10)),
    ("eps_s1.5", "moderate", dict(eps_scale=1.5)),
    ("eps_s2.12", "moderate", dict(eps_scale=2.12)),
    ("gamma1",   "moderate", dict(gamma=1.0)),
    ("gamma2",   "moderate", dict(gamma=2.0)),
    ("erf",      "moderate", dict(kernel="erf")),
    ("fork_b0.5", "clean",   dict(fork_beta=0.5)),
]

FIT_DEGS = (32, 40, 44, 80, 160)      # 80/160 matter where the span widens
FIXED_DEGS = (40, 44)
DMAX = max(FIT_DEGS)
MUS = np.geomspace(3e-3, 30.0, 13)
EFF_TARGETS = (0.99, 0.995, 0.999)
FAR_TARGETS = (0.001, 0.01)

LOG = open(OUT / "02_setouts.log", "w")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


# ── frontier metrics (Stage-1 convention, scale-free columns only) ───────────
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


# ── moment machinery (Codesign/06) ───────────────────────────────────────────
def cheb_moments(A, b, lo, hi, dmax):
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    n = A.shape[0]
    M = np.empty((n, dmax + 1))
    M[:, 0] = b
    M[:, 1] = (A @ b - c0 * b) / c1
    for k in range(2, dmax + 1):
        M[:, k] = 2.0 * ((A @ M[:, k - 1]) - c0 * M[:, k - 1]) / c1 - M[:, k - 2]
    return M


def fit_degree(M, truth, x_cls, deg):
    """Coefficient-space ridge fit; mu selected by eff @ far <= 1%."""
    Md = M[:, :deg + 1]
    MT, MF = Md[truth], Md[~truth]
    GF, GT = MF.T @ MF, MT.T @ MT
    bT = MT.T @ x_cls[truth]
    reg = 1e-10 * np.trace(GF) / (deg + 1) * np.eye(deg + 1)
    best = None
    for mu in MUS:
        c = np.linalg.solve(GF + mu * GT + reg, mu * bT)
        x = np.abs(Md @ c)
        fm = frontier_metrics(x, truth, x_cls)
        key = -fm["eff_f010"]
        if best is None or key < best[0]:
            best = (key, mu, x)
    return best[2], float(best[1])


def cheb_apply(poly, A, b):
    """|poly(A) b| by the recursion; poly a numpy Chebyshev with its domain."""
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


def design_norm_comb(degree, hw=0.10):
    x = np.linspace(-1, 1, 8000)
    y = np.zeros_like(x)
    for m in (0.5, -0.5):
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    p = npcheb.Chebyshev.fit(x, y, degree, domain=[-1, 1])
    mx = float(np.max(np.abs(p(x))))
    return p / (mx / 0.95) if mx > 1.0 else p


def discriminant(C):
    C = sp.csr_matrix(abs(C))
    deg = np.asarray(C.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    return sp.csr_matrix(sp.diags(inv) @ C @ sp.diags(inv))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # optional filters: argv = setout tags to run; EF_REPS="0,1" limits reps
    tags = set(sys.argv[1:])
    setouts = [s for s in SETOUTS if not tags or s[0] in tags]
    reps = tuple(int(r) for r in os.environ["EF_REPS"].split(",")) \
        if "EF_REPS" in os.environ else REPS
    rows = []
    for tag, regime, kn in setouts:
        nz = REGIMES[regime]
        gamma = float(kn.get("gamma", 3.0))
        alpha = float(kn.get("occ_alpha", 0.0))
        beta = float(kn.get("fork_beta", 0.0))
        kernel = kn.get("kernel", "step")
        scale = float(kn.get("eps_scale", 3.0))
        eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"],
                                    scale=scale))
        s_prime = gamma + DELTA + 4.0 * alpha
        for rep in reps:
            t0 = time.time()
            ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
            kw = dict(epsilon=eps, kernel=kernel, gamma=gamma, delta=DELTA)
            if kernel == "erf":
                kw["erf_sigma"] = eps / 3.0
            ham = qp.build_hamiltonian(ev, **kw)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            A, b, tau_abs, _ = dp_terms.dp_system(
                ham, beta=beta, eps_B=eps if beta else None,
                alpha=alpha, gamma=gamma, delta=DELTA)
            A = A.tocsr()
            n = A.shape[0]
            lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                             tol=1e-7, maxiter=5000)[0])
            hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                             tol=1e-7, maxiter=5000)[0])
            xm, _ = minres(A.tocsc(), b, rtol=1e-10, maxiter=20000)
            x_cls = np.abs(np.asarray(xm).ravel())

            amps = {}

            def record(family, degree, x, extra=""):
                fm = frontier_metrics(x, truth, x_cls)
                rows.append(dict(setout=tag, regime=regime, T=T, rep=rep,
                                 family=family, degree=degree, gamma=gamma,
                                 occ_alpha=alpha, eps_scale=scale,
                                 kernel=kernel, fork_beta=beta, eps=eps,
                                 lam_min=lo, lam_max=hi, **fm, notes=extra))
                amps[f"{family}_d{degree}"] = x.astype(np.float32)

            # classical reference (+ its attractor-aware fixed-tau point)
            act = x_cls > tau_abs
            record("classical_invA", np.nan, x_cls,
                   extra=f"fixed-tau {tau_abs:.3f}: eff "
                         f"{(act & truth).sum() / max(truth.sum(), 1):.4f} far "
                         f"{(act & ~truth).sum() / max(act.sum(), 1):.4f}")

            # fitted responses in moment space
            M = cheb_moments(A, b, lo, hi, DMAX)
            for d in FIT_DEGS:
                x, mu = fit_degree(M, truth, x_cls, d)
                record("fitted_moment", d, x, extra=f"mu={mu:.3g}")
            del M

            # line comb at the set-out's own diagonal + domain
            pad = 0.05 * (hi - lo) + 0.2
            dom = (lo - pad, hi + pad)
            for d in FIXED_DEGS:
                p = design_line_comb_inverse(degree=d, s=s_prime, domain=dom)
                record("comb_prod", d, cheb_apply(p, A, b))

            # normalized +-1/2 comb on the MODIFIED off-diagonal (H4 at occ)
            if "gamma" not in kn:      # discriminant is gamma-invariant
                Off = A.copy().tolil()
                Off.setdiag(0)
                Dm = discriminant(Off.tocsr())
                ones = np.ones(n)
                for d in FIXED_DEGS:
                    record("normalized_half_comb", d,
                           cheb_apply(design_norm_comb(d), Dm, ones))

            np.savez_compressed(CACHE / f"02_amps_{tag}_rep{rep}.npz",
                                truth=truth, sol_C=x_cls.astype(np.float32),
                                **amps)
            say(f"[{tag} rep{rep}] n={n:,} lam=[{lo:.2f},{hi:.2f}] "
                f"span {hi - lo:.1f} eps={eps * 1e3:.2f}mrad "
                f"({time.time() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT / "02_setout_frontier.csv",
                                      index=False)

    df = pd.DataFrame(rows)
    say("\n== Stage 2: median eff@far<=1% per (setout, family) at the best "
        "degree ==")
    med = (df.groupby(["setout", "family", "degree"], dropna=False)
             ["eff_f010"].median().reset_index())
    for (st, fam), g in med.groupby(["setout", "family"]):
        b_ = g.loc[g["eff_f010"].idxmax()]
        say(f"  {st:10s} {fam:20s} d={b_['degree']!s:>5}: "
            f"eff@far1% {b_['eff_f010']:.4f}")
    say("done.")


if __name__ == "__main__":
    main()
