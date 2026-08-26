#!/usr/bin/env python3
"""Stage 1+2 high-T extension — carry the frontier winners to T=1000.

George (2026-08-24): "do stage 1 and 2 up to the full 1000 tracks and see if
we keep parity with the 1bqf." Carried set-outs: base (clean + moderate),
occ_a0.05 (moderate winner, 0.9905 @ far<=1% at T=200), erf (second, 0.9853).
Families: fitted_moment d in {40, 80}, comb_prod d in {40, 44}, normalized
+-1/2 d in {40, 44}. T in {400, 700, 1000}; reps 5 at T=400, 3 at 700/1000.

Machinery identical to 02_hamiltonian_setouts.py (matrix-free Chebyshev
moments, ridge fit selected by eff @ far <= 1%, matched-eff frontier columns).
DMAX capped at 80: the moment matrix at T=1000 is n x 81 ~ 2.5 GB float64.
1BQF is NOT re-solved here (statevector ~7 h/solve at T=1000, OOM territory);
the parity comparison uses the existing store anchors in the analysis step.

Outputs: outputs/03_highT_frontier.csv (rewritten after every rep)
         outputs/cache/03_amps_{setout}_T{T}_rep{r}.npz
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
from lhcb_velo_toy.analysis import compute_epsilon           # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

DELTA = 1.0
REGIMES = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01),
}

TS = (400, 700, 1000)
REPS_BY_T = {400: (0, 1, 2, 3, 4), 700: (0, 1, 2), 1000: (0, 1, 2)}

# (setout tag, regime, knobs) — the carried winners from Stages 1-2
SETOUTS = [
    ("base",            "clean",    {}),
    ("occ_a0.05",       "clean",    dict(occ_alpha=0.05)),
    ("fork_b0.5",       "clean",    dict(fork_beta=0.5)),
    ("occ0.05_fork0.5", "clean",    dict(occ_alpha=0.05, fork_beta=0.5)),
    ("base",            "moderate", {}),
    ("occ_a0.05",       "moderate", dict(occ_alpha=0.05)),
    ("occ_a0.10",       "moderate", dict(occ_alpha=0.10)),
    ("fork_b0.5",       "moderate", dict(fork_beta=0.5)),
    ("occ0.05_fork0.5", "moderate", dict(occ_alpha=0.05, fork_beta=0.5)),
    ("erf",             "moderate", dict(kernel="erf")),
    ("base",            "heavy",    {}),
    ("occ_a0.05",       "heavy",    dict(occ_alpha=0.05)),
    ("fork_b0.5",       "heavy",    dict(fork_beta=0.5)),
    ("occ0.05_fork0.5", "heavy",    dict(occ_alpha=0.05, fork_beta=0.5)),
    ("erf",             "heavy",    dict(kernel="erf")),
]

# George 2026-08-26: do NOT vary the degree — hold it at the production value
# and show the improvement through A.  (Degree was measured to matter little on
# the base operator; where it does bite — occupancy's span growth — that is
# reported as a scoping limit, not re-optimised.)
FIT_DEGS = (40,)
FIXED_DEGS = (40,)
DMAX = max(FIT_DEGS)
MUS = np.geomspace(3e-3, 30.0, 13)
EFF_TARGETS = (0.99, 0.995, 0.999)
FAR_TARGETS = (0.001, 0.01)

LOG = open(OUT / "03_highT.log", "w")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


# ── frontier metrics (Stage-1/2 convention, scale-free columns only) ─────────
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


# ── moment machinery (Codesign/06 via Stage 2) ───────────────────────────────
def cheb_moments(A, b, lo, hi, dmax):
    """float32 storage (halves the T=1000 peak after the cgroup OOM of the
    first run); the recursion itself runs in float64 vectors."""
    c0, c1 = 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02
    n = A.shape[0]
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
    """X.T @ X accumulated in float64 by row chunks (X may be float32)."""
    d = X.shape[1]
    G = np.zeros((d, d))
    for i in range(0, X.shape[0], 500_000):
        c = X[i:i + 500_000].astype(np.float64)
        G += c.T @ c
    return G


def fit_degree(M, truth, x_cls, deg):
    """Coefficient-space ridge fit; mu selected by eff @ far <= 1%."""
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
    # optional filters: argv = setout tags; EF_TS="400" / EF_REPS="0,1" limit
    tags = set(sys.argv[1:])
    setouts = [s for s in SETOUTS if not tags or s[0] in tags]
    ts = tuple(int(t) for t in os.environ["EF_TS"].split(",")) \
        if "EF_TS" in os.environ else TS
    # resume: keep completed rows, skip finished (setout, regime, T, rep) cells
    csv_path = OUT / "03_highT_frontier.csv"
    rows, done = [], set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = set(map(tuple, prev[["setout", "regime", "T", "rep"]]
                       .drop_duplicates().itertuples(index=False)))
        say(f"resume: {len(rows)} rows, {len(done)} cells already done")
    for T in ts:
        reps = tuple(int(r) for r in os.environ["EF_REPS"].split(",")) \
            if "EF_REPS" in os.environ else REPS_BY_T[T]
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
                if (tag, regime, T, rep) in done:
                    continue
                t0 = time.time()
                ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
                kw = dict(epsilon=eps, kernel=kernel, gamma=gamma, delta=DELTA)
                if kernel == "erf":
                    kw["erf_sigma"] = eps / 3.0
                ham = qp.build_hamiltonian(ev, **kw)
                truth = np.asarray(qp.truth_from_event(ev), bool)
                # occupancy matvec is matrix-free: the explicit co-hit clique
                # graph has nnz ~ sum_h deg(h)^2 (~1e9 at T>=700, cgroup OOM),
                # but Ba = Hs Hs^T + He He^T - 2I with H the segment->hit
                # incidence, so A_occ = A + 2*alpha*(Hs Hs^T + He He^T).
                A0, b, tau_abs, _ = dp_terms.dp_system(
                    ham, beta=beta, eps_B=(eps if beta else None),
                    alpha=0.0, gamma=gamma, delta=DELTA)
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
                    attractor = (DELTA + 4.0 * alpha) / (gamma + DELTA
                                                         + 4.0 * alpha)
                    tau_abs = attractor + 0.10
                else:
                    A = A0
                lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                                 tol=1e-6, maxiter=5000)[0])
                hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                 tol=1e-6, maxiter=5000)[0])
                xm, _ = minres(A, b, rtol=1e-10, maxiter=40000)
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

                act = x_cls > tau_abs
                cls_eff = float((act & truth).sum()) / max(int(truth.sum()), 1)
                cls_far = float((act & ~truth).sum()) / max(int(act.sum()), 1)
                record("classical_invA", np.nan, x_cls,
                       extra=f"fixed-tau {tau_abs:.3f}")
                rows[-1]["eff_fixed_tau"] = cls_eff
                rows[-1]["far_fixed_tau"] = cls_far
                rows[-1]["tau_fixed"] = float(tau_abs)

                M = cheb_moments(A, b, lo, hi, DMAX)
                for d in FIT_DEGS:
                    x, mu = fit_degree(M, truth, x_cls, d)
                    record("fitted_moment", d, x, extra=f"mu={mu:.3g}")
                del M

                pad = 0.05 * (hi - lo) + 0.2
                dom = (lo - pad, hi + pad)
                for d in FIXED_DEGS:
                    p = design_line_comb_inverse(degree=d, s=s_prime,
                                                 domain=dom)
                    record("comb_prod", d, cheb_apply(p, A, b))

                if alpha == 0.0:
                    # normalized family needs the explicit off-diagonal; the
                    # occ+normalized combo is refuted at T=200 anyway (Stage 2)
                    Off = A0.copy().tolil()
                    Off.setdiag(0)
                    Dm = discriminant(Off.tocsr())
                    ones = np.ones(n)
                    for d in FIXED_DEGS:
                        record("normalized_half_comb", d,
                               cheb_apply(design_norm_comb(d), Dm, ones))
                    del Off, Dm

                np.savez_compressed(
                    CACHE / f"03_amps_{tag}_T{T}_rep{rep}.npz",
                    truth=truth, sol_C=x_cls.astype(np.float32), **amps)
                say(f"[T={T} {tag} rep{rep}] n={n:,} lam=[{lo:.2f},{hi:.2f}] "
                    f"span {hi - lo:.1f} eps={eps * 1e3:.2f}mrad "
                    f"({time.time() - t0:.0f}s)")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                del amps, A, A0, b, x_cls, truth, ev, ham
                import gc
                gc.collect()

    df = pd.DataFrame(rows)
    say("\n== Stage 1+2 high-T: median eff@far<=1% per (T, setout, family) at "
        "the best degree ==")
    med = (df[df.family != "classical_invA"]
           .groupby(["T", "setout", "family", "degree"], dropna=False)
           ["eff_f010"].median().reset_index())
    for (T, st, fam), g in med.groupby(["T", "setout", "family"]):
        b_ = g.loc[g["eff_f010"].idxmax()]
        say(f"  T={T:4d} {st:10s} {fam:20s} d={b_['degree']!s:>5}: "
            f"eff@far1% {b_['eff_f010']:.4f}")
    say("done.")


if __name__ == "__main__":
    main()
