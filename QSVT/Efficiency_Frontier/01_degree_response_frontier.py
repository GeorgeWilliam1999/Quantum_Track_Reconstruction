#!/usr/bin/env python3
"""Stage 1 — degree x response frontier at the baseline set-out.

Fills the Stage-1 gap cells frozen by 00_inventory.py: clean + moderate, T=200,
reps 0-2, response family x degree, full matched-efficiency frontier per cell.
Every planned degree is computed together with its d+4 twin (BE-05 ripple rule).

Method (all families through ONE spectral system per (regime, rep)):
  A = sI - C decomposes over connected components; per-component dense eigh
  (components are small — asserted) gives modes (lam_m, U column, w_m = u.b).
  Any scalar response f then yields exact amplitudes x = |U (f(lam) * w)| in one
  sparse matvec, isolated segments at |f(s)| * delta.  This is the Codesign/04
  `amplitudes` machinery, vectorised across components.  The normalized +-1/2
  comb runs on its own system over D^{-1/2} C D^{-1/2} (BE-03), iso at |f(0)|.

Families:
  classical_invA (1/lam), 1bqf_cos (|cos(pi lam / 2s)|)     — baselines
  comb_prod      design_line_comb_inverse hw=0.18, d in grid+twins
  band_inverse   design_band_limited_inverse defaults, same grid
  sharp_comb     line comb hw=0.06, d in {80,120}+twins
  normalized_half_comb   +-1/2 comb on the discriminant, d in {12..40}+twins
  fitted_ridge   Codesign/04 binned-ridge refit PER (regime, rep), Chebyshev-
                 realized at d in {20,28,40}+twins; fitted_binned = unrealized
                 reference; fitted_xval = rep-0 response applied to reps 1-2.

Frontier columns per cell (per-solver working points; never the classical tau):
  far_e990/995/999 (+ realized eff, tau) — matched-eff tau at the k-th true
  amplitude (XII convention); eff_f001/f010 — best eff at far <= target;
  fixed-tau eff/far via qp.quantum_metrics (gamma-aware 0.35, house rescale);
  cos_C to the classical solution.

Outputs: outputs/01_frontier_clean_moderate.csv
         outputs/cache/01_amps_{regime}_rep{r}.npz  (float32 amplitudes, for the
         Stage-3/4 loss decomposition; gitignored)
No store writes (degree is not a sol_key axis — the CSV is the record).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from numpy.polynomial import chebyshev as npcheb
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import minres

sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                   "Toy_Characterisation/_shared")
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")
import qtrk_pipeline as qp                                   # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon           # noqa: E402
from lhcb_velo_toy.solvers.quantum import (                  # noqa: E402
    design_band_limited_inverse, design_line_comb_inverse)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

T = 200
REPS = (0, 1, 2)
GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
REGIMES = {   # house triples (dp_matrix_characterisation.NOISES; drop=1% per George)
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
}

def with_twins(ds):
    return sorted(set(int(d) for d in ds) | set(int(d) + 4 for d in ds))

COMB_D = with_twins([8, 12, 16, 20, 24, 28, 32, 40, 48, 64])
BAND_D = COMB_D
SHARP_D = with_twins([80, 120])
NORM_D = with_twins([12, 20, 28, 32, 40])
FIT_D = with_twins([20, 28, 40])

# fitted-ridge constants (Codesign/04, verbatim except the mu selector which
# follows this experiment's 0.995 target instead of 04's heavy-noise 0.985)
BIN_W = 0.02
MUS = np.geomspace(3e-3, 30.0, 13)
MU_SELECT_EFF = 0.995
MAX_COMP = 3000            # dense-eigh guard; bail loudly if exceeded

EFF_TARGETS = (0.99, 0.995, 0.999)
FAR_TARGETS = (0.001, 0.01)

LOG = open(OUT / "01_frontier.log", "w")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


# ── spectral systems ─────────────────────────────────────────────────────────
def mode_system(M, b, iso_lambda):
    """Per-component eigh of sparse symmetric M -> (U, lam, w, iso_mask).

    U is (n x n_modes) sparse; amplitudes for response f are
    |U (f(lam) * w)| with x[iso] = |f(iso_lambda)| * b[iso].
    """
    n = M.shape[0]
    Off = M.copy().tolil()
    Off.setdiag(0)
    Off = Off.tocsr()
    Off.eliminate_zeros()
    ncomp, lab = connected_components(Off, directed=False)
    sizes = np.bincount(lab)
    if sizes.max() > MAX_COMP:
        raise RuntimeError(f"component of size {sizes.max()} > {MAX_COMP}: "
                           "dense per-component eigh not viable here")
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    rows, cols, vals = [], [], []
    lams, ws = [], []
    m0 = 0
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        lam, V = eigh(M[np.ix_(idx, idx)].toarray())
        wb = V.T @ b[idx]
        k = len(idx)
        rows.append(np.repeat(idx, k))
        cols.append(np.tile(np.arange(m0, m0 + k), k))
        vals.append(V.ravel())
        lams.append(lam)
        ws.append(wb)
        m0 += k
    U = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, m0))
    lam = np.concatenate(lams)
    w = np.concatenate(ws)
    iso = sizes[lab] == 1
    return dict(U=U, lam=lam, w=w, iso=iso, b=b, iso_lambda=iso_lambda,
                lam_min=float(lam.min()), lam_max=float(lam.max()),
                max_comp=int(sizes.max()),
                n_comp=int(np.count_nonzero(sizes > 1)))


def amplitudes(sysd, f):
    x = np.zeros(sysd["U"].shape[0])
    x[sysd["iso"]] = abs(float(np.asarray(f(np.array([sysd["iso_lambda"]])))
                         .ravel()[0])) * sysd["b"][sysd["iso"]]
    x += np.abs(sysd["U"] @ (np.asarray(f(sysd["lam"])) * sysd["w"])) \
        * (~sysd["iso"])
    return x


# ── frontier metrics (matched-eff tau at the k-th true amplitude) ────────────
def frontier_metrics(x, truth, sol_C):
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
        nfa = int((act & ~truth).sum())
        tag = f"e{int(round(e * 1000)):03d}"
        out[f"far_{tag}"] = nfa / na if na else np.nan
        out[f"eff_{tag}"] = float((act & truth).sum()) / n_true
        out[f"tau_{tag}"] = float(tau)
    # best eff at far <= target: scan tau downward over true amplitudes
    ks = np.arange(1, n_true + 1)
    taus = ts[ks - 1]
    # vectorised counts at each true-anchored tau
    xs = np.sort(x)
    nact = len(x) - np.searchsorted(xs, taus, side="left")
    xt = np.sort(x[truth])
    ntru = n_true - np.searchsorted(xt, taus, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    eff = ntru / n_true
    for ft in FAR_TARGETS:
        okm = far <= ft
        tag = f"f{int(round(ft * 1000)):03d}"
        out[f"eff_{tag}"] = float(eff[okm].max()) if okm.any() else 0.0
    # fixed-tau (house rescale + gamma-aware threshold) + cosine to classical.
    # sol_C must be the RAW physical classical amplitudes (isolated at 0.25):
    # quantum_metrics rescales x to the classical signal support, so a
    # unit-norm partner would push everything below the 0.35 cut.
    nrm = np.linalg.norm(x)
    solu = x / nrm if nrm > 0 else x
    mQ = qp.quantum_metrics(solu, sol_C, truth, qp.threshold_for(GAMMA, DELTA))
    out["eff_fixed"] = mQ.get("segment_efficiency")
    out["far_fixed"] = mQ.get("segment_false_rate")
    out["cos_C"] = mQ.get("cos_QC")
    return out


# ── fitted-ridge refit (Codesign/04 recipe on this system) ───────────────────
def fit_binned(sysd, truth, x_cls):
    edges = np.arange(sysd["lam_min"] - BIN_W, sysd["lam_max"] + 2 * BIN_W,
                      BIN_W)
    nb = len(edges) - 1
    bi = np.clip(np.searchsorted(edges, sysd["lam"]) - 1, 0, nb - 1)
    onehot = sp.csr_matrix((sysd["w"], (np.arange(len(bi)), bi)),
                           shape=(len(bi), nb))
    D = np.asarray((sysd["U"] @ onehot).todense())      # n_seg x nb
    incomp = ~sysd["iso"]
    tru = truth & incomp
    fal = (~truth) & incomp
    DT, DF = D[tru], D[fal]
    y_true = x_cls[tru]
    iso_row = np.zeros(nb)
    iso_row[np.clip(np.searchsorted(edges, S) - 1, 0, nb - 1)] = DELTA
    wiso = np.sqrt(float(np.count_nonzero(sysd["iso"] & ~truth))) / 30.0
    DF_full = np.vstack([DF, wiso * iso_row])
    GF, GT = DF_full.T @ DF_full, DT.T @ DT
    bT = DT.T @ y_true
    reg = 1e-8 * np.eye(nb)
    best = None
    for mu in MUS:
        p_b = np.linalg.solve(GF + mu * GT + reg, mu * bT)
        p_b = p_b / max(np.abs(p_b).max(), 1e-12)
        x_o = amplitudes(sysd, binned_fn(p_b, edges))
        fm = frontier_metrics(x_o, truth, x_cls)
        # select mu by best efficiency at far <= 1% (the experiment's success
        # axis) — far@matched-eff floods identically for every mu at moderate
        # (the fragment floor), so it cannot rank the fits there.
        key = -fm["eff_f010"]
        if best is None or key < best[0]:
            best = (key, mu, p_b)
    _, mu_b, p_opt = best
    return p_opt, edges, float(mu_b)


def binned_fn(p_bins, edges):
    def f(lam):
        bi = np.clip(np.searchsorted(edges, np.asarray(lam)) - 1,
                     0, len(p_bins) - 1)
        return p_bins[bi]
    return f


def cheb_realize(p_bins, edges, degree):
    grid = np.linspace(edges[0], edges[-1], 4000)
    targ = binned_fn(p_bins, edges)(grid)
    dom = (edges[0], edges[-1])
    u = (2 * grid - (dom[0] + dom[1])) / (dom[1] - dom[0])
    cf = npcheb.chebfit(u, targ, degree)
    sc = max(float(np.max(np.abs(npcheb.chebval(u, cf)))), 1e-12)

    def f(lam, cf=cf, dom=dom, sc=sc):
        uu = (2 * np.asarray(lam) - (dom[0] + dom[1])) / (dom[1] - dom[0])
        return npcheb.chebval(uu, cf) / sc
    return f


# ── normalized comb design (BE-03) ───────────────────────────────────────────
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
    import pandas as pd
    rows = []
    fitted_rep0 = {}
    for regime, nz in REGIMES.items():
        eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
        for rep in REPS:
            t0 = time.time()
            ev, ekey = qp.ensure_event(n_trk=T, rep=rep, **nz)
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                       gamma=GAMMA, delta=DELTA)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            n = int(ham.n_segments)
            A = ham.A.tocsr()
            sysA = mode_system(A, np.full(n, DELTA), iso_lambda=S)
            say(f"[{regime} rep{rep}] n_seg={n:,} n_true={int(truth.sum())} "
                f"eps={eps:.5f} comps={sysA['n_comp']:,} "
                f"max_comp={sysA['max_comp']} lam=[{sysA['lam_min']:.3f},"
                f"{sysA['lam_max']:.3f}] ({time.time()-t0:.0f}s)")

            # classical reference + validation against MINRES
            x_cls = amplitudes(sysA, lambda l: 1.0 / np.asarray(l))
            xm, _ = minres(A.tocsc(), DELTA * np.ones(n), rtol=1e-10,
                           maxiter=8000)
            xm = np.abs(np.asarray(xm).ravel())
            cosv = float(x_cls @ xm / (np.linalg.norm(x_cls)
                                       * np.linalg.norm(xm)))
            assert cosv > 0.999999, f"spectral-vs-MINRES cosine {cosv}"
            sol_C = x_cls          # raw physical amplitudes (isolated at 0.25)

            span = sysA["lam_max"] - sysA["lam_min"]
            pad = 0.05 * span + 0.2
            dom = (min(sysA["lam_min"], S - 2) - pad,
                   max(sysA["lam_max"], S + 2) + pad)

            # normalized system
            C = A.copy().tolil(); C.setdiag(0)
            Dm = discriminant(C.tocsr())
            sysN = mode_system(Dm, np.ones(n), iso_lambda=0.0)

            amps_cache = {}

            def record(family, degree, x, extra=""):
                fm = frontier_metrics(x, truth, sol_C)
                rows.append(dict(regime=regime, T=T, rep=rep, family=family,
                                 degree=degree, eps=eps, **fm, notes=extra))
                amps_cache[f"{family}_d{degree}"] = x.astype(np.float32)

            record("classical_invA", np.nan, x_cls)
            record("1bqf_cos", 1,
                   amplitudes(sysA, lambda l: np.abs(
                       np.cos(np.pi * np.asarray(l) / (2 * S)))))
            for d in COMB_D:
                p = design_line_comb_inverse(degree=d, s=S, domain=dom)
                record("comb_prod", d, amplitudes(sysA, lambda l: np.abs(p(l))))
            for d in BAND_D:
                p = design_band_limited_inverse(degree=d, s=S, domain=dom)
                record("band_inverse", d,
                       amplitudes(sysA, lambda l: np.abs(p(l))))
            for d in SHARP_D:
                p = design_line_comb_inverse(degree=d, s=S, hw=0.06, domain=dom)
                record("sharp_comb", d, amplitudes(sysA, lambda l: np.abs(p(l))))
            for d in NORM_D:
                p = design_norm_comb(d)
                record("normalized_half_comb", d,
                       amplitudes(sysN, lambda l: np.abs(p(l))))

            p_opt, edges, mu_b = fit_binned(sysA, truth, x_cls)
            record("fitted_binned", np.nan,
                   amplitudes(sysA, binned_fn(p_opt, edges)),
                   extra=f"mu={mu_b:.3g} nb={len(edges)-1}")
            for d in FIT_D:
                record("fitted_ridge", d,
                       amplitudes(sysA, cheb_realize(p_opt, edges, d)),
                       extra=f"mu={mu_b:.3g}")
            if rep == 0:
                fitted_rep0[regime] = (p_opt, edges)
            else:
                p0, e0 = fitted_rep0[regime]
                record("fitted_xval", np.nan,
                       amplitudes(sysA, binned_fn(p0, e0)),
                       extra="rep0 response verbatim")

            np.savez_compressed(CACHE / f"01_amps_{regime}_rep{rep}.npz",
                                truth=truth, sol_C=sol_C.astype(np.float32),
                                **amps_cache)
            say(f"[{regime} rep{rep}] {len(amps_cache)} responses "
                f"({time.time()-t0:.0f}s total)")
            df = pd.DataFrame(rows)
            df.to_csv(OUT / "01_frontier_clean_moderate.csv", index=False)

    # summary: best far at the 0.995 target per (regime, family)
    df = pd.DataFrame(rows)
    say("\n== Stage 1 frontier: best far_e995 per (regime, family), "
        "mean over reps at the best degree ==")
    agg = (df.groupby(["regime", "family", "degree"], dropna=False)
             [["far_e995", "eff_f010"]].mean().reset_index())
    for (rg, fam), gg in agg.groupby(["regime", "family"]):
        b = gg.loc[gg["far_e995"].idxmin()]
        say(f"  {rg:9s} {fam:22s} d={b['degree']!s:>6}: "
            f"far@0.995={b['far_e995']:.4f}  eff@far1%={b['eff_f010']:.4f}")
    say("done.")


if __name__ == "__main__":
    main()
