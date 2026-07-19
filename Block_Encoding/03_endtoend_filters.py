"""
03 -- end-to-end impact of the encoding choice on the QSVT comb filter.

A. Measured degree requirement per encoding: fit the line-comb response on each
   encoding's rescaled spectral domain and find the minimum Chebyshev degree
   reaching a fixed response tolerance; combine with per-call CX into total cost.
B. The alpha=1 Szegedy/discriminant encoding changes the operator to
   D^{-1/2} C D^{-1/2}: design a comb on the NORMALIZED spectrum (pure-true P4
   lines +-cos(pi/5), +-cos(2pi/5); hubs collapse to {+-1,0}; isolated -> 1) and
   measure real segment metrics on store events vs the production comb.

Outputs: outputs/03_degree_requirements.csv, outputs/03_normalized_walk_metrics.csv,
         outputs/fig03_*.png
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared")
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb

import be_lib as bl
import qtrk_pipeline as qp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
S = GAMMA + DELTA
P4_LINES_A = tuple(S - 2.0 * np.cos(k * np.pi / 5.0) for k in (1, 2, 3, 4))
# normalized P4 path adjacency eigenvalues (exact): +-cos(pi/5), +-cos(2pi/5)
P4_LINES_NORM = (np.cos(np.pi / 5), np.cos(2 * np.pi / 5),
                 -np.cos(2 * np.pi / 5), -np.cos(np.pi / 5))
HW_A = 0.18            # production comb half-width in A-units
TOL = 0.02             # response fit tolerance (L_inf on the target grid)

MCOL = dict(fable="#2a78d6", sfable="#008300", lsfable="#e87ba4",
            camps="#eda100", dictionary="#1baf7a", szegedy="#eb6834",
            dilation="#4a3aa7", hit_oracle="#e34948", classical="#52514e")


# ----------------------------------------------------------------------
# A. measured degree requirement on each encoding's domain
# ----------------------------------------------------------------------

def comb_target(x, lines, hw):
    y = np.zeros_like(x)
    for m in lines:
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    return y


def min_degree(lines_x, hw_x, dmax=600, tol=TOL):
    """Min Chebyshev degree on [-1,1] fitting the comb to L_inf <= tol."""
    x = np.linspace(-1, 1, 8000)
    y = comb_target(x, lines_x, hw_x)
    lo, hi = 4, dmax
    best = None
    d = 8
    while d <= dmax:
        p = cheb.Chebyshev.fit(x, y, d, domain=[-1, 1])
        if np.max(np.abs(p(x) - y)) <= tol:
            best = d
            break
        d = int(d * 1.3) + 1
    if best is None:
        return np.nan
    # refine down
    while best > 4:
        p = cheb.Chebyshev.fit(x, y, best - 1, domain=[-1, 1])
        if np.max(np.abs(p(x) - y)) > tol:
            break
        best -= 1
    return best


def part_A():
    dr = pd.read_csv(os.path.join(OUT, "02_method_resources.csv"))
    st = pd.read_csv(os.path.join(OUT, "02_structure.csv"))
    rows = []
    for _, r in dr[(dr.noise == "clean")].iterrows():
        if not np.isfinite(r.get("alpha", np.nan)):
            continue
        m = st[(st["T"] == r["T"]) & (st.kernel == r.kernel) & (st.noise == "clean")].iloc[0]
        if r.target == "A":
            # X = A/alpha (plus centring for dilation): map lines and width
            if r.method == "dilation":
                lo, hi = m.lam_min, m.lam_max
                sc = 2.0 / (hi - lo)
                lines_x = [sc * (l - lo) - 1 for l in P4_LINES_A]
                hw_x = HW_A * sc
            else:
                lines_x = [l / r.alpha for l in P4_LINES_A]
                hw_x = HW_A / r.alpha
        elif r.target == "C":
            lamCmax = S - m.lam_min
            lamCmin = S - m.lam_max
            linesC = [S - l for l in P4_LINES_A]
            if r.method in ("camps_C", "dictionary_C", "hit_oracle"):
                lines_x = [l / r.alpha for l in linesC]
                hw_x = HW_A / r.alpha
            else:
                continue
        elif r.target == "Dnorm":
            lines_x = list(P4_LINES_NORM)
            hw_x = 0.06   # normalized-domain width from the measured min gap
        else:
            continue
        d_req = min_degree(lines_x, hw_x)
        cx_call = r.cx if np.isfinite(r.get("cx", np.nan)) else np.nan
        rows.append(dict(T=r["T"], kernel=r.kernel, method=r.method,
                         alpha=r.alpha, degree_required=d_req,
                         cx_per_call=cx_call,
                         cx_per_solve=(d_req * cx_call
                                       if np.isfinite(cx_call) and np.isfinite(d_req)
                                       else np.nan)))
        print(f"A: {r.kernel} T={int(r['T'])} {r.method}: alpha={r.alpha:.3g} "
              f"d_req={d_req}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "03_degree_requirements.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# B. normalized-walk comb on real events
# ----------------------------------------------------------------------

def discriminant(C):
    C = sp.csr_matrix(abs(C))
    deg = np.asarray(C.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    Dm = sp.diags(inv) @ C @ sp.diags(inv)
    return sp.csr_matrix(Dm), deg


def cheb_apply(p, M, v):
    """p(M) v by the Chebyshev recursion, M with spectrum in the poly domain."""
    lo, hi = p.domain
    sc = 2.0 / (hi - lo)
    sh = (lo + hi) / (hi - lo)
    X = sc * M - sh * sp.identity(M.shape[0], format="csr")
    c = p.coef
    t_prev = v
    y = c[0] * t_prev
    if len(c) > 1:
        t_cur = X @ v
        y = y + c[1] * t_cur
        for k in range(2, len(c)):
            t_next = 2.0 * (X @ t_cur) - t_prev
            t_prev, t_cur = t_cur, t_next
            y = y + c[k] * t_cur
    return y


def design_norm_comb(degree, hw=0.06):
    x = np.linspace(-1, 1, 8000)
    y = comb_target(x, P4_LINES_NORM, hw)
    p = cheb.Chebyshev.fit(x, y, degree, domain=[-1, 1])
    mx = float(np.max(np.abs(p(x))))
    if mx > 1.0:
        p = p / (mx / 0.95)
    return p


def part_B():
    rows = []
    spec_rows = []
    for T in (100, 200, 400):
        for kernel in (("step",) if T != 200 else ("step", "erf")):
            ev, ekey = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4,
                                       sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)
            kw = dict(epsilon=EPS, gamma=GAMMA, delta=DELTA, kernel=kernel)
            if kernel == "erf":
                kw["erf_sigma"] = EPS / 3.0
            ham = qp.build_hamiltonian(ev, **kw)
            A = ham.A.tocsr()
            d = A.shape[0]
            truth = qp.truth_from_event(ev)
            tau = qp.threshold_for(gamma=GAMMA, delta=DELTA)
            sol_C, _ = qp.solve_classical(ham)
            mC = qp.metrics_at(sol_C, truth, threshold=tau)

            C = sp.csr_matrix(abs(A - sp.diags(A.diagonal())))
            C.eliminate_zeros()
            Dm, deg = discriminant(C)
            b = np.ones(d) / np.sqrt(d)

            # normalized comb (deg 60 default; alpha=1 encoding)
            for degree in (40, 60, 90):
                p = design_norm_comb(degree)
                y = cheb_apply(p, Dm, b)
                # isolated (deg 0) segments sit at Dm=0 (no self-loop here):
                # p(0) applies; they are false -> we WANT ~0. record response at 0
                l1 = float(np.sum(np.abs(p.coef)))
                psucc = float(np.vdot(y, y).real) / (l1 ** 2) * l1 ** 2  # ||y||^2
                sol = np.abs(y)
                nrm = np.linalg.norm(sol)
                sol = sol / nrm if nrm > 0 else sol
                mQ = qp.quantum_metrics(sol, sol_C, truth, tau)
                rows.append(dict(T=T, kernel=kernel, solver=f"szegedy_comb_d{degree}",
                                 eff=mQ.get("segment_efficiency"),
                                 far=mQ.get("segment_false_rate"),
                                 cos_QC=mQ.get("cos_QC"),
                                 p_succ=float(np.vdot(y, y).real / l1 ** 2),
                                 l1=l1, p_at_0=float(p(0.0)),
                                 p_at_1=float(p(1.0))))
                print(f"B: T={T} {kernel} szegedy_d{degree}: "
                      f"eff={mQ.get('segment_efficiency'):.3f} "
                      f"far={mQ.get('segment_false_rate'):.3f}", flush=True)

            # production comb reference (matrix-free, same harness)
            qd = qp.solve_qsvt(ham, degree=40)
            solq = qd["sol"] if isinstance(qd, dict) else qd[0]
            mQ4 = qp.quantum_metrics(np.asarray(solq), sol_C, truth, tau)
            rows.append(dict(T=T, kernel=kernel, solver="qsvt_comb_d40",
                             eff=mQ4.get("segment_efficiency"),
                             far=mQ4.get("segment_false_rate"),
                             cos_QC=mQ4.get("cos_QC"),
                             p_succ=(qd.get("P_anc") if isinstance(qd, dict) else np.nan)))
            rows.append(dict(T=T, kernel=kernel, solver="classical",
                             eff=mC.get("segment_efficiency"),
                             far=mC.get("segment_false_rate")))
            print(f"B: T={T} {kernel} refs done (classical eff="
                  f"{mC.get('segment_efficiency'):.3f} far={mC.get('segment_false_rate'):.3f})",
                  flush=True)

            # per-component normalized spectra (T=200 step only)
            if T == 200 and kernel == "step":
                ncomp, labels = connected_components(C, directed=False)
                tr = np.asarray(truth, bool)
                for ci in range(ncomp):
                    idx = np.nonzero(labels == ci)[0]
                    if idx.size == 1:
                        kind = "isolated_true" if tr[idx[0]] else "isolated_false"
                        spec_rows.append(dict(comp=ci, kind=kind, lam=0.0, size=1))
                        continue
                    sub = Dm[np.ix_(idx, idx)].toarray()
                    w = np.linalg.eigvalsh(sub)
                    frac = tr[idx].mean()
                    kind = ("pure_true" if frac == 1.0 else
                            "pure_false" if frac == 0.0 else "mixed")
                    for lam in w:
                        spec_rows.append(dict(comp=ci, kind=kind, lam=float(lam),
                                              size=int(idx.size)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "03_normalized_walk_metrics.csv"), index=False)
    ds = pd.DataFrame(spec_rows)
    ds.to_csv(os.path.join(OUT, "03_normalized_spectra.csv"), index=False)
    return df, ds


def figures(dfa, dfb, ds):
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.grid": True, "grid.color": "#e8e7e2",
                         "grid.linewidth": 0.6, "font.size": 11})
    # degree requirement vs alpha (T=200 step)
    sub = dfa[(dfa["T"] == 200) & (dfa.kernel == "step")].dropna(subset=["degree_required"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for _, r in sub.iterrows():
        c = MCOL.get(r.method.replace("_C", ""), "#52514e")
        ax.scatter(r.alpha, r.degree_required, s=70, color=c, zorder=3)
        ax.annotate(r.method, (r.alpha, r.degree_required), xytext=(5, 4),
                    textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("subnormalization α")
    ax.set_ylabel("measured min comb degree (tol 0.02)")
    ax.set_title("Degree cost of subnormalization (T=200, step): d ∝ α")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig03_degree_vs_alpha.png"), dpi=160)
    plt.close(fig)

    # normalized spectra scatter
    if len(ds):
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        kinds = ["pure_true", "pure_false", "mixed", "isolated_false"]
        cols = {"pure_true": "#008300", "pure_false": "#e34948",
                "mixed": "#eda100", "isolated_false": "#52514e"}
        for i, k in enumerate(kinds):
            g = ds[ds.kind == k]
            if not len(g):
                continue
            jitter = (np.random.default_rng(1).random(len(g)) - 0.5) * 0.5
            ax.scatter(g.lam, i + jitter, s=8, alpha=0.5, color=cols[k], label=k)
        for l in P4_LINES_NORM:
            ax.axvline(l, color="#008300", lw=1, ls="--", alpha=0.6)
        for l in (-1, 0, 1):
            ax.axvline(l, color="#e34948", lw=1, ls=":", alpha=0.6)
        ax.set_yticks(range(len(kinds)))
        ax.set_yticklabels(kinds)
        ax.set_xlabel("eigenvalue of D$^{-1/2}$C D$^{-1/2}$")
        ax.set_title("Normalized-walk spectrum by cluster type (T=200, step, clean)\n"
                     "dashes: P4 true lines ±cos(π/5), ±cos(2π/5); dots: hub/isolated lines {−1,0,1}")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig03_normalized_spectrum.png"), dpi=160)
        plt.close(fig)

    # metrics comparison bars
    if len(dfb):
        sub = dfb[(dfb.kernel == "step")]
        solvers = ["classical", "qsvt_comb_d40", "szegedy_comb_d60", "szegedy_comb_d90"]
        Ts = sorted(sub["T"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        w = 0.8 / len(solvers)
        cols = {"classical": "#52514e", "qsvt_comb_d40": "#4a3aa7",
                "szegedy_comb_d60": "#eb6834", "szegedy_comb_d90": "#e34948"}
        for ax, metric, title in ((axes[0], "eff", "segment efficiency"),
                                  (axes[1], "far", "segment false rate")):
            for si, sv in enumerate(solvers):
                g = sub[sub.solver == sv].set_index("T").reindex(Ts)
                ax.bar(np.arange(len(Ts)) + si * w, g[metric], width=w * 0.92,
                       color=cols[sv], label=sv)
            ax.set_xticks(np.arange(len(Ts)) + 1.5 * w)
            ax.set_xticklabels([f"T={t}" for t in Ts])
            ax.set_title(title)
        axes[0].set_ylim(0, 1.02)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle("α=1 normalized-walk comb vs production comb (step, clean, rep 0)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig03_normalized_metrics.png"), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    dfa = part_A()
    dfb, ds = part_B()
    figures(dfa, dfb, ds)
    print("done")
