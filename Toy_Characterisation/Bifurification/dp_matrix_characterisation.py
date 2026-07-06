#!/usr/bin/env python3
"""DP Hamiltonian — the MATRIX itself (George, 2026-07-06).

The pilot (dp_pilot / dp_spectrum_classical / dp_costs) established that the
composed system
    A(β, α) = A0 + β·B_fork(ε_B) + 4α·I + 2α·B_all ,   b = (δ + 4α)·1
solves and separates. This study characterises the MATRIX:

Q1  SPARSITY — where do the new terms actually live? nnz/pair scaling of C
    (continuation), B_fork (co-hit, in-window) and B_all (co-hit, any angle)
    across T × noise, the fork-window (ε_B) dial, co-hit degree structure, and
    the support disjointness C ⊥ B (opposite-side vs same-side hit sharing).

Q2  GAMMA — what does γ need to be given the added terms? KEY IDENTITY:
    γ enters A only through the diagonal, so A(γ) = A(γ_ref) + (γ−γ_ref)·I —
    the spectrum SHIFTS rigidly with γ and its WIDTH is γ-invariant. The 1BQF
    (OneBQF) reads t = π/diag, i.e. the notch AUTO-TRACKS the diagonal
    s' = γ+δ+4α, and isolated clusters sit at λ = s' EXACTLY for every (γ, α):
    the notch kill of isolated clusters is γ- and α-robust. What γ must buy:
      (a) positive definiteness       λ_min(γ) = λ_min^ref + (γ−γ_ref) > 0
      (b) one filter period           λ_max(γ) ≤ 2·s'(γ)  (else the top
          cluster aliases through |cos(πλ/2s')| and is re-admitted)
    Both are closed-form in the measured extremes at γ_ref:
      γ_pd  = γ_ref − λ_min^ref
      γ_win = λ_max^ref − γ_ref − 2δ − 8α   [from λ_max^ref+(γ−γ_ref) = 2(γ+δ+4α)]
    The price of raising γ is filter CONTRAST: the occupied band maps to an
    ever-thinner slice of cos(πλ/2s'). Recorded per config at γ=3 and γ*.

Q3  ERRORS — per-FALSE-CLASS behaviour. Mechanism taxonomy on direct edges
    (priority order):  C↔true > fork(true) > cohit(true) > C↔false >
    cohit(false) > isolated;  true split into coupled/isolated. Classes depend
    only on (event, ε) — not on (β, α) — so one labelling serves all configs.

Q4  SEPARABILITY — can a single τ separate every false class from true?
    Per class and per config: fraction admitted at wp99 (the headline
    working point) and at eff=1 (τ just under min-true), the true-false gap,
    and rank AUC — for the CLASSICAL solve and for the EXACT 1BQF readout.
    1BQF emulation is exact and circuit-free: the one-bit filter is
        f(λ) = e^{iλt/2}·cos(λt/2) = (e^{iλt} + 1)/2 ,  t = π/s'
    so  x_Q = |(e^{iAt}u + u)/2| , u ∝ 1  via scipy expm_multiply (sparse,
    no Trotter, no dense eigh) — validated against dense eigh at T=50.

Operating points: clean / moderate (= pilot) / heavy; formula ε; step kernel;
γ_ref=3, δ=1, ε_B = ε. Outputs:
  results/dp_matrix_census.csv, dp_matrix_spectrum.csv, dp_matrix_classes.csv
  figures/dp_matrix_{sparsity,gamma,class_amplitudes,separation}.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, expm_multiply, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402
import dp_terms  # noqa: E402
import bif  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True), (HERE / "results").mkdir(exist_ok=True)

GAMMA, DELTA = 3.0, 1.0                     # reference γ for all measurements
NOISES = {                                   # formula ε per point
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01),
}
T_CENSUS = [10, 20, 50, 100, 200]
T_SPEC = [50, 100, 200]
SPEC_NOISES = ["moderate", "heavy"]
SPEC_CONFIGS = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0),
                (0.0, 0.03), (0.0, 0.1), (0.0, 0.3), (0.0, 1.0), (1.0, 0.3)]
CLASS_CONFIGS = [("base", 0.0, 0.0), ("fork", 1.0, 0.0),
                 ("occ", 0.0, 0.3), ("both", 1.0, 0.3)]
T_CLASS = [50, 100, 200]
CLASS_NOISES = ["moderate", "heavy"]
EPSB_RATIOS = [0.5, 1.0, 2.0, 4.0]
GATE = 50.0                                  # validity gate max|x| <= 50
FALSE_CLASSES = ["C-true", "fork-true", "cohit-true", "C-false",
                 "cohit-false", "isolated"]
CLASS_ORDER = FALSE_CLASSES + ["TRUE-coupled", "TRUE-isolated"]

FOOT = (r"DP matrix characterisation · A($\beta,\alpha$) = A0 + $\beta B_{fork}$"
        r"($\varepsilon_B$=$\varepsilon$) + $4\alpha I$ + $2\alpha B_{all}$, "
        r"b = ($\delta$+4$\alpha$)1 · step kernel, formula $\varepsilon$, "
        r"$\gamma_{ref}$=3 $\delta$=1 · noise clean/moderate/heavy = $\sigma_{res}$ 0/10/20 µm "
        r"(drop 0/1/1%), $\sigma_{scatt}$=1e-4, $\phi_{max}$=0.2"
        "\n"
        r"1BQF: t = $\pi$/s', s' = $\gamma$+$\delta$+4$\alpha$ (notch auto-tracks the "
        r"diagonal); exact filter emulation $x_Q$ = |(e$^{iAt}$u + u)/2|, u $\propto$ 1 "
        r"(expm_multiply, no Trotter) · wp99 = $\tau$ just under the 1% true quantile "
        r"(floored) · gate max|x| $\leq$ 50 · classes on direct edges, priority "
        r"C$\leftrightarrow$true > fork(true) > cohit(true) > C$\leftrightarrow$false > "
        r"cohit(false) > isolated")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------
def get_ham(T, noise, rep=0):
    nz = NOISES[noise]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=GAMMA, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    return ham, truth, eps


def adjacency(A0, s):
    """0/1 continuation adjacency C from the base matrix A0 = s·I − C_w."""
    C = (s * sp.identity(A0.shape[0], format="csr") - A0.tocsr()).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    return (abs(C) > 1e-9).astype(np.int8).tocsr()


def pairs(G):
    return int(G.nnz) // 2


def extremes(A, name=""):
    """(λ_min, λ_max) of a sparse symmetric matrix via Lanczos."""
    try:
        lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                         maxiter=8000, tol=1e-7)[0])
        hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                         maxiter=8000, tol=1e-7)[0])
        return lo, hi
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] eigsh failed for {name}: {e}", flush=True)
        return np.nan, np.nan


def classify(ham, truth, eps_B):
    """Mechanism class per segment from direct edges (β/α-independent)."""
    n = ham.n_segments
    s = float(ham.A.diagonal()[0])
    C = adjacency(ham.A, s)
    Bf = dp_terms.fork_graph_eps(ham, float(eps_B))
    Ba = dp_terms.cohit_graph(ham)
    t = truth.astype(np.float64)

    def to_true(G):
        return (np.asarray(G @ t).ravel() > 0.5) if G.nnz else np.zeros(n, bool)

    def any_edge(G):
        return (np.asarray(G.getnnz(axis=1)).ravel() > 0) if G.nnz else np.zeros(n, bool)

    c_t, c_a = to_true(C), any_edge(C)
    f_t = to_true(Bf)
    a_t, a_a = to_true(Ba), any_edge(Ba)
    cls = np.full(n, "isolated", dtype=object)
    F = ~truth
    cls[F & a_a] = "cohit-false"
    cls[F & c_a & ~c_t] = "C-false"
    cls[F & a_t] = "cohit-true"
    cls[F & f_t] = "fork-true"
    cls[F & c_t] = "C-true"
    cls[truth] = np.where(c_a[truth] | a_a[truth], "TRUE-coupled", "TRUE-isolated")
    graphs = dict(C=C, Bf=Bf, Ba=Ba)
    return cls, graphs


def emulate_1bqf(A, n_seg):
    """Exact one-bit-filter readout |(e^{iAt}u + u)/2|, u ∝ 1 (padded-uniform).

    Padding rows are diagonal-only (λ = s' exactly) and thus notched to zero,
    so only the real-dim uniform component matters; the 1/sqrt(n_pad) scale
    carries into P_anc but not the (scale-invariant) wp99 metrics.
    """
    s_prime = float(A.diagonal()[0])
    t = np.pi / s_prime
    n_pad = 1 << int(np.ceil(np.log2(max(n_seg, 2))))
    u = np.full(n_seg, 1.0 / np.sqrt(n_pad))
    t0 = time.time()
    y = 0.5 * (expm_multiply((1j * t) * A.tocsc().astype(np.complex128), u) + u)
    p_anc = float(np.vdot(y, y).real)          # padded rows contribute exactly 0
    return np.abs(y), p_anc, s_prime, time.time() - t0


def emulate_dense_check(A, n_seg):
    """Dense-eigh reference emulation (validation only, small n)."""
    s_prime = float(A.diagonal()[0])
    t = np.pi / s_prime
    n_pad = 1 << int(np.ceil(np.log2(max(n_seg, 2))))
    lam, V = eigh(A.toarray())
    c = V.T @ np.full(n_seg, 1.0 / np.sqrt(n_pad))
    y = V @ (0.5 * (np.exp(1j * lam * t) + 1.0) * c)
    return np.abs(y)


def class_rows(x, truth, cls, meta):
    """Long-format per-class separation rows for one score vector."""
    ax = np.abs(np.asarray(x, float))
    tau_wp = qp.working_point_threshold(ax, truth)
    m_wp = qp.metrics_at(ax, truth, tau_wp)
    t_amp = ax[truth]
    tau_e1 = float(t_amp.min() - 1e-12) if truth.any() else np.nan
    act_e1 = ax > tau_e1
    far_e1 = float((act_e1 & ~truth).sum() / max(act_e1.sum(), 1))
    gap = float(t_amp.min() - ax[~truth].max()) if (truth.any() and (~truth).any()) else np.nan
    out = [dict(meta, cls="ALL", n=len(ax),
                frac_wp=float((ax > tau_wp).mean()), frac_e1=float(act_e1.mean()),
                med=float(np.median(ax)), q99=float(np.quantile(ax, 0.99)),
                mx=float(ax.max()), auc=np.nan,
                tau_wp=tau_wp, eff_wp=m_wp["segment_efficiency"],
                far_wp=m_wp["segment_false_rate"], pur_wp=m_wp["segment_purity"],
                tau_e1=tau_e1, far_e1=far_e1, gap=gap)]
    for c in CLASS_ORDER:
        m = cls == c
        if not m.any():
            continue
        xa = ax[m]
        vs_true = (np.nan if c.startswith("TRUE") or not truth.any() else
                   bif.auc(np.r_[xa, t_amp], np.r_[np.zeros(m.sum(), bool),
                                                   np.ones(int(truth.sum()), bool)]))
        out.append(dict(meta, cls=c, n=int(m.sum()),
                        frac_wp=float((xa > tau_wp).mean()),
                        frac_e1=float((xa > tau_e1).mean()),
                        med=float(np.median(xa)), q99=float(np.quantile(xa, 0.99)),
                        mx=float(xa.max()), auc=vs_true,
                        tau_wp=tau_wp, eff_wp=np.nan, far_wp=np.nan, pur_wp=np.nan,
                        tau_e1=tau_e1, far_e1=np.nan, gap=np.nan))
    return out


def append_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return df


# --------------------------------------------------------------------------
# Part 1 — sparsity census
# --------------------------------------------------------------------------
def part1_census():
    out = HERE / "results" / "dp_matrix_census.csv"
    out.unlink(missing_ok=True)
    for noise in NOISES:
        for T in T_CENSUS:
            t0 = time.time()
            ham, truth, eps = get_ham(T, noise)
            n = ham.n_segments
            s = GAMMA + DELTA
            C = adjacency(ham.A, s)
            Ba = dp_terms.cohit_graph(ham)
            deg = np.asarray(Ba.getnnz(axis=1)).ravel()
            row = dict(noise=noise, T=T, n_seg=n, n_true=int(truth.sum()),
                       epsilon=eps, C_pairs=pairs(C), Ball_pairs=pairs(Ba),
                       cohit_deg_max=int(deg.max(initial=0)),
                       cohit_deg_mean=float(deg.mean()) if n else 0.0,
                       # constant column set so incremental appends stay aligned
                       lam_min_Bfork=np.nan, lam_max_Bfork=np.nan,
                       lam_min_M=np.nan, lam_max_M=np.nan)
            for r in EPSB_RATIOS:
                Bf = dp_terms.fork_graph_eps(ham, r * eps)
                row[f"Bfork_pairs_r{r:g}"] = pairs(Bf)
                if r == 1.0:
                    row["CBf_overlap"] = int(C.multiply(Bf).nnz) // 2
                    fdeg = np.asarray(Bf.getnnz(axis=1)).ravel()
                    row["fork_deg_max"] = int(fdeg.max(initial=0))
                    # fork pairs that involve >=1 true segment (clone-of-track)
                    if Bf.nnz:
                        co = sp.triu(Bf, k=1).tocoo()
                        tv = truth
                        row["Bfork_pairs_wtrue"] = int((tv[co.row] | tv[co.col]).sum())
                    else:
                        row["Bfork_pairs_wtrue"] = 0
            row["CBa_overlap"] = int(C.multiply(Ba).nnz) // 2
            if T in T_SPEC:                    # term-spectra for the γ bounds
                row["lam_min_Bfork"], row["lam_max_Bfork"] = (
                    extremes(dp_terms.fork_graph_eps(ham, eps).astype(float), "Bfork")
                    if row["Bfork_pairs_r1"] else (0.0, 0.0))
                M = (2.0 * sp.identity(n, format="csr") + Ba.astype(float)).tocsr()
                row["lam_min_M"], row["lam_max_M"] = extremes(M, "M=2I+Ball")
            append_csv(out, [row])
            print(f"[census] {noise} T={T}: n_seg={n} C={row['C_pairs']} "
                  f"Bf={row['Bfork_pairs_r1']} (w/true {row.get('Bfork_pairs_wtrue', 0)}) "
                  f"Ba={row['Ball_pairs']} degmax={row['cohit_deg_max']} "
                  f"[{time.time() - t0:.0f}s]", flush=True)


# --------------------------------------------------------------------------
# Part 2 — spectral extremes → γ requirement
# --------------------------------------------------------------------------
def gamma_map(lam_min_ref, lam_max_ref, alpha):
    """Closed-form γ requirements from extremes measured at γ_ref = GAMMA."""
    g_pd = GAMMA - lam_min_ref                       # λ_min(γ) = 0 crossing
    g_win = lam_max_ref - GAMMA - 2 * DELTA - 8 * alpha  # λ_max(γ) = 2s'(γ)
    return g_pd, g_win, max(g_pd, g_win)


def filter_abs(lam, gamma, alpha):
    """|f(λ)| at working γ: band shifts by (γ−γ_ref), s' = γ+δ+4α."""
    s_p = gamma + DELTA + 4 * alpha
    lam_shift = lam + (gamma - GAMMA)
    return float(abs(np.cos(np.pi * lam_shift / (2 * s_p))))


def part2_spectrum():
    out = HERE / "results" / "dp_matrix_spectrum.csv"
    out.unlink(missing_ok=True)
    for noise in SPEC_NOISES:
        for T in T_SPEC:
            ham, truth, eps = get_ham(T, noise)
            for beta, alpha in SPEC_CONFIGS:
                t0 = time.time()
                A, b, _, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                                   gamma=GAMMA, delta=DELTA)
                lo, hi = extremes(A, f"A(b={beta},a={alpha})")
                g_pd, g_win, g_star = gamma_map(lo, hi, alpha)
                g_use = max(g_star, GAMMA)
                row = dict(noise=noise, T=T, n_seg=ham.n_segments, beta=beta,
                           alpha=alpha, epsilon=eps, **info,
                           lam_min=lo, lam_max=hi, width=hi - lo,
                           s_prime_g3=GAMMA + DELTA + 4 * alpha,
                           gamma_pd=g_pd, gamma_win=g_win, gamma_star=g_star,
                           ok_at_g3=bool((lo > 0) and
                                         (hi <= 2 * (GAMMA + DELTA + 4 * alpha))),
                           f_lo_g3=filter_abs(lo, GAMMA, alpha),
                           f_hi_g3=filter_abs(hi, GAMMA, alpha),
                           f_lo_star=filter_abs(lo, g_use, alpha),
                           f_hi_star=filter_abs(hi, g_use, alpha),
                           contrast_g3=filter_abs(lo, GAMMA, alpha)
                                       - filter_abs(hi, GAMMA, alpha),
                           contrast_star=filter_abs(lo, g_use, alpha)
                                        - filter_abs(hi, g_use, alpha),
                           t_wall=time.time() - t0)
                append_csv(out, [row])
                print(f"[spec] {noise} T={T} β={beta:g} α={alpha:g}: "
                      f"λ∈[{lo:.3f},{hi:.2f}] w={hi - lo:.2f} γ_pd={g_pd:.2f} "
                      f"γ_win={g_win:.2f} ok@γ3={row['ok_at_g3']} "
                      f"contrast(γ*)={row['contrast_star']:.4f} "
                      f"[{row['t_wall']:.0f}s]", flush=True)


# --------------------------------------------------------------------------
# Part 3 — false-class behaviour, classical + exact 1BQF emulation
# --------------------------------------------------------------------------
def part3_classes():
    out = HERE / "results" / "dp_matrix_classes.csv"
    out.unlink(missing_ok=True)
    jobs = []
    for noise in CLASS_NOISES:
        for T in T_CLASS:
            for rep in ([0, 1, 2] if T <= 100 else [0]):
                jobs.append((noise, T, rep))
    jobs += [("clean", 100, 0)]                     # do-no-harm reference
    for noise, T, rep in jobs:
        ham, truth, eps = get_ham(T, noise, rep=rep)
        cls, _ = classify(ham, truth, eps)
        cnt = pd.Series(cls).value_counts().to_dict()
        print(f"[class] {noise} T={T} rep{rep}: n_seg={ham.n_segments} " +
              " ".join(f"{k}={v}" for k, v in sorted(cnt.items())), flush=True)
        for name, beta, alpha in CLASS_CONFIGS:
            A, b, tau_att, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                                     gamma=GAMMA, delta=DELTA)
            t0 = time.time()
            x, _ = minres(A, b, rtol=1e-8, maxiter=8000)
            x = np.asarray(x).ravel()
            t_cls = time.time() - t0
            valid = bool(np.abs(x).max() <= GATE)
            meta = dict(noise=noise, T=T, rep=rep, config=name, beta=beta,
                        alpha=alpha, n_seg=ham.n_segments, epsilon=eps,
                        valid=valid, tau_att=tau_att, p_anc=np.nan,
                        emu_cos_dense=np.nan)
            rows = class_rows(x, truth, cls, dict(meta, solver="classical"))
            xq, p_anc, s_p, t_emu = emulate_1bqf(A, ham.n_segments)
            memu = dict(meta, solver="1bqf_emu", p_anc=p_anc)
            if T == 50 and noise == "moderate" and rep == 0 and name == "base":
                xq_ref = emulate_dense_check(A, ham.n_segments)
                memu["emu_cos_dense"] = qp.cos_sim(xq, xq_ref) \
                    if hasattr(qp, "cos_sim") else float(
                        np.dot(xq, xq_ref) / (np.linalg.norm(xq)
                                              * np.linalg.norm(xq_ref) + 1e-30))
                print(f"    [check] expm vs dense-eigh emulation cos = "
                      f"{memu['emu_cos_dense']:.12f}", flush=True)
            rows += class_rows(xq, truth, cls, memu)
            df = append_csv(out, rows)
            allc = df[df.cls == "ALL"]
            rc = allc[allc.solver == "classical"].iloc[0]
            rq = allc[allc.solver == "1bqf_emu"].iloc[0]
            print(f"  {name}: CLS eff_wp={rc.eff_wp:.3f} far_wp={rc.far_wp:.3f} "
                  f"far_e1={rc.far_e1:.3f} gap={rc.gap:+.4f} | "
                  f"1BQF eff_wp={rq.eff_wp:.3f} far_wp={rq.far_wp:.3f} "
                  f"far_e1={rq.far_e1:.3f} gap={rq.gap:+.4f} P_anc={p_anc:.3e} "
                  f"{'· INVALID' if not valid else ''} "
                  f"[minres {t_cls:.0f}s, emu {t_emu:.0f}s]", flush=True)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
NCOL = {"clean": "#79b465", "moderate": "#2a78d6", "heavy": "#e34948"}
CCOL = {"C-true": "#e34948", "fork-true": "#eb6834", "cohit-true": "#e6b422",
        "C-false": "#8e6fad", "cohit-false": "#2a78d6", "isolated": "#9a9890",
        "TRUE-coupled": "#33322e", "TRUE-isolated": "#79b465"}


def fig_sparsity(cen):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    for noise, d in cen.groupby("noise"):
        d = d.sort_values("T_")
        ax.loglog(d.T_, d.C_pairs, "o-", color=NCOL[noise], ms=4, lw=1.3,
                  label=f"C · {noise}")
        ax.loglog(d.T_, d.Ball_pairs.clip(lower=0.5), "s--", color=NCOL[noise],
                  ms=4, lw=1.3, alpha=0.75)
        ax.loglog(d.T_, d["Bfork_pairs_r1"].clip(lower=0.5), "^:",
                  color=NCOL[noise], ms=4, lw=1.3, alpha=0.75)
    ax.plot([], [], "ko-", ms=4, label="C (continuation)")
    ax.plot([], [], "ks--", ms=4, label="B_all (co-hit, any angle)")
    ax.plot([], [], "k^:", ms=4, label="B_fork (co-hit, in-window)")
    ax.set_xlabel("T (tracks)"); ax.set_ylabel("interaction pairs")
    ax.set_title("(a) pair count scaling (1BQF gate cost ∝ pairs)", loc="left",
                 fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    for noise, d in cen.groupby("noise"):
        d = d.sort_values("T_")
        y = [d[f"Bfork_pairs_r{r:g}"] for r in EPSB_RATIOS]
        for r, yy, a in zip(EPSB_RATIOS, y, [0.35, 1.0, 0.7, 0.45]):
            ax.loglog(d.T_, np.asarray(yy, float).clip(min=0.5), "o-",
                      color=NCOL[noise], ms=3, lw=1.1, alpha=a,
                      label=f"{noise} ε_B={r:g}ε" if r in (1.0, 4.0) else None)
    ax.set_xlabel("T (tracks)"); ax.set_ylabel("B_fork pairs")
    ax.set_title("(b) fork-window dial ε_B", loc="left", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[2]
    for noise, d in cen.groupby("noise"):
        d = d.sort_values("T_")
        ax.semilogx(d.T_, d.cohit_deg_max, "s-", color=NCOL[noise], ms=4,
                    lw=1.3, label=f"{noise} max")
        ax.semilogx(d.T_, d.cohit_deg_mean, "o--", color=NCOL[noise], ms=3,
                    lw=1.0, alpha=0.6, label=f"{noise} mean")
    ax.set_xlabel("T (tracks)"); ax.set_ylabel("co-hit degree (B_all)")
    ax.set_title("(c) co-hit degree → occupancy spectral radius "
                 "(λ_max(A) − s' ≈ 2α·deg)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle("DP terms — where the matrix mass lives: C ⊥ B supports; "
                 "B_all is the dense-fill driver, B_fork stays O(n_seg)", y=1.0)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.text(0.01, 0.004, FOOT, fontsize=5.6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_matrix_sparsity.png", bbox_inches="tight")
    plt.close(fig)


def fig_gamma(spec):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    sub = spec[(spec.alpha == 0)].sort_values("beta")
    ax = axes[0]
    for (noise, T), d in sub.groupby(["noise", "T_"]):
        if T != 200:
            continue
        ax.plot(d.beta, d.lam_min, "o-", color=NCOL[noise], ms=4,
                label=f"λ_min {noise}")
        ax.plot(d.beta, d.lam_max, "s--", color=NCOL[noise], ms=4,
                label=f"λ_max {noise}")
    ax.axhline(0, color="#e34948", lw=0.9, ls="--")
    ax.axhline(2 * (GAMMA + DELTA), color="#33322e", lw=0.9, ls=":",
               label="2s' (γ=3, α=0)")
    ax.set_xlabel("fork strength β"); ax.set_ylabel("eigenvalue")
    ax.set_title("(a) fork: band edges vs β (T=200, γ=3)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5)
    sub = spec[(spec.beta == 0) & (spec.alpha > 0)].sort_values("alpha")
    ax = axes[1]
    for (noise, T), d in sub.groupby(["noise", "T_"]):
        if T != 200:
            continue
        ax.loglog(d.alpha, d.lam_max, "s--", color=NCOL[noise], ms=4,
                  label=f"λ_max {noise}")
        ax.loglog(d.alpha, 2 * (GAMMA + DELTA + 4 * d.alpha), ":",
                  color="#33322e", lw=1.0,
                  label="2s'(α) window edge" if noise == "moderate" else None)
    ax.set_xlabel("occupancy strength α"); ax.set_ylabel("eigenvalue")
    ax.set_title("(b) occupancy: λ_max escapes the filter period ∀α (T=200, γ=3)",
                 loc="left", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axes[2]
    for noise, d in spec[(spec.alpha == 0) & (spec.T_ == 200)].groupby("noise"):
        d = d.sort_values("beta")
        ax.plot(d.beta, d.gamma_star.clip(lower=0), "o-", color=NCOL[noise],
                ms=4, label=f"γ* fork · {noise}")
    for noise, d in spec[(spec.beta == 0) & (spec.alpha > 0)
                         & (spec.T_ == 200)].groupby("noise"):
        d = d.sort_values("alpha")
        ax.plot(d.alpha, d.gamma_star.clip(lower=0), "s--", color=NCOL[noise],
                ms=4, label=f"γ* occ (x=α) · {noise}")
    ax.axhline(GAMMA, color="#33322e", lw=0.9, ls=":", label="γ = 3 (current)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlabel("β  (fork curves)   /   α  (occupancy curves)")
    ax.set_ylabel("required γ*")
    ax.set_title("(c) γ* = max(γ_pd, γ_win): fork is γ-fixable, occupancy is not "
                 "(contrast → 0)", loc="left", fontsize=9)
    ax.legend(fontsize=6.5)
    fig.suptitle("What γ must be: A(γ) = A(γ_ref) + (γ−γ_ref)I — γ shifts the "
                 "band, never shrinks it; the 1BQF needs λ ∈ (0, 2s') with "
                 "usable contrast", y=1.0)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.text(0.01, 0.004, FOOT, fontsize=5.6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_matrix_gamma.png", bbox_inches="tight")
    plt.close(fig)


def fig_classes(cl):
    """Per-class q99/median amplitude vs config, classical vs 1BQF (T=200 rep0)."""
    sel = cl[(cl.T_ == 200) & (cl.rep == 0) & (cl.cls != "ALL")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=True)
    for j, noise in enumerate(CLASS_NOISES):
        for i, solver in enumerate(["classical", "1bqf_emu"]):
            ax = axes[i, j]
            d = sel[(sel.noise == noise) & (sel.solver == solver)]
            xs = np.arange(len(CLASS_CONFIGS))
            for c in CLASS_ORDER:
                dc = d[d.cls == c].set_index("config").reindex(
                    [n for n, _, _ in CLASS_CONFIGS])
                if dc["med"].isna().all():
                    continue
                ax.plot(xs, dc.q99.clip(lower=1e-17), "o-", color=CCOL[c], ms=4,
                        lw=1.2, label=c if (i == 0 and j == 0) else None)
                ax.plot(xs, dc["med"].clip(lower=1e-17), "o--", color=CCOL[c],
                        ms=2.5, lw=0.8, alpha=0.5)
            dall = cl[(cl.T_ == 200) & (cl.rep == 0) & (cl.cls == "ALL")
                      & (cl.noise == noise) & (cl.solver == solver)]
            ax.plot(xs, dall.set_index("config").reindex(
                [n for n, _, _ in CLASS_CONFIGS]).tau_wp, "x:",
                color="#33322e", ms=6, lw=1.0, label="τ_wp99" if (i == 0 and j == 0) else None)
            ax.set_yscale("log")
            ax.set_xticks(xs, [n for n, _, _ in CLASS_CONFIGS])
            ax.set_title(f"{noise} · {'classical MINRES' if solver == 'classical' else 'exact 1BQF filter'}",
                         loc="left", fontsize=9)
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("|x| per class (q99 solid, median dashed)")
    fig.suptitle("Error behaviour: per-class amplitudes under the DP terms "
                 "(T=200, rep 0)", y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.955))
    fig.text(0.01, 0.004, FOOT, fontsize=5.6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_matrix_class_amplitudes.png",
                bbox_inches="tight")
    plt.close(fig)


def fig_separation(cl):
    """Verdict figure: per-class admitted fraction at wp99 and at eff=1."""
    sel = cl[(cl.cls.isin(FALSE_CLASSES))].copy()
    agg = (sel.groupby(["noise", "solver", "config", "cls"])
              [["frac_wp", "frac_e1", "n"]].mean().reset_index())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharey=True)
    for j, noise in enumerate(CLASS_NOISES):
        for i, solver in enumerate(["classical", "1bqf_emu"]):
            ax = axes[i, j]
            d = agg[(agg.noise == noise) & (agg.solver == solver)]
            xs = np.arange(len(CLASS_CONFIGS))
            w = 0.13
            for k, c in enumerate(FALSE_CLASSES):
                dc = d[d.cls == c].set_index("config").reindex(
                    [n for n, _, _ in CLASS_CONFIGS])
                ax.bar(xs + (k - 2.5) * w, dc.frac_e1.fillna(0).clip(lower=2e-4),
                       width=w * 0.92, color=CCOL[c],
                       label=c if (i == 0 and j == 0) else None)
                ax.plot(xs + (k - 2.5) * w, dc.frac_wp.fillna(0).clip(lower=2e-4),
                        "k_", ms=6, mew=1.4)
            ax.set_yscale("log")
            ax.set_ylim(2e-4, 1.5)
            ax.set_xticks(xs, [n for n, _, _ in CLASS_CONFIGS])
            ax.set_title(f"{noise} · {'classical' if solver == 'classical' else 'exact 1BQF filter'}",
                         loc="left", fontsize=9)
    axes[0, 0].legend(fontsize=6.5, ncol=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of class admitted\n(bar = at eff=1; tick = at wp99)")
    fig.suptitle("Separability verdict: which false classes survive the cut "
                 "(mean over T, reps; bars floored at 2e-4 for log display)",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.09, 1, 0.955))
    fig.text(0.01, 0.004, FOOT, fontsize=5.6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_matrix_separation.png", bbox_inches="tight")
    plt.close(fig)


def figures():
    cen = pd.read_csv(HERE / "results" / "dp_matrix_census.csv").rename(
        columns={"T": "T_"})
    spec = pd.read_csv(HERE / "results" / "dp_matrix_spectrum.csv").rename(
        columns={"T": "T_"})
    cl = pd.read_csv(HERE / "results" / "dp_matrix_classes.csv").rename(
        columns={"T": "T_"})
    fig_sparsity(cen)
    fig_gamma(spec)
    fig_classes(cl)
    fig_separation(cl)
    print("[figs] -> figures/dp_matrix_{sparsity,gamma,class_amplitudes,"
          "separation}.png", flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["census", "spectrum", "classes", "figures"]
    t00 = time.time()
    if "census" in parts:
        part1_census()
    if "spectrum" in parts:
        part2_spectrum()
    if "classes" in parts:
        part3_classes()
    if "figures" in parts:
        figures()
    print(f"[done] {time.time() - t00:.0f}s total", flush=True)
