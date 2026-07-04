#!/usr/bin/env python3
"""Classical-first spectral characterisation of the DP terms (George, 2026-07-03).

Before upgrading the fork/occupancy Hamiltonians to the 1BQF or QSVT we need to
know WHICH EIGENVALUES TO COMB: where the spectrum of
    A(β, α) = A0 + β·B_fork + 4α·I + 2α·B_all
sits, and which eigenvalues actually carry the solution/signal.

Method (n_seg is ~10k at T=50 but ~40k at T=100 and ~160k at T=200, so full
dense eigendecomposition is only feasible at T=50):

* T=50  — EXACT: dense eigh; per eigenpair the solution weight
  w_i ∝ (v_iᵀb/λ_i)² (eigencomponents of x = A⁻¹b — a comb line is needed
  wherever w is non-negligible) and truth content t_i = (v_iᵀ t̂)².
* T=100/200 — LANCZOS spectral measures: k-step Lanczos with start vector b
  (resp. t̂) gives the Gauss-quadrature nodes/weights of the b-measure
  dμ_b(λ) = Σ(v_iᵀb̂)²δ(λ−λ_i) — the same "where does b/truth live spectrally"
  answer at sparse-matvec cost. Solution weight per node ∝ ω_j/θ_j².
  Validated against the exact T=50 answer (fig 4).
* Every (T, config) — MINRES solve + Youden-J recalibrated τ vs the
  isolated-segment attractor τ (which occupancy breaks: pilot shows eff→0 at
  attractor τ while AUC≈1).

Operating point = the DP pilot's: low-but-nonzero noise, formula ε, step, γ=3
δ=1, rep 0. Comb domain marked = the pipeline qsvt design window s ± 3.8,
s = γ+δ (solve_qsvt d=40 line-comb; run_shard widens if bounds exceed).

Outputs results/dp_classical_spectrum.csv +
figures/dp_classical_{comb_map,eigdensity,tau,lanczos_check}.png
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
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402
import dp_terms  # noqa: E402
import bif  # noqa: E402

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True), (HERE / "results").mkdir(exist_ok=True)

NOISE = dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01)
GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA                       # base diagonal; qsvt comb centre
DOM = (S - 3.8, S + 3.8)                # pipeline qsvt design window
T_DENSE = 50
T_LANCZOS = [100, 200]
K_LANCZOS = 300
CONFIGS = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0),
           (0.0, 0.1), (0.0, 0.3), (0.0, 1.0), (1.0, 0.3)]
NAMES = {(0.0, 0.0): "base", (0.5, 0.0): "fork β=0.5", (1.0, 0.0): "fork β=1",
         (2.0, 0.0): "fork β=2", (0.0, 0.1): "occ α=0.1", (0.0, 0.3): "occ α=0.3",
         (0.0, 1.0): "occ α=1", (1.0, 0.3): "both β=1 α=0.3"}
C_W, C_T = "#2a78d6", "#eb6834"
FOOT = (r"DP classical-first spectral study · $\sigma_{scatt}$=1e-4, $\sigma_{res}$=10 µm, "
        r"drop=1%, $\phi_{max}$=0.2 · formula $\varepsilon$=3.178 mrad · step · $\gamma$=3 "
        r"$\delta$=1 · rep 0 · A($\beta,\alpha$) = A0 + $\beta B_{fork}$($\varepsilon_B$="
        r"$\varepsilon$) + $4\alpha I$ + $2\alpha B_{all}$, b = $\delta$+4$\alpha$ ·"
        "\n"
        r"T=50 exact dense eigh; T$\in$\{100,200\} k=300 Lanczos spectral measures of b "
        r"and $\hat t$ (Gauss nodes/weights) · solution weight $\propto\omega/\theta^2$ · "
        r"comb domain = qsvt design window s$\pm$3.8, s=$\gamma$+$\delta$=4 (d=40 line-comb)"
        r" · Youden $\tau$ = argmax(TPR−FPR)")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def lanczos_measure(A, v0, k=K_LANCZOS):
    """Gauss-quadrature nodes/weights of the spectral measure of A w.r.t. v0.

    Returns (theta, omega) with Σω = 1: dμ_v0(λ) ≈ Σ_j ω_j δ(λ − θ_j), where
    ω_j = (e1ᵀu_j)² from the k-step Lanczos tridiagonal. Full (double)
    reorthogonalisation — k=300 keeps Q at ~k·n·8 bytes (~0.4 GB at n=160k).
    """
    A = A.tocsr()
    n = A.shape[0]
    k = min(k, n)
    Q = np.zeros((k, n))
    alpha, beta = np.zeros(k), np.zeros(max(k - 1, 1))
    q = np.asarray(v0, float).ravel()
    q = q / np.linalg.norm(q)
    Q[0] = q
    r = A @ q
    alpha[0] = q @ r
    r = r - alpha[0] * q
    kk = k
    for j in range(1, k):
        r -= Q[:j].T @ (Q[:j] @ r)
        r -= Q[:j].T @ (Q[:j] @ r)
        b = np.linalg.norm(r)
        if b < 1e-10:
            kk = j
            break
        beta[j - 1] = b
        q = r / b
        Q[j] = q
        r = A @ q - b * Q[j - 1]
        alpha[j] = q @ r
        r = r - alpha[j] * q
    theta, U = eigh_tridiagonal(alpha[:kk], beta[:kk - 1])
    return theta, U[0] ** 2


def youden_tau(scores: np.ndarray, truth: np.ndarray) -> float:
    """Threshold maximising TPR - FPR over the empirical score set."""
    order = np.argsort(scores)[::-1]
    t = truth[order].astype(float)
    tpr = np.cumsum(t) / max(t.sum(), 1)
    fpr = np.cumsum(1 - t) / max((1 - t).sum(), 1)
    kbest = int(np.argmax(tpr - fpr))
    s_sorted = scores[order]
    lo = s_sorted[kbest + 1] if kbest + 1 < len(s_sorted) else s_sorted[kbest] - 1e-9
    return float(0.5 * (s_sorted[kbest] + lo))


def weight_quantile(nodes: np.ndarray, w: np.ndarray, q: float) -> float:
    """Smallest λ with cumulative (normalised) weight ≥ q."""
    order = np.argsort(nodes)
    cw = np.cumsum(w[order]) / max(w.sum(), 1e-300)
    idx = int(np.searchsorted(cw, q))
    return float(nodes[order][min(idx, len(nodes) - 1)])


def analyse_config(ham, truth, beta, alpha, dense: bool):
    """One (event, β, α) point → stats row + plotting payload."""
    A, b, tau_att, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                             gamma=GAMMA, delta=DELTA)
    t_hat = truth.astype(float)
    t_hat /= np.linalg.norm(t_hat)
    t0 = time.time()
    if dense:
        lam, V = eigh(A.toarray())
        c = V.T @ b
        u = c / lam
        x = V @ u
        w_sol = u ** 2 / np.sum(u ** 2)          # exact solution weight per λ
        w_tru = (V.T @ t_hat) ** 2               # exact truth content per λ
        nodes_b, nodes_t = lam, lam
        lam_min, lam_max = float(lam[0]), float(lam[-1])
    else:
        th_b, om_b = lanczos_measure(A, b)
        th_t, om_t = lanczos_measure(A, t_hat)
        w_sol = (om_b / th_b ** 2)
        w_sol = w_sol / w_sol.sum()
        w_tru = om_t
        nodes_b, nodes_t = th_b, th_t
        try:
            lam_min = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                                  maxiter=4000)[0])
            lam_max = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                  maxiter=4000)[0])
        except Exception:
            lam_min, lam_max = float(th_b[0]), float(th_b[-1])
        sol, _ = minres(A, b, rtol=1e-8, maxiter=8000)
        x = np.asarray(sol).ravel()
    t_spec = time.time() - t0

    tau_J = youden_tau(x, truth)
    mA = qp.metrics_at(x, truth, tau_att)
    mJ = qp.metrics_at(x, truth, tau_J)
    in_b = (nodes_b >= DOM[0]) & (nodes_b <= DOM[1])
    in_t = (nodes_t >= DOM[0]) & (nodes_t <= DOM[1])
    row = dict(
        beta=beta, alpha=alpha, exact=bool(dense), **info,
        lam_min=lam_min, lam_max=lam_max,
        lam_wq001=weight_quantile(nodes_b, w_sol, 1e-3),
        lam_wq01=weight_quantile(nodes_b, w_sol, 1e-2),
        lam_wq50=weight_quantile(nodes_b, w_sol, 0.5),
        frac_w_in_dom=float(w_sol[in_b].sum()),
        frac_t_in_dom=float(w_tru[in_t].sum() / max(w_tru.sum(), 1e-300)),
        tau_attractor=tau_att, tau_youden=tau_J,
        eff_att=mA["segment_efficiency"], far_att=mA["segment_false_rate"],
        eff_youden=mJ["segment_efficiency"], far_youden=mJ["segment_false_rate"],
        auc=bif.auc(x, truth),
        true_med=float(np.median(x[truth])),
        false_q99=float(np.quantile(x[~truth], 0.99)) if (~truth).any() else np.nan,
        t_spectral=t_spec,
    )
    payload = dict(nodes_b=nodes_b, w_sol=w_sol, nodes_t=nodes_t, w_tru=w_tru,
                   x=x, truth=truth, tau_att=tau_att, tau_J=tau_J, row=row)
    return row, payload


def main():
    rows, dense_p, lan_p = [], {}, {}
    from lhcb_velo_toy.analysis import compute_epsilon
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    for T in [T_DENSE] + T_LANCZOS:
        dense = (T == T_DENSE)
        ev, _ = qp.ensure_event(n_trk=T, rep=0, **NOISE)
        ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                   gamma=GAMMA, delta=DELTA)
        truth = np.asarray(qp.truth_from_event(ev), bool)
        print(f"[T={T}] n_seg={ham.n_segments} n_true={int(truth.sum())} "
              f"mode={'dense' if dense else 'lanczos'}", flush=True)
        for beta, alpha in CONFIGS:
            row, payload = analyse_config(ham, truth, beta, alpha, dense)
            row.update(T=T, n_seg=int(ham.n_segments), epsilon=eps)
            rows.append(row)
            (dense_p if dense else lan_p)[(T, beta, alpha)] = payload
            print(f"  β={beta:g} α={alpha:g}: λ∈[{row['lam_min']:.3f},"
                  f"{row['lam_max']:.3f}] w-q0.1%λ={row['lam_wq001']:.3f} "
                  f"w_in_dom={row['frac_w_in_dom']:.4f} "
                  f"τ_att={row['tau_attractor']:.3f}(eff {row['eff_att']:.3f}) "
                  f"τ_J={row['tau_youden']:.3f}(eff {row['eff_youden']:.3f}/"
                  f"far {row['far_youden']:.3f}) [{row['t_spectral']:.0f}s]",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "dp_classical_spectrum.csv", index=False)
    print(f"[csv] {len(df)} rows -> results/dp_classical_spectrum.csv", flush=True)

    T_SHOW = T_LANCZOS[-1]

    # ---- fig 1: THE comb map (T=200 Lanczos) --------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), CONFIGS):
        p = lan_p[(T_SHOW, *key)]
        ax.axvspan(*DOM, color="#2a78d6", alpha=0.07)
        ax.axvline(S, color="#79776f", lw=0.8, ls=":")
        ax.axvline(0, color="#e34948", lw=0.8, ls="--")
        m, s_, b_ = ax.stem(p["nodes_b"], np.maximum(p["w_sol"], 1e-12),
                            basefmt=" ")
        plt.setp(m, markersize=2.5, color=C_W)
        plt.setp(s_, linewidth=0.7, color=C_W, alpha=0.6)
        ax.scatter(p["nodes_t"], np.maximum(p["w_tru"], 1e-12), s=8, color=C_T,
                   alpha=0.7, marker="^", zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 1)
        r = p["row"]
        ax.set_title(f"{NAMES[key]} · w-q0.1% λ={r['lam_wq001']:.2f} · "
                     f"w in dom {100 * r['frac_w_in_dom']:.1f}%",
                     loc="left", fontsize=8)
    axes[0, 0].plot([], [], color=C_W, marker="o", ms=3, lw=0.7,
                    label="solution weight (b-measure / θ²)")
    axes[0, 0].plot([], [], color=C_T, marker="^", ms=4, lw=0,
                    label="truth content (t̂-measure)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("eigenvalue λ (Lanczos node θ)")
    for ax in axes[:, 0]:
        ax.set_ylabel("weight (norm.)")
    fig.suptitle(f"Which eigenvalues to comb: A(β,α) solution/truth spectral weight "
                 f"(T={T_SHOW}, rep 0; shaded = qsvt design window s±3.8)", y=0.998)
    fig.tight_layout(rect=(0, 0.08, 1, 0.965))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_classical_comb_map.png")
    plt.close(fig)

    # ---- fig 2: exact eigenvalue density families (T=50 dense) --------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    fams = [("fork", [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0)]),
            ("occupancy", [(0.0, 0.0), (0.0, 0.1), (0.0, 0.3), (0.0, 1.0)]),
            ("both", [(0.0, 0.0), (1.0, 0.3)])]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
    alllam = np.concatenate([dense_p[(T_DENSE, *k)]["nodes_b"] for k in CONFIGS])
    bins = np.linspace(alllam.min() - 0.3, alllam.max() + 0.3, 160)
    for ax, (fam, keys) in zip(axes, fams):
        ax.axvspan(*DOM, color="#2a78d6", alpha=0.07)
        ax.axvline(0, color="#e34948", lw=0.8, ls="--")
        for k, key in enumerate(keys):
            ax.hist(dense_p[(T_DENSE, *key)]["nodes_b"], bins=bins, histtype="step",
                    lw=1.3, color=cmap[k], label=NAMES[key])
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue λ")
        ax.set_title(fam, loc="left", fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Exact spectral density under the DP terms (T={T_DENSE} dense eigh; "
                 f"shaded = comb domain, red = λ=0)", y=1.0)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_classical_eigdensity.png", bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: threshold recalibration (T=200) -----------------------------
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharex=True)
    bins = np.linspace(-0.25, 1.35, 321)
    for ax, key in zip(axes.ravel(), CONFIGS):
        p = lan_p[(T_SHOW, *key)]
        x, truth = p["x"], p["truth"]
        ax.hist(np.clip(x[~truth], bins[0], bins[-1]), bins=bins,
                histtype="stepfilled", alpha=0.45, color="#eb6834", label="false")
        ax.hist(np.clip(x[truth], bins[0], bins[-1]), bins=bins, histtype="step",
                lw=1.4, color="#33322e", label="true")
        ax.axvline(p["tau_att"], color="#e34948", lw=1.0, ls="--", label="attractor τ")
        ax.axvline(p["tau_J"], color="#2a78d6", lw=1.0, label="Youden τ")
        ax.set_yscale("log")
        r = p["row"]
        ax.set_title(f"{NAMES[key]} · Youden eff {r['eff_youden']:.3f} / "
                     f"far {r['far_youden']:.3f}", loc="left", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("segment activation x")
    fig.suptitle(f"Threshold recalibration under the DP terms (T={T_SHOW}, rep 0): "
                 f"attractor τ (dashed) vs Youden τ (solid)", y=0.998)
    fig.tight_layout(rect=(0, 0.08, 1, 0.965))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_classical_tau.png")
    plt.close(fig)

    # ---- fig 4: Lanczos-vs-exact validation (base config, T=50) -------------
    ev, _ = qp.ensure_event(n_trk=T_DENSE, rep=0, **NOISE)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=GAMMA, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in zip(axes, [(0.0, 0.0), (1.0, 0.3)]):
        pe = dense_p[(T_DENSE, *key)]
        _, pl = analyse_config(ham, truth, *key, dense=False)
        for nodes, w, lab, c in [(pe["nodes_b"], pe["w_sol"], "exact (dense eigh)", "#33322e"),
                                 (pl["nodes_b"], pl["w_sol"], f"Lanczos k={K_LANCZOS}", C_W)]:
            o = np.argsort(nodes)
            ax.plot(nodes[o], np.cumsum(w[o]), lw=1.4, color=c, label=lab,
                    ls="-" if "exact" in lab else "--")
        ax.axvspan(*DOM, color="#2a78d6", alpha=0.07)
        ax.set_xlabel("eigenvalue λ")
        ax.set_title(f"{NAMES[key]} (T={T_DENSE})", loc="left", fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("cumulative solution weight")
    fig.suptitle("Lanczos spectral measure reproduces the exact solution-weight "
                 "distribution", y=1.0)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_classical_lanczos_check.png",
                bbox_inches="tight")
    plt.close(fig)
    print("[figs] -> figures/dp_classical_{comb_map,eigdensity,tau,lanczos_check}.png",
          flush=True)


if __name__ == "__main__":
    main()
