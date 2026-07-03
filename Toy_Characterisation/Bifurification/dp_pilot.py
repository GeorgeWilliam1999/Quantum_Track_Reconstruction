#!/usr/bin/env python3
"""Denby–Peterson pilot: fork (β) and per-hit occupancy (α), alone + together.

Supervisor programme point (3), first data. Low-but-nonzero noise per the
2026-07-03 programme (σ_scatt=1e-4, σ_res=10 µm, drop=1%), formula ε, step
kernel, γ=3 δ=1. For each (T, rep, β, α): classical MINRES on the modified
system at the attractor-aware τ; 1BQF matrix-free statevector rescaled onto
the classical signal support; eff / far / AUC / cos_QC + eigenvalue edges
(λ_min, λ_max via Lanczos) — does the penalty fix false structure, and does
the 1BQF notch still land once the spectrum shifts?

Outputs results/dp_pilot.csv + figures/dp_{metrics,spectra,eigs}.png
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
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402
import dp_terms  # noqa: E402
import bif  # noqa: E402  (auc)

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True), (HERE / "results").mkdir(exist_ok=True)

NOISE = dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01)
GAMMA, DELTA = 3.0, 1.0
T_GRID = [50, 100, 200]
REPS = 3
BETAS = [0.0, 0.5, 1.0, 2.0]           # fork alone (alpha=0)
ALPHAS = [0.1, 0.3, 1.0]               # occupancy alone (beta=0)
COMBOS = [(1.0, 0.3)]                  # together
FOOT = (r"DP pilot · $\sigma_{scatt}$=1e-4, $\sigma_{res}$=10 µm, drop=1% · formula "
        r"$\varepsilon$=3.178 mrad · step · $\gamma$=3 $\delta$=1 · fork = $\beta B_{fork}$ "
        r"($\varepsilon_B$=$\varepsilon$, off-diag mode) · occupancy = per-hit "
        r"$\alpha\Sigma_h(o_h-1)^2$ = $4\alpha I + 2\alpha B_{all}$, b+=4$\alpha$ ·"
        "\n"
        r"$\tau$ = isolated attractor + 0.10 (occ-aware) · classical MINRES · 1BQF "
        r"matrix-free statevector rescaled on the classical signal support · "
        r"T$\in$\{50,100,200\} 3 rep · eigenvalue edges via Lanczos")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def solve_point(ham, truth, beta, alpha, run_quantum=True):
    A, b, tau, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                         gamma=GAMMA, delta=DELTA)
    t0 = time.time()
    sol_C, _ = minres(A, b, rtol=1e-8, maxiter=8000)
    sol_C = np.asarray(sol_C).ravel()
    tC = time.time() - t0
    mC = qp.metrics_at(sol_C, truth, tau)
    lam = {}
    try:
        lam["lam_min"] = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                                     maxiter=4000)[0])
        lam["lam_max"] = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                     maxiter=4000)[0])
    except Exception:
        lam["lam_min"] = lam["lam_max"] = np.nan
    out = dict(tau=tau, **info, **lam,
               eff_C=mC["segment_efficiency"], far_C=mC["segment_false_rate"],
               auc_C=bif.auc(sol_C, truth), max_abs_x=float(np.max(np.abs(sol_C))),
               t_classical=tC)
    if run_quantum:
        from helpers import solve_quantum_statevector
        raw, p_anc, _ = solve_quantum_statevector(A, b, device="CPU")
        raw = np.abs(np.asarray(raw[: A.shape[0]], float))
        mask = sol_C > tau
        den = np.linalg.norm(raw[mask]) if mask.any() else np.linalg.norm(raw)
        sol_Q = raw * (np.linalg.norm(sol_C[mask]) / den if den > 0 else 1.0)
        mQ = qp.metrics_at(sol_Q, truth, tau)
        wp = qp.metrics_at_wp(sol_Q, truth, qp.WP_TARGET_EFF)
        cos = float(np.dot(sol_Q, sol_C) /
                    (np.linalg.norm(sol_Q) * np.linalg.norm(sol_C) + 1e-300))
        out.update(eff_Q=mQ["segment_efficiency"], far_Q=mQ["segment_false_rate"],
                   eff_Qwp=wp["segment_efficiency"], far_Qwp=wp["segment_false_rate"],
                   auc_Q=bif.auc(sol_Q, truth), cos_QC=cos, P_anc=float(p_anc))
        return out, (sol_C, sol_Q)
    return out, (sol_C, None)


def main():
    scan = ([(b, 0.0) for b in BETAS] + [(0.0, a) for a in ALPHAS] + COMBOS)
    rows, spectra = [], {}
    for T in T_GRID:
        for rep in range(REPS):
            ev, ekey = qp.ensure_event(n_trk=T, rep=rep, **NOISE)
            from lhcb_velo_toy.analysis import compute_epsilon
            eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                       gamma=GAMMA, delta=DELTA)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            for beta, alpha in scan:
                run_q = T <= 200
                out, (sC, sQ) = solve_point(ham, truth, beta, alpha, run_q)
                rows.append(dict(T=T, rep=rep, beta=beta, alpha=alpha,
                                 epsilon=eps, n_seg=int(ham.n_segments), **out))
                if T == 200 and rep == 0 and (beta, alpha) in [(0.0, 0.0), (1.0, 0.0),
                                                               (0.0, 0.3), (1.0, 0.3)]:
                    spectra[(beta, alpha)] = (sC, sQ, truth, out["tau"])
                print(f"T={T} rep={rep} β={beta:g} α={alpha:g}: "
                      f"effC={out['eff_C']:.3f} farC={out['far_C']:.3f} "
                      f"aucC={out['auc_C']:.4f} "
                      + (f"farQwp={out.get('far_Qwp', float('nan')):.3f} "
                         f"cos={out.get('cos_QC', float('nan')):.3f}" if run_q else ""),
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "dp_pilot.csv", index=False)
    print(f"[csv] {len(df)} rows -> results/dp_pilot.csv")

    # ---- fig 1: metrics vs beta / alpha at each T --------------------------
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex="col")
    for j, (knob, vals, sel) in enumerate([
            ("beta", BETAS, df.alpha == 0.0),
            ("alpha", [0.0] + ALPHAS, df.beta == 0.0)]):
        for i, (col, lab) in enumerate([("far_C", "classical false rate"),
                                        ("far_Qwp", "1BQF wp99 false rate")]):
            ax = axes[i, j]
            for T, mk in zip(T_GRID, ["o", "s", "^"]):
                g = (df[sel & (df.T == T)].groupby(knob)[col].agg(["mean", "sem"]))
                if not len(g):
                    continue
                ax.errorbar(g.index, g["mean"], yerr=g["sem"], marker=mk, ms=4,
                            lw=1.5, capsize=2, label=f"T={T}")
            ax.set_yscale("log")
            ax.set_xlabel("β (fork)" if knob == "beta" else "α (occupancy)")
            ax.set_ylabel(lab)
    # efficiency panels (right column)
    for i, col in enumerate(["eff_C", "eff_Qwp"]):
        ax = axes[i, 2]
        for (knob, sel, ls) in [("beta", df.alpha == 0.0, "-"),
                                ("alpha", df.beta == 0.0, "--")]:
            for T, mk in zip(T_GRID, ["o", "s", "^"]):
                g = df[sel & (df.T == T)].groupby(knob)[col].mean()
                if len(g):
                    ax.plot(g.index, g.values, marker=mk, ms=3.5, ls=ls, lw=1.2,
                            label=f"T={T} {knob}" if i == 0 else None)
        ax.set_ylim(0.4, 1.03)
        ax.set_xlabel("β (solid) / α (dashed)")
        ax.set_ylabel("efficiency" if i == 0 else "wp99 efficiency")
    axes[0, 0].legend(fontsize=7)
    axes[0, 2].legend(fontsize=6)
    fig.suptitle("Denby–Peterson pilot: fork β and occupancy α, alone", y=0.998)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_metrics.png")
    plt.close(fig)

    # ---- fig 2: activation spectra base vs fork vs occ vs both (T=200) -----
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    bins = np.linspace(-0.25, 1.25, 301)
    for ax, (key, ttl) in zip(axes.ravel(), [((0.0, 0.0), "base"),
                                             ((1.0, 0.0), "fork β=1"),
                                             ((0.0, 0.3), "occupancy α=0.3"),
                                             ((1.0, 0.3), "both β=1 α=0.3")]):
        if key not in spectra:
            ax.set_axis_off(); continue
        sC, sQ, truth, tau = spectra[key]
        ax.hist(np.clip(sC[~truth], *bins[[0, -1]]), bins=bins, histtype="stepfilled",
                alpha=0.45, color="#eb6834", label="false (classical)")
        ax.hist(np.clip(sC[truth], *bins[[0, -1]]), bins=bins, histtype="step",
                lw=1.4, color="#33322e", label="true (classical)")
        if sQ is not None:
            ax.hist(np.clip(sQ, *bins[[0, -1]]), bins=bins, histtype="step",
                    lw=1.0, color="#e34948", alpha=0.85, label="1BQF (rescaled)")
        ax.axvline(tau, color="#e34948", lw=0.9, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"{ttl} (τ={tau:.3f})", loc="left", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("segment activation x")
    fig.suptitle("Activation spectra under the DP terms (T=200, rep 0)", y=0.998)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_spectra.png")
    plt.close(fig)

    # ---- fig 3: eigenvalue edges vs knob ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (knob, sel) in zip(axes, [("beta", df.alpha == 0.0),
                                      ("alpha", df.beta == 0.0)]):
        for T, mk in zip(T_GRID, ["o", "s", "^"]):
            g = df[sel & (df.T == T)].groupby(knob)[["lam_min", "lam_max"]].mean()
            ax.plot(g.index, g.lam_min, marker=mk, ms=4, lw=1.5, label=f"λ_min T={T}")
            ax.plot(g.index, g.lam_max, marker=mk, ms=4, lw=1.0, ls=":", alpha=0.7)
        ax.axhline(0, color="#e34948", lw=0.8, ls="--")
        ax.set_xlabel("β (fork)" if knob == "beta" else "α (occupancy)")
        ax.set_ylabel("eigenvalue edge")
        ax.legend(fontsize=7)
    fig.suptitle("Spectrum edges under the DP terms (dotted = λ_max)", y=1.0)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_eigs.png", bbox_inches="tight")
    plt.close(fig)
    print("[figs] -> figures/dp_{metrics,spectra,eigs}.png")


if __name__ == "__main__":
    main()
