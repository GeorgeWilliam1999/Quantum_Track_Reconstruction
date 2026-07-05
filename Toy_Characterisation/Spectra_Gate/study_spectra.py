#!/usr/bin/env python3
"""Activation spectra + validity-gate maps for the older store studies.

Programme standard (2026-07-03): every study reports segment efficiency,
false rate AND the activation spectrum. ERF, Verify and Noisy_Realistic have
them; this adds the missing two — Larger_Scatter (scatter x drop) and
Larger_Scatter_Density (cone x scatter) — from the store view + streamed
vectors, with the per-solve validity gate max|x| <= 50 (Eps2 §7.12 / ERF
Youden convention) making the λ_min→0 onset measurable per study.

Usage: study_spectra.py Larger_Scatter | Larger_Scatter_Density
Outputs figures/<study>_spectra_T200.png, figures/<study>_gate_map.png,
        results/<study>_gate.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SHARED = "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared"
for p in (_SHARED, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)
import qtrk_pipeline as qp
from qtrk_pipeline.metrics import rescale_to_signal

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RES = HERE / "results"
FIGS.mkdir(exist_ok=True), RES.mkdir(exist_ok=True)

TAU, EXPLODE_MAX = 0.35, 50.0
C_TRUE, C_FALSE = "#33322e", "#eb6834"

# per-study: the two cell axes (row_axis, col_axis) + spectra subset
CFG = {
    "Larger_Scatter": dict(
        row="hit_ineff", col="sigma_scatt",
        spectra_rows=[0.0, 0.02, 0.10], spectra_cols=[1e-4, 5e-4, 1e-3],
        row_lab="drop", col_lab="σ_scatt",
        foot=(r"qtrk_store view · Larger_Scatter: $\sigma_{scatt}\in\{1,3,5,7,10\}\times10^{-4}$"
              r" × drop$\in\{0,.01,.02,.05,.10\}$ · $\sigma_{res}$=0 · $\phi_{max}$=0.2 ·"
              r" formula $\varepsilon$ (0.42–4.24 mrad) · step · $\gamma$=3 $\delta$=1 ·"
              r" T$\in$[10,1000] · classical MINRES 20 rep ($\tau$=0.35) · gate max$|x|\leq$50")),
    "Epsilon_study_2": dict(
        row="sigma_res", col="sigma_scatt",
        spectra_rows=[0.0, 0.02, 0.05], spectra_cols=[1e-4, 5e-4],
        row_lab="σ_res", col_lab="σ_scatt",
        foot=(r"qtrk_store view · Epsilon_study_2: $\sigma_{res}\in\{0,.01,.02,.05\}$ ×"
              r" $\sigma_{scatt}\in\{1,3,5\}\times10^{-4}$ · drop=0 · $\phi_{max}$=0.2 ·"
              r" formula $\varepsilon$ · step · $\gamma$=3 $\delta$=1 ·"
              "\n"
              r"T$\in$[10,1000] · classical MINRES 20 rep ($\tau$=0.35) · gate max$|x|\leq$50")),
    "Larger_Scatter_Density": dict(
        row="phi_max", col="sigma_scatt",
        spectra_rows=[0.01, 0.05, 0.1], spectra_cols=[1e-4, 3e-4],
        row_lab="φ_max", col_lab="σ_scatt",
        foot=(r"qtrk_store view · Larger_Scatter_Density: $\phi_{max}\in\{.01,.02,.05,.1\}$ ×"
              r" $\sigma_{scatt}\in\{1,3\}\times10^{-4}$ · $\sigma_{res}$=0 · drop=0 ·"
              r" formula $\varepsilon$ (0.42/1.27 mrad) · step · $\gamma$=3 $\delta$=1 ·"
              "\n"
              r"T$\in$[10,1000] · classical MINRES 20 rep ($\tau$=0.35) · gate max$|x|\leq$50")),
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def main(study: str):
    cfg = CFG[study]
    # membership, not first-requester: shared baseline cells (e.g. LS drop=0)
    # are owned by another study in the `study` column but belong here too.
    # Exact-token membership (load_metrics) — a substring test would pull every
    # Larger_Scatter_Density row into Larger_Scatter.
    n = qp.load_metrics(study=study).copy()
    print(f"[{study}] view rows: {len(n)}")

    # ---- gate: stream every classical vector once -------------------------
    cls = n[n.solver == "classical"]
    rows = []
    for r in cls.itertuples(index=False):
        try:
            sol = qp.load_solution(r.sol_key)["sol"]
        except Exception:
            continue
        rows.append({cfg["row"]: getattr(r, cfg["row"]), cfg["col"]: getattr(r, cfg["col"]),
                     "n_trk": r.n_trk, "sol_key": r.sol_key, "event_key": r.event_key,
                     "ham_key": r.ham_key, "max_abs_x": float(np.max(np.abs(sol)))})
    g = pd.DataFrame(rows)
    g["excluded"] = g.max_abs_x > EXPLODE_MAX
    g.to_csv(RES / f"{study}_gate.csv", index=False)
    print(f"[gate] {len(g)} classical solves, excluded overall: {g.excluded.mean():.3f}")

    # ---- gate map: heatmap row x T per col value --------------------------
    cols = sorted(g[cfg["col"]].unique())
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.4), squeeze=False)
    for k, cv in enumerate(cols):
        ax = axes[0, k]
        piv = g[g[cfg["col"]] == cv].pivot_table(index=cfg["row"], columns="n_trk",
                                                 values="excluded", aggfunc="mean")
        im = ax.imshow(piv.values, aspect="auto", cmap="Reds", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(piv.columns)), [str(t) for t in piv.columns], fontsize=7)
        ax.set_yticks(range(len(piv.index)), [f"{p:g}" for p in piv.index], fontsize=7)
        ax.set_xlabel("T (tracks)"); ax.set_ylabel(cfg["row_lab"])
        ax.set_title(f"{cfg['col_lab']}={cv:g}", loc="left", fontsize=9)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "#33322e")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{study}: λ_min→0 onset map (excluded_frac, gate max|x| ≤ 50)", y=1.03)
    fig.tight_layout(rect=(0, 0.09, 1, 0.99))
    fig.text(0.01, 0.004, cfg["foot"], fontsize=6, color="#52514e", va="bottom")
    fig.savefig(FIGS / f"{study}_gate_map.png", bbox_inches="tight")
    plt.close(fig)

    # ---- pooled spectra at T=200 over the subset grid ---------------------
    T_SEL = 200
    srows, scols = cfg["spectra_rows"], cfg["spectra_cols"]
    truth_cache = {}

    def truth_for(ek):
        if ek not in truth_cache:
            truth_cache[ek] = qp.truth_from_event(qp.load_event(qp.event_path(ek))).astype(bool)
        return truth_cache[ek]

    bad = set(g[g.excluded].sol_key)
    bad_pts = set(zip(g[g.excluded].event_key, g[g.excluded].ham_key))
    bins = np.linspace(-0.5, 2.0, 401)
    fig, axes = plt.subplots(len(srows), len(scols), figsize=(4.0 * len(scols), 2.6 * len(srows)),
                             sharex=True, squeeze=False)
    for i, rv in enumerate(srows):
        for j, cv in enumerate(scols):
            ax = axes[i, j]
            sel = n[(n.n_trk == T_SEL) & (np.isclose(n[cfg["row"]], rv)) &
                    (np.isclose(n[cfg["col"]], cv))]
            solC = {}
            for r in sel[sel.solver == "classical"].itertuples(index=False):
                if r.sol_key not in bad:
                    solC[(r.event_key, r.ham_key)] = qp.load_solution(r.sol_key)["sol"]
            pt, pf, pq, nexc = [], [], [], 0
            for (ek, hk), sc in solC.items():
                t = truth_for(ek)
                pt.append(sc[t]); pf.append(sc[~t])
            for r in sel[sel.solver == "quantum"].itertuples(index=False):
                key = (r.event_key, r.ham_key)
                if key in bad_pts or key not in solC:
                    nexc += 1
                    continue
                sq = rescale_to_signal(qp.load_solution(r.sol_key)["sol"], solC[key], TAU)
                pq.append(sq)
            if not pt:
                msg = "all excluded" if len(sel) else "no data"
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha="center", fontsize=8, color="#79776f")
                ax.set_title(f"{cfg['row_lab']}={rv:g}, {cfg['col_lab']}={cv:g}",
                             loc="left", fontsize=8)
                continue
            ax.hist(np.clip(np.concatenate(pf), bins[0], bins[-1]), bins=bins,
                    histtype="stepfilled", alpha=0.45, color=C_FALSE, label="false")
            ax.hist(np.clip(np.concatenate(pt), bins[0], bins[-1]), bins=bins,
                    histtype="step", lw=1.3, color=C_TRUE, label="true")
            if pq:
                ax.hist(np.clip(np.concatenate(pq), bins[0], bins[-1]), bins=bins,
                        histtype="step", lw=1.0, color="#e34948", alpha=0.8, label="1BQF (resc.)")
            ax.axvline(TAU, color="#e34948", lw=0.9, ls="--")
            ax.set_yscale("log")
            ax.set_title(f"{cfg['row_lab']}={rv:g}, {cfg['col_lab']}={cv:g}", loc="left", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("segment activation x")
    fig.suptitle(f"{study}: pooled activation spectra at T={T_SEL} (valid solves; τ=0.35 dashed)",
                 y=0.998)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    fig.text(0.01, 0.004, cfg["foot"], fontsize=6, color="#52514e", va="bottom")
    fig.savefig(FIGS / f"{study}_spectra_T200.png")
    plt.close(fig)
    print(f"[figs] -> {FIGS}/{study}_*.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Larger_Scatter")
