#!/usr/bin/env python3
"""Noisy_Ladder + Noisy_Density first look — realistic noise, fixed vs formula ε.

The first store campaign with ALL THREE noise knobs nonzero (σ_scatt = 1e-4,
σ_res = 10 µm, drop = 1%): a size ladder (φ_max = 0.2) and a density grid
(φ_max ∈ {0.01, 0.02, 0.05, 0.1}), each solved through two ε windows —
the fixed 2 mrad convention and the noise-matched formula ε = 3.178 mrad
(the fixed window is deliberately NARROWER than the formula one here).

Everything reads the recomputed qtrk_store metrics view; activation spectra
and the validity gate stream the stored solution vectors (classical raw;
1BQF via the canonical rescale_to_signal onto the classical signal support).
Per-solve validity gate: max|x| <= 50 (ERF Youden convention, Eps2 §7.12).

Outputs
  figures/noisy_ladder_eff_far.png      2x2 classical/1BQF x eff/far vs T (φ=0.2)
  figures/noisy_density_far.png         density grid: far vs T, φ ramp, both arms
  figures/noisy_activation_spectra.png  pooled per-segment spectra at T=200
  figures/noisy_explosion_map.png       excluded_frac maps (gate max|x|>50)
  results/noisy_first_look.csv          per-cell aggregates incl. gating
"""
from __future__ import annotations

import os
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

TAU = 0.35                # γ=3 classical absolute threshold
EXPLODE_MAX = 50.0        # per-solve validity gate
EPS_FIX, EPS_FORM = 0.002, 0.003178

C_FIX, C_FORM = "#eb6834", "#2a78d6"     # fixed = orange, formula = blue
C_QUANT = "#e34948"
PHI_RAMP = {0.01: "#0b3d73", 0.02: "#2a78d6", 0.05: "#74aede", 0.1: "#b9d7f0"}
FAR_FLOOR = 5e-5

FOOT = (r"qtrk_store metrics view · Noisy_Ladder ($\phi_{max}$=0.2) + Noisy_Density "
        r"($\phi_{max}\in\{0.01,0.02,0.05,0.1\}$) · $\sigma_{scatt}$=1e-4, "
        r"$\sigma_{res}$=10 µm, drop=1% · $\varepsilon$: set 2 mrad vs formula 3.178 mrad · "
        r"$\gamma$=3 $\delta$=1 · step kernel · T$\in$[10,1000]"
        "\n"
        r"classical MINRES 20 rep ($\tau$=0.35) · 1BQF matrix-free statevector 3 rep "
        r"(5 at T$\geq$700), wp99 headline · "
        r"validity gate max$|x|\leq$50 per classical solve (exploded solves counted, "
        r"spectra pooled over valid solves only) · open marker = measured 0")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def footer(fig):
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")


def load_view() -> pd.DataFrame:
    store = os.environ.get("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
    m = pd.read_csv(os.path.join(store, "manifest", "metrics.csv"))
    n = m[m.study.isin(["Noisy_Ladder", "Noisy_Density"])].copy()
    n["arm"] = np.where(n.eps_provenance == "set", "fixed", "formula")
    return n


def agg(g: pd.DataFrame, cols: list) -> pd.DataFrame:
    return g.groupby("n_trk")[cols].agg(["mean", "sem"])


def _zero_open(ax, T, mean, color):
    zero = np.asarray(mean) <= 0
    if zero.any():
        ax.plot(np.asarray(T)[zero], np.full(zero.sum(), FAR_FLOOR), marker="o",
                ms=5, mfc="white", mec=color, mew=1.2, ls="none", zorder=5)


def fig_ladder(n: pd.DataFrame):
    lad = n[n.phi_max == 0.2]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row, (solver, ec, fc, ttl) in enumerate([
            ("classical", "segment_efficiency", "segment_false_rate", "classical (τ=0.35)"),
            ("quantum", "segment_efficiency_wp", "segment_false_rate_wp", "1BQF (wp99)")]):
        ax_e, ax_f = axes[row]
        for arm, color in [("fixed", C_FIX), ("formula", C_FORM)]:
            g = lad[(lad.solver == solver) & (lad.arm == arm)]
            if not len(g):
                continue
            ge = agg(g, [ec, fc])
            T = ge.index.values
            lab = f"{arm} ε ({EPS_FIX*1e3:g} mrad)" if arm == "fixed" else \
                  f"{arm} ε ({EPS_FORM*1e3:g} mrad)"
            ax_e.errorbar(T, ge[(ec, "mean")], yerr=ge[(ec, "sem")], color=color,
                          marker="o", ms=3.5, lw=1.6, capsize=2, label=lab)
            fm = ge[(fc, "mean")].values
            ax_f.errorbar(T, np.clip(fm, FAR_FLOOR, None), yerr=ge[(fc, "sem")],
                          color=color, marker="o", ms=3.5, lw=1.6, capsize=2)
            _zero_open(ax_f, T, fm, color)
        ax_e.set_title(f"{ttl} — segment efficiency", loc="left")
        ax_f.set_title(f"{ttl} — false rate", loc="left")
        ax_e.set_ylim(0.4, 1.03)
        ax_f.set_yscale("log"); ax_f.set_ylim(2e-5, 1.5)
        for ax in (ax_e, ax_f):
            ax.set_xscale("log"); ax.set_xlim(9, 1300)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("T (tracks)")
    fig.suptitle("Realistic noise, size ladder (φ_max=0.2): fixed vs formula ε", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    footer(fig)
    fig.savefig(FIGS / "noisy_ladder_eff_far.png")
    plt.close(fig)


def fig_density(n: pd.DataFrame):
    den = n[n.phi_max < 0.2]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, (arm, ls) in enumerate([("fixed", "--"), ("formula", "-")]):
        for row, (solver, fc, ttl) in enumerate([
                ("classical", "segment_false_rate", "classical (τ=0.35)"),
                ("quantum", "segment_false_rate_wp", "1BQF (wp99)")]):
            ax = axes[row, col]
            for phi, color in PHI_RAMP.items():
                g = den[(den.solver == solver) & (den.arm == arm) & (den.phi_max == phi)]
                if not len(g):
                    continue
                ge = agg(g, [fc])
                T = ge.index.values
                fm = ge[(fc, "mean")].values
                ax.errorbar(T, np.clip(fm, FAR_FLOOR, None), yerr=ge[(fc, "sem")],
                            color=color, ls=ls, marker="o", ms=3, lw=1.4, capsize=2,
                            label=f"φ={phi:g}")
                _zero_open(ax, T, fm, color)
            ax.set_yscale("log"); ax.set_ylim(2e-5, 1.5)
            ax.set_xscale("log"); ax.set_xlim(9, 1300)
            ax.set_title(f"{ttl} — {arm} ε", loc="left")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("T (tracks)")
    for ax in axes[:, 0]:
        ax.set_ylabel("false rate")
    fig.suptitle("Realistic noise, density grid: false rate vs T per cone φ_max", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    footer(fig)
    fig.savefig(FIGS / "noisy_density_far.png")
    plt.close(fig)


def stream_gate(n: pd.DataFrame) -> pd.DataFrame:
    """max|x| for every classical solve (validity gate) — per-cell excluded_frac."""
    cls = n[n.solver == "classical"]
    rows = []
    for r in cls.itertuples(index=False):
        try:
            sol = qp.load_solution(r.sol_key)["sol"]
        except Exception:
            continue
        rows.append(dict(study=r.study, arm=r.arm, phi_max=r.phi_max, n_trk=r.n_trk,
                         rep=r.rep, sol_key=r.sol_key, event_key=r.event_key,
                         ham_key=r.ham_key,
                         max_abs_x=float(np.max(np.abs(sol)))))
    g = pd.DataFrame(rows)
    g["excluded"] = g.max_abs_x > EXPLODE_MAX
    return g


def fig_explosion(gate: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # ladder line
    ax = axes[0]
    lad = gate[gate.phi_max == 0.2]
    for arm, color in [("fixed", C_FIX), ("formula", C_FORM)]:
        e = lad[lad.arm == arm].groupby("n_trk").excluded.mean()
        ax.plot(e.index, e.values, marker="o", ms=3.5, lw=1.6, color=color, label=arm)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("T (tracks)"); ax.set_ylabel("excluded fraction")
    ax.set_title("ladder (φ=0.2): λ_min→0 onset", loc="left")
    ax.legend(fontsize=8)
    # density heatmaps per arm
    for k, (arm, cmap) in enumerate([("fixed", "Oranges"), ("formula", "Blues")]):
        ax = axes[1 + k]
        den = gate[(gate.phi_max < 0.2) & (gate.arm == arm)]
        piv = den.pivot_table(index="phi_max", columns="n_trk", values="excluded",
                              aggfunc="mean")
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                       origin="lower")
        ax.set_xticks(range(len(piv.columns)), [str(t) for t in piv.columns], fontsize=7)
        ax.set_yticks(range(len(piv.index)), [f"{p:g}" for p in piv.index], fontsize=7)
        ax.set_xlabel("T (tracks)"); ax.set_ylabel("φ_max")
        ax.set_title(f"density: excluded_frac — {arm} ε", loc="left")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "#33322e")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Validity gate max|x| ≤ 50: explosion-regime onset at realistic noise", y=1.02)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    footer(fig)
    fig.savefig(FIGS / "noisy_explosion_map.png", bbox_inches="tight")
    plt.close(fig)


def fig_spectra(n: pd.DataFrame, gate: pd.DataFrame, T_sel: int = 200, phi_sel: float = 0.2):
    """Pooled per-segment activation spectra at T=T_sel, φ=phi_sel."""
    sel = n[(n.n_trk == T_sel) & (n.phi_max == phi_sel)]
    bad = set(gate[gate.excluded].sol_key)
    bad_events = set(gate[gate.excluded].apply(lambda r: (r.event_key, r.ham_key), axis=1)) \
        if gate.excluded.any() else set()
    truth_cache = {}

    def truth_for(ekey):
        if ekey not in truth_cache:
            truth_cache[ekey] = qp.truth_from_event(qp.load_event(qp.event_path(ekey)))
        return truth_cache[ekey]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    bins = np.linspace(-0.5, 2.0, 401)
    stats = []
    for col, arm in enumerate(["fixed", "formula"]):
        solC_by = {}
        for r in sel[(sel.arm == arm) & (sel.solver == "classical")].itertuples(index=False):
            solC_by[(r.event_key, r.ham_key)] = qp.load_solution(r.sol_key)["sol"]
        for row, solver in enumerate(["classical", "quantum"]):
            ax = axes[row, col]
            pooled_t, pooled_f, nexc = [], [], 0
            for r in sel[(sel.arm == arm) & (sel.solver == solver)].itertuples(index=False):
                if solver == "classical":
                    if r.sol_key in bad:
                        nexc += 1
                        continue
                    sol = solC_by[(r.event_key, r.ham_key)]
                else:
                    solC = solC_by.get((r.event_key, r.ham_key))
                    if solC is None or (r.event_key, r.ham_key) in bad_events:
                        nexc += 1
                        continue
                    sol = rescale_to_signal(qp.load_solution(r.sol_key)["sol"], solC, TAU)
                t = truth_for(r.event_key).astype(bool)
                pooled_t.append(sol[t]); pooled_f.append(sol[~t])
            if not pooled_t:
                ax.set_axis_off(); continue
            st, sf = np.concatenate(pooled_t), np.concatenate(pooled_f)
            ax.hist(np.clip(sf, bins[0], bins[-1]), bins=bins, histtype="stepfilled",
                    alpha=0.45, color=C_FIX if col == 0 else C_FORM, label="false")
            ax.hist(np.clip(st, bins[0], bins[-1]), bins=bins, histtype="step",
                    lw=1.4, color="#33322e", label="true")
            ax.axvline(TAU, color="#e34948", lw=1.0, ls="--")
            ax.set_yscale("log")
            nm = "classical" if solver == "classical" else "1BQF (rescaled to signal)"
            ax.set_title(f"{nm} — {arm} ε   (excl. {nexc})", loc="left", fontsize=9)
            stats.append(dict(arm=arm, solver=solver, n_true=len(st), n_false=len(sf),
                              n_excluded=nexc))
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("segment activation x")
    fig.suptitle(f"Pooled activation spectra at T={T_sel}, φ_max={phi_sel:g} "
                 f"(valid solves; τ=0.35 dashed)", y=0.995)
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    footer(fig)
    fig.savefig(FIGS / "noisy_activation_spectra.png")
    plt.close(fig)
    return pd.DataFrame(stats)


def main():
    n = load_view()
    print(f"[view] Noisy rows: {len(n)}")
    fig_ladder(n)
    fig_density(n)
    gate = stream_gate(n)
    fig_explosion(gate)
    sp = fig_spectra(n, gate)
    print(sp.to_string(index=False))

    # per-cell summary CSV (incl. gated/ungated classical far + excluded_frac)
    rows = []
    gk = gate.set_index("sol_key")
    for (study, arm, phi, T, solver), g in n.groupby(
            ["study", "arm", "phi_max", "n_trk", "solver"]):
        ec, fc = (("segment_efficiency", "segment_false_rate") if solver == "classical"
                  else ("segment_efficiency_wp", "segment_false_rate_wp"))
        d = dict(study=study, arm=arm, phi_max=phi, n_trk=T, solver=solver,
                 n_solves=len(g), eff=g[ec].mean(), eff_sem=g[ec].sem(),
                 far=g[fc].mean(), far_sem=g[fc].sem())
        if solver == "classical":
            sub = gk.reindex(g.sol_key)
            d["excluded_frac"] = float(sub.excluded.mean())
            ok = g[~g.sol_key.isin(set(gate[gate.excluded].sol_key))]
            d["far_valid"] = ok[fc].mean() if len(ok) else np.nan
            d["eff_valid"] = ok[ec].mean() if len(ok) else np.nan
        rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "noisy_first_look.csv", index=False)
    print(f"[csv] {len(out)} rows -> {RES/'noisy_first_look.csv'}")
    lad = out[(out.phi_max == 0.2) & (out.n_trk.isin([200, 1000]))]
    print(lad.to_string(index=False))


if __name__ == "__main__":
    main()
