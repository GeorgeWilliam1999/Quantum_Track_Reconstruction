#!/usr/bin/env python3
"""Figures and summary numbers of the segment-pruning study.

Reads outputs/<config>_scaling.csv, outputs/<config>_solves.csv and
outputs/circuit_cost.csv; writes figures/*.pdf|png and outputs/summary.md.
Every curve is a median over the repetitions at each T with the min-max
range shaded; power-law exponents are least-squares fits in log-log over
T >= 100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CONFIGS, CONFIG_LABEL, FIG, OUT

# Okabe-Ito, fixed assignment (never cycled)
C_CFG = {"paper": "#0072B2", "no_noise": "#7F7F7F", "wide": "#D55E00"}
C_SOL = {"classical": "#000000", "onebit_circuit": "#E69F00", "onebit_exact": "#CC79A7",
         "qsvt": "#009E73"}
L_SOL = {"classical": "classical", "onebit_circuit": "one-bit filter (package circuit)",
         "onebit_exact": "one-bit filter (exact readout)", "qsvt": "QSVT, degree 40"}
LS_VAR = {"full": "-", "deg1": "--", "comp3": ":"}
L_VAR = {"full": "full matrix", "deg1": "decoupled segments dropped",
         "comp3": "components < 3 dropped"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "legend.frameon": False, "lines.linewidth": 1.6})


def load(kind):
    frames = []
    for cfg in CONFIGS:
        p = OUT / f"{cfg}_{kind}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def med(df, col, by):
    g = df.groupby(by)[col]
    return g.median(), g.min(), g.max()


def fit_exponent(T, y):
    m = (T >= 100) & (y > 0)
    if m.sum() < 2:
        return np.nan, np.nan
    a, b = np.polyfit(np.log(T[m]), np.log(y[m]), 1)
    return a, np.exp(b)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def fig_matrix_size(sc, lines):
    cfgs = [c for c in CONFIGS if c in set(sc.config)]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.4 * len(cfgs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    lines.append("## Matrix size against track count (median over repetitions)\n")
    lines.append("| configuration | quantity | fitted exponent (T >= 100) | prefactor |")
    lines.append("|---|---|---|---|")
    for ax, cfg in zip(axes, cfgs):
        d = sc[sc.config == cfg]
        for col, lab, ls in (("n_full", "all segments", "-"),
                             ("n_kept_deg1", "with a compatible neighbour", "--"),
                             ("n_kept_c3", "in a component of 3 or more", ":")):
            m, lo, hi = med(d, col, "T")
            T = m.index.values.astype(float)
            ax.plot(T, m.values, ls, color=C_CFG[cfg], label=lab)
            ax.fill_between(T, lo.values, hi.values, color=C_CFG[cfg], alpha=0.12, lw=0)
            a, b = fit_exponent(T, m.values.astype(float))
            lines.append(f"| {CONFIG_LABEL[cfg]} | {lab} | {a:.2f} | {b:.3g} |")
        Tt = np.array([T.min(), T.max()])
        ax.plot(Tt, 4 * Tt, color="#BBBBBB", lw=0.8, ls="-.", label="4 T (true segments)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tracks T"); ax.set_title(CONFIG_LABEL[cfg], fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("segments in the matrix")
            ax.legend(fontsize=7, loc="upper left")
    lines.append("")
    save(fig, "fig1_matrix_size")


def fig_qubits(sc, lines):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    lines.append("## System register (qubits), worst repetition at each T\n")
    lines.append("| configuration | T | all segments | decoupled dropped | components < 3 dropped |")
    lines.append("|---|---|---|---|---|")
    # the full register is the same in every configuration (n = 4 T^2 up to hit loss)
    g_all = sc.groupby("T")["qubits_full"].max()
    ax.step(g_all.index, g_all.values, where="mid", color="#000000", ls="-", lw=2.2,
            label="all segments (any configuration)")
    for cfg in CONFIGS:
        d = sc[sc.config == cfg]
        if d.empty:
            continue
        g = d.groupby("T")[["qubits_full", "qubits_deg1", "qubits_c3"]].max()
        ax.step(g.index, g.qubits_deg1, where="mid", color=C_CFG[cfg], ls="--",
                label=f"decoupled dropped: {CONFIG_LABEL[cfg]}")
        for T, r in g.iterrows():
            lines.append(f"| {CONFIG_LABEL[cfg]} | {T} | {r.qubits_full} | {r.qubits_deg1} | {r.qubits_c3} |")
    ax.set_xscale("log"); ax.set_xlabel("tracks T"); ax.set_ylabel("system qubits  ceil(log2 n)")
    ax.set_ylim(4, 24)
    ax.legend(fontsize=6.5, loc="lower right", ncol=1)
    lines.append("")
    save(fig, "fig2_qubits")


def fig_success(so, lines):
    d = so[(so.config == "paper") & (so.solver.isin(["onebit_circuit", "qsvt"]))]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0), sharey=True)
    lines.append("## Post-selection success probability (paper corpus, median)\n")
    lines.append("| T | solver | full | decoupled dropped | ratio |")
    lines.append("|---|---|---|---|---|")
    for ax, sol in zip(axes, ["onebit_circuit", "qsvt"]):
        for var in ("full", "deg1"):
            dd = d[(d.solver == sol) & (d.variant == var)]
            m, lo, hi = med(dd, "P", "T")
            ax.plot(m.index, m.values, LS_VAR[var], color=C_SOL[sol], marker="o", ms=3,
                    label=L_VAR[var])
            ax.fill_between(m.index, lo.values, hi.values, color=C_SOL[sol], alpha=0.12, lw=0)
        mf = d[(d.solver == sol) & (d.variant == "full")].groupby("T").P.median()
        mp = d[(d.solver == sol) & (d.variant == "deg1")].groupby("T").P.median()
        for T in mf.index:
            lines.append(f"| {T} | {L_SOL[sol]} | {mf[T]:.4g} | {mp.get(T, np.nan):.4g} | "
                         f"{mp.get(T, np.nan) / mf[T]:.1f} |")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tracks T")
        ax.set_title(L_SOL[sol], fontsize=8); ax.legend(fontsize=7)
    axes[0].set_ylabel("success probability P")
    lines.append("")
    save(fig, "fig3_success")


def fig_metrics(so, lines):
    d = so[(so.config == "paper") & (so.solver.isin(["classical", "onebit_circuit", "qsvt"]))]
    if d.empty:
        return
    FAR_FLOOR = 2e-4
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, col, ylab in zip(axes, ["eff_f010", "far_e990"],
                             ["segment efficiency at false rate <= 1 %",
                              "segment false rate at 99 % efficiency"]):
        for sol in ("classical", "onebit_circuit", "qsvt"):
            for var, lw, alpha, ms in (("full", 4.0, 0.3, 0), ("deg1", 1.4, 1.0, 3), ("comp3", 1.4, 1.0, 3)):
                dd = d[(d.solver == sol) & (d.variant == var)]
                m, lo, hi = med(dd, col, "T")
                y = m.values.astype(float)
                if col == "far_e990":
                    y = np.maximum(y, FAR_FLOOR)
                ax.plot(m.index, y, LS_VAR[var], color=C_SOL[sol], lw=lw, alpha=alpha,
                        marker="o" if ms else None, ms=ms,
                        label=f"{L_SOL[sol]}, {L_VAR[var]}")
        ax.set_xscale("log"); ax.set_xlabel("tracks T"); ax.set_ylabel(ylab)
    axes[1].set_yscale("log")
    axes[1].set_ylim(FAR_FLOOR * 0.8, 1.0)
    axes[1].text(0.02, 0.03, f"false rate 0 drawn at {FAR_FLOOR:g}", transform=axes[1].transAxes,
                 fontsize=6.5, color="#555555")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.3, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig4_metrics")
    lines.append("## Metrics at the efficiency-first working point (paper corpus, median)\n")
    lines.append("| T | solver | variant | eff at far<=1 % | far at eff 99 % | eff (coupled true only) | max amplitude difference vs full |")
    lines.append("|---|---|---|---|---|---|---|")
    g = d.groupby(["T", "solver", "variant"]).agg(eff=("eff_f010", "median"), far=("far_e990", "median"),
                                                  effc=("eff_coupled_e990", "median"),
                                                  dmax=("maxdiff_vs_full", "max")).reset_index()
    for _, r in g.iterrows():
        lines.append(f"| {r['T']} | {L_SOL[r.solver]} | {L_VAR[r.variant]} | {r.eff:.4f} | {r.far:.4f} | "
                     f"{r.effc:.4f} | {r.dmax:.1e} |")
    lines.append("")


def fig_cost(so, lines):
    d = so[(so.config == "paper") & (so.solver == "onebit_circuit")]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    for ax, col, ylab in zip(axes, ["statevector_bytes", "t_solve"],
                             ["statevector memory (bytes)", "matrix-free solve time (s)"]):
        for var in ("full", "deg1"):
            dd = d[d.variant == var]
            m, lo, hi = med(dd, col, "T")
            ax.plot(m.index, m.values, LS_VAR[var], color=C_SOL["onebit_circuit"], marker="o", ms=3,
                    label=L_VAR[var])
            ax.fill_between(m.index, lo.values, hi.values, color=C_SOL["onebit_circuit"], alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tracks T"); ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7)
    fig.suptitle("one-bit filter, package circuit emulated matrix-free", fontsize=8)
    save(fig, "fig5_cost")


def fig_circuit(lines):
    p = OUT / "circuit_cost.csv"
    if not p.exists():
        return
    cc = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, col, ylab, log in zip(axes, ["onebit_qubits", "onebit_cx", "onebit_depth"],
                                  ["one-bit circuit qubits", "two-qubit gates after transpilation",
                                   "circuit depth after transpilation"], [False, True, True]):
        for var in ("full", "deg1"):
            dd = cc[cc.variant == var].sort_values("T")
            ax.plot(dd["T"], dd[col], LS_VAR[var], color=C_SOL["onebit_circuit"], marker="o", ms=3,
                    label=L_VAR[var])
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("tracks T"); ax.set_ylabel(ylab)
    axes[0].legend(fontsize=7)
    save(fig, "fig6_circuit_cost")
    lines.append("## One-bit circuit after transpilation to {rz, sx, x, cx} (paper corpus, one event)\n")
    lines.append("| T | variant | segments | pair gates | qubits | two-qubit gates | depth | QSVT qubits (LCU / phase) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in cc.sort_values(["T", "variant"]).iterrows():
        lines.append(f"| {r['T']} | {L_VAR[r.variant]} | {r.n} | {r.n_pairs} | {r.onebit_qubits} | "
                     f"{r.onebit_cx} | {r.onebit_depth} | {r.qsvt_qubits_lcu} / {r.qsvt_qubits_phase} |")
    lines.append("")


def main():
    sc, so = load("scaling"), load("solves")
    lines = ["# Segment pruning study: summary numbers\n",
             "Generated by 02_figures.py from outputs/*.csv. Medians over repetitions unless stated.\n"]
    if not sc.empty:
        lines.append("## Events and true segments pruned\n")
        lines.append("| configuration | T | repetitions | median segments | median kept (decoupled dropped) | kept fraction | true segments | true segments decoupled (sum over reps) | max component |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        g = sc.groupby(["config", "T"]).agg(reps=("rep", "count"), n=("n_full", "median"),
                                            k=("n_kept_deg1", "median"), nt=("n_true", "median"),
                                            iso=("n_true_isolated", "sum"), mc=("max_component", "max")).reset_index()
        for _, r in g.iterrows():
            lines.append(f"| {CONFIG_LABEL[r.config]} | {r['T']} | {r.reps} | {r.n:.0f} | {r.k:.0f} | "
                         f"{r.k / r.n:.4f} | {r.nt:.0f} | {r.iso} | {r.mc} |")
        lines.append("")
        fig_matrix_size(sc, lines)
        fig_qubits(sc, lines)
    if not so.empty:
        fig_success(so, lines)
        fig_metrics(so, lines)
        fig_cost(so, lines)
        lines.append("## Exactness check\n")
        lines.append(f"Largest renormalised amplitude difference between a pruned solve and the full solve "
                     f"restricted to the kept segments, over every event, solver and variant: "
                     f"{so.maxdiff_vs_full.max():.2e}\n")
    fig_circuit(lines)
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
