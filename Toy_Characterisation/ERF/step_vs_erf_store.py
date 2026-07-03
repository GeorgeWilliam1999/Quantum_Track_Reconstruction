#!/usr/bin/env python
"""Step vs erf head-to-head at common epsilon: eff | far vs T, and sparsity scaling.

Requested 2026-07-03: side-by-side segment efficiency and false rate with track
count on the x-axis, step vs erf at a COMMON epsilon and reasonable erf widths,
plus how the sparsity (A_nnz) scales for the two kernels.

Everything is read from the qtrk_store metrics view (per-event rows; A_nnz is
recorded per solve, so sparsity needs no A rebuilds). Within each ERF noise pair
the step regression point (theta_d = 1e-6) and every erf width share the SAME
formula epsilon — the comparison is common-epsilon by construction:
  clean    (sigma_scatt=1e-4, sigma_res=0)     -> eps = 0.425 mrad
  moderate (3e-4, 0.01 mm)                     -> eps = 3.397 mrad
  heavy    (5e-4, 0.02 mm)                     -> eps = 6.646 mrad

Kernels shown: step (theta_d=1e-6) vs erf theta_d = 1e-4 ("quantum pipeline"
width) and 1e-3 ("resolution recovery" width).

Conventions: classical at the gamma-aware absolute tau = 0.35; 1BQF headline at
the efficiency-first wp99 working point (fixed-tau curve kept faded, per the
2026-06-14 decision). Panels shade the lambda_min -> 0 explosion regime measured
in erf_youden_eer.csv (step-kernel classical excluded_frac > 50%) — there the
per-event means saturate for regime reasons, not kernel physics.

Outputs: figures/erf_stepvserf_classical.png, figures/erf_stepvserf_quantum.png,
figures/erf_sparsity_scaling.png, results/erf_stepvserf_summary.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RES = HERE / "results"

PAIRS = [(1e-4, 0.0, "clean", 0.425), (3e-4, 0.01, "moderate", 3.397),
         (5e-4, 0.02, "heavy", 6.646)]
KERNELS = [(1e-6, "step ($\\theta_d$=1e-6)", "#52514e", "-"),
           (1e-4, "erf $\\theta_d$=1e-4", "#6baed6", "-"),
           (1e-3, "erf $\\theta_d$=1e-3", "#08306b", "-")]

FAR_FLOOR = 5e-5   # log-axis display floor; a measured far=0 sits here as an OPEN marker

FOOT = (r"ERF store (qtrk_store metrics view) · common formula $\varepsilon$ per pair: "
        "clean 0.425 · moderate 3.397 · heavy 6.646 mrad · "
        r"$\gamma$=3 $\delta$=1 · $\phi_{max}$=0.2 · drop=0 · T$\in$[10,1000]"
        "\n"
        "classical MINRES 20 rep ($\\tau$=0.35) · 1BQF matrix-free statevector 3 rep "
        "(1 at T$\\geq$700), wp99 headline · shaded: $\\lambda_{min}\\to0$ explosion regime "
        "(step classical excluded_frac>50%, erf_youden_eer.csv)")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def footer(fig):
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")


def load_view():
    store = os.environ.get("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
    m = pd.read_csv(os.path.join(store, "manifest", "metrics.csv"))
    erf = m[m.study == "ERF"].copy()
    erf["pair"] = [
        next((lab for ss, sr, lab, _ in PAIRS
              if abs(row.sigma_scatt - ss) < 1e-12 and abs(row.sigma_res - sr) < 1e-9), None)
        for row in erf.itertuples()]
    return erf


def explosion_onset():
    """Per pair: first T where the STEP classical excluded_frac exceeds 50%."""
    y = pd.read_csv(RES / "erf_youden_eer.csv")
    out = {}
    for _, _, pair, _ in PAIRS:
        g = y[(y.pair == pair) & (y.solver == "classical") & (y.theta_d == 1e-6)
              ].sort_values("n_trk")
        hit = g[g.excluded_frac > 0.5]
        out[pair] = int(hit.n_trk.iloc[0]) if len(hit) else None
    return out


def agg(df, cols):
    g = df.groupby("n_trk")[cols].agg(["mean", "sem"])
    return g


def plot_solver(erf, solver, headline_cols, faded_cols, fname, title):
    onset = explosion_onset()
    fig, axes = plt.subplots(3, 2, figsize=(10, 9.2), sharex=True)
    for i, (ss, sr, pair, eps) in enumerate(PAIRS):
        ax_e, ax_f = axes[i]
        for td, lab, color, ls in KERNELS:
            g = erf[(erf.pair == pair) & (erf.solver == solver) & (erf.erf_sigma == td)]
            if not len(g):
                continue
            ge = agg(g, list(dict.fromkeys(headline_cols + (faded_cols or []))))
            T = ge.index.values
            ec, fc = headline_cols
            ax_e.errorbar(T, ge[(ec, "mean")], yerr=ge[(ec, "sem")], color=color,
                          ls=ls, marker="o", ms=3.5, lw=1.6, capsize=2, label=lab)
            fmean = ge[(fc, "mean")].values
            ax_f.errorbar(T, np.clip(fmean, FAR_FLOOR, None), yerr=ge[(fc, "sem")],
                          color=color, ls=ls, marker="o", ms=3.5, lw=1.6, capsize=2)
            # a measured 0 is data, not a gap: open marker pinned at the floor
            zero = fmean <= 0
            if zero.any():
                ax_f.plot(T[zero], np.full(zero.sum(), FAR_FLOOR), marker="o",
                          ms=5, mfc="white", mec=color, mew=1.2, ls="none", zorder=5)
            if faded_cols:
                ec2, fc2 = faded_cols
                ax_e.plot(T, ge[(ec2, "mean")], color=color, ls=":", lw=1.0, alpha=0.45)
                ax_f.plot(T, np.clip(ge[(fc2, "mean")], FAR_FLOOR, None), color=color,
                          ls=":", lw=1.0, alpha=0.45)
        if onset.get(pair):
            for ax in (ax_e, ax_f):
                ax.axvspan(onset[pair], 1300, color="#e34948", alpha=0.06, zorder=0)
                ax.axvline(onset[pair], color="#e34948", lw=0.8, ls="--", alpha=0.5)
        ax_e.set_ylabel(f"{pair}\n$\\varepsilon$={eps:g} mrad\n\nsegment efficiency")
        ax_f.set_ylabel("false rate")
        ax_e.set_ylim(0.4, 1.03)
        ax_f.set_yscale("log")
        ax_f.set_ylim(2e-5, 1.5)
        ax_f.axhline(FAR_FLOOR, color="#b9b7b1", lw=0.6, ls=":", zorder=0)
        ax_f.text(1250, FAR_FLOOR, " open = 0", fontsize=6, color="#79776f",
                  va="center", ha="left", clip_on=False)
        for ax in (ax_e, ax_f):
            ax.set_xscale("log")
            ax.set_xlim(9, 1300)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[-1, 0].set_xlabel("T (tracks)")
    axes[-1, 1].set_xlabel("T (tracks)")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    footer(fig)
    fig.savefig(FIGS / fname)
    plt.close(fig)


def plot_sparsity(erf):
    # one row per (ham, event): A_nnz identical across solvers -> dedupe
    d = erf.drop_duplicates(["event_key", "ham_key"])
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    # (a) nnz vs T ; (b) nnz / n_seg vs T ; (c) erf-to-step nnz ratio vs T
    for ss, sr, pair, eps in PAIRS:
        base = d[(d.pair == pair) & (d.erf_sigma == 1e-6)].groupby("n_trk").A_nnz.mean()
        for td, lab, color, _ in KERNELS:
            g = d[(d.pair == pair) & (d.erf_sigma == td)]
            if not len(g):
                continue
            gg = g.groupby("n_trk").agg(nnz=("A_nnz", "mean"), nseg=("n_seg", "mean"))
            ls = {"clean": ":", "moderate": "--", "heavy": "-"}[pair]
            axes[0].plot(gg.index, gg.nnz, ls, color=color, marker="o", ms=3, lw=1.4)
            axes[1].plot(gg.index, gg.nnz / gg.nseg, ls, color=color, marker="o", ms=3, lw=1.4)
            if td != 1e-6:
                ratio = (gg.nnz / base.reindex(gg.index)).dropna()
                axes[2].plot(ratio.index, ratio.values, ls, color=color, marker="o",
                             ms=3, lw=1.4)
    # n_seg = 4 T^2 reference on (a)
    Ts = np.array([10, 1000])
    axes[0].plot(Ts, 4 * Ts.astype(float) ** 2, color="#c3c2b7", lw=1.0, ls="-.")
    axes[0].text(320, 4 * 550 ** 2, "$n_{seg}=4T^2$", fontsize=7.5, color="#52514e",
                 rotation=18)
    for ax, ttl, ylab in ((axes[0], "A_nnz vs T", "A_nnz"),
                          (axes[1], "fill: A_nnz / n_seg", "A_nnz / n_seg"),
                          (axes[2], "erf-to-step nnz ratio", "nnz(erf) / nnz(step)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T (tracks)")
        ax.set_title(ttl, fontsize=9.5)
        ax.set_ylabel(ylab)
    handles = ([Line2D([], [], color=c, lw=1.6, label=lab) for _, lab, c, _ in KERNELS]
               + [Line2D([], [], color="#52514e", ls=ls, lw=1.2, label=p)
                  for p, ls in (("clean", ":"), ("moderate", "--"), ("heavy", "-"))])
    axes[0].legend(handles=handles, fontsize=6.8, loc="upper left", ncol=2)
    fig.suptitle("Sparsity scaling, step vs erf (common ε per pair; A_nnz from the store view)", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    footer(fig)
    fig.savefig(FIGS / "erf_sparsity_scaling.png")
    plt.close(fig)


def main():
    erf = load_view()
    print(f"[view] ERF rows: {len(erf)}")
    plot_solver(erf, "classical", ["segment_efficiency", "segment_false_rate"], None,
                "erf_stepvserf_classical.png",
                "Step vs erf at common ε — classical (τ=0.35): efficiency | false rate vs T")
    plot_solver(erf, "quantum",
                ["segment_efficiency_wp", "segment_false_rate_wp"],
                ["segment_efficiency", "segment_false_rate"],
                "erf_stepvserf_quantum.png",
                "Step vs erf at common ε — 1BQF (wp99 headline, dotted = fixed τ=0.35): efficiency | false rate vs T")
    plot_sparsity(erf)
    # tidy summary CSV
    rows = []
    for solver in ("classical", "quantum"):
        for ss, sr, pair, eps in PAIRS:
            for td, *_ in KERNELS:
                g = erf[(erf.pair == pair) & (erf.solver == solver) & (erf.erf_sigma == td)]
                if not len(g):
                    continue
                for T, gg in g.groupby("n_trk"):
                    rows.append(dict(
                        pair=pair, epsilon_mrad=eps, theta_d=td, solver=solver, n_trk=T,
                        n_events=len(gg),
                        eff=gg.segment_efficiency.mean(), eff_sem=gg.segment_efficiency.sem(),
                        far=gg.segment_false_rate.mean(), far_sem=gg.segment_false_rate.sem(),
                        eff_wp=gg.segment_efficiency_wp.mean(), far_wp=gg.segment_false_rate_wp.mean(),
                        nnz=gg.A_nnz.mean(), n_seg=gg.n_seg.mean(),
                        fill=(gg.A_nnz / gg.n_seg).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "erf_stepvserf_summary.csv", index=False)
    print(f"[csv] {len(out)} rows -> {RES/'erf_stepvserf_summary.csv'}")
    # headline print
    h = out[(out.solver == "classical") & (out.n_trk == 200)]
    for _, r in h.iterrows():
        print(f"  T=200 {r['pair']:9} td={r.theta_d:g}: eff={r.eff:.3f} far={r.far:.3f} "
              f"nnz={r.nnz:,.0f} fill={r.fill:.2f}")
    print("[figs] 3 figures ->", FIGS)


if __name__ == "__main__":
    main()
