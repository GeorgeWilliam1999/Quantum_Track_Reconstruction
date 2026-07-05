#!/usr/bin/env python
"""Step vs kink-matched erf at common epsilon: eff | far vs T, and sparsity.

Requested 2026-07-03 (first pass, ad-hoc widths) and refined 2026-07-04: the
erf kernel is shown at its PHYSICALLY MEANINGFUL width, theta_d = eps/3.  The
acceptance is eps = 3*sigma_kink (compute_epsilon, scale=3), so eps/3 IS the
true RMS 3D kink angle — the erf soft edge then matches the actual angular
smearing of true-segment kinks instead of an arbitrary width.  Per noise pair
(same formula epsilon for step and erf — common-epsilon by construction):

  clean    (sigma_scatt=1e-4, sigma_res=0)   eps=0.425 mrad  theta_d=0.1416 mrad
  moderate (3e-4, 0.01 mm)                   eps=3.397 mrad  theta_d=1.1322 mrad
  heavy    (5e-4, 0.02 mm)                   eps=6.646 mrad  theta_d=2.2153 mrad

Data: 20 events per point for BOTH solvers (2026-07-04 20-rep quantum campaign;
the first pass had only 3 quantum events per point).  Everything is read from
the qtrk_store metrics view; A_nnz is recorded per solve, so sparsity needs no
A rebuilds.

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
import re
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

# (sigma_scatt, sigma_res, label, eps [mrad], kink width theta_d = eps/3)
PAIRS = [(1e-4, 0.0, "clean", 0.425, 1.4159802e-4),
         (3e-4, 0.01, "moderate", 3.397, 1.1322446e-3),
         (5e-4, 0.02, "heavy", 6.646, 2.2153451e-3)]

STEP_TD = 1e-6                      # step reference (erf at negligible width)
C_STEP, C_KINK = "#52514e", "#08519c"

FAR_FLOOR = 5e-5   # log-axis display floor; a measured far=0 sits here as an OPEN marker
EXPLODE_MAX = 50.0  # per-solve validity gate on the CLASSICAL vector (Eps2 §7.12)
MIN_VALID = 5       # a (pair, kernel, T) point needs >=5/20 valid events to be shown

FOOT = (r"ERF store (qtrk_store metrics view) · common formula $\varepsilon$ per pair: "
        "clean 0.425 · moderate 3.397 · heavy 6.646 mrad · kink-matched width "
        r"$\theta_d=\varepsilon/3=\sigma_{kink}$ · $\gamma$=3 $\delta$=1 · "
        r"$\phi_{max}$=0.2 · drop=0 · T$\in$[10,1000]"
        "\n"
        "classical MINRES ($\\tau$=0.35) · 1BQF matrix-free statevector, wp99 headline · "
        "means over VALID events only (classical max$|x|\\leq$50 gate, quantum gated via "
        "classical partner; point needs $\\geq$5/20 valid — see erf_stepvserf_validity.png) · "
        "shaded: $\\lambda_{min}\\to0$ explosion regime")

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
    # membership via the `studies` column (token match), NOT `study`: shared
    # cells are attributed to whichever study "owns" them and a `study==`
    # filter silently drops them.
    is_erf = m.studies.map(lambda s: "ERF" in re.split(r"[|,;\s]+", str(s)))
    erf = m[is_erf].copy()
    erf["pair"] = [
        next((lab for ss, sr, lab, _, _ in PAIRS
              if abs(row.sigma_scatt - ss) < 1e-12 and abs(row.sigma_res - sr) < 1e-9), None)
        for row in erf.itertuples()]
    # Per-solve VALIDITY GATE (2026-07-05 consistency audit): a lambda_min->0
    # explosion (max|x| up to 1e24, every segment active) saturates that
    # event's metrics for regime reasons, not kernel physics — one exploded
    # event dragged the clean T=200 step mean from 0.9% to 5.8% far while the
    # pooled (gated) Youden section saw no difference, which is exactly the
    # section-to-section inconsistency this gate closes.  Quantum vectors are
    # unit-norm, so quantum rows are gated via their CLASSICAL partner
    # (same event+ham), matching build_metrics / the Youden convention.
    cls = (erf[erf.solver == "classical"]
           [["event_key", "ham_key", "max_abs_x"]]
           .rename(columns={"max_abs_x": "cls_max_abs"}))
    erf = erf.merge(cls, on=["event_key", "ham_key"], how="left")
    erf["valid"] = erf.cls_max_abs <= EXPLODE_MAX
    return erf


def sel_width(df, td):
    """Rows at erf width td (isclose: the kink widths are non-round floats)."""
    return df[np.isclose(df.erf_sigma.values, td, rtol=1e-3)]


def kernels_for(pair_kink_td):
    return [(STEP_TD, "step ($\\theta_d$=1e-6)", C_STEP),
            (pair_kink_td, "erf $\\theta_d=\\varepsilon/3$ (kink-matched)", C_KINK)]


def explosion_onset():
    """Per pair: first T where the STEP classical excluded_frac exceeds 50%."""
    y = pd.read_csv(RES / "erf_youden_eer.csv")
    out = {}
    for _, _, pair, _, _ in PAIRS:
        g = y[(y.pair == pair) & (y.solver == "classical") & (y.theta_d == 1e-6)
              ].sort_values("n_trk")
        hit = g[g.excluded_frac > 0.5]
        out[pair] = int(hit.n_trk.iloc[0]) if len(hit) else None
    return out


def agg(df, cols):
    return df.groupby("n_trk")[cols].agg(["mean", "sem"])


def plot_solver(erf, solver, headline_cols, faded_cols, fname, title):
    onset = explosion_onset()
    fig, axes = plt.subplots(3, 2, figsize=(10, 9.2), sharex=True)
    for i, (ss, sr, pair, eps, kink_td) in enumerate(PAIRS):
        ax_e, ax_f = axes[i]
        sub = erf[(erf.pair == pair) & (erf.solver == solver)]
        for td, lab, color in kernels_for(kink_td):
            g = sel_width(sub, td)
            if not len(g):
                continue
            gv = g[g.valid]
            cnt = gv.groupby("n_trk").size()
            gv = gv[gv.n_trk.isin(cnt[cnt >= MIN_VALID].index)]
            if not len(gv):
                continue
            ge = agg(gv, list(dict.fromkeys(headline_cols + (faded_cols or []))))
            T = ge.index.values
            ec, fc = headline_cols
            ax_e.errorbar(T, ge[(ec, "mean")], yerr=ge[(ec, "sem")], color=color,
                          marker="o", ms=3.5, lw=1.6, capsize=2, label=lab)
            fmean = ge[(fc, "mean")].values
            ax_f.errorbar(T, np.clip(fmean, FAR_FLOOR, None), yerr=ge[(fc, "sem")],
                          color=color, marker="o", ms=3.5, lw=1.6, capsize=2)
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
        ax_e.set_ylabel(f"{pair}\n$\\varepsilon$={eps:g} mrad · "
                        f"$\\theta_d$={kink_td*1e3:.3g} mrad\n\nsegment efficiency")
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


def plot_validity(erf):
    """Excluded fraction (classical max|x|>50) vs T per pair and kernel."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    cls = erf[erf.solver == "classical"]
    for ax, (ss, sr, pair, eps, kink_td) in zip(axes, PAIRS):
        sub = cls[cls.pair == pair]
        for td, lab, color in kernels_for(kink_td):
            g = sel_width(sub, td)
            if not len(g):
                continue
            ex = g.groupby("n_trk").apply(lambda d: 1.0 - d.valid.mean())
            ax.plot(ex.index, ex.values, "-o", ms=3.5, lw=1.5, color=color, label=lab)
        ax.axhline(1 - MIN_VALID / 20, color="#e34948", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("T (tracks)")
        ax.set_title(f"{pair} · $\\varepsilon$={eps:g} mrad", fontsize=9)
    axes[0].set_ylabel("excluded fraction (max$|x|>$50)")
    axes[0].legend(fontsize=7.5)
    axes[1].text(11, 1 - MIN_VALID / 20 + 0.02, "points above the red line are DROPPED "
                 "from the eff/far figures (<5/20 valid)", fontsize=7, color="#e34948")
    fig.suptitle("Validity map: the $\\lambda_{min}\\to0$ explosion regime by kernel "
                 "(fraction of the 20 events per point failing the max$|x|\\leq$50 gate)",
                 y=0.98)
    fig.tight_layout(rect=(0, 0.09, 1, 0.91))
    footer(fig)
    fig.savefig(FIGS / "erf_stepvserf_validity.png")
    plt.close(fig)


def plot_sparsity(erf):
    # one row per (ham, event): A_nnz identical across solvers -> dedupe
    d = erf.drop_duplicates(["event_key", "ham_key"])
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    # (a) nnz vs T ; (b) nnz / n_seg vs T ; (c) kink-erf-to-step nnz ratio vs T
    for ss, sr, pair, eps, kink_td in PAIRS:
        dp = d[d.pair == pair]
        base = sel_width(dp, STEP_TD).groupby("n_trk").A_nnz.mean()
        ls = {"clean": ":", "moderate": "--", "heavy": "-"}[pair]
        for td, lab, color in kernels_for(kink_td):
            g = sel_width(dp, td)
            if not len(g):
                continue
            gg = g.groupby("n_trk").agg(nnz=("A_nnz", "mean"), nseg=("n_seg", "mean"))
            axes[0].plot(gg.index, gg.nnz, ls, color=color, marker="o", ms=3, lw=1.4)
            axes[1].plot(gg.index, gg.nnz / gg.nseg, ls, color=color, marker="o", ms=3, lw=1.4)
            if td != STEP_TD:
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
                          (axes[2], "kink-erf-to-step nnz ratio", "nnz(erf) / nnz(step)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("T (tracks)")
        ax.set_title(ttl, fontsize=9.5)
        ax.set_ylabel(ylab)
    handles = ([Line2D([], [], color=C_STEP, lw=1.6, label="step"),
                Line2D([], [], color=C_KINK, lw=1.6, label="erf $\\theta_d=\\varepsilon/3$")]
               + [Line2D([], [], color="#52514e", ls=ls, lw=1.2, label=p)
                  for p, ls in (("clean", ":"), ("moderate", "--"), ("heavy", "-"))])
    axes[0].legend(handles=handles, fontsize=6.8, loc="upper left", ncol=2)
    fig.suptitle("Sparsity scaling, step vs kink-matched erf (common ε per pair; "
                 "A_nnz from the store view)", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    footer(fig)
    fig.savefig(FIGS / "erf_sparsity_scaling.png")
    plt.close(fig)


def main():
    erf = load_view()
    print(f"[view] ERF rows: {len(erf)}")
    plot_solver(erf, "classical", ["segment_efficiency", "segment_false_rate"], None,
                "erf_stepvserf_classical.png",
                "Step vs kink-matched erf ($\\theta_d=\\varepsilon/3$) at common ε — "
                "classical (τ=0.35): efficiency | false rate vs T")
    plot_solver(erf, "quantum",
                ["segment_efficiency_wp", "segment_false_rate_wp"],
                ["segment_efficiency", "segment_false_rate"],
                "erf_stepvserf_quantum.png",
                "Step vs kink-matched erf ($\\theta_d=\\varepsilon/3$) at common ε — "
                "1BQF (wp99 headline, dotted = fixed τ=0.35): efficiency | false rate vs T")
    plot_validity(erf)
    plot_sparsity(erf)
    # tidy summary CSV — means over VALID events; n_valid/n_total recorded
    rows = []
    for solver in ("classical", "quantum"):
        for ss, sr, pair, eps, kink_td in PAIRS:
            sub = erf[(erf.pair == pair) & (erf.solver == solver)]
            for (td, _, _), kname in zip(kernels_for(kink_td), ("step", "erf_kink")):
                g = sel_width(sub, td)
                if not len(g):
                    continue
                for T, ga in g.groupby("n_trk"):
                    gg = ga[ga.valid]
                    row = dict(pair=pair, epsilon_mrad=eps, kernel=kname, theta_d=td,
                               solver=solver, n_trk=T, n_valid=len(gg), n_total=len(ga),
                               excluded_frac=1.0 - len(gg) / max(len(ga), 1),
                               nnz=ga.A_nnz.mean(), n_seg=ga.n_seg.mean(),
                               fill=(ga.A_nnz / ga.n_seg).mean())
                    if len(gg) >= MIN_VALID:
                        row.update(
                            eff=gg.segment_efficiency.mean(), eff_sem=gg.segment_efficiency.sem(),
                            far=gg.segment_false_rate.mean(), far_sem=gg.segment_false_rate.sem(),
                            eff_wp=gg.segment_efficiency_wp.mean(),
                            far_wp=gg.segment_false_rate_wp.mean())
                    rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "erf_stepvserf_summary.csv", index=False)
    print(f"[csv] {len(out)} rows -> {RES/'erf_stepvserf_summary.csv'}")
    # headline print
    h = out[(out.solver == "classical") & (out.n_trk == 200)]
    for _, r in h.iterrows():
        print(f"  T=200 {r['pair']:9} {r['kernel']:8} td={r.theta_d:g}: "
              f"eff={r.eff:.3f} far={r.far:.3f} nnz={r.nnz:,.0f} fill={r.fill:.2f}")
    print("[figs] 3 figures ->", FIGS)


if __name__ == "__main__":
    main()
