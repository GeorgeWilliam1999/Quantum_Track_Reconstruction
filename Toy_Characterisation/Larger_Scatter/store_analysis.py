#!/usr/bin/env python
"""store_analysis.py — Larger_Scatter (T3) analysis from the qtrk_store.

Replaces the never-run local-pkl aggregation.  Reads the recomputed metrics
VIEW (segment efficiency / purity / false-rate at the gamma-aware absolute
threshold) straight from ``qtrk_store/manifest/metrics.csv`` — no re-solve, no
local ``results/`` pickles.

Shared-event subtlety
---------------------
The clean cells (p_drop = 0) at sigma_scatt in {1e-4, 3e-4, 5e-4} are *shared
events* (identical event_key) with Epsilon_study_2 / Larger_Scatter_Density, so
the primary ``study`` column attributes them elsewhere.  We therefore select on
the comma-separated ``studies`` membership column, which recovers the full
5 x 5 x 8 grid (200/200 classical cells, 20 reps at T=200).

Outputs:  results/ls_store_summary.csv  +  figures/*.png
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
METRICS = Path(os.environ["QTRK_STORE"]) / "manifest" / "metrics.csv"

STUDY = "Larger_Scatter"
SCATTS = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
DROPS  = [0.0, 0.01, 0.02, 0.05, 0.10]
TS     = [10, 20, 50, 100, 200, 400, 700, 1000]


def in_study(cell: str, name: str) -> bool:
    return name in re.split(r"[|,;\s]+", str(cell))


def load() -> pd.DataFrame:
    df = pd.read_csv(METRICS)
    df = df[df["studies"].fillna("").apply(lambda s: in_study(s, STUDY))].copy()
    # HEADLINE = wp99 high-efficiency working point (segment_*_wp from build_metrics
    # >= 2026-06-14); the fixed gamma-aware cut (0.35 at gamma=3) kept as *_fix.
    wp = "segment_efficiency_wp" in df.columns
    E = "segment_efficiency_wp" if wp else "segment_efficiency"
    F = "segment_false_rate_wp" if wp else "segment_false_rate"
    P = "segment_purity_wp" if wp else "segment_purity"
    df["eff"] = df[E] * 100
    df["far"] = df[F] * 100
    df["pur"] = df[P] * 100
    df["eff_fix"] = df["segment_efficiency"] * 100
    df["far_fix"] = df["segment_false_rate"] * 100
    return df


def _sem(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["solver", "sigma_scatt", "hit_ineff", "n_trk"])
    rows = []
    for (sol, ss, he, T), sub in g:
        rows.append(dict(
            solver=sol, sigma_scatt=ss, hit_ineff=he, n_trk=T, n_rep=len(sub),
            eff=sub.eff.mean(), eff_sem=_sem(sub.eff),
            far=sub.far.mean(), far_sem=_sem(sub.far),
            pur=sub.pur.mean(), pur_sem=_sem(sub.pur),
            cos_QC=sub.cos_QC.mean() if sub.cos_QC.notna().any() else np.nan,
        ))
    out = pd.DataFrame(rows).sort_values(["solver", "sigma_scatt", "hit_ineff", "n_trk"])
    out.to_csv(RES / "ls_store_summary.csv", index=False)
    return out


def pick(ag, solver, ss, he, col):
    s = ag[(ag.solver == solver) & (np.isclose(ag.sigma_scatt, ss)) &
           (np.isclose(ag.hit_ineff, he))].sort_values("n_trk")
    return s.n_trk.values, s[col].values, s[col + "_sem"].values if col + "_sem" in s else None


def fig_vs_T(ag):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, 5))
    # scatter-only (drop=0)
    for c, ss in zip(cmap, SCATTS):
        T, e, es = pick(ag, "classical", ss, 0.0, "eff")
        ax[0, 0].errorbar(T, e, es, marker="o", color=c, label=f"{ss*1e4:.0f}e-4")
        T, f, fs = pick(ag, "classical", ss, 0.0, "far")
        ax[1, 0].errorbar(T, f, fs, marker="o", color=c, label=f"{ss*1e4:.0f}e-4")
    # drop-only (sigma_scatt=1e-4)
    for c, he in zip(cmap, DROPS):
        T, e, es = pick(ag, "classical", 1e-4, he, "eff")
        ax[0, 1].errorbar(T, e, es, marker="s", color=c, label=f"{he*100:.0f}%")
        T, f, fs = pick(ag, "classical", 1e-4, he, "far")
        ax[1, 1].errorbar(T, f, fs, marker="s", color=c, label=f"{he*100:.0f}%")
    ax[0, 0].set_title("Scatter-only (p_drop=0): efficiency"); ax[0, 0].legend(title=r"$\sigma_{scatt}$", fontsize=8)
    ax[0, 1].set_title("Drop-only ($\\sigma_{scatt}$=1e-4): efficiency"); ax[0, 1].legend(title="p_drop", fontsize=8)
    ax[1, 0].set_title("Scatter-only: false rate"); ax[1, 1].set_title("Drop-only: false rate")
    for a in ax.flat:
        a.set_xscale("log"); a.set_xlabel("T (tracks/event)"); a.grid(alpha=0.3)
    for a in ax[0]: a.set_ylabel("segment efficiency [%]"); a.set_ylim(40, 102)
    for a in ax[1]: a.set_ylabel("segment false rate [%]")
    fig.suptitle("Larger_Scatter (T3) — classical segment metrics vs T (qtrk_store)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "ls_eff_far_vs_T.png", dpi=130); plt.close(fig)


def fig_heatmaps(ag, T=200):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, (col, cmap, lab) in enumerate([("eff", "viridis", "efficiency [%]"),
                                          ("far", "magma_r", "false rate [%]")]):
        M = np.full((len(SCATTS), len(DROPS)), np.nan)
        for i, ss in enumerate(SCATTS):
            for j, he in enumerate(DROPS):
                row = ag[(ag.solver == "classical") & np.isclose(ag.sigma_scatt, ss) &
                         np.isclose(ag.hit_ineff, he) & (ag.n_trk == T)]
                if len(row): M[i, j] = row[col].values[0]
        im = ax[k].imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax[k].set_xticks(range(len(DROPS))); ax[k].set_xticklabels([f"{d*100:.0f}%" for d in DROPS])
        ax[k].set_yticks(range(len(SCATTS))); ax[k].set_yticklabels([f"{s*1e4:.0f}e-4" for s in SCATTS])
        ax[k].set_xlabel("p_drop"); ax[k].set_ylabel(r"$\sigma_{scatt}$ [rad]"); ax[k].set_title(f"T={T}: {lab}")
        for i in range(len(SCATTS)):
            for j in range(len(DROPS)):
                if not np.isnan(M[i, j]):
                    ax[k].text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                               color="w" if (col=="eff" and M[i,j]<70) or (col=="far" and M[i,j]>25) else "k", fontsize=8)
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle(f"Larger_Scatter (T3) — classical, T={T}: scattering x inefficiency surfaces", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "ls_heatmaps_T200.png", dpi=130); plt.close(fig)


def fig_quantum(ag):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.plasma(np.linspace(0, 0.85, 5))
    for c, ss in zip(cmap, SCATTS):
        T, q, _ = pick(ag, "quantum", ss, 0.01, "cos_QC")
        m = ~np.isnan(q)
        if m.any(): ax[0].plot(np.array(T)[m], np.array(q)[m], "o-", color=c, label=f"{ss*1e4:.0f}e-4")
    ax[0].set_title("1BQF–classical cosine vs T (p_drop=1%)"); ax[0].legend(title=r"$\sigma_{scatt}$", fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("T"); ax[0].set_ylabel(r"$\cos\theta_{QC}$"); ax[0].grid(alpha=0.3)
    # quantum vs classical efficiency to T=1000 (matrix-free 1BQF now reaches high T)
    for c, ss in zip(cmap, SCATTS):
        Tc, ec, _ = pick(ag, "classical", ss, 0.01, "eff")
        Tq, eq, _ = pick(ag, "quantum", ss, 0.01, "eff")
        ax[1].plot(Tc, ec, "-", color=c, alpha=0.5)
        ax[1].plot(Tq, eq, "o--", color=c, label=f"{ss*1e4:.0f}e-4")
    ax[1].set_title("efficiency: classical (—) vs 1BQF (o--), p_drop=1%")
    ax[1].set_xscale("log"); ax[1].set_xlabel("T"); ax[1].set_ylabel("efficiency [%]"); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("Larger_Scatter (T3) — quantum 1BQF (to T=1000, matrix-free)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "ls_quantum.png", dpi=130); plt.close(fig)


def fig_combined(ag, T=200):
    # efficiency loss from clean baseline: scatter-only, drop-only, combined
    base = ag[(ag.solver=="classical") & np.isclose(ag.sigma_scatt,1e-4) &
              np.isclose(ag.hit_ineff,0.0) & (ag.n_trk==T)]["eff"]
    base = base.values[0] if len(base) else 100.0
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(SCATTS[1:]))
    for he, mk in [(0.0,"scatter-only"),(0.05,"+5% drop"),(0.10,"+10% drop")]:
        ys=[]
        for ss in SCATTS[1:]:
            r=ag[(ag.solver=="classical") & np.isclose(ag.sigma_scatt,ss) &
                 np.isclose(ag.hit_ineff,he) & (ag.n_trk==T)]["eff"]
            ys.append(base-(r.values[0] if len(r) else np.nan))
        ax.plot(x, ys, "o-", label=mk)
    ax.set_xticks(x); ax.set_xticklabels([f"{s*1e4:.0f}e-4" for s in SCATTS[1:]])
    ax.set_xlabel(r"$\sigma_{scatt}$ [rad]"); ax.set_ylabel(f"efficiency loss vs clean baseline [pp], T={T}")
    ax.set_title("Larger_Scatter (T3) — combined scattering x inefficiency", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "ls_combined.png", dpi=130); plt.close(fig)


def main():
    df = load()
    print(f"[load] {len(df)} shared-aware rows for {STUDY}")
    ag = aggregate(df)
    print(f"[agg]  {len(ag)} cells -> results/ls_store_summary.csv")
    fig_vs_T(ag); fig_heatmaps(ag); fig_quantum(ag); fig_combined(ag)
    # headline numbers for the report
    def at(solver, ss, he, T, col):
        r = ag[(ag.solver==solver) & np.isclose(ag.sigma_scatt,ss) &
               np.isclose(ag.hit_ineff,he) & (ag.n_trk==T)]
        return float(r[col].values[0]) if len(r) else float("nan")
    print("\n=== HEADLINE (classical, T=200) ===")
    print(f"clean (1e-4,0%):           eff={at('classical',1e-4,0,200,'eff'):.1f}% far={at('classical',1e-4,0,200,'far'):.1f}%")
    print(f"max scatter (1e-3,0%):     eff={at('classical',1e-3,0,200,'eff'):.1f}% far={at('classical',1e-3,0,200,'far'):.1f}%")
    print(f"max drop (1e-4,10%):       eff={at('classical',1e-4,0.10,200,'eff'):.1f}% far={at('classical',1e-4,0.10,200,'far'):.1f}%")
    print(f"combined (1e-3,10%):       eff={at('classical',1e-3,0.10,200,'eff'):.1f}% far={at('classical',1e-3,0.10,200,'far'):.1f}%")
    print(f"clean (1e-4,0%) T=1000:    eff={at('classical',1e-4,0,1000,'eff'):.1f}% far={at('classical',1e-4,0,1000,'far'):.1f}%")
    print(f"combined (1e-3,10%) T=1000:eff={at('classical',1e-3,0.10,1000,'eff'):.1f}% far={at('classical',1e-3,0.10,1000,'far'):.1f}%")


if __name__ == "__main__":
    main()
