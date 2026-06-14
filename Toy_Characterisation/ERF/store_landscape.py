#!/usr/bin/env python
"""store_landscape.py — ERF (Task 5) full multiplicity landscape from qtrk_store.

The existing report is the single fixed-event (T=50) step-vs-erf comparison.
This extends it to the full Condor sweep that now lives in the store:
    kernel = erf, theta_d (erf_sigma) in {1e-6 .. 1e-3},
    sigma_scatt in {1e-4, 3e-4, 5e-4}, sigma_res in {0, 0.01, 0.02}, T in {10..1000}.
theta_d = 1e-6 is the step regression point (erf width -> 0 reproduces the hard cut).

Reads the recomputed metrics VIEW (segment metrics at the absolute 0.35
threshold) — no local pkls, no re-solve.

Outputs:  results/erf_store_landscape.csv  +  figures/erf_landscape_*.png
NOTE: Youden-J / EER threshold optimisation needs the pooled per-segment score
distributions (qp.load_solution + truth) and is left as a documented follow-up.
"""
from __future__ import annotations
import os, re
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

STUDY = "ERF"
TDS = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
# ERF used PAIRED noise (not a Cartesian sigma_scatt x sigma_res grid):
#   clean / moderate / heavy  =  (sigma_scatt, sigma_res)
PAIRS = [(1e-4, 0.0, "clean"), (3e-4, 0.01, "moderate"), (5e-4, 0.02, "heavy")]
TS = [10, 20, 50, 100, 200, 400, 700, 1000]


def in_study(cell, name): return name in re.split(r"[|,;\s]+", str(cell))
def _sem(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def load():
    df = pd.read_csv(METRICS)
    df = df[df["studies"].fillna("").apply(lambda s: in_study(s, STUDY))].copy()
    # HEADLINE = wp99 high-efficiency working point (segment_*_wp from build_metrics
    # >= 2026-06-14); the fixed gamma-aware cut (0.35 at gamma=3) is kept as *_fix.
    wp = "segment_efficiency_wp" in df.columns
    E = "segment_efficiency_wp" if wp else "segment_efficiency"
    F = "segment_false_rate_wp" if wp else "segment_false_rate"
    P = "segment_purity_wp" if wp else "segment_purity"
    df["eff"] = df[E] * 100
    df["far"] = df[F] * 100
    df["pur"] = df[P] * 100
    df["eff_fix"] = df.segment_efficiency * 100
    df["far_fix"] = df.segment_false_rate * 100
    return df


def aggregate(df):
    rows = []
    keys = ["solver", "erf_sigma", "sigma_scatt", "sigma_res", "n_trk"]
    for k, sub in df.groupby(keys):
        d = dict(zip(keys, k))
        d.update(n_rep=len(sub),
                 eff=sub.eff.mean(), eff_sem=_sem(sub.eff),
                 far=sub.far.mean(), far_sem=_sem(sub.far),
                 pur=sub.pur.mean(),
                 cos_QC=sub.cos_QC.mean() if sub.cos_QC.notna().any() else np.nan)
        rows.append(d)
    out = pd.DataFrame(rows).sort_values(keys)
    out.to_csv(RES / "erf_store_landscape.csv", index=False)
    return out


def _pair(ag, ss, sr, td=None, T=None):
    m = (ag.solver == "classical") & np.isclose(ag.sigma_scatt, ss) & np.isclose(ag.sigma_res, sr)
    if td is not None: m &= np.isclose(ag.erf_sigma, td)
    if T is not None: m &= (ag.n_trk == T)
    return ag[m]


def fig_resolution_recovery(ag):
    """Headline: per noise PAIR, efficiency & false-rate vs T, step (theta_d=1e-6)
    vs widest erf (theta_d=1e-3)."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for k, (ss, sr, name) in enumerate(PAIRS):
        for td, color, lab in [(1e-6, "k", "step (θ_d=1e-6)"),
                               (1e-4, "tab:blue", "erf θ_d=1e-4"),
                               (1e-3, "tab:red", "erf θ_d=1e-3")]:
            s = _pair(ag, ss, sr, td).sort_values("n_trk")
            ax[0, k].errorbar(s.n_trk, s.eff, s.eff_sem, marker="o", color=color, label=lab)
            ax[1, k].errorbar(s.n_trk, s.far, s.far_sem, marker="o", color=color, label=lab)
        ax[0, k].set_title(f"{name}: σ_scatt={ss:g}, σ_res={sr:g}")
        for r in (0, 1):
            ax[r, k].set_xscale("log"); ax[r, k].grid(alpha=0.3)
        ax[0, k].set_ylim(78, 101); ax[1, k].set_ylim(-3, 100)
        ax[1, k].set_xlabel("T")
        ax[0, k].legend(fontsize=8)
    ax[0, 0].set_ylabel("segment efficiency [%]"); ax[1, 0].set_ylabel("segment false rate [%]")
    fig.suptitle("ERF (T5) — full multiplicity sweep: erf vs step per noise pair (qtrk_store)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "erf_landscape_resolution_recovery.png", dpi=130); plt.close(fig)


def fig_eff_vs_td(ag, T=200):
    """Efficiency & false rate vs theta_d, one line per noise pair, at fixed T."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(PAIRS)))
    for c, (ss, sr, name) in zip(cmap, PAIRS):
        s = _pair(ag, ss, sr, T=T).sort_values("erf_sigma")
        ax[0].errorbar(s.erf_sigma, s.eff, s.eff_sem, marker="o", color=c, label=name)
        ax[1].errorbar(s.erf_sigma, s.far, s.far_sem, marker="o", color=c, label=name)
    for a, lab in zip(ax, ["segment efficiency [%]", "segment false rate [%]"]):
        a.set_xscale("log"); a.set_xlabel(r"$\theta_d$ (erf width);  $\theta_d{=}10^{-6}\approx$ step")
        a.set_ylabel(lab); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"ERF (T5) — kernel-width scan at T={T} (per noise pair)", fontweight="bold")
    fig.subplots_adjust(top=0.88); fig.savefig(FIG / "erf_landscape_td_scan.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_heatmap(ag, T=200):
    """Efficiency heatmap theta_d x noise-pair at fixed T (classical)."""
    M = np.full((len(TDS), len(PAIRS)), np.nan)
    for i, td in enumerate(TDS):
        for j, (ss, sr, name) in enumerate(PAIRS):
            r = _pair(ag, ss, sr, td, T)
            if len(r): M[i, j] = r.eff.values[0]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=40, vmax=100)
    ax.set_xticks(range(len(PAIRS))); ax.set_xticklabels([p[2] for p in PAIRS])
    ax.set_yticks(range(len(TDS))); ax.set_yticklabels([f"{t:g}" for t in TDS])
    ax.set_xlabel("noise pair"); ax.set_ylabel(r"$\theta_d$ (erf width)")
    for i in range(len(TDS)):
        for j in range(len(PAIRS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                        color="w" if M[i, j] < 70 else "k", fontsize=9)
    fig.colorbar(im, label="efficiency [%]")
    ax.set_title(f"ERF (T5) — classical efficiency, T={T}", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "erf_landscape_heatmap_T200.png", dpi=130); plt.close(fig)


def main():
    df = load()
    print(f"[load] {len(df)} shared-aware rows for {STUDY}")
    ag = aggregate(df)
    print(f"[agg]  {len(ag)} cells -> results/erf_store_landscape.csv")
    fig_resolution_recovery(ag); fig_eff_vs_td(ag); fig_heatmap(ag)

    def at(ss, sr, td, T, col):
        r = _pair(ag, ss, sr, td, T)
        return float(r[col].values[0]) if len(r) else float("nan")
    print("\n=== HEADLINE (classical) ===")
    for ss, sr, name in PAIRS:
        for T in (200,):
            print(f"{name:9s} (σ_sc={ss:g},σ_res={sr:g}) T={T}: "
                  f"step eff={at(ss,sr,1e-6,T,'eff'):5.1f}% far={at(ss,sr,1e-6,T,'far'):5.1f}%  "
                  f"-> erf(1e-3) eff={at(ss,sr,1e-3,T,'eff'):5.1f}% far={at(ss,sr,1e-3,T,'far'):5.1f}%")


if __name__ == "__main__":
    main()
