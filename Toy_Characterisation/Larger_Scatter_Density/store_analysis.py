#!/usr/bin/env python
"""store_analysis.py — Larger_Scatter_Density (T4) analysis from the qtrk_store.

Tight-cone x scattering: a narrower angular cone phi_max packs the same T tracks
into less solid angle, raising local track density ~ T / phi_max^2 and stressing
the segment discriminator more than either knob alone.

Reads the recomputed metrics VIEW from qtrk_store (no local pkls, no re-solve).
The phi_max = 0.2 baseline column lives under shared events, so we select on the
comma-separated ``studies`` membership column (recovers the full 5 x 2 x 8 grid).

Outputs:  results/lsd_store_summary.csv  +  figures/*.png
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SHARED = "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared"
for p in (_SHARED, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)
import qtrk_pipeline as qp

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
RES = HERE / "results"; RES.mkdir(exist_ok=True)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

STUDY = "Larger_Scatter_Density"
PHIS = [0.2, 0.1, 0.05, 0.02, 0.01]
SCATTS = [1e-4, 3e-4]
TS = [10, 20, 50, 100, 200, 400, 700, 1000]
MIN_VALID = 5   # a cell needs >=5 valid events for a mean to be quoted


def _sem(a):
    a = np.asarray(a, float)
    return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0


def load():
    # exact-token membership + the standard validity gate (max|x|<=50 on the
    # classical partner): the tight-cone/high-T corner is lambda_min->0
    # explosion territory and per-cell means over exploded solves are artefacts.
    df = qp.add_validity(qp.load_metrics(study=STUDY))
    # Classical-only density study -> classical stays at the fixed gamma-aware cut
    # (0.35 at gamma=3), its established operating point (user 2026-06-14: wp99 is
    # the 1BQF headline only; classical keeps tau=0.35).
    df["eff"] = df.segment_efficiency * 100
    df["far"] = df.segment_false_rate * 100
    df["pur"] = df.segment_purity * 100
    df["density"] = df.n_trk / df.phi_max**2
    return df


def aggregate(df):
    rows = []
    for (sol, ph, ss, T), sub in df.groupby(["solver", "phi_max", "sigma_scatt", "n_trk"]):
        v = sub[sub.valid]
        row = dict(solver=sol, phi_max=ph, sigma_scatt=ss, n_trk=T,
                   n_rep=len(sub), n_valid=len(v),
                   excluded_frac=1.0 - len(v) / len(sub) if len(sub) else np.nan,
                   density=T / ph**2)
        # quantum cells run fewer reps than classical (3-4 vs 20): a cell needs
        # min(MIN_VALID, n_rep) valid events for its mean to be quoted
        if len(v) >= min(MIN_VALID, len(sub)) and len(v) > 0:
            row.update(eff=v.eff.mean(), eff_sem=_sem(v.eff),
                       far=v.far.mean(), far_sem=_sem(v.far),
                       pur=v.pur.mean(), pur_sem=_sem(v.pur),
                       cos_QC=v.cos_QC.mean() if v.cos_QC.notna().any() else np.nan)
        else:
            row.update(eff=np.nan, eff_sem=np.nan, far=np.nan, far_sem=np.nan,
                       pur=np.nan, pur_sem=np.nan, cos_QC=np.nan)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["solver", "phi_max", "sigma_scatt", "n_trk"])
    out.to_csv(RES / "lsd_store_summary.csv", index=False)
    return out


def pick(ag, solver, ph, ss, col):
    s = ag[(ag.solver == solver) & np.isclose(ag.phi_max, ph) &
           np.isclose(ag.sigma_scatt, ss)].sort_values("n_trk")
    return s.n_trk.values, s[col].values, s.get(col + "_sem")


def fig_vs_T(ag, ss=1e-4):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(PHIS)))
    for c, ph in zip(cmap, PHIS):
        T, e, es = pick(ag, "classical", ph, ss, "eff")
        ax[0].errorbar(T, e, es, marker="o", color=c, label=f"{ph:g}")
        T, f, fs = pick(ag, "classical", ph, ss, "far")
        ax[1].errorbar(T, f, fs, marker="o", color=c, label=f"{ph:g}")
    ax[0].set_ylabel("segment efficiency [%]"); ax[0].set_title("efficiency vs T")
    ax[1].set_ylabel("segment false rate [%]"); ax[1].set_title("false rate vs T")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("T (tracks/event)"); a.grid(alpha=0.3)
        a.legend(title=r"$\phi_{max}$ [rad]", fontsize=8)
    fig.suptitle(f"Larger_Scatter_Density (T4) — classical, $\\sigma_{{scatt}}$={ss:g} (qtrk_store)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "lsd_eff_far_vs_T.png", dpi=130); plt.close(fig)


def fig_density_collapse(ag):
    # false rate collapses onto the density proxy T/phi_max^2
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, ss in enumerate(SCATTS):
        cmap = plt.cm.coolwarm(np.linspace(0, 1, len(PHIS)))
        for c, ph in zip(cmap, PHIS):
            s = ag[(ag.solver == "classical") & np.isclose(ag.phi_max, ph) &
                   np.isclose(ag.sigma_scatt, ss)].sort_values("density")
            ax[k].plot(s.density, s.far, "o-", color=c, label=f"{ph:g}")
        ax[k].set_xscale("log"); ax[k].set_xlabel(r"local density  $T/\phi_{max}^2$")
        ax[k].set_ylabel("segment false rate [%]"); ax[k].grid(alpha=0.3)
        ax[k].set_title(f"$\\sigma_{{scatt}}$={ss:g}"); ax[k].legend(title=r"$\phi_{max}$", fontsize=8)
    fig.suptitle("Larger_Scatter_Density (T4) — false rate vs the density proxy", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "lsd_density_collapse.png", dpi=130); plt.close(fig)


def fig_heatmap(ag, T=200):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, ss in enumerate(SCATTS):
        M = np.full((len(PHIS), 1), np.nan)
        labels = []
        vals_eff, vals_far = [], []
        for ph in PHIS:
            r = ag[(ag.solver == "classical") & np.isclose(ag.phi_max, ph) &
                   np.isclose(ag.sigma_scatt, ss) & (ag.n_trk == T)]
            vals_eff.append(r.eff.values[0] if len(r) else np.nan)
            vals_far.append(r.far.values[0] if len(r) else np.nan)
            labels.append(f"{ph:g}")
        x = np.arange(len(PHIS))
        ax[k].bar(x - 0.2, vals_eff, 0.4, label="efficiency [%]", color="tab:blue")
        ax[k].bar(x + 0.2, vals_far, 0.4, label="false rate [%]", color="tab:red")
        ax[k].set_xticks(x); ax[k].set_xticklabels(labels)
        ax[k].set_xlabel(r"$\phi_{max}$ [rad]"); ax[k].set_title(f"T={T}, $\\sigma_{{scatt}}$={ss:g}")
        ax[k].grid(alpha=0.3, axis="y"); ax[k].legend(fontsize=8)
    fig.suptitle(f"Larger_Scatter_Density (T4) — tightening the cone at fixed T={T}", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "lsd_cone_bars_T200.png", dpi=130); plt.close(fig)


def main():
    df = load()
    print(f"[load] {len(df)} shared-aware rows for {STUDY}")
    ag = aggregate(df)
    print(f"[agg]  {len(ag)} cells -> results/lsd_store_summary.csv")
    fig_vs_T(ag); fig_density_collapse(ag); fig_heatmap(ag)

    def at(sol, ph, ss, T, col):
        r = ag[(ag.solver == sol) & np.isclose(ag.phi_max, ph) &
               np.isclose(ag.sigma_scatt, ss) & (ag.n_trk == T)]
        return float(r[col].values[0]) if len(r) else float("nan")
    print("\n=== HEADLINE (classical, sigma_scatt=1e-4; valid solves only) ===")
    for T in (200, 1000):
        print(f"T={T}:  wide phi=0.2 far={at('classical',0.2,1e-4,T,'far'):.1f}% "
              f"(excl {at('classical',0.2,1e-4,T,'excluded_frac'):.0%})  | "
              f"tight phi=0.01 far={at('classical',0.01,1e-4,T,'far'):.1f}% "
              f"(excl {at('classical',0.01,1e-4,T,'excluded_frac'):.0%})  "
              f"eff {at('classical',0.2,1e-4,T,'eff'):.1f}->{at('classical',0.01,1e-4,T,'eff'):.1f}%")
    ex = ag[(ag.solver == "classical") & (ag.excluded_frac > 0)]
    if len(ex):
        print(f"\n[gate] {len(ex)} classical cells have exclusions; "
              f"{(ag[ag.solver == 'classical'].n_valid < MIN_VALID).sum()} cells blanked (<{MIN_VALID} valid)")


if __name__ == "__main__":
    main()
