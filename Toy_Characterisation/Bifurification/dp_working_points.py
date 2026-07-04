#!/usr/bin/env python3
"""DP terms Q1 + Q4: fair working-point comparison and activation peak structure.

Q1 — does adding fork/occupancy IMPROVE eff/far vs the base system? Fair
comparison needs matched events and threshold-free / common-working-point
metrics, because the terms move the activation scale (attractor τ is wrong
under occupancy). Per (T, rep, config) this solves classically (MINRES) and
reports: AUC, far at eff ≥ 0.99 and ≥ 0.95 (τ-sweep working points), and the
Youden point. The 1BQF side of Q1 comes from the pilot CSV (far_Qwp, auc_Q on
the same events) — merged in the analysis notebooks.

Q4 — does the activation spectrum change; fewer DISTINCT peak sets? Pooled
(3-rep) histograms per config with scipy find_peaks (log-space, prominence)
counting distinct peaks for true / false / all.

Step kernel only. Outputs results/dp_working_points.csv +
figures/dp_wp_spectra_peaks.png
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
from scipy.signal import find_peaks
from scipy.sparse.linalg import minres

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
(HERE / "results").mkdir(exist_ok=True), (HERE / "figures").mkdir(exist_ok=True)

NOISE = dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01)
GAMMA, DELTA = 3.0, 1.0
T_GRID = [100, 200]
REPS = 3
CONFIGS = [("base", 0.0, 0.0), ("fork b=0.5", 0.5, 0.0), ("fork b=1", 1.0, 0.0),
           ("occ a=0.1", 0.0, 0.1), ("occ a=0.3", 0.0, 0.3), ("both", 1.0, 0.3)]
FOOT = (r"DP working points · $\sigma_{scatt}$=1e-4, $\sigma_{res}$=10 µm, drop=1%, "
        r"$\phi_{max}$=0.2 · formula $\varepsilon$=3.178 mrad · STEP kernel · $\gamma$=3 "
        r"$\delta$=1 · classical MINRES, matched events, 3 rep pooled ·"
        "\n"
        r"working points from the per-config $\tau$-sweep (ROC), NOT the attractor $\tau$ · "
        r"peaks: find_peaks on log10(1+counts), prominence 0.5, 480 bins on [-0.3, 1.5]")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def roc_points(x, truth):
    """far (n_false_active/n_active) at eff>=0.99 / 0.95 via tau sweep + Youden."""
    order = np.argsort(x)[::-1]
    t = truth[order].astype(float)
    tp = np.cumsum(t)
    n_act = np.arange(1, len(x) + 1)
    eff = tp / max(truth.sum(), 1)
    far = (n_act - tp) / n_act
    fpr = (n_act - tp) / max((~truth).sum(), 1)
    out = {}
    for target, tag in [(0.99, "far_eff99"), (0.95, "far_eff95")]:
        k = np.searchsorted(eff, target)          # eff is non-decreasing along the sweep
        out[tag] = float(far[min(k, len(far) - 1)])
    j = int(np.argmax(eff - fpr))
    out["eff_youden"], out["far_youden"] = float(eff[j]), float(far[j])
    return out


def peak_count(vals, bins):
    h, _ = np.histogram(vals, bins=bins)
    p, _ = find_peaks(np.log10(1.0 + h), prominence=0.5)
    return len(p)


def main():
    from lhcb_velo_toy.analysis import compute_epsilon
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    bins = np.linspace(-0.3, 1.5, 481)
    rows, pool = [], {}
    for T in T_GRID:
        for rep in range(REPS):
            ev, _ = qp.ensure_event(n_trk=T, rep=rep, **NOISE)
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                       gamma=GAMMA, delta=DELTA)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            for name, beta, alpha in CONFIGS:
                A, b, tau_att, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                                         gamma=GAMMA, delta=DELTA)
                x, _ = minres(A, b, rtol=1e-8, maxiter=8000)
                x = np.asarray(x).ravel()
                m_att = qp.metrics_at(x, truth, tau_att)
                rows.append(dict(T=T, rep=rep, config=name, beta=beta, alpha=alpha,
                                 n_seg=int(ham.n_segments), auc=bif.auc(x, truth),
                                 eff_att=m_att["segment_efficiency"],
                                 far_att=m_att["segment_false_rate"],
                                 **roc_points(x, truth)))
                pool.setdefault((T, name), []).append((x, truth))
                print(f"T={T} rep={rep} {name}: auc={rows[-1]['auc']:.5f} "
                      f"far@eff99={rows[-1]['far_eff99']:.4f} "
                      f"far@eff95={rows[-1]['far_eff95']:.4f}", flush=True)
    df = pd.DataFrame(rows)

    # ---- Q4: pooled spectra + peak counts (largest T) -----------------------
    T_SHOW = T_GRID[-1]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex=True)
    pk_rows = []
    for ax, (name, beta, alpha) in zip(axes.ravel(), CONFIGS):
        xs = pool[(T_SHOW, name)]
        xt = np.concatenate([x[t] for x, t in xs])
        xf = np.concatenate([x[~t] for x, t in xs])
        n_pk = dict(config=name, T=T_SHOW,
                    peaks_true=peak_count(xt, bins), peaks_false=peak_count(xf, bins),
                    peaks_all=peak_count(np.concatenate([xt, xf]), bins))
        pk_rows.append(n_pk)
        ax.hist(np.clip(xf, bins[0], bins[-1]), bins=bins, histtype="stepfilled",
                alpha=0.42, color="#eb6834", label="false")
        ax.hist(np.clip(xt, bins[0], bins[-1]), bins=bins, histtype="step", lw=1.3,
                color="#33322e", label="true")
        ax.set_yscale("log")
        ax.set_title(f"{name} · peaks t/f/all = {n_pk['peaks_true']}/"
                     f"{n_pk['peaks_false']}/{n_pk['peaks_all']}", loc="left", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("segment activation x")
    fig.suptitle(f"Activation peak structure under the DP terms (T={T_SHOW}, 3 reps pooled, "
                 f"step kernel)", y=0.995)
    fig.tight_layout(rect=(0, 0.085, 1, 0.96))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_wp_spectra_peaks.png", bbox_inches="tight")
    plt.close(fig)

    pk = pd.DataFrame(pk_rows)
    df = df.merge(pk, on=["T", "config"], how="left")
    df.to_csv(HERE / "results" / "dp_working_points.csv", index=False)
    print(f"[csv] {len(df)} rows -> results/dp_working_points.csv")
    print(df.groupby(["T", "config"])[["auc", "far_eff99", "far_eff95"]].mean().to_string())
    print(pk.to_string(index=False))


if __name__ == "__main__":
    main()
