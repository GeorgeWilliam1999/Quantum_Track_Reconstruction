"""
Aggregate presentation figures driven by the canonical metrics view
(qtrk_store/manifest/metrics.csv).  Classical coverage is complete across the
full T grid and all noise cells, so every figure here is built fresh from the
recomputed (absolute-0.35) metrics — no reuse of pre-fix PNGs.

Produces (figures/):
  seg_metrics_vs_T            segment eff/purity/false-rate vs T (clean step)
  noise_collapse_heatmap      classical seg-eff over (sigma_res x sigma_scatt)
  scatter_drop_heatmap        Larger_Scatter: eff over (sigma_scatt x hit drop)
  density_vs_phi              Larger_Scatter_Density: eff vs cone width phi_max
  scaling_nseg_nnz            n_seg & A_nnz vs T (sparse ~O(T^2))
  timing_vs_T                 classical solve time vs T
  erf_vs_step_noise           seg-eff: erf widths vs noise (where erf helps)
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

M = pd.read_csv(cm.METRICS_CSV)
C = M[M.solver == "classical"].copy()
print(f"loaded metrics.csv: {len(M)} rows ({len(C)} classical)")


def agg(df, by, col):
    """mean + standard error of the mean of `col` grouped by `by`."""
    g = df.groupby(by)[col]
    out = g.agg(["mean", "count", "std"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


# ---------------------------------------------------------------------------
# 1. Segment metrics vs T (clean, step kernel) — the classical baseline
# ---------------------------------------------------------------------------
def fig_seg_metrics_vs_T():
    d = C[(C.study == "Epsilon_study_2") & (C.kernel == "step") &
          (C.sigma_res == 0.0) & (np.isclose(C.sigma_scatt, 1e-4))]
    if d.empty:
        print("  [skip] seg_metrics_vs_T: no clean cell"); return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for col, lab, c in [("segment_efficiency", "Efficiency", cm.C["true"]),
                        ("segment_purity", "Purity", cm.C["classical"]),
                        ("segment_false_rate", "False rate", cm.C["quantum"])]:
        a = agg(d, "n_trk", col)
        ax.errorbar(a.n_trk, a["mean"], yerr=a["sem"], marker="o",
                    capsize=3, label=lab, color=c)
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("Segment metric"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Classical segment reconstruction vs multiplicity\n"
                 r"(clean: $\sigma_{\rm scatt}=10^{-4}$, $\sigma_{\rm res}=0$, step kernel)")
    ax.legend(loc="center left"); ax.axhline(1.0, ls=":", c="grey", lw=1)
    cm.savefig(fig, "seg_metrics_vs_T")


# ---------------------------------------------------------------------------
# 2. Classical failure mode: efficiency robust, PURITY collapses under
#    resolution smearing.   epsilon scales with the noise so true segments stay
#    accepted (efficiency ~1), but the widened acceptance lets false triplets
#    leak in -> purity falls.  Two panels make the asymmetry the headline.
# ---------------------------------------------------------------------------
def _annot(ax, piv):
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.55 else "black", fontsize=10)


def fig_noise_failure_mode(Tsel=200):
    d = C[(C.study == "Epsilon_study_2") & (C.kernel == "step") & (C.n_trk == Tsel)]
    if d.empty:
        print(f"  [skip] noise_failure_mode: no T={Tsel}"); return
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    for ax, col, title, cmap in [
        (axes[0], "segment_efficiency", "Efficiency: robust", "viridis"),
        (axes[1], "segment_purity", "Purity: collapses", "inferno")]:
        piv = d.pivot_table(index="sigma_res", columns="sigma_scatt",
                            values=col, aggfunc="mean")
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap=cmap,
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:g}" for v in piv.columns], rotation=30)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:g}" for v in piv.index])
        ax.set_xlabel(r"$\sigma_{\rm scatt}$ (rad)")
        ax.set_ylabel(r"$\sigma_{\rm res}$ (mm)")
        ax.set_title(title); _annot(ax, piv); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Classical failure mode at $T={Tsel}$: resolution smearing "
                 "spares efficiency but collapses purity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    cm.savefig(fig, "noise_failure_mode")


# ---------------------------------------------------------------------------
# 3. Scatter x hit-drop heatmap (Larger_Scatter)
# ---------------------------------------------------------------------------
def fig_scatter_drop(Tsel=200):
    d = C[(C.study == "Larger_Scatter") & (C.n_trk == Tsel)]
    if d.empty:
        print(f"  [skip] scatter_drop: no T={Tsel}"); return
    piv = d.pivot_table(index="hit_ineff", columns="sigma_scatt",
                        values="segment_efficiency", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    masked = np.ma.masked_invalid(piv.values)
    cmap = plt.cm.magma.copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:g}" for v in piv.index])
    ax.set_xlabel(r"$\sigma_{\rm scatt}$ (rad)")
    ax.set_ylabel("hit inefficiency (drop fraction)")
    ax.set_title(f"Segment efficiency: scattering $\\times$ hit inefficiency\n($T={Tsel}$, step kernel)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="#888", fontsize=10)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="Segment efficiency")
    ax.grid(False)
    cm.savefig(fig, "scatter_drop_heatmap")


# ---------------------------------------------------------------------------
# 4. Density: efficiency vs cone width phi_max
# ---------------------------------------------------------------------------
def fig_density(Tsel=700):
    d = C[C.study == "Larger_Scatter_Density"]
    if d.empty:
        print("  [skip] density"); return
    d = d[d.n_trk == Tsel] if (d.n_trk == Tsel).any() else d[d.n_trk == d.n_trk.max()]
    Tsel = int(d.n_trk.iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for ss, ls in zip(sorted(d.sigma_scatt.unique()), ["-", "--"]):
        dd = d[np.isclose(d.sigma_scatt, ss)]
        for col, c, lab in [("segment_efficiency", cm.C["true"], "eff"),
                            ("segment_purity", cm.C["quantum"], "purity")]:
            a = agg(dd, "phi_max", col).sort_values("phi_max")
            ax.errorbar(a.phi_max, a["mean"], yerr=a["sem"], marker="o", capsize=3,
                        color=c, ls=ls,
                        label=fr"{lab}, $\sigma_{{\rm s}}={ss:g}$")
    ax.set_xscale("log"); ax.set_xlabel(r"generation cone half-width  $\phi_{\max}$ (rad)  — tighter $\to$ denser")
    ax.set_ylabel("Segment metric"); ax.set_ylim(-0.03, 1.05)
    ax.invert_xaxis()
    ax.set_title(f"Density via cone width ($T={Tsel}$): efficiency holds, purity collapses")
    ax.legend(ncol=2, fontsize=9); cm.savefig(fig, "density_vs_phi")


# ---------------------------------------------------------------------------
# 6b. Purity collapse with multiplicity at fixed resolution smearing
# ---------------------------------------------------------------------------
def fig_purity_vs_T_noisy():
    d = C[(C.study == "Epsilon_study_2") & (C.kernel == "step") &
          (np.isclose(C.sigma_scatt, 1e-4))]
    if d.empty:
        print("  [skip] purity_vs_T_noisy"); return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    cmap = plt.cm.plasma
    srs = sorted(d.sigma_res.unique())
    for k, sr in enumerate(srs):
        sub = d[d.sigma_res == sr]
        a = agg(sub, "n_trk", "segment_purity").sort_values("n_trk")
        ax.errorbar(a.n_trk, a["mean"], yerr=a["sem"], marker="o", capsize=3,
                    color=cmap(k / max(len(srs) - 1, 1)),
                    label=fr"$\sigma_{{\rm res}}={sr:g}$ mm")
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("Segment purity"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Purity vs multiplicity: smearing $\\times$ density compound")
    ax.legend(); cm.savefig(fig, "purity_vs_T_noisy")


# ---------------------------------------------------------------------------
# 5. Sparsity scaling: n_seg & A_nnz vs T
# ---------------------------------------------------------------------------
def fig_scaling():
    d = C[(C.study == "Epsilon_study_2") & (C.kernel == "step") &
          (C.sigma_res == 0.0) & (np.isclose(C.sigma_scatt, 1e-4))]
    if d.empty:
        d = C[C.kernel == "step"]
    a_seg = agg(d, "n_trk", "n_seg"); a_nnz = agg(d, "n_trk", "A_nnz")
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.loglog(a_seg.n_trk, a_seg["mean"], "o-", color=cm.C["classical"],
              label=r"$n_{\rm seg}$ (system size)")
    ax.loglog(a_nnz.n_trk, a_nnz["mean"], "s-", color=cm.C["quantum"],
              label=r"$A_{\rm nnz}$ (non-zeros)")
    T = np.array(sorted(d.n_trk.unique()), float)
    ref = a_seg["mean"].iloc[0] * (T / T[0]) ** 2
    ax.loglog(T, ref, "k--", lw=1.4, alpha=0.7, label=r"$\propto T^2$ guide")
    ax.set_xlabel("Track multiplicity  $T$"); ax.set_ylabel("count")
    ax.set_title("Sparse end-to-end: $A_{\\rm nnz}\\sim n_{\\rm seg}\\sim O(T^2)$, never $O(T^3)$")
    ax.legend(); cm.savefig(fig, "scaling_nseg_nnz")


def fig_timing():
    d = C[(C.kernel == "step") & (C.t_solve > 0)]
    if d.empty:
        print("  [skip] timing"); return
    a = agg(d, "n_trk", "t_solve")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.loglog(a.n_trk, a["mean"], "o-", color=cm.C["classical"])
    ax.fill_between(a.n_trk, a["mean"] - a["sem"], a["mean"] + a["sem"],
                    alpha=0.2, color=cm.C["classical"])
    ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("classical solve time (s)")
    ax.set_title("Classical sparse solve time vs multiplicity")
    cm.savefig(fig, "timing_vs_T")


# ---------------------------------------------------------------------------
# 6. ERF vs step under noise (classical): where the soft kernel helps
# ---------------------------------------------------------------------------
def fig_erf_vs_step_noise():
    d = M[(M.study == "ERF") & (M.solver == "classical")].copy()
    if d.empty:
        print("  [skip] erf_vs_step_noise"); return
    # pick the noisiest paired cell present (erf is expected to help under smearing)
    cell = d.sort_values(["sigma_res", "sigma_scatt"]).iloc[-1]
    sr, ss = cell.sigma_res, cell.sigma_scatt
    dd = d[(d.sigma_res == sr) & (np.isclose(d.sigma_scatt, ss))]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for td in sorted(dd.erf_sigma.unique()):
        sub = dd[np.isclose(dd.erf_sigma, td)]
        a = agg(sub, "n_trk", "segment_efficiency").sort_values("n_trk")
        lab = (r"step ($\theta_d\!\to\!0$)" if td <= 1e-6 else fr"$\theta_d={td:g}$")
        ax.plot(a.n_trk, a["mean"], marker="o", label=lab)
    ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
    ax.set_ylabel("Segment efficiency")
    ax.set_title(fr"ERF soft kernel under smearing ($\sigma_{{\rm res}}={sr:g}$, $\sigma_{{\rm scatt}}={ss:g}$)")
    ax.legend(ncol=2, fontsize=9); cm.savefig(fig, "erf_vs_step_noise")


if __name__ == "__main__":
    print("== aggregate figures ==")
    fig_seg_metrics_vs_T()
    fig_noise_failure_mode()
    fig_purity_vs_T_noisy()
    fig_scatter_drop()
    fig_density()
    fig_scaling()
    fig_timing()
    fig_erf_vs_step_noise()
    print("done.")
