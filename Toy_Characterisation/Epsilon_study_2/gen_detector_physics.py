#!/usr/bin/env python3
r"""
Detector-physics translation layer for the Epsilon_study_2 segment metrics.

The toy is driven by two noise knobs, sigma_scatt and sigma_res.  This script
rewrites every result in terms of the *physical* quantities the supervisor
cares about,

    sigma_scatt  <->  transverse momentum pT  AND  material thickness x/X0
                      (multiple-Coulomb-scattering, Highland/Lynch-Dahl)
    sigma_res    <->  single-hit resolution = pixel pitch / sqrt(12)

both of which enter the segment algorithm through the single combination

    sigma_p^2 = sigma_scatt^2 + 6 arctan^2(sigma_res / dz)          (per projection)
              = [13.6 MeV/(beta p c)]^2 (x/X0)(1 + 0.038 ln x/X0)^2     <- pT, thickness
              + 6 (pitch/sqrt(12))^2 / dz^2                             <- resolution

and the Hamiltonian acceptance is eps = s * sqrt(2) * sigma_p (s = 3).

Outputs
  figures/epsilon_sensitivity/dp1_sigma_p_decomposition.png
  figures/epsilon_sensitivity/dp2_crossover_map.png
  figures/epsilon_sensitivity/dp3_scatt_to_pT.png
  figures/epsilon_sensitivity/dp4_softtrack_extension.png     (NEW DATA)
  figures/epsilon_sensitivity/dp5_false_tax_decomposition.png
  outputs/detector_physics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import helpers  # noqa: E402
from helpers import solve_quantum_statevector  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (  # noqa: E402
    DEFAULT_DZ, DEFAULT_SCALE, DEFAULT_THETA_MIN,
    segment_truth_mask, solver_segment_metrics,
)
from lhcb_velo_toy.solvers import SimpleHamiltonianFast  # noqa: E402
from qtrk_pipeline.metrics import rescale_to_signal, quantum_metrics_wp  # noqa: E402

FIGDIR = HERE / "figures" / "epsilon_sensitivity"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# physical constants and the (flagged) detector anchor
# ---------------------------------------------------------------------------
DZ = DEFAULT_DZ            # 33 mm inter-plane spacing (5 planes, z = 33..165 mm)
S = DEFAULT_SCALE          # 3 sigma acceptance scale
TH_MIN = DEFAULT_THETA_MIN  # 1.5e-5 rad pixel-pitch angular floor
HIGHLAND = 13.6            # MeV, Highland constant
PIX_VELO = 0.055          # mm, VELO pixel pitch
SRES_VELO = PIX_VELO / np.sqrt(12.0)   # 15.9 um binary resolution

# NOMINAL per-module material budget. PLACEHOLDER: replace with the measured
# VELO Run-3 value (the RF foil dominates). Every pT number below scales as
# sqrt(x/X0), so this anchor only shifts the momentum labels, not the physics.
XX0_NOMINAL = 0.01

# production noise grid actually simulated in Epsilon_study_2
GRID_SS = np.array([1e-4, 3e-4, 5e-4])
GRID_SR = np.array([0.0, 0.01, 0.02, 0.05])


# ---------------------------------------------------------------------------
# the translation functions  (all physics lives in these six lines)
# ---------------------------------------------------------------------------
def sigma_p(sr: float, ss: float) -> float:
    """Per-projection kink RMS: quadrature sum of scattering and resolution."""
    return float(np.sqrt(ss ** 2 + 6.0 * np.arctan(sr / DZ) ** 2))


def sigma_res_term(sr):
    """Resolution contribution to sigma_p:  sqrt(6) arctan(sigma_res/dz)."""
    return np.sqrt(6.0) * np.arctan(np.asarray(sr, dtype=float) / DZ)


def highland(p_MeV: np.ndarray, xX0: float = XX0_NOMINAL) -> np.ndarray:
    """Projected RMS multiple-scattering angle per plane (Highland/Lynch-Dahl)."""
    p = np.asarray(p_MeV, dtype=float)
    logterm = 1.0 + 0.038 * np.log(xX0)         # beta = 1, z = 1
    return (HIGHLAND / p) * np.sqrt(xX0) * logterm


def p_from_scatt(ss: float, xX0: float = XX0_NOMINAL) -> float:
    """Invert Highland: momentum (MeV) that produces scattering angle ss."""
    if ss <= 0:
        return np.inf
    f = lambda p: highland(p, xX0) - ss
    return float(brentq(f, 1.0, 1.0e7))


def pitch_from_sres(sr: float) -> float:
    """Pixel pitch (mm) for a binary single-hit resolution sigma_res."""
    return sr * np.sqrt(12.0)


def p_cross(xX0: float, sr: float) -> float:
    """Momentum where scattering term == resolution term (the crossover)."""
    target = sigma_res_term(sr)
    if target <= 0:
        return np.inf
    f = lambda p: highland(p, xX0) - target
    return float(brentq(f, 1.0, 1.0e7))


# ---------------------------------------------------------------------------
# analytic empirical laws (from gen_epsilon_sensitivity_scan / sparsity study)
# ---------------------------------------------------------------------------
def p_kink(eps: float, sr: float, ss: float) -> float:
    x = eps ** 2 - 2.0 * TH_MIN ** 2
    return 1.0 if x <= 0 else float(np.exp(-x / (2.0 * sigma_p(sr, ss) ** 2)))


def eff_analytic(p):
    q = 1.0 - np.asarray(p, dtype=float)
    return q ** 2 * (1.0 + 0.5 * (1.0 - q))


# ===========================================================================
# PART A  --  soft-track extension run  (NEW DATA, classical + 1BQF)
# ===========================================================================
T, NREP_C, NREP_Q = 30, 10, 2
# canonical false-rate coefficient far ~ C_FAR * eps^2, fit in the dedicated
# direct-eps sweep of section 7 (20 reps, large-eps lever arm; the formula-eps
# false rate measured here is too small/quantised to fit c reliably on its own)
C_FAR = 267.0
GEOM = helpers.make_geometry()
# extend sigma_scatt from the production grid (<=5e-4) deep into the
# scattering-dominated regime (3e-3 ~ pT 0.36 GeV at x/X0 = 1%)
SS_EXT = np.array([1e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3])
# two families: sigma_res = 0 (pure scattering) and 0.01 (VELO-like)
SR_FAMILIES = [0.0, 0.01]


def solve_classical(ev, eps):
    ham = SimpleHamiltonianFast(epsilon=eps, gamma=helpers.GAMMA,
                                delta=helpers.DELTA, theta_d=1e-4)
    ham.construct_segments(ev)
    ham.construct_hamiltonian(ev, convolution=False)
    truth = segment_truth_mask(ham)
    sol_C = np.asarray(ham.solve_classicaly()).ravel()
    mC = solver_segment_metrics(truth, sol_C, threshold=0.35)
    return ham, truth, sol_C, mC


def solve_quantum(ham, sol_C, truth):
    raw, p_anc, n_sys = solve_quantum_statevector(ham.A, ham.b, device="CPU")
    # 1BQF HEADLINE = wp99 high-efficiency working point: per-solve tau just below
    # the 1% quantile of the TRUE-segment amplitudes (eff/far are rescale-invariant).
    # The fixed tau=0.35 cut pinned the 1BQF at the artefactual ~75% plateau
    # (user 2026-06-14, wp99 refresh).  Classical stays at its fixed tau=0.35 op-point.
    mQ = quantum_metrics_wp(raw, sol_C, truth)
    return mQ, int(n_sys)


soft = {}
t_start = time.time()
for sr in SR_FAMILIES:
    rows = []
    for ss in SS_EXT:
        eps = float(compute_epsilon(sr, ss))
        effC, farC, effQ, farQ = [], [], [], []
        n_nnz = n_sys = 0
        for rep in range(NREP_C):
            ev = helpers.safe_generate(T, seed=80_000 + rep, geom=GEOM,
                                       measurement_error=sr, collision_noise=ss)
            ham, truth, sol_C, mC = solve_classical(ev, eps)
            effC.append(mC["segment_efficiency"])
            farC.append(mC["segment_false_rate"])
            n_nnz = int(ham.A.nnz)
            if rep < NREP_Q:
                mQ, n_sys = solve_quantum(ham, sol_C, truth)
                effQ.append(mQ["segment_efficiency"])
                farQ.append(mQ["segment_false_rate"])
        row = dict(
            sr=sr, ss=float(ss), eps=eps, sigma_p=sigma_p(sr, ss),
            pT_MeV=p_from_scatt(ss, XX0_NOMINAL),
            eff_C=float(np.mean(effC)), eff_C_sem=float(np.std(effC) / np.sqrt(len(effC))),
            far_C=float(np.mean(farC)), far_C_sem=float(np.std(farC) / np.sqrt(len(farC))),
            eff_Q=float(np.mean(effQ)) if effQ else None,
            far_Q=float(np.mean(farQ)) if farQ else None,
            A_nnz=n_nnz, n_sys=n_sys,
        )
        rows.append(row)
        print(f"[A] sr={sr:g} ss={ss:.2e} pT={row['pT_MeV']/1e3:.2f}GeV "
              f"eps={eps:.3e} effC={row['eff_C']:.3f} farC={row['far_C']:.3f} "
              f"effQ={row['eff_Q'] if row['eff_Q'] is None else round(row['eff_Q'],3)} "
              f"farQ={row['far_Q'] if row['far_Q'] is None else round(row['far_Q'],3)}",
              flush=True)
    soft[f"sr{sr:g}"] = rows
print(f"[A] soft-track sweep done in {time.time()-t_start:.0f}s", flush=True)

# fit the false-rate coefficient far ~ c * eps^2 from the sr=0 family (pure
# scattering, the cleanest one) so the analytic overlays use the data's own c
fam0 = soft["sr0"]
eps2 = np.array([r["eps"] ** 2 for r in fam0])
far0 = np.array([r["far_C"] for r in fam0])
c_far_local = float(np.sum(far0 * eps2) / np.sum(eps2 ** 2))   # noisy cross-check
print(f"[A] local far=c*eps^2 cross-check c={c_far_local:.1f} (canonical {C_FAR})",
      flush=True)


# ===========================================================================
# PART B  --  figures
# ===========================================================================
def _toptick_pT(ax, xX0=XX0_NOMINAL):
    """Add a top axis labelled in pT (GeV) mapped from sigma_scatt."""
    sec = ax.secondary_xaxis(
        "top",
        functions=(np.vectorize(lambda s: p_from_scatt(max(s, 1e-12), xX0) / 1e3),
                   np.vectorize(lambda p: highland(p * 1e3, xX0))))
    sec.set_xlabel(rf"$p_T$ [GeV]  (at $x/X_0={xX0*100:g}\%$)")
    return sec


# ---- DP1 : sigma_p decomposition --------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

ss_axis = np.logspace(-5, np.log10(3e-3), 300)
for sr in GRID_SR:
    sp = np.sqrt(ss_axis ** 2 + 6.0 * np.arctan(sr / DZ) ** 2)
    axA.loglog(ss_axis, sp, lw=2,
               label=rf"$\sigma_{{res}}={sr:g}$ mm (pitch {pitch_from_sres(sr)*1e3:.0f} $\mu$m)")
    floor = sigma_res_term(sr)
    if floor > 0:
        axA.axhline(floor, ls=":", color=axA.lines[-1].get_color(), alpha=0.6)
axA.plot(ss_axis, ss_axis, "k--", lw=1, alpha=0.5,
         label=r"pure scattering $\sigma_p=\sigma_{scatt}$")
axA.axvspan(GRID_SS.min(), GRID_SS.max(), color="grey", alpha=0.15)
axA.text(np.sqrt(GRID_SS.min() * GRID_SS.max()), 1.3e-5,
         "production\ngrid", ha="center", fontsize=8)
sstar = sigma_res_term(0.01)
axA.axvline(sstar, color="C1", ls="-.", alpha=0.7)
axA.annotate(rf"$\sigma^*_{{scatt}}=\sqrt{{6}}\,\arctan(\sigma_{{res}}/\Delta z)$"
             "\n" rf"$={sstar:.1e}$ rad",
             xy=(sstar, 4e-4), xytext=(1.1e-3, 1.5e-4), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="C1"))
axA.set_xlabel(r"$\sigma_{scatt}$ [rad]")
axA.set_ylabel(r"$\sigma_p=\sqrt{\sigma_{scatt}^2+6\arctan^2(\sigma_{res}/\Delta z)}$ [rad]")
axA.set_title(r"(a) kink scale vs scattering  ($p_T\!\downarrow$ as $\sigma_{scatt}\!\uparrow$)")
axA.legend(fontsize=7.5, loc="lower right")
axA.grid(alpha=0.3, which="both")
_toptick_pT(axA)

sr_axis = np.linspace(0, 0.05, 300)
for ss in (1e-4, 5e-4, 2e-3):
    sp = np.sqrt(ss ** 2 + 6.0 * np.arctan(sr_axis / DZ) ** 2)
    axB.plot(sr_axis * 1e3, sp * 1e3, lw=2,
             label=rf"$\sigma_{{scatt}}={ss:g}$ rad "
                   rf"($p_T\!\approx${p_from_scatt(ss)/1e3:.1f} GeV)")
axB.axvline(SRES_VELO * 1e3, color="k", ls="--", alpha=0.6)
axB.text(SRES_VELO * 1e3 + 0.5, 0.3, "VELO\n55 $\\mu$m", fontsize=8)
secB = axB.secondary_xaxis("top",
                           functions=(lambda x: x * np.sqrt(12.0),
                                      lambda x: x / np.sqrt(12.0)))
secB.set_xlabel(r"pixel pitch [$\mu$m]")
axB.set_xlabel(r"$\sigma_{res}$ [$\mu$m]")
axB.set_ylabel(r"$\sigma_p$ [mrad]")
axB.set_title(r"(b) kink scale vs resolution  (pitch $\uparrow$ as $\sigma_{res}\!\uparrow$)")
axB.legend(fontsize=8)
axB.grid(alpha=0.3)
fig.suptitle(r"DP1  $\sigma_p^2=\sigma_{scatt}^2+6\arctan^2(\sigma_{res}/\Delta z)$:"
             r"  scattering ($p_T$, $x/X_0$) and resolution (pitch) add in quadrature",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIGDIR / "dp1_sigma_p_decomposition.png", dpi=130)
plt.close(fig)
print("[B] dp1_sigma_p_decomposition.png", flush=True)


# ---- DP2 : crossover map ----------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

# panel A: (pT, x/X0) at VELO resolution
pT = np.logspace(np.log10(0.2), np.log10(20), 220)     # GeV
xx0 = np.logspace(np.log10(0.002), np.log10(0.1), 200)
PT, XX = np.meshgrid(pT, xx0)
scatt = highland(PT * 1e3, XX)
res_term = sigma_res_term(SRES_VELO)
ratio = np.log10(scatt ** 2 / res_term ** 2)            # >0 scattering-limited
pcm = axA.pcolormesh(PT, XX * 100, ratio, cmap="RdBu_r",
                     vmin=-2, vmax=2, shading="auto")
axA.contour(PT, XX * 100, ratio, levels=[0], colors="k", linewidths=2)
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel(r"$p_T$ [GeV]")
axA.set_ylabel(r"material $x/X_0$ [%]")
axA.set_title(rf"(a) dominant kink source vs $p_T$ and thickness"
              "\n" rf"(VELO $\sigma_{{res}}={SRES_VELO*1e3:.1f}\,\mu$m, $\Delta z={DZ:g}$ mm)")
axA.axhline(XX0_NOMINAL * 100, color="grey", ls=":", alpha=0.7)
cb = fig.colorbar(pcm, ax=axA)
cb.set_label(r"$\log_{10}(\sigma_{scatt}^2 / [6\arctan^2(\sigma_{res}/\Delta z)])$")
axA.text(0.4, 6, "scattering-limited\n(soft / thick)", fontsize=8)
axA.text(6, 0.3, "resolution-limited\n(hard / thin)", fontsize=8)

# panel B: (pT, pitch) at nominal thickness
pitch = np.logspace(np.log10(10), np.log10(300), 200)   # um
PT2, PIT = np.meshgrid(pT, pitch)
scatt2 = highland(PT2 * 1e3, XX0_NOMINAL)
res2 = sigma_res_term((PIT * 1e-3) / np.sqrt(12.0))     # pitch um -> sigma_res mm
ratio2 = np.log10(scatt2 ** 2 / res2 ** 2)
pcm2 = axB.pcolormesh(PT2, PIT, ratio2, cmap="RdBu_r",
                      vmin=-2, vmax=2, shading="auto")
axB.contour(PT2, PIT, ratio2, levels=[0], colors="k", linewidths=2)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel(r"$p_T$ [GeV]")
axB.set_ylabel(r"pixel pitch [$\mu$m]")
axB.set_title(rf"(b) dominant kink source vs $p_T$ and pitch"
              "\n" rf"(at $x/X_0={XX0_NOMINAL*100:g}\%$)")
axB.axhline(PIX_VELO * 1e3, color="grey", ls=":", alpha=0.7)
axB.text(PIX_VELO * 1e3 * 1.1, 12, "VELO 55 $\\mu$m", fontsize=8)
cb2 = fig.colorbar(pcm2, ax=axB)
cb2.set_label(r"$\log_{10}(\sigma_{scatt}^2 / [6\arctan^2(\sigma_{res}/\Delta z)])$")
fig.suptitle(r"DP2  the crossover momentum $p_{cross}=\sqrt{2}\,(\Delta z/\mathrm{pitch})"
             r"\,13.6\,\mathrm{MeV}\sqrt{x/X_0}$ — black line where the two terms are equal",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGDIR / "dp2_crossover_map.png", dpi=130)
plt.close(fig)
print("[B] dp2_crossover_map.png", flush=True)


# ---- DP3 : sigma_scatt -> pT at three thicknesses ---------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ss_axis = np.logspace(-4, np.log10(3e-3), 200)
for xx in (0.005, 0.01, 0.02):
    pT_curve = np.array([p_from_scatt(s, xx) / 1e3 for s in ss_axis])
    ax.loglog(ss_axis, pT_curve, lw=2, label=rf"$x/X_0={xx*100:g}\%$")
for ss in GRID_SS:
    ax.axvline(ss, color="grey", ls=":", alpha=0.6)
ax.axvspan(GRID_SS.min(), GRID_SS.max(), color="grey", alpha=0.12)
ax.axhline(p_cross(XX0_NOMINAL, SRES_VELO) / 1e3, color="C3", ls="--",
           label=rf"$p_{{cross}}\approx{p_cross(XX0_NOMINAL, SRES_VELO)/1e3:.1f}$ GeV"
                 rf" (VELO, $x/X_0=1\%$)")
ax.set_xlabel(r"$\sigma_{scatt}$ [rad]")
ax.set_ylabel(r"$p_T$ [GeV]")
ax.set_title(r"DP3  $\sigma_{scatt}=\dfrac{13.6\,\mathrm{MeV}}{p_T}\sqrt{x/X_0}\,(1+0.038\ln x/X_0)$"
             "\n" r"the production grid lives at $p_T\!\approx\!2$–$11$ GeV (resolution-limited)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "dp3_scatt_to_pT.png", dpi=130)
plt.close(fig)
print("[B] dp3_scatt_to_pT.png", flush=True)


# ---- DP4 : soft-track extension (DATA) --------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
colors = {0.0: "C0", 0.01: "C2"}
for sr in SR_FAMILIES:
    rows = soft[f"sr{sr:g}"]
    ss = np.array([r["ss"] for r in rows])
    effC = np.array([r["eff_C"] for r in rows])
    effCe = np.array([r["eff_C_sem"] for r in rows])
    farC = np.array([r["far_C"] for r in rows])
    farCe = np.array([r["far_C_sem"] for r in rows])
    lab = rf"$\sigma_{{res}}={sr:g}$ mm"
    axA.errorbar(ss, effC, yerr=effCe, marker="o", color=colors[sr],
                 lw=2, capsize=2, label="classical, " + lab)
    axB.errorbar(ss, farC, yerr=farCe, marker="o", color=colors[sr],
                 lw=2, capsize=2, label="classical, " + lab)
    effQ = [r["eff_Q"] for r in rows]
    farQ = [r["far_Q"] for r in rows]
    if any(e is not None for e in effQ):
        mask = [e is not None for e in effQ]
        axA.plot(ss[mask], np.array(effQ)[mask], marker="s", ls="--",
                 color=colors[sr], alpha=0.8, label="1BQF, " + lab)
        axB.plot(ss[mask], np.array(farQ)[mask], marker="s", ls="--",
                 color=colors[sr], alpha=0.8, label="1BQF, " + lab)

# analytic far overlay: far = c * eps^2 = c * 2 s^2 (ss^2 + 6 arctan^2(sr/dz))
ss_fine = np.logspace(-4, np.log10(3e-3), 200)
for sr in SR_FAMILIES:
    eps_fine = np.array([compute_epsilon(sr, s) for s in ss_fine])
    axB.plot(ss_fine, C_FAR * eps_fine ** 2, ":", color=colors[sr], alpha=0.8)
axB.text(0.03, 0.93, rf"$\overline{{\mathrm{{far}}}}=c\,\varepsilon^2$,"
                      rf" $c={C_FAR:.0f}$ rad$^{{-2}}$ (sec. 7)",
         transform=axB.transAxes, fontsize=8, va="top")

for ax in (axA, axB):
    ax.axvspan(GRID_SS.min(), GRID_SS.max(), color="grey", alpha=0.12)
    ax.axvline(sigma_res_term(0.01), color="C1", ls="-.", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{scatt}$ [rad]")
    _toptick_pT(ax)
    ax.grid(alpha=0.3, which="both")
axA.set_ylabel("segment efficiency")
axA.set_ylim(0.6, 1.03)
axA.set_title(r"(a) efficiency held flat by formula $\varepsilon$ (noise-independent)")
axA.legend(fontsize=7.5, loc="lower left")
axA.text(0.97, 0.04, r"classical $\tau=0.35$, 1BQF wp99",
         transform=axA.transAxes, fontsize=7.5, ha="right", va="bottom",
         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
axB.set_ylabel("segment false rate")
axB.set_yscale("log")
axB.set_title(r"(b) false tax grows $\propto\sigma_p^2$ once $\sigma_{scatt}>\sigma^*_{scatt}$")
axB.legend(fontsize=7.5, loc="upper left")
fig.suptitle(r"DP4  soft-track extension (T=30): walking $\sigma_{scatt}$ from 11 GeV down to 0.36 GeV"
             r" at fixed resolution — NEW DATA in the scattering-limited regime", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGDIR / "dp4_softtrack_extension.png", dpi=130)
plt.close(fig)
print("[B] dp4_softtrack_extension.png", flush=True)


# ---- DP5 : false-tax decomposition (analytic) -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, sr in zip(axes, (0.01, SRES_VELO)):
    ss_fine = np.logspace(-4, np.log10(3e-3), 300)
    far_res = C_FAR * 2 * S ** 2 * (sigma_res_term(sr) ** 2) * np.ones_like(ss_fine)
    far_sca = C_FAR * 2 * S ** 2 * ss_fine ** 2
    ax.fill_between(ss_fine, 0, far_res, color="C0", alpha=0.5,
                    label=r"resolution part  $c\,2s^2\,6\arctan^2(\sigma_{res}/\Delta z)$")
    ax.fill_between(ss_fine, far_res, far_res + far_sca, color="C3", alpha=0.5,
                    label=r"scattering part  $c\,2s^2\,\sigma_{scatt}^2$")
    ax.plot(ss_fine, far_res + far_sca, "k", lw=1.5, label=r"total $\propto\sigma_p^2$")
    sstar = sigma_res_term(sr)
    ax.axvline(sstar, color="k", ls="-.", alpha=0.6)
    ax.text(sstar * 1.05, ax.get_ylim()[1] * 0.05 if False else far_res[0] * 1.2,
            rf"$\sigma^*_{{scatt}}={sstar:.1e}$", rotation=90, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_{scatt}$ [rad]")
    ax.set_ylabel("predicted segment false rate")
    pitch_um = pitch_from_sres(sr) * 1e3
    ax.set_title(rf"$\sigma_{{res}}={sr:.4g}$ mm  (pitch {pitch_um:.0f} $\mu$m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    _toptick_pT(ax)
fig.suptitle(r"DP5  false rate $=c\,\varepsilon^2=c\,2s^2(\sigma_{scatt}^2+6\arctan^2(\sigma_{res}/\Delta z))$:"
             r" the two detector terms add, crossing at $\sigma^*_{scatt}$", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGDIR / "dp5_false_tax_decomposition.png", dpi=130)
plt.close(fig)
print("[B] dp5_false_tax_decomposition.png", flush=True)


# ===========================================================================
# PART C  --  dump every computed number for the report to cite
# ===========================================================================
grid_translation = []
for ss in GRID_SS:
    grid_translation.append(dict(
        sigma_scatt=float(ss),
        pT_GeV={f"xX0_{x}": p_from_scatt(ss, x) / 1e3 for x in (0.005, 0.01, 0.02)},
    ))
res_translation = []
for sr in GRID_SR:
    res_translation.append(dict(
        sigma_res_mm=float(sr), pitch_um=pitch_from_sres(sr) * 1e3,
        res_term_rad=float(sigma_res_term(sr)),
    ))

out = dict(
    constants=dict(dz_mm=DZ, scale=S, theta_min=TH_MIN, highland_MeV=HIGHLAND,
                   pix_velo_um=PIX_VELO * 1e3, sres_velo_um=SRES_VELO * 1e3,
                   xX0_nominal=XX0_NOMINAL),
    crossover=dict(
        p_cross_GeV_velo=p_cross(XX0_NOMINAL, SRES_VELO) / 1e3,
        sstar_at_sr0p01=float(sigma_res_term(0.01)),
        formula="p_cross = sqrt(2)*(dz/pitch)*13.6MeV*sqrt(x/X0)*(1+0.038 ln x/X0)",
    ),
    grid_to_pT=grid_translation,
    res_to_pitch=res_translation,
    c_far_canonical=C_FAR,
    c_far_local_crosscheck=c_far_local,
    softtrack=soft,
)
with open(OUTDIR / "detector_physics.json", "w") as f:
    json.dump(out, f, indent=2)
print("[C] outputs/detector_physics.json written", flush=True)
print(f"\nDONE in {time.time()-t_start:.0f}s "
      f"| p_cross(VELO,1%) = {p_cross(XX0_NOMINAL, SRES_VELO)/1e3:.2f} GeV "
      f"| grid pT (x/X0=1%) = "
      f"{[round(p_from_scatt(s)/1e3,1) for s in GRID_SS]} GeV", flush=True)
