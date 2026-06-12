#!/usr/bin/env python3
"""
Epsilon formula — derivation illustrations + Monte-Carlo validation.

Produces figures/epsilon_derivation/*.png and
outputs/epsilon_derivation_numbers.json for the Notion report
"Epsilon_study_2 — physical validity of the calculated epsilon".

Checks performed against the real StateEventGenerator:
  1. noise-free floor (does theta_min = 1.5e-5 bound the residual kink?)
  2. per-projection variance bookkeeping
       scattering: kink at mid hit = ONE slope kick  -> sigma_s^2 per proj
       resolution: (1,-2,1)/dz hit weights           -> 6 sigma_r^2/dz^2 per proj
     3D kink ~ Rayleigh(sigma_p), sigma_p^2 = sigma_s^2 + 6 sigma_r^2/dz^2
  3. true coverage + false acceptance at eps (current) and eps/sqrt(2)
     (the "remove a factor of sqrt(2)" option).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_shared"))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import helpers  # noqa: E402  (Toy_Characterisation/_shared)
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import (  # noqa: E402
    DEFAULT_DZ, DEFAULT_SCALE, DEFAULT_THETA_MIN,
)

FIGDIR = HERE / "figures" / "epsilon_derivation"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)

DZ = DEFAULT_DZ          # 33 mm
S = DEFAULT_SCALE        # 3
TH_MIN = DEFAULT_THETA_MIN  # 1.5e-5 rad
GEOM = helpers.make_geometry()


def sigma_proj(sigma_res: float, sigma_scatt: float) -> float:
    """Predicted per-projection kink std at the mid hit."""
    return float(np.sqrt(sigma_scatt**2 + 6.0 * (sigma_res / DZ) ** 2))


# ---------------------------------------------------------------------------
# Kink collection (vectorised)
# ---------------------------------------------------------------------------

def hits_by_module(event):
    mods: dict[int, list] = {}
    for h in event.hits:
        mods.setdefault(h.module_id, []).append(h)
    out = {}
    for m, hs in mods.items():
        out[m] = (np.array([[h.x, h.y, h.z] for h in hs]),
                  np.array([h.track_id for h in hs]),
                  np.array([h.hit_id for h in hs]))
    return out


def true_kinks(event):
    """3D kink angle at every interior hit of every truth track."""
    pos = {h.hit_id: np.array([h.x, h.y, h.z]) for h in event.hits}
    angs = []
    for trk in event.tracks:
        ids = trk.hit_ids
        for k in range(1, len(ids) - 1):
            v1 = pos[ids[k]] - pos[ids[k - 1]]
            v2 = pos[ids[k + 1]] - pos[ids[k]]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angs.append(np.arccos(np.clip(c, -1.0, 1.0)))
    return np.array(angs)


def true_kinks_proj(event):
    """Per-projection slope kinks (dtx, dty) at interior hits (for fig 3)."""
    pos = {h.hit_id: np.array([h.x, h.y, h.z]) for h in event.hits}
    dtx, dty = [], []
    for trk in event.tracks:
        ids = trk.hit_ids
        for k in range(1, len(ids) - 1):
            p0, p1, p2 = pos[ids[k - 1]], pos[ids[k]], pos[ids[k + 1]]
            t1 = (p1[:2] - p0[:2]) / (p1[2] - p0[2])
            t2 = (p2[:2] - p1[:2]) / (p2[2] - p1[2])
            dtx.append(t2[0] - t1[0])
            dty.append(t2[1] - t1[1])
    return np.array(dtx), np.array(dty)


def false_kink_hist(event, edges):
    """Histogram of FALSE-triplet kink angles over consecutive module
    triples (all combinatorial triplets minus same-track ones)."""
    mods = hits_by_module(event)
    mids = sorted(mods.keys())
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    n_false = 0
    for mi in range(1, len(mids) - 1):
        P0, t0, _ = mods[mids[mi - 1]]
        P1, t1, _ = mods[mids[mi]]
        P2, t2, _ = mods[mids[mi + 1]]
        for j in range(len(P1)):
            v1 = P1[j] - P0                       # (n0, 3)
            v2 = P2 - P1[j]                       # (n2, 3)
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            cos = (v1 @ v2.T) / np.outer(n1, n2)
            ang = np.arccos(np.clip(cos, -1.0, 1.0))
            is_true = np.outer(t0 == t1[j], t2 == t1[j])
            f = ang[~is_true]
            n_false += f.size
            counts += np.histogram(f, bins=edges)[0]
    return counts, n_false


# ---------------------------------------------------------------------------
# Monte-Carlo points
# ---------------------------------------------------------------------------

POINTS = {
    "noise_free":  dict(sr=0.0,  ss=0.0),
    "scatt_only":  dict(sr=0.0,  ss=5e-4),
    "res_only":    dict(sr=0.02, ss=0.0),
    "grid_mid":    dict(sr=0.02, ss=3e-4),
    "grid_worst":  dict(sr=0.05, ss=5e-4),
    "validation":  dict(sr=0.0,  ss=1e-4),
}

T_TRUE, NREP_TRUE = 100, 60      # ~60*100*3 = 18k true kinks / point
T_FALSE, NREP_FALSE = 100, 6     # ~6 * 3*100^3 = 18M false triplets / point

results: dict[str, dict] = {}
true_samples: dict[str, np.ndarray] = {}
proj_samples: dict[str, tuple] = {}

print("[mc] collecting true kinks ...")
for name, p in POINTS.items():
    angs, dtxs, dtys = [], [], []
    for rep in range(NREP_TRUE):
        ev = helpers.safe_generate(T_TRUE, seed=10_000 + rep, geom=GEOM,
                                   measurement_error=p["sr"],
                                   collision_noise=p["ss"])
        angs.append(true_kinks(ev))
        if name == "grid_mid":
            dx, dy = true_kinks_proj(ev)
            dtxs.append(dx)
            dtys.append(dy)
    a = np.concatenate(angs)
    true_samples[name] = a
    if dtxs:
        proj_samples[name] = (np.concatenate(dtxs), np.concatenate(dtys))
    sp = sigma_proj(p["sr"], p["ss"])
    eps = compute_epsilon(p["sr"], p["ss"])
    results[name] = dict(
        sigma_res=p["sr"], sigma_scatt=p["ss"],
        n_true_kinks=int(a.size),
        sigma_proj_pred=sp,
        # for a Rayleigh(sigma_p), E[theta^2] = 2 sigma_p^2
        sigma_proj_meas=float(np.sqrt(np.mean(a**2) / 2.0)),
        kink_rms_3d=float(np.sqrt(np.mean(a**2))),
        kink_max=float(a.max()),
        kink_q999=float(np.quantile(a, 0.999)),
        eps_formula=float(eps),
        eps_no_sqrt2=float(eps / np.sqrt(2.0)),
        true_cov_eps=float(np.mean(a < eps)),
        true_cov_no_sqrt2=float(np.mean(a < eps / np.sqrt(2.0))),
    )
    print(f"  {name:11s} sigma_p pred={sp:.3e} meas="
          f"{results[name]['sigma_proj_meas']:.3e}  "
          f"cov(eps)={results[name]['true_cov_eps']:.5f}  "
          f"cov(eps/sqrt2)={results[name]['true_cov_no_sqrt2']:.5f}")

print("[mc] collecting false kinks (grid_mid, grid_worst) ...")
EDGES = np.linspace(0.0, 0.4, 4001)   # 0.1 mrad bins up to 400 mrad
false_hists = {}
for name in ("grid_mid", "grid_worst"):
    p = POINTS[name]
    counts = np.zeros(len(EDGES) - 1, dtype=np.int64)
    ntot = 0
    for rep in range(NREP_FALSE):
        ev = helpers.safe_generate(T_FALSE, seed=20_000 + rep, geom=GEOM,
                                   measurement_error=p["sr"],
                                   collision_noise=p["ss"])
        c, n = false_kink_hist(ev, EDGES)
        counts += c
        ntot += n
    false_hists[name] = (counts, ntot)
    eps = results[name]["eps_formula"]
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])
    acc_eps = counts[centers < eps].sum()
    acc_half = counts[centers < eps / np.sqrt(2.0)].sum()
    results[name].update(
        n_false_triplets=int(ntot),
        false_acc_eps=int(acc_eps),
        false_acc_no_sqrt2=int(acc_half),
        false_ratio_no_sqrt2=float(acc_half / max(acc_eps, 1)),
        false_frac_eps=float(acc_eps / ntot),
        false_per_event_eps=float(acc_eps / NREP_FALSE),
        false_per_event_no_sqrt2=float(acc_half / NREP_FALSE),
    )
    print(f"  {name:11s} false acc: eps={acc_eps}  eps/sqrt2={acc_half}  "
          f"ratio={results[name]['false_ratio_no_sqrt2']:.3f}")


# ---------------------------------------------------------------------------
# Fig 1 — geometry of the kink (the derivation picture)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.5, 5.2))
zs = [0, 1, 2]
labels = ["plane $k-1$", "plane $k$", "plane $k+1$"]
for z, lab in zip(zs, labels):
    ax.axvline(z, color="0.75", lw=6, zorder=0)
    ax.text(z, 1.30, lab, ha="center", fontsize=11)
    ax.text(z, 1.22, r"$\Delta z = 33$ mm" if z < 2 else "",
            ha="left", fontsize=9, color="0.4")

# true straight trajectory (no scattering)
ax.plot([-0.35, 2.3], [-0.14, 0.92], ls="--", color="0.45", lw=1.5,
        label="undisturbed straight trajectory")
# scattered TRUE trajectory: kink at plane k
x_tr = np.array([-0.35, 0, 1, 2, 2.3])
y_tr = np.array([-0.14, 0.0, 0.40, 1.05, 1.245])  # extra slope after plane k
ax.plot(x_tr[:3], y_tr[:3], color="tab:blue", lw=2.2,
        label="true trajectory (scattering kick at each plane)")
ax.plot(x_tr[2:], y_tr[2:], color="tab:blue", lw=2.2)

# true crossing points + measured hits (smeared)
true_pts = [(0, 0.0), (1, 0.40), (2, 1.05)]
meas_pts = [(0, 0.10), (1, 0.33), (2, 1.13)]
for (zt, yt), (zm, ym) in zip(true_pts, meas_pts):
    ax.plot(zt, yt, "o", color="tab:blue", ms=7, zorder=5)
    ax.plot(zm, ym, "s", color="tab:red", ms=8, zorder=6)
    ax.annotate("", xy=(zm, ym), xytext=(zt, yt),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.4))
ax.plot([], [], "o", color="tab:blue", label="true crossing point")
ax.plot([], [], "s", color="tab:red",
        label=r"recorded hit  ($\pm\sigma_{\rm res}$ smear in $x$ and $y$)")

# measured segments
mx = [p[0] for p in meas_pts]
my = [p[1] for p in meas_pts]
ax.plot(mx, my, color="tab:red", lw=2.4, ls="-",
        label=r"measured segments $\to$ kink angle $\theta$ at plane $k$")

# kink arc at measured mid hit
ax.annotate(r"$\theta_{\rm meas}$", xy=(1.18, 0.50), fontsize=13,
            color="tab:red")
ax.annotate(r"scattering kick $\Delta t\sim\mathcal{N}(0,\sigma_{\rm scatt}^2)$"
            "\nper projection, applied at the plane",
            xy=(1.02, 0.42), xytext=(0.45, 0.85), fontsize=10,
            color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue"))
ax.annotate(r"displacement at next plane: $\Delta z\,\tan(\Delta t)$",
            xy=(1.97, 1.07), xytext=(0.35, 1.13), fontsize=10, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.25"))

ax.set_xlim(-0.5, 3.05)
ax.set_ylim(-0.35, 1.55)
ax.axis("off")
ax.legend(loc="lower right", fontsize=9, frameon=True)
ax.set_title("Where the measured kink comes from: one scattering kick at the "
             "shared plane + three independent hit smears", fontsize=11)
fig.tight_layout()
fig.savefig(FIGDIR / "kink_geometry.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 2 — resolution propagation: the (1, -2, 1)/dz weights
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2),
                         gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axes[0]
for z, lab in zip([0, 1, 2], ["$x_{k-1}$", "$x_k$", "$x_{k+1}$"]):
    ax.axvline(z, color="0.8", lw=5, zorder=0)
    ax.text(z, -0.62, lab, ha="center", fontsize=12)
ax.plot([0, 1, 2], [0, 0, 0], ls="--", color="0.5", lw=1.5)
ax.plot([0, 1, 2], [0.18, -0.30, 0.22], color="tab:red", lw=2.2, marker="s")
for z, d in zip([0, 1, 2], [0.18, -0.30, 0.22]):
    ax.annotate("", xy=(z, d), xytext=(z, 0),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3))
    ax.text(z + 0.06, d / 2, rf"$\delta_{{{z + 1}}}$", color="tab:red",
            fontsize=11)
ax.text(1.0, 0.42,
        r"$\theta_{\rm res} \approx \Delta t_x"
        r"= \dfrac{\delta_3 - 2\delta_2 + \delta_1}{\Delta z}$",
        ha="center", fontsize=13)
ax.set_xlim(-0.45, 2.45)
ax.set_ylim(-0.75, 0.62)
ax.axis("off")
ax.set_title("hit smears enter the kink with weights $(1,\,-2,\,1)/\Delta z$",
             fontsize=11)

ax = axes[1]
w = np.array([1, -2, 1])
ax.bar(["$\\delta_1$", "$\\delta_2$", "$\\delta_3$"], w,
       color=["tab:red"] * 3, alpha=0.75)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("weight in $\\Delta t_x$  (units of $1/\\Delta z$)")
ax.set_title(r"$\mathrm{Var} = (1^2{+}2^2{+}1^2)\,\sigma_{\rm res}^2/\Delta z^2"
             r" = 6\,\sigma_{\rm res}^2/\Delta z^2$ per projection",
             fontsize=10.5)
for i, v in enumerate(w):
    ax.text(i, v + (0.12 if v > 0 else -0.22), f"{v:+d}", ha="center",
            fontsize=12)
ax.set_ylim(-2.6, 1.7)
fig.tight_layout()
fig.savefig(FIGDIR / "resolution_weights.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 3 — the two projections: where the factor sqrt(2) lives
# ---------------------------------------------------------------------------

p = POINTS["grid_mid"]
sp = sigma_proj(p["sr"], p["ss"])
eps_mid = results["grid_mid"]["eps_formula"]
dtx, dty = proj_samples["grid_mid"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                         gridspec_kw=dict(width_ratios=[1, 1.3]))
ax = axes[0]
ax.scatter(dtx / sp, dty / sp, s=3, alpha=0.25, color="tab:blue",
           rasterized=True)
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(S * np.sqrt(2) * np.cos(th), S * np.sqrt(2) * np.sin(th),
        color="tab:green", lw=2,
        label=r"current: $\varepsilon = \sqrt{2}\,s\,\sigma_p$")
ax.plot(S * np.cos(th), S * np.sin(th), color="tab:orange", lw=2, ls="--",
        label=r"option: $\varepsilon' = s\,\sigma_p$  ($\sqrt{2}$ removed)")
ax.set_xlabel(r"$\Delta t_x / \sigma_p$")
ax.set_ylabel(r"$\Delta t_y / \sigma_p$")
ax.set_title("true kinks live in TWO projections\n"
             r"(MC at $\sigma_{\rm res}=0.02$, $\sigma_{\rm scatt}=3\times10^{-4}$)",
             fontsize=10.5)
ax.legend(fontsize=9, loc="upper right")
ax.set_aspect("equal")
lim = 5.4
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

ax = axes[1]
a = true_samples["grid_mid"]
x = np.linspace(0, 5.4 * sp, 400)
ax.hist(a, bins=80, range=(0, 5.4 * sp), density=True, alpha=0.45,
        color="tab:blue", label="MC true kink $\\theta$")
ax.plot(x, x / sp**2 * np.exp(-x**2 / (2 * sp**2)), "k-", lw=2,
        label=r"Rayleigh$(\sigma_p)$,  $\sigma_p^2=\sigma_{\rm scatt}^2"
              r"+6\sigma_{\rm res}^2/\Delta z^2$")
ax.axvline(eps_mid, color="tab:green", lw=2,
           label=(r"$\varepsilon$: misses $e^{-s^2}\approx0.012\%$"))
ax.axvline(eps_mid / np.sqrt(2), color="tab:orange", lw=2, ls="--",
           label=(r"$\varepsilon/\sqrt{2}$: misses $e^{-s^2/2}\approx1.1\%$"))
ax.set_xlabel(r"3D kink angle $\theta$ [rad]")
ax.set_ylabel("density")
ax.set_title("the 3D kink is Rayleigh, not Gaussian: "
             "its tail sets the coverage", fontsize=10.5)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGDIR / "rayleigh_two_projections.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 4 — MC validation of the variance bookkeeping (3 noise points)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
for ax, name, ttl in zip(
        axes,
        ["scatt_only", "res_only", "grid_worst"],
        [r"scattering only ($\sigma_s=5\times10^{-4}$): one kick, two proj.",
         r"resolution only ($\sigma_r=0.02$ mm): $6\sigma_r^2/\Delta z^2$ per proj.",
         r"worst grid cell ($\sigma_r=0.05$, $\sigma_s=5\times10^{-4}$)"]):
    a = true_samples[name]
    r = results[name]
    sp = r["sigma_proj_pred"]
    x = np.linspace(0, 5.4 * sp, 400)
    ax.hist(a, bins=80, range=(0, 5.4 * sp), density=True, alpha=0.45,
            color="tab:blue", label="MC (generator)")
    ax.plot(x, x / sp**2 * np.exp(-x**2 / (2 * sp**2)), "k-", lw=1.8,
            label="predicted Rayleigh")
    ax.axvline(r["eps_formula"], color="tab:green", lw=2,
               label=r"$\varepsilon$ (formula)")
    ax.axvline(r["eps_no_sqrt2"], color="tab:orange", ls="--", lw=2,
               label=r"$\varepsilon/\sqrt{2}$")
    ax.set_title(ttl, fontsize=9.5)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.text(0.97, 0.55,
            (f"$\\sigma_p$ pred {sp:.2e}\n"
             f"$\\sigma_p$ meas {r['sigma_proj_meas']:.2e}\n"
             f"cov($\\varepsilon$) = {r['true_cov_eps']*100:.3f}%\n"
             f"cov($\\varepsilon/\\sqrt{{2}}$) = {r['true_cov_no_sqrt2']*100:.2f}%"),
            transform=ax.transAxes, ha="right", va="center", fontsize=9,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
axes[0].set_ylabel("density")
axes[0].legend(fontsize=8.5)
fig.suptitle("The formula's variance bookkeeping vs the actual generator "
             "(true kinks, T=100, 60 events per point)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIGDIR / "mc_validation.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 5 — the trade-off: coverage vs false acceptance, both options marked
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, name in zip(axes, ["grid_mid", "grid_worst"]):
    r = results[name]
    a = true_samples[name]
    counts, ntot = false_hists[name]
    centers = 0.5 * (EDGES[:-1] + EDGES[1:])
    cum_false = np.cumsum(counts) / NREP_FALSE   # accepted false / event
    eps = r["eps_formula"]
    m = np.linspace(0.3, 1.5, 200)
    cov = np.array([np.mean(a < mm * eps) for mm in m])
    fl = np.interp(m * eps, centers, cum_false)

    ax2 = ax.twinx()
    ax.plot(m, 100 * cov, color="tab:blue", lw=2, label="true-kink coverage")
    ax2.plot(m, fl, color="tab:red", lw=2, label="accepted FALSE triplets / event")
    ax.axvline(1.0, color="tab:green", lw=2)
    ax.axvline(1 / np.sqrt(2), color="tab:orange", ls="--", lw=2)
    ax.text(1.0, 0.4, r" $\varepsilon$", color="tab:green",
            transform=ax.get_xaxis_transform(), fontsize=12)
    ax.text(1 / np.sqrt(2), 0.4, r" $\varepsilon/\sqrt{2}$", color="tab:orange",
            transform=ax.get_xaxis_transform(), fontsize=12)
    ax.set_xlabel(r"threshold / $\varepsilon_{\rm formula}$")
    ax.set_ylabel("true-kink coverage [%]", color="tab:blue")
    ax2.set_ylabel("accepted false triplets / event", color="tab:red")
    ax.set_ylim(95, 100.3)
    ax.set_title(f"$\\sigma_r={r['sigma_res']}$ mm, "
                 f"$\\sigma_s={r['sigma_scatt']:g}$ rad   (T={T_FALSE})",
                 fontsize=10.5)
fig.suptitle("Efficiency vs false inclusion: what removing the "
             r"$\sqrt{2}$ buys and costs", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIGDIR / "tradeoff_sqrt2.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig 6 — epsilon across the study grid, both variants
# ---------------------------------------------------------------------------

SRS = [0.0, 0.01, 0.02, 0.05]
SSS = [1e-4, 3e-4, 5e-4]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
srx = np.linspace(0, 0.055, 300)
for ss, col in zip(SSS, ["tab:blue", "tab:purple", "tab:brown"]):
    eps_c = [compute_epsilon(sr, ss) for sr in srx]
    ax.plot(srx, eps_c, color=col, lw=2,
            label=rf"$\varepsilon$, $\sigma_s={ss:g}$")
    ax.plot(srx, np.array(eps_c) / np.sqrt(2), color=col, lw=1.6, ls="--")
for ss in SSS:
    for sr in SRS:
        ax.plot(sr, compute_epsilon(sr, ss), "ko", ms=4)
ax.plot([], [], "k--", label=r"same, $\sqrt{2}$ removed")
ax.plot([], [], "ko", ms=4, label="study grid points")
ax.axhline(np.sqrt(2) * TH_MIN, color="0.5", lw=1,
           label=r"floor $\sqrt{2}\,\theta_{\min}$")
ax.set_yscale("log")
ax.set_xlabel(r"$\sigma_{\rm res}$ [mm]")
ax.set_ylabel(r"$\varepsilon$ [rad]")
ax.set_title("Calculated epsilon over the Epsilon_study_2 noise grid")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "epsilon_grid.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------------------
# Numbers dump
# ---------------------------------------------------------------------------

summary = dict(
    dz_mm=DZ, scale=S, theta_min=TH_MIN,
    geometry="5 planes, z = 33..165 mm, half-aperture 40 mm, slopes <= ~0.2",
    eps_validation_point=float(compute_epsilon(0.0, 1e-4)),
    eps_floor_formula=float(np.sqrt(2) * TH_MIN),
    rayleigh_miss_eps=float(np.exp(-S**2)),
    rayleigh_miss_no_sqrt2=float(np.exp(-S**2 / 2)),
    track_survival_3kinks_eps=float((1 - np.exp(-S**2)) ** 3),
    track_survival_3kinks_no_sqrt2=float((1 - np.exp(-S**2 / 2)) ** 3),
    points=results,
)
with (OUTDIR / "epsilon_derivation_numbers.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2)[:2000])
print("[done] figures ->", FIGDIR)
