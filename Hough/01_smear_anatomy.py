"""
Deep dive part 1 — anatomy of the vote smear, from first principles.

Questions answered here:
  (Q1) WHY do one track's 5 votes not coincide?  ->  the vertex-z term
       d_k = t (1 - z_pv/z_k): each plane sees a different chord slope from the
       ORIGIN to the hit when the track actually started at z_pv != 0.
       The smear is RADIAL (along t) and deterministic: delta_k = -(z_pv/z_k) t.
  (Q2) How big is each smear source?  ->  controlled experiments: switch
       vertex spread and scattering on/off independently and measure the
       per-track vote spread; overlay the closed-form predictions.
  (Q3) Does the product law spread ~ |t| * |z_pv| hold?  ->  scatter test.

Figures: fig01_vote_anatomy, fig02_smear_decomposition, fig03_smear_scaling.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hough_study_lib import (Z_PLANES, INVZ, INVZ_MEAN, INVZ_STD, INVZ_PP,
                             hit_directions, truth_tables, event_arrays)
import helpers  # Toy_Characterisation/_shared

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "deep_dive")
os.makedirs(OUT, exist_ok=True)
GEOM = helpers.make_geometry()


# ---------------------------------------------------------------------------
# fig01 — geometric anatomy of the vertex smear (one exaggerated track)
# ---------------------------------------------------------------------------
def fig01():
    z_pv, tx = 3.0, 0.2                       # 3 sigma vertex, max slope
    hits_x = tx * (Z_PLANES - z_pv)           # exact straight line, no noise
    votes = hits_x / Z_PLANES                 # what the point-vote computes

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # (a) position space: the track line vs the origin-chords
    ax[0].plot([z_pv, 180], [0, tx * (180 - z_pv)], "k-", lw=2,
               label=f"true track  (from $z_{{pv}}$={z_pv:.0f} mm)")
    ax[0].scatter(Z_PLANES, hits_x, c="crimson", zorder=5, label="hits")
    for zk, xk in zip(Z_PLANES, hits_x):
        ax[0].plot([0, zk], [0, xk], "--", color="steelblue", lw=1)
    ax[0].plot([], [], "--", color="steelblue", label="origin chords  $x_k/z_k$ = the vote")
    ax[0].scatter([z_pv], [0], marker="*", s=180, c="k", zorder=6, label="PV (displaced)")
    for zk in Z_PLANES:
        ax[0].axvline(zk, color="gray", lw=0.4, alpha=0.5)
    ax[0].set(xlabel="z [mm]", ylabel="x [mm]",
              title="(a) Why the votes smear: 5 chords from the ORIGIN,\n"
                    "but the track starts at $z_{pv}\\neq 0$ (here 3$\\sigma$, exaggerated)")
    ax[0].legend(loc="upper left", fontsize=9)

    # (b) vote space: radial structure
    ax[1].axhline(0, color="gray", lw=0.4); ax[1].axvline(0, color="gray", lw=0.4)
    ax[1].plot([0, 0.21], [0, 0], "k:", lw=1, label="radial direction (towards origin)")
    ax[1].scatter(votes, np.zeros_like(votes), c=np.arange(5), cmap="viridis",
                  s=90, zorder=5)
    for v, zk in zip(votes, Z_PLANES):
        ax[1].annotate(f"z={zk:.0f}", (v, 0.004), fontsize=8, rotation=60)
    ax[1].scatter([tx], [0], marker="x", s=120, c="crimson", zorder=6,
                  label="true $t_x$")
    ax[1].scatter([votes.mean()], [0], marker="+", s=160, c="k", zorder=6,
                  label="vote centroid")
    span = votes.max() - votes.min()
    ax[1].annotate("", xy=(votes.min(), -0.01), xytext=(votes.max(), -0.01),
                   arrowprops=dict(arrowstyle="<->", color="crimson"))
    ax[1].text(votes.mean(), -0.018,
               f"spread $=|t|\\,|z_{{pv}}|(1/z_1-1/z_5)$ = {span:.4f}",
               ha="center", color="crimson", fontsize=10)
    ax[1].set(xlim=(0.175, 0.205), ylim=(-0.025, 0.02),
              xlabel="$t_x$ vote", yticks=[],
              title="(b) The 5 votes in Hough space: smeared RADIALLY,\n"
                    "$\\delta_k = -(z_{pv}/z_k)\\,t$ — first plane worst (1/33 > 1/165)")
    ax[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig01_vote_anatomy.png"), dpi=130)
    plt.close(fig)
    print("fig01 done | exaggerated spread:", span,
          "| formula:", 0.2 * 3.0 * INVZ_PP)


# ---------------------------------------------------------------------------
# fig02/03 — controlled source decomposition + product law
# ---------------------------------------------------------------------------
CONFIGS = [
    ("binning only\n($z_{pv}=0,\\ \\sigma_s=0$)", dict(pvz=0.0, scat=0.0), "gray"),
    ("scattering only\n($z_{pv}=0,\\ \\sigma_s=10^{-4}$)", dict(pvz=0.0, scat=1e-4), "seagreen"),
    ("vertex only\n($\\sigma_{z}=1$mm$,\\ \\sigma_s=0$)", dict(pvz=1.0, scat=0.0), "steelblue"),
    ("both (store config)", dict(pvz=1.0, scat=1e-4), "crimson"),
]


def generate_controlled(pvz, scat, n_trk=50, reps=40):
    """Generate events with vertex spread / scattering independently switched."""
    helpers.PV_SIGMA = {"x": 0.0, "y": 0.0, "z": pvz}   # monkeypatch the module global
    tables = []
    for r in range(reps):
        ev = helpers.safe_generate(n_trk, 77000 + r, GEOM,
                                   measurement_error=0.0, collision_noise=scat)
        D, planes, truth = hit_directions(ev)
        tables.append(truth_tables(ev, D, planes, truth))
    import pandas as pd
    return pd.concat(tables, ignore_index=True)


def analytic_vertex_rms_sample(n=200000, sigz=1.0, rng=None):
    """Predicted distribution of spread_rms for the vertex term alone:
    rms = |t| |z_pv| std(1/z), t uniform on [-0.2,0.2]^2, z_pv ~ N(0, sigz)."""
    rng = rng or np.random.default_rng(1)
    t = rng.uniform(-0.2, 0.2, size=(n, 2))
    zpv = rng.normal(0, sigz, size=n)
    return np.hypot(t[:, 0], t[:, 1]) * np.abs(zpv) * INVZ_STD


def analytic_scatter_rms():
    """Closed-form per-plane scattering rms of the votes (sigma_s = 1e-4):
    rms_k = sigma_s * sqrt(sum_{j<k} (z_k - z_j)^2) / z_k ; rms over planes."""
    s = 1e-4
    per_plane = [s * np.sqrt(((Z_PLANES[k] - Z_PLANES[:k]) ** 2).sum()) / Z_PLANES[k]
                 for k in range(5)]
    return np.array(per_plane)


def fig02_03():
    dfs = {}
    for label, cfg, color in CONFIGS:
        dfs[label] = generate_controlled(**cfg)
        print(f"  config '{label.splitlines()[0]}': median spread_rms = "
              f"{dfs[label].spread_rms.median():.2e}")
    helpers.PV_SIGMA = {"x": 0.0, "y": 0.0, "z": 1.0}   # restore

    bins = np.logspace(-5.5, -1.6, 70)
    w256 = 0.5 / 256
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for label, cfg, color in CONFIGS:
        if cfg["pvz"] == 0.0 and cfg["scat"] == 0.0:
            continue   # exactly zero (votes coincide) — annotated below, not drawn
        v = dfs[label].spread_rms.to_numpy()
        ax.hist(v, bins=bins, histtype="step", lw=2, color=color, label=label)
    pred = analytic_vertex_rms_sample(n=2000)
    ax.hist(pred, bins=bins, histtype="step", lw=1.6, ls="--", color="navy",
            label="analytic vertex term\n$|t|\\,|z_{pv}|\\,\\mathrm{std}(1/z)$ (sampled)")
    sc = analytic_scatter_rms()
    ax.axvline(np.sqrt((sc ** 2).mean()), color="darkgreen", ls=":", lw=2,
               label=f"analytic scattering rms ({np.sqrt((sc**2).mean()):.1e})")
    ax.axvline(w256, color="k", lw=2, alpha=0.6,
               label=f"bin width (256 grid) = {w256:.1e}")
    ax.axvline(w256 / np.sqrt(12), color="k", lw=1, ls="--", alpha=0.6,
               label="bin quantisation rms $w/\\sqrt{12}$")
    ax.text(0.02, 0.55, "binning-only config ($z_{pv}=0,\\sigma_s=0$):\n"
                        "spread $\\equiv 0$ to machine precision\n"
                        "(all 5 votes coincide exactly) — not drawn",
            transform=ax.transAxes, fontsize=9, color="gray",
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    ax.set(xscale="log", xlabel="per-track vote spread  rms$_k\\,|d_k - c|$",
           ylabel="tracks / bin  (2000 tracks per configuration)",
           title="Decomposition of the vote smear — controlled experiments vs closed form\n"
                 "(vertex term dominates scattering by ~18×; 'both' is indistinguishable from 'vertex only')")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig02_smear_decomposition.png"), dpi=130)
    plt.close(fig)
    print("fig02 done")

    # product law: vertex-only config, spread vs |t||z_pv|
    dfv = dfs[CONFIGS[2][0]]
    x = dfv.t_mag * dfv.z_pv.abs()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax[0].scatter(x, dfv.spread_rms, s=4, alpha=0.35, c="steelblue", label="tracks")
    xx = np.linspace(0, x.max(), 50)
    ax[0].plot(xx, xx * INVZ_STD, "k-", lw=2,
               label="prediction  rms $= |t||z_{pv}|\\,\\mathrm{std}(1/z)$"
                     f"  (std$(1/z)$={INVZ_STD:.2e}/mm)")
    ax[0].set(xlabel="$|t|\\cdot|z_{pv}|$  [mm]", ylabel="measured vote spread rms",
              title="(a) The product law (vertex-only events)")
    ax[0].legend(fontsize=9)
    h = ax[1].scatter(dfv.t_mag, dfv.spread_rms, s=6, alpha=0.5,
                      c=dfv.z_pv.abs(), cmap="plasma")
    fig.colorbar(h, ax=ax[1], label="$|z_{pv}|$ [mm]")
    ax[1].axhline(0.5 / 256, color="k", lw=1.5, label="bin width (256)")
    ax[1].axhline(claimr := 3 * (0.5 / 256) * 2, color="crimson", lw=1.5, ls="--",
                  label=f"claim radius ({claimr:.1e})")
    ax[1].set(xlabel="$|t|$ (slope magnitude)", ylabel="vote spread rms",
              title="(b) Spread grows with slope AND vertex offset:\n"
                    "edge-of-acceptance tracks from 2–3$\\sigma$ vertices smear worst")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig03_smear_scaling.png"), dpi=130)
    plt.close(fig)
    print("fig03 done")


if __name__ == "__main__":
    print(f"constants: mean(1/z)={INVZ_MEAN:.6f}  std(1/z)={INVZ_STD:.6f}  "
          f"pp(1/z)={INVZ_PP:.6f}  [per mm]")
    fig01()
    fig02_03()
