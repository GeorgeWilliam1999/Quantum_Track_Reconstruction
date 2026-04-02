#!/usr/bin/env python3
"""
Segment-Grass Analysis
======================
Investigates the anomalous occupancy sub-peaks ("grass") that appear in the
hit-occupancy histograms at the ±0.2 rad angle setting but not at narrower
angles.

Root cause: geometric clipping of extreme-angle tracks on the outermost
detector modules when the primary vertex is generated far upstream.

Produces all figures for the LaTeX report.
"""

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.stats import norm

# ── package import ──────────────────────────────────────────────
from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast

# ── output dirs ────────────────────────────────────────────────
OUTDIR = Path(__file__).parent
FIG_DIR = OUTDIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── plot style ─────────────────────────────────────────────────
plt.rcParams["figure.dpi"] = 150
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 1.0, "lines.linewidth": 1.8, "lines.markersize": 5,
    "font.family": "serif",
})

# ── geometry & physics ─────────────────────────────────────────
N_MODULES   = 5
Z_FIRST     = 100.0   # mm
DZ          = 33.0     # mm
HALF_X      = 50.0     # mm
HALF_Y      = 50.0     # mm
PV_SIGMA_Z  = 50.0     # mm (Gaussian σ for PV generation)

SIGMA_RES   = 0.0
SIGMA_SCATT = 1e-4
SCALE       = 3.0
GAMMA       = 1.5
DELTA       = 1.0

Z_POSITIONS = [Z_FIRST + i * DZ for i in range(N_MODULES)]

def make_geometry():
    return PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF_X] * N_MODULES,
        ly=[HALF_Y] * N_MODULES,
        z=Z_POSITIONS,
    )

def compute_epsilon():
    theta_s = SCALE * SIGMA_SCATT
    theta_r = np.arctan((SCALE * SIGMA_RES) / DZ) if DZ != 0 else 0.0
    return float(np.sqrt(2 * theta_s**2 + 12 * theta_r**2 + 2 * (1.5e-5)**2))

EPSILON = compute_epsilon()


def safe_generate(geo, n_tracks, phi_max=0.2, theta_max=0.2):
    """Generate an event, retrying until all tracks have >= 3 hits."""
    for _ in range(50):
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[n_tracks],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=0.0, collision_noise=1e-8,
        )
        gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": PV_SIGMA_Z})
        particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
        gen.generate_particles(particles)
        event = gen.generate_complete_events()
        if min(len(t.hit_ids) for t in event.tracks) >= 3:
            return gen, event
    return gen, event


def build_hit_to_segments(ham):
    h2s = defaultdict(list)
    for idx, (fid, tid) in enumerate(ham._segment_to_hit_ids):
        h2s[fid].append(idx)
        h2s[tid].append(idx)
    return dict(h2s)


# =====================================================================
#  FIGURE 1 — Event display: clipped vs unclipped events
# =====================================================================
def fig1_event_displays():
    """Side-by-side event displays showing a clipped and unclipped event."""
    geo = make_geometry()
    np.random.seed(42)

    # Generate events until we find one with clipping and one without
    clipped_event = clipped_gen = None
    clean_event = clean_gen = None

    for _ in range(500):
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[30],
            phi_min=-0.2, phi_max=0.2,
            theta_min=-0.2, theta_max=0.2,
            measurement_error=0.0, collision_noise=1e-8,
        )
        gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": PV_SIGMA_Z})
        particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * 30]
        gen.generate_particles(particles)
        event = gen.generate_complete_events()

        n_full = sum(1 for t in event.tracks if len(t.hit_ids) == N_MODULES)
        n_clipped = 30 - n_full
        pv_z = gen.primary_vertices[0][2]

        if n_clipped >= 3 and clipped_event is None:
            clipped_event, clipped_gen = event, gen
        if n_clipped == 0 and clean_event is None and pv_z > -10:
            clean_event, clean_gen = event, gen

        if clipped_event is not None and clean_event is not None:
            break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, event, gen, title_extra in [
        (axes[0], clean_event, clean_gen, "No clipping"),
        (axes[1], clipped_event, clipped_gen, "Geometric clipping"),
    ]:
        pv = gen.primary_vertices[0]
        hits_by_track = defaultdict(list)
        for h in event.hits:
            hits_by_track[h.track_id].append(h)

        # Draw detector modules
        for z in Z_POSITIONS:
            ax.add_patch(Rectangle(
                (z - 1, -HALF_X), 2, 2 * HALF_X,
                facecolor="lightyellow", edgecolor="gray",
                linewidth=0.8, zorder=0))

        # Draw tracks
        n_clipped_trk = 0
        for trk in event.tracks:
            hits = sorted(hits_by_track[trk.track_id], key=lambda h: h.z)
            zz = [h.z for h in hits]
            xx = [h.x for h in hits]
            is_clipped = len(hits) < N_MODULES
            if is_clipped:
                n_clipped_trk += 1
                ax.plot(zz, xx, "o-", color="red", alpha=0.7, markersize=4,
                        linewidth=1.5, zorder=2)
            else:
                ax.plot(zz, xx, "o-", color="steelblue", alpha=0.4,
                        markersize=3, linewidth=0.8, zorder=1)

        # PV marker
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.plot(pv[2], pv[0], "k*", markersize=12, zorder=5, label="PV")

        n_full = sum(1 for t in event.tracks if len(t.hit_ids) == N_MODULES)
        ax.set_title(f"{title_extra}\n"
                     f"PV $z = {pv[2]:.1f}$ mm | "
                     f"{n_full} full + {n_clipped_trk} clipped tracks",
                     fontweight="bold")
        ax.set_xlabel("$z$ (mm)")
        ax.set_ylabel("$x$ (mm)")
        ax.set_ylim(-70, 70)
        ax.set_xlim(pv[2] - 20, Z_POSITIONS[-1] + 20)

        legend_elements = [
            Line2D([0], [0], color="steelblue", marker="o", lw=1, markersize=4,
                   label=f"Full tracks ({n_full})"),
            Line2D([0], [0], color="red", marker="o", lw=1.5, markersize=4,
                   label=f"Clipped tracks ({n_clipped_trk})"),
            Line2D([0], [0], color="k", marker="*", lw=0, markersize=10,
                   label="Primary vertex"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Figure 1: Event Displays — Track Clipping at $\\pm 0.2$ rad",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_event_displays.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_event_displays.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig1_event_displays")


# =====================================================================
#  FIGURE 2 — Clipping probability vs PV z for each module
# =====================================================================
def fig2_clipping_probability():
    """Analytical + simulated clipping probability vs PV z."""
    tx_max = np.tan(0.2)
    z_pv_range = np.linspace(-200, 100, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, N_MODULES))

    for m_idx in range(N_MODULES):
        z_mod = Z_POSITIONS[m_idx]
        # Clipping threshold: tx_max * (z_mod - z_pv) > HALF_X
        # => z_pv < z_mod - HALF_X / tx_max  (for positive tx)
        z_crit = z_mod - HALF_X / tx_max
        p_clip = norm.cdf(z_crit, loc=0, scale=PV_SIGMA_Z)
        ax.axvline(z_crit, color=colors[m_idx], ls="--", lw=0.8, alpha=0.6)
        ax.annotate(f"$z_{{\\rm crit}}^{{({m_idx})}} = {z_crit:.0f}$",
                    xy=(z_crit, 0.5 + 0.08 * m_idx), fontsize=8,
                    color=colors[m_idx], rotation=90, va="bottom",
                    ha="right")

    # PV z distribution
    pv_pdf = norm.pdf(z_pv_range, loc=0, scale=PV_SIGMA_Z)
    pv_pdf_norm = pv_pdf / pv_pdf.max()
    ax.fill_between(z_pv_range, 0, pv_pdf_norm, color="lightskyblue",
                    alpha=0.3, label="PV $z$ distribution (arb. units)")

    # Highlight the dangerous region
    z_crit_mod4 = Z_POSITIONS[4] - HALF_X / tx_max
    mask = z_pv_range < z_crit_mod4
    ax.fill_between(z_pv_range[mask], 0, pv_pdf_norm[mask],
                    color="red", alpha=0.25,
                    label=f"Clipping region (module 4)")

    # Table annotation
    table_text = "Module | $z_{\\rm crit}$ (mm) | P(clip)\n"
    table_text += "─" * 35 + "\n"
    for m_idx in range(N_MODULES):
        z_mod = Z_POSITIONS[m_idx]
        z_crit = z_mod - HALF_X / tx_max
        p_clip = norm.cdf(z_crit, loc=0, scale=PV_SIGMA_Z) * 100
        table_text += f"   {m_idx}     |   {z_crit:>7.1f}     | {p_clip:>5.1f}%\n"

    ax.text(0.98, 0.95, table_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Primary vertex $z$ (mm)")
    ax.set_ylabel("Normalised density / threshold")
    ax.set_title("Figure 2: PV $z$ Distribution vs Module Clipping Thresholds\n"
                 f"($\\phi_{{\\rm max}} = 0.2$ rad, detector half-width = {HALF_X} mm)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-200, 100)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_clipping_probability.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_clipping_probability.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig2_clipping_probability")


# =====================================================================
#  FIGURE 3 — Occupancy histograms: ±0.2 vs ±0.1 vs ±0.04
# =====================================================================
def fig3_occupancy_comparison():
    """Occupancy histograms for three angle settings, highlighting sub-peaks."""
    geo = make_geometry()
    angle_settings = [0.2, 0.1, 0.04]
    angle_colors = {0.2: "royalblue", 0.1: "darkorange", 0.04: "forestgreen"}
    n_tracks = 50
    n_repeats = 30

    occ_data = {}
    track_info = {}  # For ±0.2 only: which tracks caused anomalies

    for angle in angle_settings:
        all_occ = []
        all_occ_by_module = defaultdict(list)

        for rep in range(n_repeats):
            gen, event = safe_generate(geo, n_tracks, phi_max=angle,
                                       theta_max=angle)
            ham = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA,
                                       delta=DELTA)
            ham.construct_hamiltonian(event)
            h2s = build_hit_to_segments(ham)

            hit_mod = {h.hit_id: h.module_id for h in event.hits}
            for hid, segs in h2s.items():
                occ = len(segs)
                all_occ.append(occ)
                all_occ_by_module[hit_mod[hid]].append(occ)

        occ_data[angle] = {
            "all": np.array(all_occ),
            "by_module": {m: np.array(v)
                          for m, v in all_occ_by_module.items()},
        }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, angle in enumerate(angle_settings):
        ax = axes[idx]
        arr = occ_data[angle]["all"]
        hi = int(np.max(arr)) + 2
        counts, bins, patches = ax.hist(
            arr, bins=np.arange(0, hi + 1) - 0.5,
            alpha=0.6, color=angle_colors[angle],
            edgecolor="black", lw=0.3)

        # Highlight anomalous bins (not at expected 2n or n values)
        expected = {n_tracks, 2 * n_tracks}
        for i, (c, b) in enumerate(zip(counts, bins[:-1])):
            bval = int(b + 0.5)
            if c > 0 and bval not in expected:
                patches[i].set_facecolor("red")
                patches[i].set_alpha(0.7)

        ax.set_xlabel("Segments using this hit")
        ax.set_ylabel("Count")
        label = f"$\\pm{angle}$ rad"
        ax.set_title(f"{label}  ({n_tracks} tracks × {n_repeats} events)",
                     fontweight="bold")
        ax.grid(True, alpha=0.2)

        # Annotate anomalous bins
        vals, cnts = np.unique(arr, return_counts=True)
        anomalous = [(v, c) for v, c in zip(vals, cnts)
                     if v not in expected and c > 0]
        if anomalous:
            text = "Sub-peaks:\n" + "\n".join(
                f"  occ={int(v)}: {int(c)}" for v, c in anomalous[:6])
            ax.text(0.98, 0.95, text, transform=ax.transAxes, fontsize=7,
                    va="top", ha="right", fontfamily="monospace",
                    bbox=dict(facecolor="mistyrose", alpha=0.8,
                              boxstyle="round,pad=0.3"))
        else:
            ax.text(0.98, 0.95, "No anomalous bins",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="right",
                    bbox=dict(facecolor="honeydew", alpha=0.8,
                              boxstyle="round,pad=0.3"))

    plt.suptitle("Figure 3: Hit Occupancy Distributions — "
                 "Anomalous Sub-peaks at $\\pm 0.2$ rad Only",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_occupancy_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_occupancy_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig3_occupancy_comparison")

    return occ_data


# =====================================================================
#  FIGURE 4 — Direct track-level diagnosis: which tracks are clipped?
# =====================================================================
def fig4_track_diagnosis():
    """For a single event with clipping, show exactly which tracks are
    clipped and how that maps to the occupancy histogram."""
    geo = make_geometry()
    np.random.seed(99)

    # Find an event with moderate clipping
    best_event = best_gen = None
    for _ in range(500):
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[50],
            phi_min=-0.2, phi_max=0.2,
            theta_min=-0.2, theta_max=0.2,
            measurement_error=0.0, collision_noise=1e-8,
        )
        gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": PV_SIGMA_Z})
        particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * 50]
        gen.generate_particles(particles)
        event = gen.generate_complete_events()

        n_clipped = sum(1 for t in event.tracks if len(t.hit_ids) < N_MODULES)
        if 3 <= n_clipped <= 8:
            best_event, best_gen = event, gen
            break

    event, gen = best_event, best_gen
    pv = gen.primary_vertices[0]

    # Classify tracks
    hits_by_track = defaultdict(list)
    for h in event.hits:
        hits_by_track[h.track_id].append(h)

    clipped_tracks = []
    full_tracks = []
    for trk in event.tracks:
        hits = sorted(hits_by_track[trk.track_id], key=lambda h: h.z)
        mod_set = {h.module_id for h in hits}
        missing = [m for m in range(N_MODULES) if m not in mod_set]
        tx = gen.particles[0][trk.track_id]["tx"]
        ty = gen.particles[0][trk.track_id]["ty"]
        phi = np.arctan(tx)
        theta = np.arctan(ty)

        info = {"track_id": trk.track_id, "hits": hits,
                "missing_modules": missing,
                "n_hits": len(hits), "phi": phi, "theta": theta,
                "tx": tx, "ty": ty}
        if missing:
            clipped_tracks.append(info)
        else:
            full_tracks.append(info)

    # Build occupancy histogram for this single event
    ham = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA, delta=DELTA)
    ham.construct_hamiltonian(event)
    h2s = build_hit_to_segments(ham)
    hit_mod = {h.hit_id: h.module_id for h in event.hits}

    occ_all = np.array([len(segs) for segs in h2s.values()])
    occ_by_mod = defaultdict(list)
    for hid, segs in h2s.items():
        occ_by_mod[hit_mod[hid]].append(len(segs))

    n_hits_per_mod = defaultdict(int)
    for h in event.hits:
        n_hits_per_mod[h.module_id] += 1

    # ── Create 2×2 figure ──
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # (a) Event display with clipped tracks highlighted
    ax_disp = fig.add_subplot(gs[0, 0])
    for z in Z_POSITIONS:
        ax_disp.add_patch(Rectangle(
            (z - 1.0, -HALF_X), 2, 2 * HALF_X,
            facecolor="lightyellow", edgecolor="gray", lw=0.6, zorder=0))

    for info in full_tracks:
        hits = info["hits"]
        ax_disp.plot([h.z for h in hits], [h.x for h in hits],
                     "o-", color="steelblue", alpha=0.3, markersize=2.5,
                     lw=0.6, zorder=1)
    for info in clipped_tracks:
        hits = info["hits"]
        ax_disp.plot([h.z for h in hits], [h.x for h in hits],
                     "o-", color="red", alpha=0.8, markersize=4,
                     lw=1.5, zorder=2)
        # Show missing modules
        for m in info["missing_modules"]:
            z_miss = Z_POSITIONS[m]
            x_extrap = pv[0] + info["tx"] * (z_miss - pv[2])
            ax_disp.plot(z_miss, x_extrap, "x", color="darkred",
                         markersize=8, mew=2, zorder=3)

    ax_disp.plot(pv[2], pv[0], "k*", markersize=12, zorder=5)
    ax_disp.axhline(HALF_X, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_disp.axhline(-HALF_X, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_disp.set_xlabel("$z$ (mm)")
    ax_disp.set_ylabel("$x$ (mm)")
    ax_disp.set_title(f"(a) Event display — PV $z = {pv[2]:.1f}$ mm\n"
                      f"{len(full_tracks)} full + "
                      f"{len(clipped_tracks)} clipped tracks",
                      fontweight="bold")
    ax_disp.set_ylim(-75, 75)
    ax_disp.set_xlim(pv[2] - 15, Z_POSITIONS[-1] + 15)
    ax_disp.grid(True, alpha=0.15)

    legend_el = [
        Line2D([0], [0], color="steelblue", marker="o", lw=0.8, markersize=3,
               label="Full tracks"),
        Line2D([0], [0], color="red", marker="o", lw=1.5, markersize=4,
               label="Clipped tracks"),
        Line2D([0], [0], color="darkred", marker="x", lw=0, markersize=8,
               mew=2, label="Extrapolated miss"),
        Line2D([0], [0], color="gray", ls="--", lw=0.8,
               label="Detector edge"),
    ]
    ax_disp.legend(handles=legend_el, fontsize=7, loc="upper left")

    # (b) Angle distribution of all tracks, highlighting clipped
    ax_ang = fig.add_subplot(gs[0, 1])
    full_phi = [info["phi"] for info in full_tracks]
    full_theta = [info["theta"] for info in full_tracks]
    clip_phi = [info["phi"] for info in clipped_tracks]
    clip_theta = [info["theta"] for info in clipped_tracks]

    ax_ang.scatter(full_phi, full_theta, c="steelblue", s=20, alpha=0.5,
                   label="Full tracks", zorder=1)
    ax_ang.scatter(clip_phi, clip_theta, c="red", s=60, marker="^",
                   edgecolors="darkred", lw=0.8, alpha=0.9,
                   label="Clipped tracks", zorder=2)

    # Angular boundary
    rect = plt.Rectangle((-0.2, -0.2), 0.4, 0.4, fill=False,
                          edgecolor="gray", ls="--", lw=1)
    ax_ang.add_patch(rect)
    ax_ang.set_xlabel("$\\phi$ (rad)")
    ax_ang.set_ylabel("$\\theta$ (rad)")
    ax_ang.set_title("(b) Track angles — clipped tracks\n"
                     "cluster at extreme angles", fontweight="bold")
    ax_ang.legend(fontsize=8)
    ax_ang.grid(True, alpha=0.2)
    ax_ang.set_xlim(-0.25, 0.25)
    ax_ang.set_ylim(-0.25, 0.25)

    # (c) Per-module occupancy histogram
    ax_occ_mod = fig.add_subplot(gs[1, 0])
    mod_labels = []
    mod_data = []
    expected_occ = {}
    for m in range(N_MODULES):
        n_h = n_hits_per_mod[m]
        if m == 0:
            exp = n_hits_per_mod[1]  # only connects to module 1
        elif m == N_MODULES - 1:
            exp = n_hits_per_mod[m - 1]  # only connects to module m-1
        else:
            exp = n_hits_per_mod[m - 1] + n_hits_per_mod[m + 1]
        expected_occ[m] = exp
        mod_labels.append(f"Mod {m}\n(z={Z_POSITIONS[m]:.0f})")
        mod_data.append(np.array(occ_by_mod[m]))

    bp = ax_occ_mod.boxplot(mod_data, labels=mod_labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        c = "salmon" if any(v != expected_occ[i] for v in mod_data[i]) \
            else "lightsteelblue"
        patch.set_facecolor(c)

    # Overlay expected values
    for i in range(N_MODULES):
        ax_occ_mod.axhline(expected_occ[i], color="gray", ls=":", lw=0.5)
        ax_occ_mod.text(i + 1.3, expected_occ[i], f"exp={expected_occ[i]}",
                        fontsize=7, color="gray", va="bottom")

    ax_occ_mod.set_ylabel("Segments per hit (occupancy)")
    ax_occ_mod.set_title(f"(c) Hit occupancy per module\n"
                         f"(hits/mod: {[n_hits_per_mod[m] for m in range(N_MODULES)]})",
                         fontweight="bold")
    ax_occ_mod.grid(True, alpha=0.2, axis="y")

    # (d) Overall occupancy histogram with expected vs anomalous
    ax_occ = fig.add_subplot(gs[1, 1])
    n_total = len(event.hits)
    hi = int(np.max(occ_all)) + 2
    counts, bins, patches = ax_occ.hist(
        occ_all, bins=np.arange(0, hi + 1) - 0.5,
        alpha=0.6, color="cornflowerblue", edgecolor="black", lw=0.4)

    expected_vals = set()
    for m in range(N_MODULES):
        expected_vals.add(expected_occ[m])

    for i, (c, b) in enumerate(zip(counts, bins[:-1])):
        bval = int(b + 0.5)
        if c > 0 and bval not in expected_vals:
            patches[i].set_facecolor("red")
            patches[i].set_alpha(0.8)

    ax_occ.set_xlabel("Segments using this hit")
    ax_occ.set_ylabel("Count")
    ax_occ.set_title("(d) Overall occupancy histogram\n"
                     "(red = anomalous sub-peaks from clipping)",
                     fontweight="bold")
    ax_occ.grid(True, alpha=0.2)

    # Text summary
    vals, cnts = np.unique(occ_all, return_counts=True)
    table = "Occupancy | Count | Source\n" + "─" * 32 + "\n"
    for v, c in zip(vals, cnts):
        src = "expected" if v in expected_vals else "CLIPPING"
        table += f"   {int(v):>4d}   | {int(c):>4d}  | {src}\n"
    ax_occ.text(0.98, 0.95, table, transform=ax_occ.transAxes,
                fontsize=7, va="top", ha="right", fontfamily="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9,
                          boxstyle="round,pad=0.3"))

    plt.suptitle("Figure 4: Track-Level Diagnosis — Connecting Clipped Tracks "
                 "to Occupancy Anomalies",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(FIG_DIR / "fig4_track_diagnosis.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_track_diagnosis.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig4_track_diagnosis")

    return clipped_tracks, full_tracks


# =====================================================================
#  FIGURE 5 — Occupancy vs PV z scatter (many events)
# =====================================================================
def fig5_occupancy_vs_pvz():
    """Show how mean occupancy on module 3 varies with PV z position."""
    geo = make_geometry()
    np.random.seed(7)

    pv_zs = []
    mean_occ_mod3 = []
    n_clipped_list = []
    n_tracks = 50

    for trial in range(200):
        gen = StateEventGenerator(
            detector_geometry=geo, events=1, n_particles=[n_tracks],
            phi_min=-0.2, phi_max=0.2,
            theta_min=-0.2, theta_max=0.2,
            measurement_error=0.0, collision_noise=1e-8,
        )
        gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": PV_SIGMA_Z})
        particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
        gen.generate_particles(particles)
        event = gen.generate_complete_events()

        pv_z = gen.primary_vertices[0][2]
        n_clipped = sum(1 for t in event.tracks if len(t.hit_ids) < N_MODULES)

        ham = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA, delta=DELTA)
        ham.construct_hamiltonian(event)
        h2s = build_hit_to_segments(ham)
        hit_mod = {h.hit_id: h.module_id for h in event.hits}

        occ_m3 = [len(segs) for hid, segs in h2s.items()
                  if hit_mod[hid] == 3]
        if occ_m3:
            mean_occ_mod3.append(np.mean(occ_m3))
        else:
            mean_occ_mod3.append(0)
        pv_zs.append(pv_z)
        n_clipped_list.append(n_clipped)

    pv_zs = np.array(pv_zs)
    mean_occ_mod3 = np.array(mean_occ_mod3)
    n_clipped_arr = np.array(n_clipped_list)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: mean occupancy on module 3 vs PV z
    sc = ax1.scatter(pv_zs, mean_occ_mod3, c=n_clipped_arr, cmap="RdYlBu_r",
                     s=25, edgecolors="gray", lw=0.3, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax1, label="Clipped tracks in event")
    ax1.axhline(2 * n_tracks, color="gray", ls="--", lw=1,
                label=f"Expected = {2*n_tracks}")
    z_crit = Z_POSITIONS[4] - HALF_X / np.tan(0.2)
    ax1.axvline(z_crit, color="red", ls=":", lw=1.2,
                label=f"Module-4 clip threshold ({z_crit:.0f} mm)")
    ax1.set_ylabel("Mean occupancy (module 3)")
    ax1.set_title("Figure 5: Module-3 Occupancy vs Primary Vertex $z$",
                  fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Bottom: number of clipped tracks vs PV z
    ax2.bar(pv_zs, n_clipped_arr, width=2.5, color="salmon",
            edgecolor="darkred", lw=0.3, alpha=0.7)
    ax2.axvline(z_crit, color="red", ls=":", lw=1.2)
    ax2.set_xlabel("Primary vertex $z$ (mm)")
    ax2.set_ylabel("# clipped tracks")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_occupancy_vs_pvz.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_occupancy_vs_pvz.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig5_occupancy_vs_pvz")


# =====================================================================
#  FIGURE 6 — Module hit-count distributions for three angle settings
# =====================================================================
def fig6_hits_per_module():
    """Show that module hit counts are constant for ±0.1, ±0.04
    but fluctuate for ±0.2."""
    geo = make_geometry()
    angle_settings = [0.2, 0.1, 0.04]
    colors = {0.2: "royalblue", 0.1: "darkorange", 0.04: "forestgreen"}
    n_tracks = 50
    n_repeats = 100

    fig, axes = plt.subplots(1, N_MODULES, figsize=(16, 3.5), sharey=True)

    for angle in angle_settings:
        mod_counts = defaultdict(list)
        for _ in range(n_repeats):
            gen, event = safe_generate(geo, n_tracks, phi_max=angle,
                                       theta_max=angle)
            per_mod = defaultdict(int)
            for h in event.hits:
                per_mod[h.module_id] += 1
            for m in range(N_MODULES):
                mod_counts[m].append(per_mod[m])

        for m in range(N_MODULES):
            arr = np.array(mod_counts[m])
            vals, cnts = np.unique(arr, return_counts=True)
            axes[m].bar(vals + (angle - 0.12) * 2, cnts, width=0.5,
                        color=colors[angle], alpha=0.6,
                        label=f"$\\pm{angle}$" if m == 0 else None,
                        edgecolor="black", lw=0.3)

    for m in range(N_MODULES):
        axes[m].set_xlabel("Hits on module")
        axes[m].set_title(f"Module {m}\n($z={Z_POSITIONS[m]:.0f}$ mm)",
                          fontweight="bold", fontsize=10)
        axes[m].grid(True, alpha=0.2)
    axes[0].set_ylabel("Count (events)")
    axes[0].legend(fontsize=8, title="Angle range")

    plt.suptitle("Figure 6: Hits per Module — Variable at $\\pm 0.2$ rad "
                 "(modules 3–4), Constant at Narrower Angles",
                 fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_hits_per_module.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_hits_per_module.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig6_hits_per_module")


# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    print("Running Segment-Grass analysis ...\n")
    fig1_event_displays()
    fig2_clipping_probability()
    fig3_occupancy_comparison()
    fig4_track_diagnosis()
    fig5_occupancy_vs_pvz()
    fig6_hits_per_module()
    print(f"\nAll figures saved to: {FIG_DIR.resolve()}")
