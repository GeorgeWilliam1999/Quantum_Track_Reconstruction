"""
Generate figures for baseline_report.tex.
Run with: /data/bfys/gscriven/conda/envs/Q_env/bin/python generate_baseline_figs.py
Output PNGs are written to the Toy_Characterisation directory.
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast, get_tracks
from lhcb_velo_toy.analysis import EventValidator
from lhcb_velo_toy.analysis.plotting.event_display import (
    plot_event_3d,
    plot_reco_vs_truth,
)

plt.rcParams["figure.dpi"] = 120

# ── helpers ─────────────────────────────────────────────────────
def make_geometry(n_modules=5, z_first=100.0, z_spacing=33.0,
                  half_x=50.0, half_y=50.0):
    z_positions = [z_first + i * z_spacing for i in range(n_modules)]
    return PlaneGeometry(
        module_id=list(range(n_modules)),
        lx=[half_x] * n_modules,
        ly=[half_y] * n_modules,
        z=z_positions,
    )

def generate_event(geo, n_tracks, phi_max=0.2, theta_max=0.2):
    gen = StateEventGenerator(
        detector_geometry=geo, events=1, n_particles=[n_tracks],
        phi_min=-phi_max, phi_max=phi_max,
        theta_min=-theta_max, theta_max=theta_max,
        measurement_error=0.0, collision_noise=1e-8,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    gen.generate_particles([[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks])
    return gen, gen.generate_complete_events()

def compute_epsilon(sigma_res, sigma_scatt, dz, scale=1.0, theta_min=1.5e-5):
    theta_s = scale * sigma_scatt
    theta_r = np.arctan((scale * sigma_res) / dz) if dz != 0 else 0.0
    return float(np.sqrt(2*theta_s**2 + 12*theta_r**2 + 2*theta_min**2))

# ── parameters ──────────────────────────────────────────────────
SIGMA_SCATT = 1e-3
DZ_MM       = 33.0
SCALE       = 3.0
EPSILON     = compute_epsilon(0.0, SIGMA_SCATT, DZ_MM, scale=SCALE)
GAMMA, DELTA = 1.5, 1.0
BASELINE  = DELTA / (DELTA + GAMMA)
THRESHOLD = (1 + BASELINE) / 2

print(f"ε = {EPSILON*1e3:.3f} mrad  threshold = {THRESHOLD:.3f}")

geo_clean = make_geometry()

# ── clean event ──────────────────────────────────────────────────
np.random.seed(42)
for _ in range(20):
    gen_clean, event_clean = generate_event(geo_clean, 3)
    if min(len(t.hit_ids) for t in event_clean.tracks) >= 3:
        break

print(f"Clean event: {len(event_clean.tracks)} tracks, {len(event_clean.hits)} hits")

# ── Fig 1: 3D truth display ──────────────────────────────────────
fig = plot_event_3d(event_clean, title="Clean Event — Truth Tracks")
plt.savefig("baseline_fig1_truth_3d.png", bbox_inches="tight", dpi=120)
plt.close()
print("Saved baseline_fig1_truth_3d.png")

# ── Hamiltonian solve ────────────────────────────────────────────
ham_clean = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA, delta=DELTA)
ham_clean.construct_hamiltonian(event_clean)
x_clean = ham_clean.solve_classicaly()
reco_clean = get_tracks(ham_clean, x_clean, event_clean, threshold=THRESHOLD)
print(f"Reco: {len(reco_clean)} tracks")

# ── Fig 2: reco vs truth (clean event) ──────────────────────────
fig = plot_reco_vs_truth(event_clean, reco_clean)
plt.savefig("baseline_fig2_reco_truth_clean.png", bbox_inches="tight", dpi=120)
plt.close()
print("Saved baseline_fig2_reco_truth_clean.png")

# ── clean-event segment angles ───────────────────────────────────
def pairwise_angles_from_tracks(tracks, hits_dict):
    angles = []
    for trk in tracks:
        hids = trk.hit_ids
        for k in range(len(hids) - 2):
            ha = hits_dict[hids[k]]; hb = hits_dict[hids[k+1]]; hc = hits_dict[hids[k+2]]
            v1 = np.array([hb.x-ha.x, hb.y-ha.y, hb.z-ha.z])
            v2 = np.array([hc.x-hb.x, hc.y-hb.y, hc.z-hb.z])
            cos_a = np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)
            angles.append(np.arccos(cos_a))
    return np.array(angles) if angles else np.array([])

hits_dict_clean = {h.hit_id: h for h in event_clean.hits}
true_seg_angles = pairwise_angles_from_tracks(event_clean.tracks, hits_dict_clean)
reco_seg_angles = pairwise_angles_from_tracks(reco_clean, hits_dict_clean)

hi = max(0.01, max(true_seg_angles.max() if len(true_seg_angles) else 0,
                   reco_seg_angles.max() if len(reco_seg_angles) else 0) * 1.5)
bins = np.linspace(0, hi, 60)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(true_seg_angles, bins=bins, color="forestgreen", alpha=0.8, edgecolor="black", lw=0.5,
             label=f"True ({len(true_seg_angles)})")
axes[0].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε = {EPSILON:.4f}")
axes[0].set_title("True Segment-Pair Angles — Clean Event", fontweight="bold")
axes[0].set_xlabel("Angle (rad)"); axes[0].set_ylabel("Count"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(reco_seg_angles, bins=bins, color="royalblue", alpha=0.8, edgecolor="black", lw=0.5,
             label=f"Reco ({len(reco_seg_angles)})")
axes[1].axvline(EPSILON, color="red", ls="--", lw=2, label=f"ε = {EPSILON:.4f}")
axes[1].set_title("Reco Segment-Pair Angles — Clean Event", fontweight="bold")
axes[1].set_xlabel("Angle (rad)"); axes[1].set_ylabel("Count"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("Pairwise Segment Angles — Clean Event", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("baseline_fig3_seg_angles_clean.png", bbox_inches="tight", dpi=120)
plt.close()
print("Saved baseline_fig3_seg_angles_clean.png")

# ── 10-event mixed study ─────────────────────────────────────────
track_counts = [2, 3, 4, 5, 6, 3, 4, 5, 2, 4]
events_data = []

for i, n_tracks in enumerate(track_counts):
    np.random.seed(i * 17 + 3)
    for _ in range(20):
        _, event = generate_event(geo_clean, n_tracks)
        if min(len(t.hit_ids) for t in event.tracks) >= 3:
            break
    ham = SimpleHamiltonianFast(epsilon=EPSILON, gamma=GAMMA, delta=DELTA)
    ham.construct_hamiltonian(event)
    x = ham.solve_classicaly()
    reco_tracks = get_tracks(ham, x, event, threshold=THRESHOLD)
    val = EventValidator(event, reco_tracks)
    _, metrics = val.match_tracks(purity_min=0.7)
    true_segs = [(t.hit_ids[j], t.hit_ids[j+1]) for t in event.tracks for j in range(len(t.hit_ids)-1)]
    reco_segs = [(t.hit_ids[j], t.hit_ids[j+1]) for t in reco_tracks for j in range(len(t.hit_ids)-1)]
    events_data.append(dict(event_id=i, n_tracks_true=len(event.tracks), n_tracks_reco=len(reco_tracks),
                            event=event, reco_tracks=reco_tracks,
                            true_segments=true_segs, reco_segments=reco_segs, metrics=metrics))

print("10 events generated")

# ── Fig 4: aggregate segment angles across 10 events ────────────
combined_true, combined_false = [], []
for ed in events_data:
    ev = ed["event"]
    true_seg_set = {(t.hit_ids[j], t.hit_ids[j+1]) for t in ev.tracks for j in range(len(t.hit_ids)-1)}
    hits_by_mod = {}
    for h in ev.hits:
        hits_by_mod.setdefault(h.module_id, []).append(h)
    mod_ids = sorted(hits_by_mod.keys())
    for mi in range(1, len(mod_ids)-1):
        for h_mid in hits_by_mod[mod_ids[mi]]:
            for h_prev in hits_by_mod[mod_ids[mi-1]]:
                v1 = np.array([h_mid.x-h_prev.x, h_mid.y-h_prev.y, h_mid.z-h_prev.z])
                n1 = np.linalg.norm(v1)
                if n1 == 0: continue
                v1 /= n1
                for h_next in hits_by_mod[mod_ids[mi+1]]:
                    v2 = np.array([h_next.x-h_mid.x, h_next.y-h_mid.y, h_next.z-h_mid.z])
                    n2 = np.linalg.norm(v2)
                    if n2 == 0: continue
                    v2 /= n2
                    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                    if ((h_prev.hit_id, h_mid.hit_id) in true_seg_set and
                            (h_mid.hit_id, h_next.hit_id) in true_seg_set and
                            h_prev.track_id is not None and
                            h_prev.track_id == h_mid.track_id == h_next.track_id):
                        combined_true.append(angle)
                    else:
                        combined_false.append(angle)

combined_true = np.array(combined_true)
combined_false = np.array(combined_false)
print(f"10-event pairs: {len(combined_true):,} true, {len(combined_false):,} false")

bins = np.linspace(0, np.pi/2, 80)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].hist(combined_true, bins=bins, color="forestgreen", alpha=0.8, edgecolor="black", lw=0.3,
             label=f"True ({len(combined_true):,})"); axes[0].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε={EPSILON:.4f}")
axes[0].set_title("True Pairs (same track)", fontweight="bold"); axes[0].set_xlabel("Angle (rad)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(combined_false, bins=bins, color="indianred", alpha=0.8, edgecolor="black", lw=0.3,
             label=f"False ({len(combined_false):,})"); axes[1].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε={EPSILON:.4f}")
axes[1].set_title("False Pairs", fontweight="bold"); axes[1].set_xlabel("Angle (rad)"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].hist(combined_true, bins=bins, color="forestgreen", alpha=0.6, edgecolor="black", lw=0.3, density=True, label=f"True ({len(combined_true):,})")
axes[2].hist(combined_false, bins=bins, color="indianred", alpha=0.5, edgecolor="black", lw=0.3, density=True, label=f"False ({len(combined_false):,})")
axes[2].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε={EPSILON:.4f}")
axes[2].set_title("True vs False (normalised)", fontweight="bold"); axes[2].set_xlabel("Angle (rad)"); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.suptitle("Pairwise Segment Angles — 10-event mixed-occupancy sample", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("baseline_fig4_seg_angles_10events.png", bbox_inches="tight", dpi=120)
plt.close()
print("Saved baseline_fig4_seg_angles_10events.png")

# ── large sample 20 × 50 ────────────────────────────────────────
N_EVENTS_LARGE, N_TRACKS_LARGE = 20, 50
all_true, all_false = [], []
for ev_i in range(N_EVENTS_LARGE):
    np.random.seed(ev_i * 31 + 7)
    for _ in range(20):
        _, evt = generate_event(geo_clean, N_TRACKS_LARGE)
        if min(len(t.hit_ids) for t in evt.tracks) >= 3:
            break
    true_seg_set = {(t.hit_ids[j], t.hit_ids[j+1]) for t in evt.tracks for j in range(len(t.hit_ids)-1)}
    hits_by_mod = {}
    for h in evt.hits:
        hits_by_mod.setdefault(h.module_id, []).append(h)
    mod_ids = sorted(hits_by_mod.keys())
    for mi in range(1, len(mod_ids)-1):
        for h_mid in hits_by_mod[mod_ids[mi]]:
            for h_prev in hits_by_mod[mod_ids[mi-1]]:
                v1 = np.array([h_mid.x-h_prev.x, h_mid.y-h_prev.y, h_mid.z-h_prev.z])
                n1 = np.linalg.norm(v1)
                if n1 == 0: continue
                v1 /= n1
                for h_next in hits_by_mod[mod_ids[mi+1]]:
                    v2 = np.array([h_next.x-h_mid.x, h_next.y-h_mid.y, h_next.z-h_mid.z])
                    n2 = np.linalg.norm(v2)
                    if n2 == 0: continue
                    v2 /= n2
                    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                    if ((h_prev.hit_id, h_mid.hit_id) in true_seg_set and
                            (h_mid.hit_id, h_next.hit_id) in true_seg_set and
                            h_prev.track_id is not None and
                            h_prev.track_id == h_mid.track_id == h_next.track_id):
                        all_true.append(angle)
                    else:
                        all_false.append(angle)
    if (ev_i + 1) % 10 == 0:
        print(f"  large-sample: {ev_i+1}/{N_EVENTS_LARGE} done")

all_true = np.array(all_true); all_false = np.array(all_false)
print(f"Large sample: {len(all_true):,} true, {len(all_false):,} false")

bins = np.linspace(0, 0.5, 120)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].hist(all_true, bins=bins, color="forestgreen", alpha=0.8, edgecolor="black", lw=0.3,
             label=f"True ({len(all_true):,})"); axes[0].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε={EPSILON:.4f}")
axes[0].set_title("True Segment Pairs (same track)", fontweight="bold"); axes[0].set_xlabel("Angle (rad)"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(all_false, bins=bins, color="indianred", alpha=0.8, edgecolor="black", lw=0.3,
             label=f"False ({len(all_false):,})"); axes[1].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε={EPSILON:.4f}")
axes[1].set_title("False Segment Pairs", fontweight="bold"); axes[1].set_xlabel("Angle (rad)"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].hist(all_true, bins=bins, color="forestgreen", alpha=0.6, edgecolor="black", lw=0.3, density=True, label=f"True ({len(all_true):,})")
axes[2].hist(all_false, bins=bins, color="indianred", alpha=0.5, edgecolor="black", lw=0.3, density=True, label=f"False ({len(all_false):,})")
axes[2].axvline(EPSILON, color="blue", ls="--", lw=2, label=f"ε={EPSILON:.4f}")
axes[2].set_title("True vs False (normalised)", fontweight="bold"); axes[2].set_xlabel("Angle (rad)"); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.suptitle(f"Pairwise Segment Angles — {N_EVENTS_LARGE} events × {N_TRACKS_LARGE} tracks",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("baseline_fig5_seg_angles_large.png", bbox_inches="tight", dpi=120)
plt.close()
print("Saved baseline_fig5_seg_angles_large.png")

print("\nAll figures generated successfully.")
