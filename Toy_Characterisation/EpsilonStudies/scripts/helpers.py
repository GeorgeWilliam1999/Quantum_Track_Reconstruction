"""
Shared helper functions for the Condor pipeline.

Extracted from characterisation.ipynb and hit_competition_study.ipynb
so that the worker script and aggregation notebook use identical logic.
"""

import numpy as np
from collections import defaultdict

from lhcb_velo_toy.generation import PlaneGeometry, StateEventGenerator
from lhcb_velo_toy.solvers import SimpleHamiltonianFast, get_tracks
from lhcb_velo_toy.analysis import EventValidator

# ═══════════════════════════════════════════════════════════════════
#  Physics constants (shared across both notebooks)
# ═══════════════════════════════════════════════════════════════════
SIGMA_RES        = 0.0       # measurement resolution (mm)
SIGMA_SCATT      = 1e-4      # baseline scattering angle (rad)
DZ_MM            = 33.0      # module spacing (mm)
SCALE            = 3.0       # number of sigma to accept
GAMMA            = 1.5       # self-interaction penalty
DELTA            = 1.0       # bias weight
N_MODULES        = 5         # number of detector modules
Z_FIRST          = 100.0     # z position of first module (mm)

BASELINE  = DELTA / (DELTA + GAMMA)      # = 0.4
THRESHOLD = (1 + BASELINE) / 2           # = 0.7


# ═══════════════════════════════════════════════════════════════════
#  Geometry
# ═══════════════════════════════════════════════════════════════════
def make_geometry(n_modules=N_MODULES, z_first=Z_FIRST, z_spacing=DZ_MM,
                  half_x=50.0, half_y=50.0):
    """Create a PlaneGeometry with evenly spaced modules."""
    z_positions = [z_first + i * z_spacing for i in range(n_modules)]
    return PlaneGeometry(
        module_id=list(range(n_modules)),
        lx=[half_x] * n_modules,
        ly=[half_y] * n_modules,
        z=z_positions,
    )


# ═══════════════════════════════════════════════════════════════════
#  Event generation
# ═══════════════════════════════════════════════════════════════════
def generate_event(geo, n_tracks, measurement_error=0.0, collision_noise=1e-8,
                   phi_max=0.2, theta_max=0.2):
    """Generate a full event end-to-end."""
    gen = StateEventGenerator(
        detector_geometry=geo,
        events=1,
        n_particles=[n_tracks],
        phi_min=-phi_max, phi_max=phi_max,
        theta_min=-theta_max, theta_max=theta_max,
        measurement_error=measurement_error,
        collision_noise=collision_noise,
    )
    gen.generate_random_primary_vertices({"x": 0.1, "y": 0.1, "z": 50.0})
    particles = [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
    gen.generate_particles(particles)
    event = gen.generate_complete_events()
    return gen, event


def safe_generate(geo, n_tracks, collision_noise=1e-8,
                  phi_max=0.2, theta_max=0.2):
    """Generate event, retrying until all tracks have >= 3 hits."""
    for _ in range(20):
        _, event = generate_event(geo, n_tracks,
                                  collision_noise=collision_noise,
                                  phi_max=phi_max, theta_max=theta_max)
        if min(len(t.hit_ids) for t in event.tracks) >= 3:
            return event
    return event  # best effort


# ═══════════════════════════════════════════════════════════════════
#  Epsilon computation
# ═══════════════════════════════════════════════════════════════════
def compute_epsilon(sigma_res, sigma_scatt, dz, scale=1.0, theta_min=1.5e-5):
    """Compute angular acceptance threshold epsilon.

    Uses the paper formula:
        epsilon = sqrt(2*theta_s^2 + 12*theta_r^2 + 2*theta_min^2)
    """
    theta_s = scale * sigma_scatt
    theta_r = np.arctan((scale * sigma_res) / dz) if dz != 0 else 0.0
    return float(np.sqrt(2 * theta_s**2 + 12 * theta_r**2 + 2 * theta_min**2))


# ═══════════════════════════════════════════════════════════════════
#  Angle analysis
# ═══════════════════════════════════════════════════════════════════
def pairwise_angle(h_prev, h_mid, h_next):
    """Angle between segment (prev->mid) and segment (mid->next)."""
    v1 = np.array([h_mid.x - h_prev.x, h_mid.y - h_prev.y, h_mid.z - h_prev.z])
    v2 = np.array([h_next.x - h_mid.x, h_next.y - h_mid.y, h_next.z - h_mid.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def collect_segment_pair_angles(event):
    """Collect pairwise angles for ALL segment pairs sharing a middle hit.

    Returns (true_angles, false_angles) as lists.
    """
    hits_dict = {h.hit_id: h for h in event.hits}

    true_seg_set = set()
    for trk in event.tracks:
        hids = trk.hit_ids
        for k in range(len(hids) - 1):
            true_seg_set.add((hids[k], hids[k + 1]))

    hits_by_module = {}
    for h in event.hits:
        hits_by_module.setdefault(h.module_id, []).append(h)
    sorted_mod_ids = sorted(hits_by_module.keys())

    true_angles = []
    false_angles = []

    for mi in range(1, len(sorted_mod_ids) - 1):
        mod_prev = sorted_mod_ids[mi - 1]
        mod_mid  = sorted_mod_ids[mi]
        mod_next = sorted_mod_ids[mi + 1]

        for h_mid in hits_by_module[mod_mid]:
            for h_prev in hits_by_module[mod_prev]:
                seg_in_true = (h_prev.hit_id, h_mid.hit_id) in true_seg_set
                for h_next in hits_by_module[mod_next]:
                    seg_out_true = (h_mid.hit_id, h_next.hit_id) in true_seg_set
                    angle = pairwise_angle(h_prev, h_mid, h_next)
                    if (seg_in_true and seg_out_true and
                            h_prev.track_id is not None and
                            h_prev.track_id == h_mid.track_id == h_next.track_id):
                        true_angles.append(angle)
                    else:
                        false_angles.append(angle)

    return true_angles, false_angles


# ═══════════════════════════════════════════════════════════════════
#  Hit-competition helpers (from hit_competition_study.ipynb)
# ═══════════════════════════════════════════════════════════════════
def label_segments(ham, event):
    """Boolean array: True for segments matching a truth-track pair."""
    true_seg_set = set()
    for trk in event.tracks:
        hids = trk.hit_ids
        for k in range(len(hids) - 1):
            true_seg_set.add((hids[k], hids[k + 1]))
    return np.array([
        ham._segment_to_hit_ids[i] in true_seg_set
        for i in range(ham.n_segments)
    ])


def build_hit_to_segments(ham):
    """Dict: hit_id -> list of segment indices using that hit."""
    h2s = defaultdict(list)
    for idx, (fid, tid) in enumerate(ham._segment_to_hit_ids):
        h2s[fid].append(idx)
        h2s[tid].append(idx)
    return dict(h2s)


# ═══════════════════════════════════════════════════════════════════
#  Standard pipeline: build Hamiltonian + solve
# ═══════════════════════════════════════════════════════════════════
def build_and_solve(event, epsilon=None, gamma=GAMMA, delta=DELTA):
    """Build Hamiltonian and solve classically.

    Returns (ham, x, epsilon_used).
    """
    if epsilon is None:
        epsilon = compute_epsilon(SIGMA_RES, SIGMA_SCATT, DZ_MM, scale=SCALE)
    ham = SimpleHamiltonianFast(epsilon=epsilon, gamma=gamma, delta=delta)
    ham.construct_hamiltonian(event)
    x = ham.solve_classicaly()
    return ham, x, epsilon


def reconstruct_and_validate(event, ham, x, threshold=THRESHOLD, purity_min=0.7):
    """Reconstruct tracks from solution and validate against truth.

    Returns (reco_tracks, matches, metrics).
    """
    reco_tracks = get_tracks(ham, x, event, threshold=threshold)
    if len(reco_tracks) == 0:
        return reco_tracks, [], {
            'efficiency': 0.0,
            'ghost_rate': 0.0,
            'clone_fraction': 0.0,
        }
    val = EventValidator(event, reco_tracks)
    matches, metrics = val.match_tracks(purity_min=purity_min)
    if 'clone_fraction' not in metrics:
        metrics['clone_fraction'] = 0.0
    return reco_tracks, matches, metrics


# ═══════════════════════════════════════════════════════════════════
#  Standard epsilon for the default configuration
# ═══════════════════════════════════════════════════════════════════
EPSILON = compute_epsilon(SIGMA_RES, SIGMA_SCATT, DZ_MM, scale=SCALE)
