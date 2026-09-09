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

# ── Package re-exports ──────────────────────────────────────────────
# compute_epsilon / pairwise_angle / collect_segment_pair_angles used to be
# defined here.  They now live in lhcb_velo_toy.analysis.segment_metrics, so
# this study picks up package fixes instead of drifting away from them.
# Verified identical before removal (2026-09-09): bit-exact at this study's
# operating point, 1 ulp worst case over a synthetic parameter sweep.
#
# NOTE the package default is scale=3.0 where the local copy defaulted to
# scale=1.0.  Every call site here passes scale=SCALE explicitly, so nothing
# changes -- but pass it explicitly in new code too.
from lhcb_velo_toy.analysis.segment_metrics import (  # noqa: F401
    compute_epsilon,
    pairwise_angle,
    collect_segment_pair_angles,
)

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
