"""
qtrk_pipeline — decoupled data production for the segment-level Toy studies.

Layers (each keyed independently, so work is never repeated):

    keys        event_key / ham_key / sol_key / seed_for   (content addressing)
    events      generate / save / load / ensure_event       (Layer 0+1)
    hamiltonian build_hamiltonian (regenerate A on demand)   (Layer 2)
    solve       solve_classical / solve_quantum (1BQF)       (Layer 3)
    metrics     truth_from_event / metrics_at / quantum_metrics  (Layer 4 view)
    manifest    StudySpec / build_manifest / standard_specs  (planning)
    store       on-disk layout + solution IO

Canonical recipe (one solved point)::

    from qtrk_pipeline import (ensure_event, build_hamiltonian,
                               solve_classical, truth_from_event, metrics_at)
    ev, ekey = ensure_event(n_trk=50, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    ham      = build_hamiltonian(ev, epsilon=eps, kernel="step")
    sol_C, t = solve_classical(ham)
    truth    = truth_from_event(ev)
    m        = metrics_at(sol_C, truth, threshold=0.35)

Metrics are a VIEW: never trust a stored metric, recompute from sol+truth.
"""
from __future__ import annotations

from .keys import event_key, ham_key, sol_key, seed_for, fnum
from .store import (
    store_root, events_dir, solutions_dir, manifest_dir,
    event_path, solution_path,
    save_solution, load_solution, solution_exists, load_metrics,
    add_validity, EXPLODE_MAX,
)
from .events import generate_event, save_event, load_event, ensure_event
from .hamiltonian import build_hamiltonian, segments_only
from .solve import solve_classical, solve_quantum, solve_qsvt, spectral_bounds_for
from .metrics import (
    ABS_THRESHOLD, WP_TARGET_EFF, threshold_for, truth_from_event, rescale_to,
    rescale_to_signal, cos_sim, metrics_at, quantum_metrics,
    working_point_threshold, metrics_at_wp, quantum_metrics_wp,
)
from .manifest import (
    StudySpec, expand_spec, build_manifest, standard_specs, DEFAULT_TRACK_GRID,
)

__all__ = [
    "event_key", "ham_key", "sol_key", "seed_for", "fnum",
    "store_root", "events_dir", "solutions_dir", "manifest_dir",
    "event_path", "solution_path",
    "save_solution", "load_solution", "solution_exists", "load_metrics",
    "add_validity", "EXPLODE_MAX",
    "generate_event", "save_event", "load_event", "ensure_event",
    "build_hamiltonian", "segments_only",
    "solve_classical", "solve_quantum", "solve_qsvt", "spectral_bounds_for",
    "ABS_THRESHOLD", "WP_TARGET_EFF", "threshold_for", "truth_from_event",
    "rescale_to", "rescale_to_signal", "cos_sim", "metrics_at", "quantum_metrics",
    "working_point_threshold", "metrics_at_wp", "quantum_metrics_wp",
    "StudySpec", "expand_spec", "build_manifest", "standard_specs",
    "DEFAULT_TRACK_GRID",
]
