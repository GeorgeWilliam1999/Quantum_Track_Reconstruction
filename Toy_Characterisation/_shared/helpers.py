"""
Shared helpers for the segment-level Toy_Characterisation experiments
(Epsilon_study_2, Larger_Scatter, Larger_Scatter_Density, ERF).

This module is intentionally *thin*: most logic now lives in
``lhcb_velo_toy.analysis.segment_metrics``. We keep here only:
- the standard 5-module / 33 mm geometry shared by all experiments,
- ``safe_generate`` (event generation with retries),
- ``apply_hit_inefficiency`` (random hit dropout),
- ``solve_quantum_statevector`` (statevector OneBitHHL helper),
- ``rescale_quantum`` (rescale q-solution to match classical L2 norm).

All experiment workers import from here so geometry / event-gen settings
stay in lockstep.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Re-exports so worker code can import everything from one module.
from lhcb_velo_toy.analysis.segment_metrics import (  # noqa: F401
    compute_epsilon,
    pairwise_angle,
    collect_segment_pair_angles,
    compute_segment_metrics,
    fast_segment_metrics,
    segment_truth_mask,
    solver_segment_metrics,
    DEFAULT_DZ,
    DEFAULT_SCALE,
    DEFAULT_THETA_MIN,
)


# ---------------------------------------------------------------------------
# Detector geometry — single source of truth (matches Verify_new_results).
# ---------------------------------------------------------------------------

DZ_MM = 33.0
N_MODULES = 5
Z_FIRST = 33.0
HALF_X = HALF_Y = 40.0
PV_SIGMA = {"x": 0.0, "y": 0.0, "z": 1.0}

# Hamiltonian defaults (matches segment_level_analysis.ipynb §1).
GAMMA = 3.0
DELTA = 1.0


def make_geometry():
    """Build the standard 5-plane geometry used by all experiments."""
    from lhcb_velo_toy.generation import PlaneGeometry
    z_positions = [Z_FIRST + DZ_MM * i for i in range(N_MODULES)]
    return PlaneGeometry(
        module_id=list(range(N_MODULES)),
        lx=[HALF_X] * N_MODULES,
        ly=[HALF_Y] * N_MODULES,
        z=z_positions,
    )


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def safe_generate(
    n_tracks: int,
    seed: int,
    geom,
    measurement_error: float = 0.0,
    collision_noise: float = 1e-4,
    phi_max: float = 0.2,
    theta_max: float = 0.2,
    max_retries: int = 20,
):
    """
    Generate one event, retrying until every track has >= 3 hits.

    ``measurement_error`` corresponds to ``sigma_res`` and
    ``collision_noise`` to ``sigma_scatt`` in the segment-level report.
    """
    from lhcb_velo_toy.generation import StateEventGenerator
    np.random.seed(int(seed))
    last = None
    for _ in range(max_retries):
        gen = StateEventGenerator(
            detector_geometry=geom,
            events=1,
            n_particles=[n_tracks],
            phi_min=-phi_max, phi_max=phi_max,
            theta_min=-theta_max, theta_max=theta_max,
            measurement_error=measurement_error,
            collision_noise=collision_noise,
        )
        gen.generate_random_primary_vertices(PV_SIGMA)
        gen.generate_particles(
            [[{"type": "pion", "mass": 139.6, "q": 1}] * n_tracks]
        )
        ev = gen.generate_complete_events()
        last = ev
        if ev.tracks and min(len(t.hit_ids) for t in ev.tracks) >= 3:
            return ev
    return last


def apply_hit_inefficiency(event, p_drop: float, rng: np.random.Generator):
    """
    Randomly remove a fraction ``p_drop`` of hits from ``event`` in-place
    (also pruning the matching entries in each track's ``hit_ids``).

    Returns the modified event for convenience. ``p_drop = 0`` is a no-op.
    """
    if p_drop <= 0.0:
        return event
    keep_mask = rng.random(len(event.hits)) >= p_drop
    kept_ids = {h.hit_id for h, keep in zip(event.hits, keep_mask) if keep}
    event.hits = [h for h, keep in zip(event.hits, keep_mask) if keep]
    # Sync per-track hit_ids.
    for trk in event.tracks:
        trk.hit_ids = [hid for hid in trk.hit_ids if hid in kept_ids]
    # Sync per-module hit_ids if present on the geometry / modules.
    modules = getattr(event, "modules", None)
    if modules is not None:
        for m in modules:
            m.hit_ids = [hid for hid in m.hit_ids if hid in kept_ids]
    return event


# ---------------------------------------------------------------------------
# Quantum solver (statevector readout, sparse-A path)
# ---------------------------------------------------------------------------

def solve_quantum_statevector(A, b, device: str = 'CPU', engine: str | None = None):
    """
    Run the 1BQF / OneBitHHL solver in exact statevector mode.

    ``engine`` selects HOW the exact statevector is produced — the result is
    bit-identical across engines (validated cos=1.0, ΔP~1e-14), so this never
    changes a ``sol_key`` (the key carries solver/device/readout, not engine):

    ``engine='matrixfree'`` (DEFAULT) — applies the OneBQF gates directly to a
        numpy statevector (no qiskit circuit / transpile / Aer). Statevector-only
        memory (≤256 MB at T=1000) and seconds per solve at any T/ε; this is what
        removes the host-RAM OOM that the Aer circuit-assemble caused at T≥400.
        ``device`` is ignored (CPU numpy; it already beats Aer-GPU).
    ``engine='aer'`` — the legacy AerSimulator path (``device='CPU'`` multi-threaded
        C++, ``device='GPU'`` cuStateVec). Kept for cross-checks; it transpiles the
        circuit and Aer ingests ~millions of gates → 13–119 GB at high T/ε.

    Override the default with env ``QTRK_OBQF_ENGINE=aer|matrixfree``.

    Returns ``(sol_vec, p_success, n_system_qubits)``.
    ``A`` must be a sparse matrix (OneBQF sparse path).
    """
    import os
    if engine is None:
        engine = os.environ.get('QTRK_OBQF_ENGINE', 'matrixfree')
    if str(engine).lower() == 'matrixfree':
        from obqf_matrixfree import solve_matrixfree
        return solve_matrixfree(A, b)

    from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL

    dev = device.upper()
    if dev in ('GPU', 'CPU'):
        # AerSimulator for BOTH GPU and CPU. The CPU path is MULTI-THREADED C++
        # (OpenMP) — ~3x slower than GPU at T=100, ~7x at T=200, widening at high
        # T, but it runs on abundant CPU slots instead of waiting for scarce GPUs.
        # (Single-threaded Statevector.from_instruction is the 'CPU_EXACT' fallback
        # below and is much slower.)
        import os
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False,
                        readout='statevector')
        hhl.build_circuit()
        # CRITICAL: cap Aer's memory budget. AerSimulator queries the NODE's total
        # RAM (not the Condor cgroup), so it over-allocates work buffers as if it
        # owns the machine and OOMs the cgroup past ~22 qubits (T>=400). Passing
        # max_memory_mb makes it stay within budget (cache-blocking if needed):
        # T=400 dropped 64 GB -> ~5 GB. Set AER_MAX_MEM_MB to (request_mem - headroom)
        # in the submit env. Disable fusion (no speed benefit here, sometimes slower).
        aer_kw = dict(method='statevector', fusion_enable=False)
        max_mem = int(os.environ.get('AER_MAX_MEM_MB', 0))
        if max_mem > 0:
            aer_kw['max_memory_mb'] = max_mem
        if dev == 'GPU':
            sim = AerSimulator(device='GPU', **aer_kw)
        else:
            threads = int(os.environ.get('OMP_NUM_THREADS', 0)) or (os.cpu_count() or 8)
            sim = AerSimulator(device='CPU', max_parallel_threads=threads, **aer_kw)
        tqc = transpile(hhl.circuit, sim, optimization_level=1)
        job = sim.run(tqc, shots=1)
        sv = np.asarray(job.result().data(0)['statevector'])
        sol_raw, success = hhl.get_solution_from_statevector(sv)
        n_sys = hhl.num_system_qubits
        nrm = float(np.linalg.norm(sol_raw))
        if nrm > 0:
            sol_raw = sol_raw / nrm
        return sol_raw, float(success), n_sys
    else:
        from qiskit.quantum_info import Statevector
        # NOTE: do NOT pass readout='statevector' here — that inserts a
        # save_statevector instruction that Statevector.from_instruction()
        # cannot consume.  Strip final measurements instead.
        hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False)
        hhl.build_circuit()
        qc = hhl.circuit.remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc)
        amps = np.asarray(sv.data)
        n_sys = hhl.num_system_qubits
        amps = amps.reshape(2, 2 ** n_sys, 2)
        probs_anc1 = (np.abs(amps[1]) ** 2).sum(axis=-1)
        success = float(probs_anc1.sum())
        prob_norm = probs_anc1 / success if success > 0 else np.zeros_like(probs_anc1)
        sol_padded = np.sqrt(prob_norm)
        nrm = float(np.linalg.norm(sol_padded))
        if nrm > 0:
            sol_padded /= nrm
        return sol_padded[: hhl.original_dim], success, n_sys


def solve_quantum_aer_sampling(A, b, shots: int):
    """
    Run the 1BQF / OneBitHHL solver via AerSimulator (sampling readout).

    NOTE: The OneBQF circuit has O(A_nnz) QRAM gates, so circuit simulation
    time scales as O(A_nnz × 2^n_sys) regardless of backend.  This function
    is NOT faster than solve_quantum_statevector for large T; both are limited
    by the same circuit simulation cost.  Use only for small T (≤50) where
    the AerSimulator path is needed for testing or comparison purposes.

    For T≥400 use run_quantum=0 (classical solver only).

    Returns the same ``(sol_vec, p_success, n_system_qubits)`` triple as
    ``solve_quantum_statevector`` so callers are interchangeable.
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL

    hhl = OneBitHHL(A, b, num_time_qubits=1, shots=shots, debug=False)
    hhl.build_circuit()
    sim = AerSimulator(method='statevector', device='CPU')
    tqc = transpile(hhl.circuit, sim, optimization_level=1)
    job = sim.run(tqc, shots=shots)
    hhl.counts = job.result().get_counts()
    sol_raw, total_success = hhl.get_solution()
    n_sys = hhl.num_system_qubits
    # sol_raw is already sqrt(normalised probs) → unit norm by construction;
    # apply the same safety normalisation as the statevector path.
    nrm = float(np.linalg.norm(sol_raw))
    if nrm > 0:
        sol_raw = sol_raw / nrm
    p_anc = float(total_success) / shots if shots > 0 else 0.0
    return sol_raw, p_anc, n_sys


def rescale_quantum(sol_Q: np.ndarray, sol_C: np.ndarray) -> np.ndarray:
    """Rescale ``sol_Q`` so its L2 norm matches ``sol_C`` (canonical convention)."""
    nQ = float(np.linalg.norm(sol_Q))
    nC = float(np.linalg.norm(sol_C))
    return sol_Q * (nC / nQ) if nQ > 0 else np.zeros_like(sol_Q)


# ---------------------------------------------------------------------------
# Standard worker-record schema
# ---------------------------------------------------------------------------

def make_record(*, status: str, **fields) -> dict:
    """Canonical worker output schema. Phases extend ``fields`` as needed."""
    out = {"status": status}
    out.update(fields)
    return out


__all__ = [
    "DZ_MM", "N_MODULES", "Z_FIRST", "HALF_X", "HALF_Y", "PV_SIGMA",
    "GAMMA", "DELTA",
    "make_geometry", "safe_generate", "apply_hit_inefficiency",
    "solve_quantum_statevector", "solve_quantum_aer_sampling", "rescale_quantum",
    "make_record",
    # Re-exports from lhcb_velo_toy
    "compute_epsilon", "pairwise_angle", "collect_segment_pair_angles",
    "compute_segment_metrics", "fast_segment_metrics",
    "segment_truth_mask", "solver_segment_metrics",
    "DEFAULT_DZ", "DEFAULT_SCALE", "DEFAULT_THETA_MIN",
]
