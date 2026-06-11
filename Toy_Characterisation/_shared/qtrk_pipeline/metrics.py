"""
Metrics as a VIEW — never stored as canonical, always recomputed.

The relative-threshold bug happened because a metric got frozen into a pickle.
Here metrics are a pure function of (solution vector, truth, threshold) computed
by ONE code path, so that class of bug is structurally impossible: regenerate
the whole table in seconds and it is always correct.

Truth is recomputed from the stored EVENT (segment skeleton only), so the
metrics stage never needs to rebuild the full matrix A.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SHARED = Path(__file__).resolve().parent.parent
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from .hamiltonian import segments_only  # noqa: E402

__all__ = [
    "ABS_THRESHOLD", "truth_from_event", "rescale_to", "rescale_to_signal",
    "cos_sim", "metrics_at", "quantum_metrics",
]

# The validated absolute operating point FOR gamma=3, delta=1 (Hopfield false
# attractor delta/(delta+gamma)=0.25, true plateau ~0.375).  Do NOT use a
# relative tau*max.
ABS_THRESHOLD = 0.35

# Provisional margin above the false attractor (0.25 + 0.10 = 0.35 at gamma=3).
# PENDING user confirmation — the principled alternative is a histogram-driven
# (Youden-J / EER) threshold per cell.  See the Notion source-of-truth page.
_THRESHOLD_MARGIN = 0.10


def threshold_for(gamma: float = 3.0, delta: float = 1.0,
                  margin: float = _THRESHOLD_MARGIN) -> float:
    """gamma-aware absolute segment threshold.

    The false/isolated Hopfield attractor sits at ``delta/(delta+gamma)``; the
    operating point is a fixed ``margin`` above it.  Reduces to the validated
    0.35 at gamma=3, delta=1.  For gamma!=3 the fixed 0.35 is wrong (e.g. at
    gamma=1 the attractor is 0.5 > 0.35, so every false segment reads active).
    """
    return float(delta) / (float(delta) + float(gamma)) + float(margin)


def truth_from_event(event) -> np.ndarray:
    """Per-segment truth mask, recomputed from the event (no A build)."""
    from helpers import segment_truth_mask
    ham = segments_only(event)
    return segment_truth_mask(ham)


def rescale_to(sol: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rescale ``sol`` so ‖sol‖ == ‖reference‖ (canonical quantum convention).

    Thin wrapper over the shared ``helpers.rescale_quantum`` so there is one
    implementation of the convention across the codebase.
    """
    from helpers import rescale_quantum
    return rescale_quantum(np.asarray(sol, np.float64), np.asarray(reference, np.float64))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb + 1e-30))


def metrics_at(sol: np.ndarray, truth: np.ndarray,
               threshold: float = ABS_THRESHOLD) -> dict:
    """Segment efficiency / purity / false-rate at an ABSOLUTE threshold."""
    from helpers import solver_segment_metrics
    return solver_segment_metrics(np.asarray(truth, bool), np.asarray(sol),
                                  threshold=float(threshold))


def rescale_to_signal(sol_Q_raw: np.ndarray, sol_C: np.ndarray,
                      threshold: float = ABS_THRESHOLD) -> np.ndarray:
    """Put a unit-norm quantum solution on the classical SIGNAL scale.

    The 1BQF statevector solution is unit-norm, so to compare it against the
    classical ABSOLUTE threshold it must be rescaled back to the classical
    amplitude scale.  Matching the *full* L2 norm (``rescale_to(.., sol_C)``) is
    WRONG: ‖sol_C‖ is dominated by the O(n_seg) false segments pinned at the
    Hopfield attractor ``delta/(delta+gamma)`` (95-99 % of the norm), so it grows
    ~``sqrt(n_seg)``.  That makes the rescale factor multiplicity-dependent and
    inflates the quantum amplitudes, pushing borderline false segments over the
    fixed threshold — an artefact that made the quantum false rate diverge from
    the classical one (≈44 % vs ≈2 % at T=400) while the underlying solve is
    faithful (cos on the signal support ≈ 0.97, AUC ≈ 1).

    Fix: rescale on the classical SIGNAL support (``sol_C > threshold``, i.e. the
    classically-active segments — the meaningful, false-bulk-free scale).  This
    is truth-free (it uses the classical partner, not the labels) and removes the
    n-dependence.  See the Notion "quantum metrics convention" note.
    """
    sol_Q_raw = np.asarray(sol_Q_raw, np.float64)
    sol_C = np.asarray(sol_C, np.float64)
    mask = sol_C > float(threshold)
    if not mask.any():                              # nothing active -> fall back
        return rescale_to(sol_Q_raw, sol_C)
    denom = float(np.linalg.norm(sol_Q_raw[mask]))
    if denom == 0.0:
        return np.zeros_like(sol_Q_raw)
    return sol_Q_raw * (float(np.linalg.norm(sol_C[mask])) / denom)


def quantum_metrics(sol_Q_raw: np.ndarray, sol_C: np.ndarray,
                    truth: np.ndarray, threshold: float = ABS_THRESHOLD) -> dict:
    """Metrics for a quantum solve: rescale on the classical SIGNAL, then threshold.

    Rescales the unit-norm quantum solution to the classical amplitude scale on
    the active (signal) support — NOT the full ‖sol_C‖, which is false-bulk
    dominated and grows with multiplicity (see ``rescale_to_signal``).  Returns
    the segment metrics dict plus ``cos_QC``, the fidelity to classical measured
    on that same signal support (the full-vector cosine is dragged down to ~0.1
    by the 0.25 false bulk and understates the agreement).
    """
    sol_C = np.asarray(sol_C, np.float64)
    sol_Q = rescale_to_signal(sol_Q_raw, sol_C, threshold)
    out = metrics_at(sol_Q, truth, threshold)
    mask = sol_C > float(threshold)
    if mask.any():
        out["cos_QC"] = cos_sim(sol_Q[mask], sol_C[mask])
    else:
        out["cos_QC"] = cos_sim(sol_Q, sol_C)
    return out
