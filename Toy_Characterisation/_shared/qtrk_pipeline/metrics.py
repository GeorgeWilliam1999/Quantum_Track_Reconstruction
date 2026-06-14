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
    "ABS_THRESHOLD", "WP_TARGET_EFF", "truth_from_event", "rescale_to",
    "rescale_to_signal", "cos_sim", "metrics_at", "quantum_metrics",
    "working_point_threshold", "metrics_at_wp", "quantum_metrics_wp",
]

# The validated absolute operating point FOR gamma=3, delta=1 (Hopfield false
# attractor delta/(delta+gamma)=0.25, true plateau ~0.375).  Do NOT use a
# relative tau*max.
ABS_THRESHOLD = 0.35

# Provisional margin above the false attractor (0.25 + 0.10 = 0.35 at gamma=3).
# PENDING user confirmation — the principled alternative is a histogram-driven
# (Youden-J / EER) threshold per cell.  See the Notion source-of-truth page.
_THRESHOLD_MARGIN = 0.10

# Default efficiency target for the HEADLINE operating point (user, 2026-06-14).
# The fixed ABS_THRESHOLD above is the low-false-rate cut; for filtered solvers
# (1BQF / QSVT) it chops genuine true segments (the "~75 % efficiency plateau"
# is that artefact, not physics).  The canonical operating point is instead the
# efficiency-first working point: hold ~WP_TARGET_EFF of the true segments and
# pay the false rate (LHCb-style: keep all true, kill false).  See
# working_point_threshold + the wp99 refresh log.
WP_TARGET_EFF = 0.99


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


# ---------------------------------------------------------------------------
# Efficiency-first working point (wp99) — the HEADLINE operating point.
#
# user, 2026-06-14: report every 1BQF/filtered-solver result at the working
# point that holds ~100 % segment efficiency (get ALL true segments; killing
# false segments is how LHCb works), accepting the false rate paid.  This
# REPLACES the fixed ABS_THRESHOLD=0.35 cut as the headline.  Same convention
# as QSVT/Segment_level_studies/01 §4 (fig qsvt_working_points).
# ---------------------------------------------------------------------------

def working_point_threshold(sol: np.ndarray, truth: np.ndarray,
                            target_eff: float = WP_TARGET_EFF) -> float:
    """Efficiency-first per-solve segment threshold (the wp99 working point).

    Place ``tau`` just BELOW the ``(1 - target_eff)`` quantile of the TRUE-segment
    amplitudes, so that >= ``target_eff`` of the true segments stay active.  The
    magnitude of ``sol`` is used (active <=> ``|sol| > tau``).

    NOTE this is SCALE-INVARIANT: scaling ``sol`` by a constant scales ``tau`` by
    the same factor, so the active set — and hence eff / false-rate — does not
    depend on the quantum rescale convention (only ``cos_QC`` does).  For a
    quantum solve you still rescale to the classical SIGNAL first so ``cos_QC``
    and the reported ``tau`` sit on the classical amplitude scale.
    """
    s = np.abs(np.asarray(sol, np.float64))
    t = np.asarray(truth, bool)
    st = np.sort(s[t])
    if st.size == 0:
        return 0.0
    k = int(np.floor((1.0 - float(target_eff)) * st.size))
    k = min(max(k, 0), st.size - 1)
    return float(st[k] - 1e-9)


def metrics_at_wp(sol: np.ndarray, truth: np.ndarray,
                  target_eff: float = WP_TARGET_EFF) -> dict:
    """Segment metrics at the efficiency-first working point.

    Drop-in replacement for ``metrics_at`` for the HEADLINE numbers.  Returns the
    usual metrics dict plus ``tau_wp`` (the threshold used) and ``target_eff``.
    """
    sol = np.abs(np.asarray(sol, np.float64))
    tau = working_point_threshold(sol, truth, target_eff)
    out = metrics_at(sol, truth, threshold=tau)
    out["tau_wp"] = tau
    out["target_eff"] = float(target_eff)
    return out


def quantum_metrics_wp(sol_Q_raw: np.ndarray, sol_C: np.ndarray,
                       truth: np.ndarray,
                       target_eff: float = WP_TARGET_EFF,
                       signal_threshold: float = ABS_THRESHOLD) -> dict:
    """Quantum-solve metrics at the efficiency-first working point.

    Like ``quantum_metrics`` but the threshold is the per-solve wp99 working
    point on the (signal-rescaled) quantum amplitudes, NOT the fixed cut.
    Rescales the unit-norm quantum solution onto the classical SIGNAL support
    (``sol_C > signal_threshold`` — pass the gamma-aware cut for gamma!=3 so the
    reported ``tau_wp`` / ``cos_QC`` sit on the right classical scale; defaults to
    the gamma=3 value 0.35), then sets ``tau`` just below the
    ``(1 - target_eff)`` quantile of the quantum amplitudes on the TRUE segments.
    eff / false-rate are rescale-INVARIANT (see ``working_point_threshold``), so
    they are correct regardless of ``signal_threshold``; only ``tau_wp`` and
    ``cos_QC`` depend on it.
    """
    sol_C = np.asarray(sol_C, np.float64)
    sol_Q = rescale_to_signal(sol_Q_raw, sol_C, signal_threshold)
    out = metrics_at_wp(sol_Q, truth, target_eff)
    mask = sol_C > float(signal_threshold)
    if mask.any():
        out["cos_QC"] = cos_sim(np.asarray(sol_Q, np.float64)[mask], sol_C[mask])
    else:
        out["cos_QC"] = cos_sim(sol_Q, sol_C)
    return out
