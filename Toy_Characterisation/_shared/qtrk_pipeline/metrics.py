"""Metrics as a VIEW — thin re-export of the canonical package.

Promoted into ``lhcb_velo_toy.analysis.working_points`` (centralise-2.1
Phase 5, 2026-08-26); this module remains only so existing
``qtrk_pipeline.metrics`` / ``qp.metrics_at`` / ``qp.threshold_for`` imports
keep working. The threshold policy (ABS_THRESHOLD=0.35 at gamma=3, the
gamma-aware ``threshold_for``, the wp99 efficiency-first headline working
point) and the signal-support quantum rescale convention live in the
package — one implementation, no drift.

``rescale_to`` keeps its historical qtrk name (package name:
``rescale_quantum``).
"""
from __future__ import annotations

from lhcb_velo_toy.analysis.working_points import (  # noqa: F401
    ABS_THRESHOLD,
    WP_TARGET_EFF,
    WP_TAU_FLOOR,
    cos_sim,
    metrics_at,
    metrics_at_wp,
    quantum_metrics,
    quantum_metrics_wp,
    rescale_quantum as rescale_to,
    rescale_to_signal,
    threshold_for,
    truth_from_event,
    working_point_threshold,
)

__all__ = [
    "ABS_THRESHOLD", "WP_TARGET_EFF", "truth_from_event", "rescale_to",
    "rescale_to_signal", "cos_sim", "metrics_at", "quantum_metrics",
    "working_point_threshold", "metrics_at_wp", "quantum_metrics_wp",
    "threshold_for", "WP_TAU_FLOOR",
]
