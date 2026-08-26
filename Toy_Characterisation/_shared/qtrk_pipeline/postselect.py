"""Hit-uniqueness post-selection — thin re-export of the canonical package.

Promoted into ``lhcb_velo_toy.solvers.postselect`` (centralise-2.1 Phase 5,
2026-08-26); this module remains only so existing ``qtrk_pipeline.postselect``
imports keep working. The docstrings, modes (greedy/strict/margin/anyrole)
and behaviour live in the package — one implementation, no drift.
"""
from __future__ import annotations

from lhcb_velo_toy.solvers.postselect import (  # noqa: F401
    hit_uniqueness_filter,
    uniqueness_frontier,
)

__all__ = ["hit_uniqueness_filter", "uniqueness_frontier"]
