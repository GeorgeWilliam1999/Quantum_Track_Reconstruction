"""Matrix-free OneBQF statevector engine — thin re-export of the package.

Promoted verbatim into ``lhcb_velo_toy.solvers.quantum.matrixfree``
(centralise-2.1 Phase 5, 2026-08-26); this module remains only so existing
``from obqf_matrixfree import solve_matrixfree`` imports (helpers.py, the
Condor workers) keep working. One implementation, no drift.
"""
from __future__ import annotations

from lhcb_velo_toy.solvers.quantum.matrixfree import (  # noqa: F401
    solve_one_bit_matrixfree as solve_matrixfree,
)

__all__ = ["solve_matrixfree"]
