"""
On-demand matrix A construction from a stored event.

This module does NOT implement the Hamiltonian — all matrix construction is
``lhcb_velo_toy.solvers.SimpleHamiltonianFast`` (``construct_segments`` +
``construct_hamiltonian``).  It is a thin pipeline wrapper that adds exactly two
things the library does not:
  1. a string ``kernel`` API (``"step"``/``"erf"`` -> ``convolution`` bool +
     ``erf_sigma``), and
  2. the **sparse-A invariant guard** (a campaign safety check, not library
     behaviour).

A is never persisted (decision: store events, regenerate A on demand), so this
is the single place the regeneration + guard live:

- ``step`` kernel: A.nnz must stay ~O(n_seg) -> hard assert ``nnz < 5*n_seg``.
- ``erf``  kernel: legitimately denser (keeps pairs out to ~eps+6*theta_d), and
  for wide theta_d at high T grows faster than O(n_seg).  We record nnz and warn
  past a loose ratio, but do not crash — feasibility of those corners is a
  campaign-planning decision, not a bug.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

__all__ = ["build_hamiltonian", "segments_only"]

# Hard cap for the step kernel (sparse invariant).  Overridable via env
# QTRK_STEP_MAX_RATIO for cells where the fill is genuinely dense physics
# (e.g. Noisy_Density T=1000 phi_max=0.01 formula-eps reaches 5.0-5.2x with
# real couplings, not the explicit-zero bug this guard was built to catch).
STEP_MAX_RATIO = float(os.environ.get("QTRK_STEP_MAX_RATIO", 5.0))
ERF_WARN_RATIO = 250.0    # soft warn for erf; above this a point may be heavy


def build_hamiltonian(
    event,
    epsilon: float,
    kernel: str = "step",
    erf_sigma: float = 1e-4,
    gamma: float = 3.0,
    delta: float = 1.0,
    assert_sparse: bool = True,
):
    """Build A and b on ``event``; return the constructed Hamiltonian.

    The returned object exposes ``.A`` (csc), ``.b``, ``.n_segments`` and the
    cached ``._segment_track_ids`` used for the truth mask.
    """
    from lhcb_velo_toy.solvers import SimpleHamiltonianFast

    kern = str(kernel).lower()
    ham = SimpleHamiltonianFast(epsilon=float(epsilon), gamma=float(gamma),
                                delta=float(delta), theta_d=float(erf_sigma))
    # materialize_segments=False: build only the numpy arrays, NOT the O(n_seg)
    # Python Segment objects (4M of them at T=1000 — the real high-T cost).
    # construct_hamiltonian and the truth mask both work off the cached arrays;
    # nothing downstream needs ham.segments.
    ham.construct_segments(event, materialize_segments=False)
    ham.construct_hamiltonian(event, convolution=(kern == "erf"))

    nnz = int(ham.A.nnz)
    nseg = int(ham.n_segments)
    ratio = nnz / max(nseg, 1)
    if assert_sparse:
        if kern == "step" and ratio > STEP_MAX_RATIO:
            raise RuntimeError(
                f"DENSE step A: nnz={nnz} = {ratio:.1f}x n_seg={nseg}. "
                "The step path has lost sparsity — do NOT submit a batch."
            )
        if kern == "erf" and ratio > ERF_WARN_RATIO:
            warnings.warn(
                f"erf A is heavy: nnz={nnz} = {ratio:.0f}x n_seg={nseg} "
                f"(theta_d={erf_sigma:g}). High-T wide-theta_d corners scale "
                "faster than O(n_seg); confirm feasibility before mass submit.",
                RuntimeWarning,
            )
    return ham


def segments_only(event, epsilon: float = 1.0, gamma: float = 3.0,
                  delta: float = 1.0):
    """Build only the segment skeleton (no A) — enough for the truth mask.

    Much cheaper than ``build_hamiltonian`` (no O(T^3) candidate scan).
    ``materialize_segments=False`` skips building the O(n_seg) Python ``Segment``
    objects (4M of them at T=1000 — the real cost); the truth mask only needs the
    cached ``_segment_track_ids`` numpy array, which is populated either way.
    """
    from lhcb_velo_toy.solvers import SimpleHamiltonianFast

    ham = SimpleHamiltonianFast(epsilon=float(epsilon), gamma=float(gamma),
                                delta=float(delta))
    ham.construct_segments(event, materialize_segments=False)
    return ham
