"""
Classical and quantum (1BQF / OneBitHHL) solvers over a built Hamiltonian.

Both return float32 vectors for storage.  The quantum solution is returned
RAW (unit-norm, as the simulator produces it) together with P_anc / n_sys;
rescaling to the classical L2 norm is deferred to the metrics stage, so a
quantum solve never depends on having run the classical one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_SHARED = Path(__file__).resolve().parent.parent
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

__all__ = ["solve_classical", "solve_quantum", "solve_qsvt", "spectral_bounds_for"]


def spectral_bounds_for(ham, pad: float = 0.10) -> tuple[float, float]:
    """Cheap spectral bounds for A = sI - C — thin wrapper over the package.

    The implementation was promoted to
    ``lhcb_velo_toy.solvers.quantum.spectral_bounds`` (centralise-2.1
    Phase 5, 2026-08-26); this wrapper only extracts s = gamma + delta from
    the Hamiltonian object.
    """
    from lhcb_velo_toy.solvers.quantum import spectral_bounds

    s = float(getattr(ham, "gamma", 3.0)) + float(getattr(ham, "delta", 1.0))
    return spectral_bounds(ham.A, s=s, pad=pad)


def solve_classical(ham) -> tuple[np.ndarray, float]:
    """Return ``(sol_float32, t_solve)`` for the classical sparse solve.

    A is **symmetric** and may be **indefinite**: ``A = (gamma+delta)I - C`` and
    the coupling spectral radius can exceed the diagonal (e.g. gamma=1 at high T
    -> ``2I - C`` has negative eigenvalues).  The library ``solve_classicaly``
    uses **CG with no maxiter**, which only works for SPD matrices — on the
    indefinite case it stalls for hours (observed: Verify gamma=1, T=700 hung
    ~4 h).  We instead use a direct solve for small systems and **MINRES** (the
    correct symmetric-indefinite Krylov method, converges in <1 s here) for
    large ones, both bounded.
    """
    from scipy.sparse.linalg import spsolve, minres

    t0 = time.time()
    A, b = ham.A, ham.b
    n = int(ham.n_segments)
    if n < 5000:
        sol = spsolve(A, b)
    else:
        sol, info = minres(A, b, rtol=1e-8, maxiter=5000)
        if info != 0:
            res = float(np.linalg.norm(A @ sol - b) / (np.linalg.norm(b) + 1e-30))
            if res > 1e-2:   # genuinely failed, not just a loose stop
                import warnings
                warnings.warn(f"MINRES did not converge (info={info}, "
                              f"rel_resid={res:.1e}) at n_seg={n}", RuntimeWarning)
    sol = np.asarray(sol).ravel().astype(np.float32)
    return sol, time.time() - t0


def solve_quantum(
    ham, device: str = "CPU", readout: str = "statevector", shots: int = 0,
) -> dict:
    """Run the 1BQF solver; return raw (unrescaled) solution + diagnostics.

    Returns ``{'sol': float32, 'P_anc': float, 'n_sys': int, 'n_qubits': int,
    't_solve': float}``.  ``readout='sampling'`` uses the AerSimulator sampling
    path (needs ``shots>0``); otherwise the exact statevector path is used.
    """
    from helpers import solve_quantum_statevector, solve_quantum_aer_sampling

    t0 = time.time()
    if str(readout).lower() == "sampling":
        if shots <= 0:
            raise ValueError("readout='sampling' requires shots > 0")
        sol_raw, p_anc, n_sys = solve_quantum_aer_sampling(ham.A, ham.b, int(shots))
    else:
        sol_raw, p_anc, n_sys = solve_quantum_statevector(ham.A, ham.b, device=device)
    return dict(
        sol=np.asarray(sol_raw, dtype=np.float32),
        P_anc=float(p_anc), n_sys=int(n_sys), n_qubits=int(n_sys) + 2,
        t_solve=time.time() - t0,
    )


def solve_qsvt(ham, degree: int = 40, spectral_bounds=None, domain=None,
               filter_design: str = "comb") -> dict:
    """Run the QSVT polynomial-filter solver; return the raw solution.

    Uses ``lhcb_velo_toy.solvers.quantum.QSVT`` (LCU-of-Chebyshev over the
    qubitization walk operator; the 1BQF cosine is the degree-1 member of the
    same family).  The filter is gamma-aware, centred on the constant diagonal
    s = gamma + delta:

    * ``filter_design='comb'`` (default) — the **line-comb inverse**: narrow
      ~1/lambda passes at the four true-track (P4) eigenvalue lines only.
      Rejects the tangle/chain modes between the lines, which keeps the false
      rate ~0 at high track density (the band design's false promotions grow
      to tens of percent by T=400).
    * ``filter_design='minimax'`` — the same comb fit in the Chebyshev
      (minimax) norm via an LP; floors the metric at roughly half the degree
      of the least-squares comb (D4 resource reduction).
    * ``filter_design='band'`` — the band-limited inverse of the initial
      study (~1/lambda on the contiguous true band, notched at the failure
      modes).  Kept for comparison.

    Like :func:`solve_quantum`, the solution is returned RAW (unit-norm
    |amplitudes|, the post-selected readout); rescaling to the classical
    signal support is deferred to the metrics stage.  ``P_anc`` is the
    post-selection success probability (LCU register and block-encoding
    ancilla both 0).

    ``spectral_bounds=(lo, hi)``, if given, must contain the spectrum of A and
    skips the (slow) Lanczos estimate; the campaign computes it cheaply from
    the coupling spectral radius rho(C):  spec(A) = s - spec(C).
    """
    from lhcb_velo_toy.solvers.quantum import (
        QSVT, design_band_limited_inverse, design_line_comb_inverse,
        design_minimax_comb)

    t0 = time.time()
    s = float(getattr(ham, "gamma", 3.0)) + float(getattr(ham, "delta", 1.0))
    if str(filter_design) == "minimax":
        poly = design_minimax_comb(degree=int(degree), s=s, domain=domain)
    elif str(filter_design) == "comb":
        poly = design_line_comb_inverse(degree=int(degree), s=s, domain=domain)
    else:
        poly = design_band_limited_inverse(degree=int(degree), s=s, domain=domain)
    qv = QSVT(ham.A, np.asarray(ham.b, float).ravel(), poly=poly,
              spectral_bounds=spectral_bounds)
    sol, p_succ = qv.solve_statevector()
    return dict(
        sol=np.asarray(sol, dtype=np.float32),
        P_anc=float(p_succ), n_sys=int(qv.num_system_qubits),
        n_qubits=int(qv.total_qubits), degree=int(degree),
        filter_design=str(filter_design),
        t_solve=time.time() - t0,
    )
