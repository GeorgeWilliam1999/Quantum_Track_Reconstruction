"""Matrix-free OneBQF (1-bit HHL) statevector engine.

Reproduces EXACTLY what ``OneBQF.build_circuit()`` + an exact statevector simulator
produce, but **without** building a qiskit circuit, transpiling, or invoking Aer.
Host memory is therefore just the ``2**n`` statevector + O(1) (≤256 MB at T=1000),
not Aer's ~7 KB × (millions of transpiled gates) that OOM'd the high-T / large-ε
quantum jobs at 24–119 GB.

Cost is ``O(n_sys · 2**n)`` for the Hadamards plus ``O(#interaction_pairs)`` for the
evolution — each interaction pair is a controlled two-level rotation that touches
O(1) amplitudes (vs a transpiled multi-controlled-RX of ~1.6k basis gates). Measured:
T=400 in 2.6 s / 0.7 GB, T=1000 in 12 s / 1.5 GB, vs Aer-GPU 365 s / Aer-CPU 4 h.

**Bit-identical** to ``helpers.solve_quantum_statevector(A, b, engine='aer')``:
validated cos = 1.000000000000, ΔP_anc ~1e-14 (clean *and* σ_res=0.05) at T=10/20/50.

Qubit order matches OneBQF: ``QuantumCircuit(time(1), system(n_sys), ancilla(1))`` →
little-endian state index ``k = time(bit 0) + 2·sys + 2**(n_sys+1)·anc``.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse as _sp

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

__all__ = ["solve_matrixfree"]


def _h_on_bit(psi: np.ndarray, q: int, n: int) -> None:
    """Hadamard on qubit ``q`` of an ``n``-qubit little-endian statevector (in place)."""
    v = psi.reshape(1 << (n - 1 - q), 2, 1 << q)
    a = v[:, 0, :].copy()
    b = v[:, 1, :].copy()
    v[:, 0, :] = (a + b) * _INV_SQRT2
    v[:, 1, :] = (a - b) * _INV_SQRT2
    # reshape is a view -> psi already updated.


def _controlled_U(psi: np.ndarray, pairs, theta: float, diag_phase: float,
                  n_sys: int) -> None:
    """Apply controlled-(on the time qubit)-U to ``psi`` in place.

    ``U = [∏ over pairs (i,j) RX(theta) in span{|i>,|j>}] · e^{i·diag_phase}`` on the
    time=1 branch.  Only the time=1, all-ancilla amplitudes are touched.
    """
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    anc_stride = 1 << (n_sys + 1)
    for (i, j) in pairs:
        ki0 = 1 + 2 * i            # time=1, sys=i, anc=0
        kj0 = 1 + 2 * j
        ki1 = ki0 + anc_stride     # anc=1
        kj1 = kj0 + anc_stride
        ai = psi[ki0]; aj = psi[kj0]
        psi[ki0] = c * ai + s * aj
        psi[kj0] = s * ai + c * aj
        ai = psi[ki1]; aj = psi[kj1]
        psi[ki1] = c * ai + s * aj
        psi[kj1] = s * ai + c * aj
    if diag_phase != 0.0:
        psi.reshape(-1, 2)[:, 1] *= np.exp(1j * diag_phase)


def solve_matrixfree(A, b):
    """1BQF statevector solve, matrix-free.

    Returns ``(sol_vec_float32, P_anc, n_sys)`` with the same convention and (to
    float32) the same values as :func:`helpers.solve_quantum_statevector`.
    ``A`` must be sparse with a constant diagonal (the OneBQF requirement).
    """
    A = A.tocsr() if _sp.issparse(A) else _sp.csr_matrix(np.asarray(A, float))
    d = A.shape[0]
    n_sys = max(1, int(np.ceil(np.log2(d))))
    diag = np.asarray(A.diagonal()).ravel()
    if not np.allclose(diag, diag[0]):
        raise ValueError("Matrix A must have a constant diagonal for this scheme.")
    diagonal_val = float(diag[0])
    t = np.pi / diagonal_val
    off = _sp.triu(A + A.T, k=1).tocoo()      # symmetrised off-diagonal support (i<j)
    pairs = list(zip(off.row.tolist(), off.col.tolist()))

    n = n_sys + 2                              # time(1) + system + ancilla(1)
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0

    # H on system qubits (bits 1..n_sys) -> uniform |b> on the system register
    for q in range(1, n_sys + 1):
        _h_on_bit(psi, q, n)

    theta = 2.0 * t                            # power = 1 (num_time_qubits = 1)
    phase = -diagonal_val * t

    # phase estimation: H(time); controlled-U; IQFT(1)=H(time)
    _h_on_bit(psi, 0, n)
    _controlled_U(psi, pairs, theta, phase, n_sys)
    _h_on_bit(psi, 0, n)

    # ancilla gadget: x(time); cx(time->anc); x(time)  ==  anc ^= (time == 0)
    anc_bit = n_sys + 1
    v = psi.reshape(1 << (n - 1 - anc_bit), 2, 1 << anc_bit)
    low0 = v[:, 0, :].reshape(-1, 2)           # anc=0, columns = time {0,1}
    low1 = v[:, 1, :].reshape(-1, 2)           # anc=1
    tmp = low0[:, 0].copy()                    # swap anc for time==0
    low0[:, 0] = low1[:, 0]
    low1[:, 0] = tmp

    # uncompute: QFT(1)=H(time); controlled-U^dagger; H(time)
    _h_on_bit(psi, 0, n)
    _controlled_U(psi, pairs, -theta, -phase, n_sys)
    _h_on_bit(psi, 0, n)

    # readout (== OneBQF.get_solution_from_statevector): sum |.|^2 over time, anc=1
    N_sys = 1 << n_sys
    psi3 = psi.reshape(2, N_sys, 2)            # [anc, sys, time]
    prob = np.sum(np.abs(psi3[1]) ** 2, axis=1)
    success = float(prob.sum())
    if success == 0.0:
        return np.zeros(d, dtype=np.float32), 0.0, n_sys
    prob = prob / success
    sol = np.sqrt(prob)
    nrm = np.linalg.norm(sol)
    if nrm > 0:
        sol = sol / nrm
    return sol[:d].astype(np.float32), success, n_sys
