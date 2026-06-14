#!/usr/bin/env python3
"""Validate a matrix-free OneBQF statevector emulator against the live Aer path.

The emulator applies the SAME high-level circuit (helpers' OneBitHHL.build_circuit)
gate-by-gate to a numpy statevector, WITHOUT transpiling to basis gates and WITHOUT
Aer — so memory is the 2^n statevector + O(1), not Aer's ~7 KB x (millions of gates).
Multi-controlled gates are applied by index arithmetic (control mask + base 2x2),
never materialising a 2^k matrix.

Bit-identity is expected up to float reassociation (~1e-10): same unitary, same order.

Usage: python _validate_matrixfree.py T
"""
import sys, time, resource
import numpy as np
from pathlib import Path
_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))
import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator

def gmat(op):
    try:
        return op.to_matrix()
    except Exception:
        return Operator(op).data

T = int(sys.argv[1])
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

eps = H.compute_epsilon(0.0, 1e-4)
ev = H.safe_generate(T, seed=T*100+200000, geom=H.make_geometry(),
                     measurement_error=0.0, collision_noise=1e-4)
ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
ham.construct_segments(ev, materialize_segments=False)
A, b = ham.construct_hamiltonian(ev, convolution=False)


def emulate(hhl):
    """Apply hhl.circuit to a fresh |0..0> numpy statevector, return the statevector."""
    qc = hhl.circuit
    n = qc.num_qubits
    bit = {q: qc.find_bit(q).index for q in qc.qubits}
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    idx = np.arange(1 << n, dtype=np.int64)

    def apply_1q(M, t):
        b0 = ((idx >> t) & 1) == 0
        i0 = idx[b0]; i1 = i0 | (1 << t)
        a = psi[i0]; c = psi[i1]
        psi[i0] = M[0, 0]*a + M[0, 1]*c
        psi[i1] = M[1, 0]*a + M[1, 1]*c

    def apply_ctrl_1q(M, ctrls, ctrl_state, t):
        mask = np.ones(1 << n, dtype=bool)
        for k, q in enumerate(ctrls):
            want = (ctrl_state >> k) & 1
            mask &= (((idx >> q) & 1) == want)
        b0 = mask & (((idx >> t) & 1) == 0)
        i0 = idx[b0]; i1 = i0 | (1 << t)
        a = psi[i0]; c = psi[i1]
        psi[i0] = M[0, 0]*a + M[0, 1]*c
        psi[i1] = M[1, 0]*a + M[1, 1]*c

    def apply_2q(M, q0, q1):
        # M acts on (q1,q0) little-endian 2-qubit subspace; apply per basis pairing
        for s in range(4):
            pass  # unused; 2q handled generically below

    for inst in qc.data:
        op = inst.operation
        name = op.name
        if name in ("save_statevector", "barrier"):
            continue
        qubits = [bit[q] for q in inst.qubits]
        if isinstance(op, ControlledGate):
            base = op.base_gate
            ctrl_state = op.ctrl_state
            ncc = op.num_ctrl_qubits
            ctrls = qubits[:ncc]; targs = qubits[ncc:]
            if base.num_qubits == 1:
                apply_ctrl_1q(gmat(base), ctrls, ctrl_state, targs[0])
            else:  # e.g. controlled 2q — generic fallback
                _apply_general(psi, gmat(op), qubits, n)
        elif op.num_qubits == 1:
            apply_1q(gmat(op), qubits[0])
        elif op.num_qubits == 2:
            _apply_general(psi, gmat(op), qubits, n)
        else:
            _apply_general(psi, gmat(op), qubits, n)
    return psi


def _apply_general(psi, M, qubits, n):
    """Apply a (2^k x 2^k) gate matrix M on the given qubit indices.

    np.reshape([2]*n): axis j holds bit (n-1-j) (axis 0 = MSB). Qiskit's gate matrix
    M is little-endian in its own qubit list: qubits[0] is the LSB of the gate index.
    So move axes so qubits[-1] becomes the front (MSB of the k-block) and qubits[0]
    the last front axis (LSB), matching M's index convention.
    """
    k = len(qubits)
    tens = psi.reshape([2]*n)
    axes = [n-1-q for q in reversed(qubits)]   # front axis 0 = qubits[-1] (MSB) ... last = qubits[0] (LSB)
    tens = np.moveaxis(tens, axes, range(k))
    shp = tens.shape
    tens = tens.reshape(2**k, -1)
    tens = M @ tens
    tens = tens.reshape((2,)*k + shp[k:])
    tens = np.moveaxis(tens, range(k), axes)
    psi[:] = tens.reshape(-1)


# --- matrix-free emulation ---
hhl_mf = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False, readout='statevector')
hhl_mf.build_circuit()
t0 = time.time()
sv_mf = emulate(hhl_mf)
t_mf = time.time() - t0
sol_mf, succ_mf = hhl_mf.get_solution_from_statevector(sv_mf)
peak_mf = peak()

# --- reference Aer path (the live helpers code); optional (slow at high T) ---
do_aer = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
if do_aer:
    t0 = time.time()
    sol_aer, succ_aer, n_sys = H.solve_quantum_statevector(A, b, device='CPU')
    t_aer = time.time() - t0
    m = min(len(sol_mf), len(sol_aer))
    a = np.asarray(sol_mf[:m], float); c = np.asarray(sol_aer[:m], float)
    cos = float(np.dot(a, c) / (np.linalg.norm(a)*np.linalg.norm(c) + 1e-30))
    maxabs = float(np.max(np.abs(a - c)))
    print(f"T={T} n_qubits={hhl_mf.num_system_qubits+2} "
          f"succ_mf={succ_mf:.6f} succ_aer={succ_aer:.6f} d_succ={abs(succ_mf-succ_aer):.2e} "
          f"cos={cos:.10f} max|dsol|={maxabs:.2e} "
          f"t_mf={t_mf:.1f}s t_aer={t_aer:.1f}s peak_mf={peak_mf:.0f}MB")
else:
    print(f"T={T} n_qubits={hhl_mf.num_system_qubits+2} succ_mf={succ_mf:.6f} "
          f"t_mf={t_mf:.1f}s peak_mf={peak_mf:.0f}MB  (matrix-free only)")
