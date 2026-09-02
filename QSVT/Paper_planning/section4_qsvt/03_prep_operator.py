#!/usr/bin/env python3
"""The PREP operator of the LCU build (paper section 3.1, figure after fig. 9).

PREP loads the Chebyshev coefficients of the production comb onto the index
register: |0>^m  ->  sum_k sqrt(|c_k|/||c||_1) |k>, with the signs of c_k a
separate diagonal.  This script

  1. builds the production comb  p = design_line_comb_inverse(d=40, s=4,
     hw=0.18, domain=(0.2,7.8))  and takes its Chebyshev coefficients c_k;
  2. builds the Moettoenen binary tree of partial weights and the uniformly
     controlled R_y angles that walk down it;
  3. VALIDATES the cascade by dense simulation against the target amplitudes;
  4. counts the rotations and CNOTs actually required once the 23 padding
     amplitudes (k = 41..63) are taken into account, and cross-checks the CX
     count against a transpiled qiskit StatePreparation;
  5. reduces the sign pattern sgn(c_k) to its algebraic normal form, which is
     the gate content of the sign diagonal.

Output: outputs/03_prep_operator.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

for _p in ("/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",):
    if _p not in sys.path: sys.path.insert(0, _p)
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse

HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
S, DEGREE, HW, DOMAIN = 4.0, 40, 0.18, (0.2, 7.8)


def tree_angles(v, m):
    """Uniformly controlled R_y angles for a REAL target vector v (MSB-first).

    Bottom-up: a(leaf) = v_k, a(node) = hypot(a_left, a_right) >= 0, and the
    node's rotation is theta = 2 atan2(a_right, a_left).  Then the node passes
    a(node) cos(theta/2) = a_left to its left child and a(node) sin(theta/2) =
    a_right to its right child, signs included.  A real 2^m-vector carries
    2^m - 1 relative signs and the tree has exactly 2^m - 1 angles, so allowing
    theta in (-2pi, 2pi] costs nothing and no sign diagonal is needed.
    Returns angles[j][i] and the signed subtree amplitudes A[j][i]."""
    A = [None] * (m + 1)
    A[m] = np.asarray(v, float).copy()
    for j in range(m - 1, -1, -1):
        lo, hi = A[j + 1][0::2], A[j + 1][1::2]
        A[j] = np.hypot(lo, hi)
    ang = []
    for j in range(m):
        a = np.zeros(2 ** j)
        for i in range(2 ** j):
            a[i] = 0.0 if A[j][i] == 0 else 2 * math.atan2(A[j + 1][2 * i + 1], A[j + 1][2 * i])
        ang.append(a)
    return ang, A


def apply_cascade(ang, m):
    """Dense simulation of the cascade on |0>^m, MSB-first ordering."""
    psi = np.zeros(2 ** m); psi[0] = 1.0
    for j, a in enumerate(ang):
        new = np.zeros_like(psi)
        blk = 2 ** (m - j - 1)                  # size of a subtree at level j+1
        for i in range(2 ** j):                 # prefix i selects the rotation
            c, s = math.cos(a[i] / 2), math.sin(a[i] / 2)
            lo, hi = 2 * i * blk, (2 * i + 1) * blk
            src = psi[lo:lo + blk]              # amplitude currently on prefix i
            new[lo:lo + blk] += c * src
            new[hi:hi + blk] += s * src
        psi = new
    return psi


def anf(bits, m):
    """Algebraic normal form over GF(2) by the Moebius transform.
    bits[k] in {0,1} indexed by k with bit 0 = MSB of the register label."""
    f = bits.copy().astype(np.uint8)
    for j in range(m):
        step = 1 << j
        for k in range(2 ** m):
            if k & step:
                f[k] ^= f[k ^ step]
    return f  # f[mask] = 1 means the monomial prod_{bits in mask} x_i is present


def main():
    m = math.ceil(math.log2(DEGREE + 1))
    N = 2 ** m
    p = design_line_comb_inverse(degree=DEGREE, s=S, hw=HW, domain=DOMAIN)
    c = np.zeros(N); c[:DEGREE + 1] = np.asarray(p.coef, float)
    l1 = float(np.abs(c).sum())
    w = np.abs(c) / l1
    target = np.sqrt(w)

    signed = np.sign(c) * target      # what the unpreparation must load
    signed[c == 0] = target[c == 0]

    ang, P = tree_angles(target, m)                 # PREP  (non-negative)
    angS, PS = tree_angles(signed, m)               # PREP with the signs
    psi = apply_cascade(ang, m); psiS = apply_cascade(angS, m)
    err = float(np.max(np.abs(psi - target)))
    errS = float(np.max(np.abs(psiS - signed)))

    # gate content: a rotation is needed only where the subtree carries weight
    # and the split is non-trivial (angle not 0 and not pi means both children live)
    live = [int(np.sum(np.abs(P[j]) > 0)) for j in range(m)]  # nodes reached
    nontrivial = [int(np.sum((np.abs(P[j]) > 0) & (np.abs(ang[j]) > 1e-12) & (np.abs(ang[j] - math.pi) > 1e-12)))
                  for j in range(m)]
    ry_full = N - 1                                           # dense Moettoenen
    cx_full = N - 2
    ry_live = int(sum(live))
    # CNOTs: a uniformly controlled R_y on j controls costs 2^j CX; the last one
    # of the cascade can be dropped, hence N-2 for the dense case.
    cx_live = sum(2 ** j for j in range(m) if live[j] > 0)
    cx_live = max(0, cx_live - 1)

    # cross-check against qiskit's own state preparation
    qk = {}
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import StatePreparation
        from qiskit.quantum_info import Statevector
        qc = QuantumCircuit(m)
        # qiskit is little-endian: |k> has bit 0 = qubit 0 = LSB.  Our label k
        # already is the integer, so pass the amplitude vector directly.
        qc.append(StatePreparation(target.astype(complex)), range(m))
        sv = np.asarray(Statevector(qc).data).real
        qk["statevector_err"] = float(np.max(np.abs(np.abs(sv) - target)))
        tq = transpile(qc, basis_gates=["cx", "ry", "rz", "x", "h"], optimization_level=3)
        ops = dict(tq.count_ops())
        qk["cx"] = int(ops.get("cx", 0)); qk["1q"] = int(sum(v for k, v in ops.items() if k != "cx"))
        qk["depth"] = int(tq.depth())
        qc2 = QuantumCircuit(m)
        qc2.append(StatePreparation(signed.astype(complex)), range(m))
        sv2 = np.asarray(Statevector(qc2).data).real
        qk["signed_statevector_err"] = float(np.max(np.abs(np.abs(sv2) - np.abs(signed))))
        t2 = transpile(qc2, basis_gates=["cx", "ry", "rz", "x", "h"], optimization_level=3)
        o2 = dict(t2.count_ops())
        qk["signed_cx"] = int(o2.get("cx", 0)); qk["signed_depth"] = int(t2.depth())
    except Exception as e:               # pragma: no cover
        qk["error"] = repr(e)

    # the sign diagonal
    neg = (c < 0).astype(np.uint8)
    runs = []
    k = 0
    while k < DEGREE + 1:
        j = k
        while j + 1 <= DEGREE and neg[j + 1] == neg[k]: j += 1
        runs.append((int(k), int(j), int(neg[k]))); k = j + 1
    a = anf(neg, m)
    monos = [bin(mask).count("1") for mask in range(N) if a[mask]]
    from collections import Counter
    mono_deg = dict(sorted(Counter(monos).items()))

    res = dict(
        degree=DEGREE, m=m, n_index_states=N, n_coeff=DEGREE + 1, n_padding=N - DEGREE - 1,
        l1=l1, cmax=float(np.abs(c).max()), c=c[:DEGREE + 1].round(12).tolist(),
        amp=target.round(12).tolist(),
        angles_deg=[np.degrees(a_).round(6).tolist() for a_ in ang],
        subtree_weight=[np.asarray(P[j]).round(12).tolist() for j in range(m + 1)],
        cascade_max_err=err, cascade_signed_max_err=errS,
        angles_signed_deg=[np.degrees(a_).round(6).tolist() for a_ in angS],
        live_nodes_per_level=live, nontrivial_per_level=nontrivial,
        ry_dense=ry_full, cx_dense=cx_full, ry_live=ry_live, cx_live=cx_live,
        qiskit=qk,
        sign_runs=runs, n_negative=int(neg.sum()),
        sign_anf_monomials=int(a.sum()), sign_anf_by_degree=mono_deg,
    )
    json.dump(res, open(OUT / "03_prep_operator.json", "w"), indent=1)

    print(f"degree {DEGREE}  m={m}  states {N}  coefficients {DEGREE+1}  padding zeros {N-DEGREE-1}")
    print(f"||c||_1 = {l1:.6f}   max|c_k| = {np.abs(c).max():.6f}")
    print(f"cascade vs target: max |dA| = {err:.3e}   (signed prep: {errS:.3e})")
    print(f"level:      " + "  ".join(f"{j}" for j in range(m)))
    print(f"live nodes: " + "  ".join(f"{v}" for v in live))
    print(f"non-trivial:" + "  ".join(f"{v}" for v in nontrivial))
    print(f"dense Moettoenen: {ry_full} R_y, {cx_full} CX   |   live-tree: {ry_live} R_y, {cx_live} CX")
    print(f"qiskit StatePreparation: {qk}")
    print(f"negative coefficients: {int(neg.sum())} of {DEGREE+1}; sign runs {runs}")
    print(f"sign diagonal ANF: {int(a.sum())} monomials by degree {mono_deg}")
    print("first angles (deg): level0", np.degrees(ang[0]).round(3),
          " level1", np.degrees(ang[1]).round(3))
    print("signed  angles (deg): level0", np.degrees(angS[0]).round(3),
          " level1", np.degrees(angS[1]).round(3))
    print("angle range: PREP [%.1f, %.1f] deg, signed [%.1f, %.1f] deg" % (
        min(np.degrees(a).min() for a in ang), max(np.degrees(a).max() for a in ang),
        min(np.degrees(a).min() for a in angS), max(np.degrees(a).max() for a in angS)))
    print("[saved]", OUT / "03_prep_operator.json")


if __name__ == "__main__":
    main()
