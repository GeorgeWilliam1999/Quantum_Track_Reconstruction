#!/usr/bin/env python3
"""Gate-level cost of the one-bit circuit with and without the prune.

The one-bit circuit applies one multi-controlled rotation per compatible
pair, controlled on the time qubit and on every other system qubit, so the
cost of each pair gate grows with the register width even though the number
of pair gates does not change under the prune.  This script builds the
package circuit (``OneBQF``, statevector readout) for small events, full and
pruned, transpiles both to the {rz, sx, x, cx} basis and records qubits,
pair count, two-qubit gate count and depth.  Also the QSVT resource model
(qubits and calls for the LCU and phase-sequence realisations) for the same
systems, from ``QSVT.resources``.

Outputs: outputs/circuit_cost.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qiskit import transpile

from common import (OUT, S, bounds_for, build_full, epsilon_for, load_event, pruned_copy)
from lhcb_velo_toy.solvers.quantum import QSVT, OneBQF

TS = [2, 3, 4, 5, 6, 8, 10, 12, 15]
CFG = "paper"
BASIS = ["rz", "sx", "x", "cx"]


def cost(ham, variant, T):
    A, b = ham.A, np.asarray(ham.b, float)
    t0 = time.time()
    c = OneBQF(A, b, readout="sampling")
    qc = c.build_circuit().remove_final_measurements(inplace=False)
    tq = transpile(qc, basis_gates=BASIS, optimization_level=1)
    ops = tq.count_ops()
    q = QSVT.comb(A, b, spectral_bounds=bounds_for(ham))
    rl, rp = q.resources("lcu"), q.resources("phase")
    return dict(T=T, variant=variant, n=int(A.shape[0]), n_pairs=len(c.interaction_pairs),
                onebit_qubits=tq.num_qubits, onebit_n_sys=c.num_system_qubits,
                onebit_cx=int(ops.get("cx", 0)), onebit_gates=int(sum(ops.values())),
                onebit_depth=int(tq.depth()),
                qsvt_qubits_lcu=int(rl["total_qubits"]), qsvt_calls_lcu=int(rl["calls_per_shot"]),
                qsvt_qubits_phase=int(rp["total_qubits"]), qsvt_calls_phase=int(rp["calls_per_shot"]),
                t_transpile=time.time() - t0)


def main():
    eps = epsilon_for(CFG)
    rows = []
    for T in TS:
        ev, _ = load_event(CFG, T, 0)
        ham = build_full(ev, eps)
        for variant in ("full", "deg1"):
            h = ham if variant == "full" else pruned_copy(ham)
            r = cost(h, variant, T)
            rows.append(r)
            print(f"T={T:3d} {variant:5s} n={r['n']:5d} pairs={r['n_pairs']:4d} "
                  f"qubits={r['onebit_qubits']:2d} cx={r['onebit_cx']:7d} depth={r['onebit_depth']:7d} "
                  f"qsvt lcu {r['qsvt_qubits_lcu']}q / phase {r['qsvt_qubits_phase']}q  "
                  f"({r['t_transpile']:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "circuit_cost.csv", index=False)


if __name__ == "__main__":
    main()
