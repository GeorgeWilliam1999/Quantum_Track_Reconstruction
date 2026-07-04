#!/usr/bin/env python3
"""DP terms: 1BQF resource cost (Q2) and matrix sparsity (Q3). Step kernel.

Q2 — qubits & 2-qubit gates: the 1BQF circuit applies one n_sys-controlled RX
per interaction PAIR (off-diagonal nonzero pair of A), so the 2-qubit gate
count scales with the pair count; qubit count depends only on n_seg. We build
and transpile the REAL OneBitHHL circuit (basis cx/rz/sx/x) at small T to
measure cx-per-pair, verify the pair-proportionality on the occupancy system
at T=10, and project the extra cost at analysis T from the measured pair
counts. Configs are term-presence only (β/α values change weights, not nnz).

Q3 — sparsity: nnz/fill of A(β,α) vs the base across T, including whether the
step invariant (nnz < 5·n_seg) survives.

Outputs results/dp_costs.csv (+ printed tables).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402
import dp_terms  # noqa: E402

HERE = Path(__file__).resolve().parent
(HERE / "results").mkdir(exist_ok=True)

NOISE = dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01)
GAMMA, DELTA = 3.0, 1.0
CONFIGS = [("base", 0.0, 0.0), ("fork", 1.0, 0.0), ("occ", 0.0, 0.3), ("both", 1.0, 0.3)]
T_SPARSITY = [10, 20, 50, 100, 200]
T_GATES = [10, 20]              # real transpiled circuits (base+fork); occ at T=10 only
OCC_PAIR_CAP = 6000             # skip transpile beyond this many pairs (hours)


def build(T):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, **NOISE)
    from lhcb_velo_toy.analysis import compute_epsilon
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=GAMMA, delta=DELTA)
    return ham


def gate_count(A, b):
    from qiskit import transpile
    from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL
    hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False)
    hhl.build_circuit()
    qc = hhl.circuit.remove_final_measurements(inplace=False)
    t0 = time.time()
    tqc = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    ops = tqc.count_ops()
    return dict(n_qubits=int(tqc.num_qubits), cx=int(ops.get("cx", 0)),
                total_ops=int(sum(ops.values())), t_transpile=time.time() - t0)


def main():
    rows = []
    for T in T_SPARSITY:
        ham = build(T)
        n = int(ham.n_segments)
        for name, beta, alpha in CONFIGS:
            A, b, _, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                               gamma=GAMMA, delta=DELTA)
            offdiag = int(A.nnz) - n
            pairs = offdiag // 2
            row = dict(T=T, config=name, n_seg=n, A_nnz=int(A.nnz),
                       fill=A.nnz / n, pairs=pairs,
                       pairs_vs_base=np.nan, sparse_invariant=bool(A.nnz < 5 * n),
                       **{k: info[k] for k in ("Bfork_nnz", "Ball_nnz")})
            if T in T_GATES and (name in ("base", "fork") or pairs <= OCC_PAIR_CAP):
                g = gate_count(A, b)
                row.update(g)
                row["cx_per_pair"] = g["cx"] / max(pairs, 1)
                print(f"[gates] T={T} {name}: qubits={g['n_qubits']} pairs={pairs} "
                      f"cx={g['cx']:,} ({row['cx_per_pair']:.0f}/pair) "
                      f"[{g['t_transpile']:.0f}s]", flush=True)
            rows.append(row)
        base_pairs = next(r["pairs"] for r in rows if r["T"] == T and r["config"] == "base")
        for r in rows:
            if r["T"] == T:
                r["pairs_vs_base"] = r["pairs"] / max(base_pairs, 1)
        print(f"[sparsity] T={T} n_seg={n}: " +
              "  ".join(f"{r['config']}: fill={r['fill']:.2f} ({r['pairs_vs_base']:.1f}x pairs)"
                        for r in rows if r["T"] == T), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "dp_costs.csv", index=False)
    print(f"[csv] {len(df)} rows -> results/dp_costs.csv")

    # projection: extra cx at analysis T from measured cx_per_pair (largest measured T)
    cal = df.dropna(subset=["cx_per_pair"]) if "cx_per_pair" in df else pd.DataFrame()
    if len(cal):
        cpp = float(cal[cal["T"] == cal["T"].max()].cx_per_pair.mean())
        print(f"\n[projection] using cx_per_pair={cpp:.0f} (measured at T={int(cal['T'].max())}; "
              f"grows ~linearly with n_sys — lower bound at larger T):")
        for T in [100, 200]:
            sub = df[df["T"] == T]
            bp = int(sub[sub.config == "base"].pairs.iloc[0])
            for name in ("fork", "occ", "both"):
                p = int(sub[sub.config == name].pairs.iloc[0])
                print(f"  T={T} {name}: +{p - bp:,} pairs -> ~+{(p - bp) * cpp:,.0f} cx "
                      f"({p / bp:.1f}x the base 2q-gate cost)", flush=True)


if __name__ == "__main__":
    main()
