#!/usr/bin/env python3
"""Layer-B profiler: 1BQF circuit build + transpile RAM/time vs optimization_level.

Run ONE (T, opt_level) per process so getrusage peak RSS is clean.
Usage: python _profile_layerB.py T opt_level [sigma_res] [sigma_scatt] [do_sim]
  do_sim=1 -> also run the AerSimulator statevector (expensive, small T only).
Emits one JSON line on stdout.
"""
import sys, os, time, json, resource, gc
import numpy as np
from pathlib import Path

_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))

import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL

T = int(sys.argv[1])
opt_level = int(sys.argv[2])
sigma_res = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
sigma_scatt = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
do_sim = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

eps = H.compute_epsilon(sigma_res, sigma_scatt)
geom = H.make_geometry()

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

ev = H.safe_generate(T, seed=T * 100 + 200000, geom=geom,
                     measurement_error=sigma_res, collision_noise=sigma_scatt)
ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
ham.construct_segments(ev, materialize_segments=False)
A, b = ham.construct_hamiltonian(ev, convolution=False)
n_seg = ham.n_segments

# --- build the 1BQF circuit (statevector readout) ---
rss_pre_build = rss_mb()
t0 = time.time()
hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False, readout='statevector')
hhl.build_circuit()
t_build = time.time() - t0
rss_post_build = rss_mb()

n_sys = hhl.num_system_qubits
n_qubits = n_sys + 2          # time(1) + system + ancilla(1)
n_pairs = len(hhl.interaction_pairs)
circ_size = hhl.circuit.size()
circ_depth = hhl.circuit.depth()
# statevector memory: 2^n_qubits complex128 (16 bytes)
sv_MB = (2 ** n_qubits) * 16 / 1e6

# --- transpile at the requested optimization_level ---
from qiskit_aer import AerSimulator
from qiskit import transpile
sim = AerSimulator(method='statevector', fusion_enable=False,
                   device='CPU', max_parallel_threads=4)
gc.collect()
rss_pre_tr = rss_mb()
t0 = time.time()
tqc = transpile(hhl.circuit, sim, optimization_level=opt_level)
t_transpile = time.time() - t0
rss_post_tr = rss_mb()
tr_size = tqc.size()
tr_depth = tqc.depth()
# 2q-gate count of the transpiled circuit
ops = tqc.count_ops()
n_cx = int(ops.get('cx', 0) + ops.get('cz', 0) + ops.get('ecr', 0))

sim_t = sv_branch_success = None
if do_sim:
    t0 = time.time()
    job = sim.run(tqc, shots=1)
    sv = np.asarray(job.result().data(0)['statevector'])
    sim_t = time.time() - t0
    sol, success = hhl.get_solution_from_statevector(sv)
    sv_branch_success = float(success)
    np.save(f"/tmp/_lb_sol_T{T}_opt{opt_level}.npy", np.asarray(sol, dtype=np.float64))

out = dict(
    T=T, opt_level=opt_level, sigma_res=sigma_res, sigma_scatt=sigma_scatt, eps=eps,
    n_seg=n_seg, n_sys=n_sys, n_qubits=n_qubits, sv_MB=round(sv_MB, 2),
    n_interaction_pairs=n_pairs,
    circ_size=circ_size, circ_depth=circ_depth,
    t_build=round(t_build, 3),
    rss_pre_build_MB=round(rss_pre_build, 1), rss_post_build_MB=round(rss_post_build, 1),
    rss_pre_transpile_MB=round(rss_pre_tr, 1), rss_post_transpile_MB=round(rss_post_tr, 1),
    transpile_dRSS_MB=round(rss_post_tr - rss_pre_tr, 1),
    t_transpile=round(t_transpile, 3),
    tr_size=tr_size, tr_depth=tr_depth, tr_n2q=n_cx,
    sim_t=round(sim_t, 2) if sim_t is not None else None,
    success=round(sv_branch_success, 5) if sv_branch_success is not None else None,
    peak_rss_MB=round(rss_mb(), 1),
)
print(json.dumps(out))
