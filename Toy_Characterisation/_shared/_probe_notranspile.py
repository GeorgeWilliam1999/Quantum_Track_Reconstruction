#!/usr/bin/env python3
"""Compare Aer host RAM for three circuit-prep paths at fixed T:
   path=opt1 : transpile(optimization_level=1) then run  (current helpers path)
   path=opt0 : transpile(optimization_level=0) then run
   path=none : run the untranspiled circuit directly (let Aer decompose)
Prints peak RSS + success. Usage: _probe_notranspile.py T path"""
import sys, time, resource, numpy as np
from pathlib import Path
_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))
import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL
from qiskit_aer import AerSimulator
from qiskit import transpile

T = int(sys.argv[1]); path = sys.argv[2]
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0

eps = H.compute_epsilon(0.0, 1e-4)
ev = H.safe_generate(T, seed=T*100+200000, geom=H.make_geometry(),
                     measurement_error=0.0, collision_noise=1e-4)
ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
ham.construct_segments(ev, materialize_segments=False)
A, b = ham.construct_hamiltonian(ev, convolution=False)
hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False, readout='statevector')
hhl.build_circuit()
sim = AerSimulator(method='statevector', fusion_enable=False, device='CPU', max_parallel_threads=8)

t0 = time.time()
if path == 'none':
    circ = hhl.circuit
    n_gates = circ.size()
else:
    lvl = 1 if path == 'opt1' else 0
    circ = transpile(hhl.circuit, sim, optimization_level=lvl)
    n_gates = circ.size()
t_prep = time.time() - t0
peak_after_prep = peak()

t0 = time.time()
job = sim.run(circ, shots=1)
sv = np.asarray(job.result().data(0)['statevector'])
t_run = time.time() - t0
sol, success = hhl.get_solution_from_statevector(sv)
print(f"T={T} path={path:5s} gates_fed={n_gates:>9d} t_prep={t_prep:6.1f}s "
      f"t_run={t_run:6.1f}s peak_after_prep={peak_after_prep:7.0f}MB "
      f"PEAK={peak():7.0f}MB success={success:.6f}")
