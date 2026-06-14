#!/usr/bin/env python3
"""Probe where the 1BQF host RAM goes: build vs transpile vs Aer ingestion+run.
Prints staged peak RSS. Usage: _probe_aer_mem.py T opt_level [aer_max_mb]"""
import sys, os, time, resource, numpy as np
from pathlib import Path
_SHARED = Path(__file__).resolve().parent
sys.path.insert(0, str(_SHARED))
import helpers as H
from lhcb_velo_toy.solvers import SimpleHamiltonianFast
from lhcb_velo_toy.solvers.quantum.one_bit_hhl import OneBitHHL

T = int(sys.argv[1]); opt = int(sys.argv[2])
aer_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 0

def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
def rss():
    with open('/proc/self/statm') as f: return int(f.read().split()[1])*4096/1e6

eps = H.compute_epsilon(0.0, 1e-4)
ev = H.safe_generate(T, seed=T*100+200000, geom=H.make_geometry(),
                     measurement_error=0.0, collision_noise=1e-4)
ham = SimpleHamiltonianFast(epsilon=eps, gamma=3.0, delta=1.0)
ham.construct_segments(ev, materialize_segments=False)
A, b = ham.construct_hamiltonian(ev, convolution=False)
print(f"[T={T} opt={opt} aer_max_mb={aer_mb}] after A-build: rss={rss():.0f} peak={peak():.0f} MB", flush=True)

from qiskit_aer import AerSimulator
from qiskit import transpile
hhl = OneBitHHL(A, b, num_time_qubits=1, shots=1, debug=False, readout='statevector')
hhl.build_circuit()
print(f"  after build_circuit ({hhl.circuit.size()} gates): rss={rss():.0f} peak={peak():.0f} MB", flush=True)

aer_kw = dict(method='statevector', fusion_enable=False, device='CPU', max_parallel_threads=8)
if aer_mb > 0: aer_kw['max_memory_mb'] = aer_mb
sim = AerSimulator(**aer_kw)
tqc = transpile(hhl.circuit, sim, optimization_level=opt)
print(f"  after transpile opt{opt} ({tqc.size()} gates): rss={rss():.0f} peak={peak():.0f} MB", flush=True)

t0 = time.time()
job = sim.run(tqc, shots=1)
# poll the result; ingestion peak should appear quickly
sv = np.asarray(job.result().data(0)['statevector'])
print(f"  after sim.run done (t={time.time()-t0:.0f}s): rss={rss():.0f} peak={peak():.0f} MB", flush=True)
sol, success = hhl.get_solution_from_statevector(sv)
print(f"  success={success:.5f}  PEAK_TOTAL={peak():.0f} MB", flush=True)
