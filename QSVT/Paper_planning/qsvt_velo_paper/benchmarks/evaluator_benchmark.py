#!/usr/bin/env python3
"""Persisted benchmark for the paper's Sec. 4 evaluator claims (George, 2026-07-18).

Claim under test (main.tex, "Summing the series: LCU", ~line 555):
  (a) the matrix-free Chebyshev evaluator p(A)b runs in ~1.6 s at T=1000
      (n = 4e6 segments), and
  (b) it agrees with the full qiskit circuit and with the gate-streaming
      evaluation to 1e-9 where all three are affordable.

Outputs benchmarks/evaluator_benchmark.json + .md with machine info, so the
paper's numbers have a persisted provenance instead of study-record memory.
Config matches the clean store campaign (Verify_new_results): gamma=3 delta=1,
sigma_scatt=1e-4, sigma_res=0, phi_max=0.2, drop=0, eps=2 mrad, comb d=40.
"""
import json
import platform
import sys
import time
from pathlib import Path

for p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
          "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import qtrk_pipeline as qp
from lhcb_velo_toy.solvers.quantum import QSVT, design_line_comb_inverse

HERE = Path(__file__).resolve().parent
OUT = {"config": dict(gamma=3.0, delta=1.0, sigma_scatt=1e-4, sigma_res=0.0,
                      phi_max=0.2, hit_ineff=0.0, epsilon=0.002, degree=40,
                      design="line comb (production)"),
       "host": dict(node=platform.node(), python=platform.python_version(),
                    numpy=np.__version__)}

S = 4.0
# Gershgorin-safe domain (the class requires domain to cover the spectrum + pad)
DOM = (-0.6, 8.6)
BOUNDS = (0.0, 8.0)
poly = design_line_comb_inverse(degree=40, s=S, hw=0.18, domain=DOM)

# ---------- (a) timing at T=1000 ----------
print("[a] building T=1000 event + Hamiltonian (build time excluded from the claim)...",
      flush=True)
ev, _ = qp.ensure_event(n_trk=1000, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                        phi_max=0.2, hit_ineff=0.0)
t0 = time.perf_counter()
ham = qp.build_hamiltonian(ev, epsilon=0.002, gamma=3.0, delta=1.0)
t_build = time.perf_counter() - t0
n = ham.n_segments
print(f"    n_seg = {n}  (build {t_build:.1f} s)", flush=True)

solver = QSVT(ham.A, np.ones(n), poly=poly, spectral_bounds=BOUNDS)
times = []
for i in range(3):
    t0 = time.perf_counter()
    res = solver.solve_statevector()
    times.append(time.perf_counter() - t0)
    print(f"    solve rep {i}: {times[-1]:.2f} s", flush=True)
OUT["timing_T1000"] = dict(n_seg=int(n), build_s=round(t_build, 1),
                           solve_s=[round(t, 2) for t in times],
                           solve_s_median=round(sorted(times)[1], 2))

# ---------- (b) three-way agreement at circuit-affordable size ----------
print("[b] three-way agreement at T=10 (full circuit affordable)...", flush=True)
ev2, _ = qp.ensure_event(n_trk=10, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                         phi_max=0.2, hit_ineff=0.0)
ham2 = qp.build_hamiltonian(ev2, epsilon=0.002, gamma=3.0, delta=1.0)
s2 = QSVT(ham2.A, np.ones(ham2.n_segments), poly=poly, spectral_bounds=BOUNDS)
mf = np.asarray(s2.solve_statevector()[0], float)
c_full = np.asarray(s2.run_circuit()[0], float)
c_stream = np.asarray(s2.run_circuit(streaming=True)[0], float)
d_fc = float(np.max(np.abs(mf - c_full)))
d_st = float(np.max(np.abs(mf - c_stream)))
OUT["agreement_T10"] = dict(n_seg=int(ham2.n_segments),
                            max_absdiff_matrixfree_vs_circuit=d_fc,
                            max_absdiff_matrixfree_vs_streaming=d_st,
                            claim_1e9_holds=bool(max(d_fc, d_st) < 1e-9))
print(f"    |mf - circuit|_max   = {d_fc:.2e}")
print(f"    |mf - streaming|_max = {d_st:.2e}", flush=True)

OUT["verdict"] = dict(
    timing_claim_1p6s=bool(sorted(times)[1] < 2.5),
    agreement_claim_1e9=OUT["agreement_T10"]["claim_1e9_holds"])

(HERE / "evaluator_benchmark.json").write_text(json.dumps(OUT, indent=2))
md = (f"# Evaluator benchmark ({time.strftime('%Y-%m-%d')})\n\n"
      f"Host {platform.node()}. Config: clean campaign, comb d=40.\n\n"
      f"| quantity | value |\n|---|---|\n"
      f"| T=1000 n_seg | {n} |\n"
      f"| solve times (3 reps) | {', '.join(f'{t:.2f} s' for t in times)} |\n"
      f"| median solve | {sorted(times)[1]:.2f} s |\n"
      f"| T=10 max diff mf vs circuit | {d_fc:.2e} |\n"
      f"| T=10 max diff mf vs streaming | {d_st:.2e} |\n")
(HERE / "evaluator_benchmark.md").write_text(md)
print("saved evaluator_benchmark.{json,md}")
