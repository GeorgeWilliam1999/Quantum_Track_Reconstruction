#!/usr/bin/env python3
"""End-to-end self-test for qtrk_pipeline. Run inside Q_env."""
from __future__ import annotations
import os, tempfile, numpy as np

os.environ.setdefault("QTRK_STORE", tempfile.mkdtemp(prefix="qtrk_selftest_"))
print(f"[selftest] QTRK_STORE = {os.environ['QTRK_STORE']}")

import qtrk_pipeline as qp
from lhcb_velo_toy.analysis import compute_epsilon

T, REP = 10, 0
SS, SR = 1e-4, 0.0
eps = compute_epsilon(SR, SS)

# 1) generate + save + reload
ev, ekey = qp.ensure_event(n_trk=T, rep=REP, sigma_scatt=SS, sigma_res=SR)
print(f"[1] event {ekey}: n_hits={ev.n_hits} n_tracks={ev.n_tracks}")
ev_re = qp.load_event(qp.event_path(ekey))
assert ev_re.n_hits == ev.n_hits and ev_re.n_tracks == ev.n_tracks

# 2) REPRODUCIBILITY: A from reloaded event == A from fresh event
ham_fresh = qp.build_hamiltonian(ev, epsilon=eps, kernel="step")
ham_re    = qp.build_hamiltonian(ev_re, epsilon=eps, kernel="step")
assert ham_fresh.n_segments == ham_re.n_segments, "n_seg mismatch"
assert ham_fresh.A.nnz == ham_re.A.nnz, "nnz mismatch"
dA = abs(ham_fresh.A - ham_re.A).max()
print(f"[2] step  n_seg={ham_re.n_segments} A_nnz={ham_re.A.nnz} "
      f"ratio={ham_re.A.nnz/ham_re.n_segments:.2f}  max|dA|={dA:.2e}")
assert dA == 0.0, "A NOT reproducible from reloaded event!"

# 3) classical solve + truth + metrics
truth = qp.truth_from_event(ev_re)
assert truth.size == ham_re.n_segments
sol_C, tC = qp.solve_classical(ham_re)
mC = qp.metrics_at(sol_C, truth, threshold=qp.ABS_THRESHOLD)
print(f"[3] classical t={tC:.3f}s  eff={mC['segment_efficiency']:.3f} "
      f"far={mC['segment_false_rate']:.3f} pur={mC['segment_purity']:.3f} "
      f"n_active={mC['n_active']} n_true={int(truth.sum())}")

# 4) quantum solve (CPU statevector) + rescale + metrics
qd = qp.solve_quantum(ham_re, device="CPU", readout="statevector")
mQ = qp.quantum_metrics(qd["sol"], sol_C, truth, threshold=qp.ABS_THRESHOLD)
print(f"[4] quantum n_qubits={qd['n_qubits']} P_anc={qd['P_anc']:.4f} "
      f"t={qd['t_solve']:.2f}s  eff={mQ['segment_efficiency']:.3f} "
      f"cos_QC={mQ['cos_QC']:.3f}")

# 5) solution store round-trip
sk = qp.sol_key(ekey, qp.ham_key(eps, "step", 1e-4, 3.0, 1.0, "formula"),
                "classical")
qp.save_solution(sk, sol_C, event_key_=ekey,
                 ham_key_=qp.ham_key(eps, "step", 1e-4, 3.0, 1.0, "formula"),
                 solver="classical", n_seg=ham_re.n_segments,
                 n_true=int(truth.sum()), epsilon=eps, kernel="step", t_solve=tC)
back = qp.load_solution(sk)
assert np.array_equal(back["sol"], sol_C.astype(np.float32))
assert back["n_seg"] == ham_re.n_segments
print(f"[5] solution store round-trip OK  dtype={back['sol'].dtype} "
      f"keys={sorted(k for k in back if k!='sol')}")

# 6) ERF path sanity at T=10
ham_erf = qp.build_hamiltonian(ev_re, epsilon=eps, kernel="erf", erf_sigma=1e-4)
print(f"[6] erf   n_seg={ham_erf.n_segments} A_nnz={ham_erf.A.nnz} "
      f"ratio={ham_erf.A.nnz/ham_erf.n_segments:.2f}")

# 7) manifest coverage + budget over all five studies
events, solves = qp.build_manifest(qp.standard_specs(), write=False)
import pandas as pd
def est_bytes(row):
    nseg = 4 * row.n_trk**2
    return nseg * 4  # float32 vector
solves = solves.copy()
solves["bytes"] = solves.apply(est_bytes, axis=1)
gb = solves["bytes"].sum() / 1e9
ev_gb = (events.n_trk.apply(lambda T: 5*T*28).sum()) / 1e9  # ~28 B/hit columns
print(f"[7] manifest: {len(events)} events, {len(solves)} solves")
print(f"    est solution storage ~ {gb:.1f} GB ; est event storage ~ {ev_gb:.2f} GB")
print(solves.groupby('solver').size().to_string())
print("ALL CHECKS PASSED")
