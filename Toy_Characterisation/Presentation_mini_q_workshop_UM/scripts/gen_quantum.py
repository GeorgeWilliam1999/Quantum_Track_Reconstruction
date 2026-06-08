"""
Fresh small/medium-T quantum (1BQF) sweep for the presentation.

The store's quantum coverage is mid-campaign (only T=200, step).  This script
regenerates a clean low-T quantum-vs-classical comparison from scratch via the
package, for BOTH kernels, so the deck's quantum story (fidelity vs T, erf vs
step, P_anc, quantum segment metrics) rests entirely on new data.

Outputs (assets/):
  quantum_sweep.csv     one row per (T, kernel, rep): classical+quantum metrics,
                        cos_QC, P_anc, n_qubits, t_q   (resumable / append-only)
  solvecs_T{T}.npz      raw classical + rescaled-quantum solution vectors + truth
                        for one representative event (for histograms/activations)

Run:  python gen_quantum.py            # default grid
      python gen_quantum.py 100        # also attempt T=100 (slow, ~tens of min)
"""
from __future__ import annotations
import sys, time, csv; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import qtrk_pipeline as qp
from helpers import compute_epsilon

CSV = cm.ASSETS / "quantum_sweep.csv"
FIELDS = ["T", "kernel", "erf_sigma", "rep", "n_seg", "n_sys", "n_qubits",
          "A_nnz", "t_q", "P_anc", "cos_QC",
          "effC", "purC", "farC", "effQ", "purQ", "farQ"]
ERF_SIGMA = 1e-3          # widest erf in the study -> biggest spectral reshaping
SAVE_VEC_T = {20, 50}     # T values for which to dump representative solvecs


def load_done():
    done = set()
    if CSV.exists():
        with open(CSV) as f:
            for r in csv.DictReader(f):
                done.add((int(r["T"]), r["kernel"], int(r["rep"])))
    return done


def append_row(row):
    new = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_point(T, kernel, rep, save_vec=False):
    eps = compute_epsilon(0.0, 1e-4)
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=1e-4, sigma_res=0.0)
    truth = qp.truth_from_event(ev)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel=kernel, erf_sigma=ERF_SIGMA)
    solC, _ = qp.solve_classical(ham)
    mC = qp.metrics_at(solC, truth, 0.35)
    t0 = time.time()
    qd = qp.solve_quantum(ham, device="CPU", readout="statevector")
    t_q = time.time() - t0
    mQ = qp.quantum_metrics(qd["sol"], solC, truth, 0.35)
    row = dict(T=T, kernel=kernel, erf_sigma=ERF_SIGMA, rep=rep,
               n_seg=ham.n_segments, n_sys=qd["n_sys"], n_qubits=qd["n_qubits"],
               A_nnz=ham.A.nnz, t_q=round(t_q, 2), P_anc=qd["P_anc"],
               cos_QC=mQ["cos_QC"],
               effC=mC["segment_efficiency"], purC=mC["segment_purity"],
               farC=mC["segment_false_rate"],
               effQ=mQ["segment_efficiency"], purQ=mQ["segment_purity"],
               farQ=mQ["segment_false_rate"])
    if save_vec:
        solQ_resc = qp.rescale_to(np.asarray(qd["sol"], float),
                                  np.asarray(solC, float))
        np.savez_compressed(
            cm.ASSETS / f"solvecs_T{T}_{kernel}.npz",
            solC=np.asarray(solC, np.float32),
            solQ=np.asarray(solQ_resc, np.float32),
            truth=np.asarray(truth, bool),
            eps=eps, P_anc=qd["P_anc"], cos_QC=mQ["cos_QC"],
            n_seg=ham.n_segments)
    return row


if __name__ == "__main__":
    grid_T = [10, 20, 50]
    if len(sys.argv) > 1:
        grid_T += [int(a) for a in sys.argv[1:]]
    reps = [0, 1, 2]
    done = load_done()
    print(f"grid T={grid_T} kernels=[step,erf] reps={reps}; {len(done)} done")
    for T in grid_T:
        for kernel in ["step", "erf"]:
            for rep in reps:
                if (T, kernel, rep) in done:
                    continue
                save_vec = (rep == 0 and T in SAVE_VEC_T)
                try:
                    row = run_point(T, kernel, rep, save_vec=save_vec)
                    append_row(row)
                    print(f"  T={T:4d} {kernel:4s} rep{rep}  cosQC={row['cos_QC']:.3f} "
                          f"P_anc={row['P_anc']:.4f} effQ={row['effQ']:.2f} "
                          f"purQ={row['purQ']:.2f} t_q={row['t_q']}s", flush=True)
                except Exception as e:
                    print(f"  T={T} {kernel} rep{rep} FAILED: {type(e).__name__}: {e}", flush=True)
    print("quantum sweep done.")
