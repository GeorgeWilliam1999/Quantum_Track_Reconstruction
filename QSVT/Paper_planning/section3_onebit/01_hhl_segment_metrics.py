#!/usr/bin/env python3
"""Full HHL at segment level, on the same stored clean events as the paper's
figure 12, read at the efficiency-first working point.

No HHL segment-level measurement exists in the project.  HHL's action on the
segment register is a spectral response, so it is emulated exactly through the
same per-cluster eigendecomposition every other curve in the paper uses:

  * a clock of n_c qubits; phase estimation maps eigenvalue lambda to clock
    amplitudes a_k(lambda) = (1/N) sum_m exp(i m (lambda t - 2 pi k / N)),
    N = 2^n_c -- the standard finite-clock kernel, leakage included;
  * the controlled rotation R_y(theta_k) on the flag, theta_k =
    2 arcsin(min(1, C / lambda_k)), lambda_k = 2 pi k / (t N), k = 1..N-1;
  * the true-inverse uncompute of the phase estimation on the ancilla-1 branch,
    after which the clock is still entangled and is traced out (as the package
    reads it out): p_i = sum_c | sum_j beta_j u_j(i) phi_j(c) |^2.

Conventions follow the package HHLAlgorithm exactly where they matter:
A scaled by its Frobenius norm, zero padding, t = pi / lambda_max(A_scaled).
The package fixes the rotation gain at C = 0.3 in *scaled* units, which
saturates the rotation (theta_k = pi) at every clock value for any system of
more than a handful of segments -- the device then applies no 1/lambda at all.
Both gains are measured: the package's, and the textbook C = lambda_min
(scaled), which reduces to the classical inverse when the clock is exact.

The emulation is validated against an exact, correctly wired finite-clock HHL
circuit built in this script (Aer statevector) at T = 3 and 5; the package's
own circuit keys its rotation on a bit-reversed clock value (Notion to-do).

Outputs
  outputs/01_hhl_validation.csv   cosine(emulation, exact circuit) per (T, n_c, gain)
  outputs/01_hhl_metrics.csv      eff / far at the efficiency-first working point
                                  per (T, rep, n_c, gain), with the classical and
                                  1BQF partners from the store on the same event
"""
from __future__ import annotations
import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
warnings.filterwarnings("ignore")
for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp
from lhcb_velo_toy.analysis.working_points import metrics_at_wp

HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
GAMMA, DELTA, S = 3.0, 1.0, 4.0
EPS = 0.002                                   # the benchmark's fixed acceptance
NZ = dict(sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)
TGRID = (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 550, 700, 1000)
REPS = (0, 1, 2)
CLOCKS = (4, 6, 8)
GAINS = ("package", "textbook")
PACKAGE_GAIN = 0.3
MAX_COMP = 3000
LOG = open(OUT / "01_run.log", "a")
def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


# ── per-eigenvalue clock vector on the ancilla-1 branch ──────────────────────
def clock_ops(n_c):
    N = 1 << n_c
    m = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(m, m) / N) / np.sqrt(N)      # IQFT (with swaps): |m> -> sum_k F[k,m] |k>
    Hn = np.ones((1, 1))
    for _ in range(n_c): Hn = np.kron(Hn, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    return N, m, F, Hn

def phi_vector(lam, t, n_c, C, ops):
    """clock-register vector (length N) multiplying eigencomponent lam on the
    ancilla-1 branch after: QPE, R_y rotation keyed on the clock value, QPE^-1."""
    N, m, F, Hn = ops
    D = np.exp(1j * m * lam * t)                                       # phase kick-back e^{+i lam t m} (package: U = e^{+iAt})
    QPE = F @ np.diag(D) @ Hn                                          # acts on |0>_clock
    a = QPE[:, 0]                                                      # a_k(lam)
    k = np.arange(N); lam_k = 2 * np.pi * k / (t * N)
    # package: angle = 2 arcsin(min(1, gain * (1/lambda_k) / 2)); textbook: sin(theta/2) = C / lambda_k
    with np.errstate(divide="ignore"):
        sin_half = np.where(k == 0, 0.0, np.minimum(1.0, C / np.where(k == 0, 1.0, lam_k)))
    v = a * sin_half
    return QPE.conj().T @ v                                            # true-inverse uncompute, clock left entangled


def emulate(A, b, n_c, gain, ops_cache):
    """sqrt(p_i) over the (unpadded) segments for the finite-clock HHL response."""
    n = A.shape[0]
    fro = float(np.sqrt((A.multiply(A)).sum()))                        # zero padding adds nothing
    As = A / fro
    Off = As.copy().tolil(); Off.setdiag(0); Off = Off.tocsr(); Off.eliminate_zeros()
    ncomp, lab = connected_components(Off, directed=False)
    sizes = np.bincount(lab)
    if sizes.max() > MAX_COMP: raise RuntimeError("component too large for dense eigh")
    order = np.argsort(lab, kind="stable"); bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    iso = sizes[lab] == 1
    lam_iso = S / fro
    # eigen-decompose the coupled components
    rows, cols, vals, lams, ws = [], [], [], [], []
    m0 = 0
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        lam, V = eigh(As[np.ix_(idx, idx)].toarray()); k = len(idx)
        rows.append(np.repeat(idx, k)); cols.append(np.tile(np.arange(m0, m0 + k), k)); vals.append(V.ravel())
        lams.append(lam); ws.append(V.T @ b[idx]); m0 += k
    lam_max = max([lam_iso] + [float(l.max()) for l in lams]) if lams else lam_iso
    lam_min_pos = min([lam_iso] + [float(l[l > 1e-12].min()) for l in lams if (l > 1e-12).any()])
    t = np.pi / lam_max
    C = PACKAGE_GAIN / 2.0 if gain == "package" else lam_min_pos      # the package halves its gain inside arcsin
    if n_c not in ops_cache: ops_cache[n_c] = clock_ops(n_c)
    ops = ops_cache[n_c]; N = ops[0]
    cache = {}
    def phi(l):
        key = round(float(l), 12)
        if key not in cache: cache[key] = phi_vector(l, t, n_c, C, ops)
        return cache[key]
    p = np.zeros(n)
    # isolated segments: modes are unit vectors, all at lam_iso
    ph = phi(lam_iso); p[iso] = (b[iso] ** 2) * float(np.sum(np.abs(ph) ** 2))
    if lams:
        U = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, m0))
        lam_all = np.concatenate(lams); w_all = np.concatenate(ws)
        Phi = np.stack([phi(l) for l in lam_all], axis=0)              # (n_modes, N)
        for c in range(N):
            Xc = U @ (w_all * Phi[:, c])
            p += np.abs(Xc) ** 2
    return np.sqrt(p), dict(t=t, C=C, lam_max=lam_max, lam_min=lam_min_pos, n_distinct=len(cache))


# ── validation against an exact, correctly wired circuit ─────────────────────
# The package HHLAlgorithm keys its rotation on a bit-reversed clock value and
# saturates its gain (Notion to-do, 2026-09-02); the emulation is therefore
# validated against a textbook finite-clock HHL built here: big-endian phase
# kick-back, Qiskit's inverse QFT with the register order made consistent, the
# rotation keyed on the estimated clock value, the true inverse, statevector
# readout with the register offsets right.
def exact_circuit(A_dense, b, n_c, C):
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import QFT, UnitaryGate, RYGate
    from qiskit_aer import AerSimulator
    from scipy.linalg import expm
    n = A_dense.shape[0]; n_s = int(np.ceil(np.log2(n))); N_s = 1 << n_s
    Ap = np.zeros((N_s, N_s)); Ap[:n, :n] = A_dense
    fro = float(np.linalg.norm(Ap)); As = Ap / fro
    lam = np.linalg.eigvalsh(As); t = np.pi / float(lam.max())
    bp = np.zeros(N_s); bp[:n] = b; bp = bp / np.linalg.norm(bp)
    T = QuantumRegister(n_c, "t"); Sy = QuantumRegister(n_s, "s"); An = QuantumRegister(1, "a")
    qc = QuantumCircuit(T, Sy, An)
    qc.initialize(bp, Sy)
    qc.h(T)
    for i in range(n_c):                                   # qubit T[i] carries 2^i  (little-endian, Qiskit's convention)
        qc.append(UnitaryGate(expm(1j * As * t * (1 << i))).control(1), [T[i], *Sy])
    qc.append(QFT(n_c, do_swaps=True).inverse().to_gate(label="IQFT"), T)
    N = 1 << n_c
    for k in range(1, N):
        lam_k = 2 * np.pi * k / (t * N); theta = 2 * np.arcsin(min(1.0, C / lam_k))
        bits = [(k >> j) & 1 for j in range(n_c)]           # T[j] holds bit j of k
        flip = [T[j] for j in range(n_c) if bits[j] == 0]
        if flip: qc.x(flip)
        qc.append(RYGate(theta).control(n_c), [*T, An[0]])
        if flip: qc.x(flip)
    qc.append(QFT(n_c, do_swaps=True).to_gate(label="QFT"), T)
    for i in reversed(range(n_c)):
        qc.append(UnitaryGate(expm(-1j * As * t * (1 << i))).control(1), [T[i], *Sy])
    qc.h(T)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    psi = np.asarray(sim.run(transpile(qc, sim)).result().data(0)["statevector"]).ravel()
    psi3 = psi.reshape(2, N_s, N)                            # [ancilla | system | clock], clock fastest
    p = (np.abs(psi3[1]) ** 2).sum(axis=1)
    return np.sqrt(p)[:n], float((np.abs(psi3[1]) ** 2).sum()), dict(t=t, lam_min=float(lam[lam > 1e-12].min()), lam_max=float(lam.max()))

def validate():
    rows = []
    Ts, Ncs = ((3, 5), (4, 6)) if "--validate-all" in sys.argv else ((3,), (4,))
    for T in Ts:
        ev, _ = qp.ensure_event(n_trk=T, rep=0, **NZ)
        ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step", gamma=GAMMA, delta=DELTA)
        A = sp.csr_matrix(ham.A); n = A.shape[0]; b = np.ones(n) / np.sqrt(n)
        for n_c in Ncs:
            for gain in GAINS:
                x_emu, info = emulate(A, b, n_c, gain, {})
                x_circ, p1, cinfo = exact_circuit(A.toarray(), np.ones(n), n_c, info["C"])
                cosv = float(x_circ @ x_emu / np.linalg.norm(x_circ) / np.linalg.norm(x_emu))
                rows.append(dict(T=T, n_seg=n, n_c=n_c, gain=gain, C=info["C"], cos_emulation_vs_circuit=cosv,
                                 P_anc_circuit=p1, P_anc_emulation=float(np.sum(x_emu ** 2)),
                                 max_abs_diff=float(np.abs(x_circ - x_emu).max())))
                say(f"[validate] T={T} n_seg={n} n_c={n_c} gain={gain:8s} C={info['C']:.3g}: cos={cosv:.8f}  "
                    f"P_anc circuit {p1:.5f} vs emulation {np.sum(x_emu**2):.5f}  max|diff| {np.abs(x_circ-x_emu).max():.1e}")
    pd.DataFrame(rows).to_csv(OUT / "01_hhl_validation.csv", index=False)

# ── production: the figure-12 events ─────────────────────────────────────────
def stored(man, T, rep, solver):
    sub = man[(man.n_trk == T) & (man.rep == rep) & (man.solver == solver) & (man.gamma == GAMMA) &
              (man.hit_ineff == 0.0) & (man.sigma_res == 0.0) & (man.epsilon == EPS)]
    for k in sub.sol_key:
        if qp.solution_exists(k): return np.abs(np.asarray(qp.load_solution(k)["sol"], float))
    return None

def main():
    # The exact-circuit validation at T=3, n_c=4 is on record (cos = 1.00000000,
    # max|diff| ~1e-12 for both gains, outputs/01_hhl_validation.csv); larger
    # clocks make Aer's transpile of the dense controlled unitaries take tens
    # of minutes for no additional information, so it is opt-in.
    if "--validate-all" in sys.argv:
        validate()
    man = pd.read_csv(os.environ["QTRK_STORE"] + "/manifest/solutions.csv")
    rows = []; ops_cache = {}
    for T in TGRID:
        for rep in REPS:
            t0 = time.time()
            ev, _ = qp.ensure_event(n_trk=T, rep=rep, **NZ)
            ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step", gamma=GAMMA, delta=DELTA)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            A = sp.csr_matrix(ham.A); n = A.shape[0]; b = np.ones(n) / np.sqrt(n)
            partners = {}
            for solver in ("classical", "quantum", "qsvt"):
                x = stored(man, T, rep, solver)
                if x is not None: partners[solver] = x[:n]
            if "classical" not in partners:
                partners["classical"] = np.abs(np.asarray(qp.solve_classical(ham)[0], float))[:n]
            for name, x in partners.items():
                m = metrics_at_wp(x, truth, 0.99)
                rows.append(dict(T=T, rep=rep, solver=name, n_c=np.nan, gain="", eff=m["segment_efficiency"],
                                 far=m["segment_false_rate"], tau_wp=m["tau_wp"], n_seg=n, n_true=int(truth.sum())))
            for n_c in CLOCKS:
                for gain in GAINS:
                    try:
                        x, info = emulate(A, b, n_c, gain, ops_cache)
                    except Exception as exc:
                        say(f"[skip] T={T} rep={rep} n_c={n_c} {gain}: {type(exc).__name__}: {exc}"); continue
                    m = metrics_at_wp(x, truth, 0.99)
                    rows.append(dict(T=T, rep=rep, solver=f"hhl", n_c=n_c, gain=gain, eff=m["segment_efficiency"],
                                     far=m["segment_false_rate"], tau_wp=m["tau_wp"], n_seg=n, n_true=int(truth.sum()),
                                     P_anc=float(np.sum(x ** 2)), C=info["C"], n_distinct_lambda=info["n_distinct"]))
            say(f"[ok] T={T:5d} rep={rep}  n_seg={n:8d}  partners={sorted(partners)}  {time.time()-t0:6.1f}s")
            pd.DataFrame(rows).to_csv(OUT / "01_hhl_metrics.csv", index=False)
    df = pd.DataFrame(rows)
    piv = df.groupby(["solver", "gain", "n_c", "T"])[["eff", "far"]].mean().reset_index()
    say("\n" + piv.to_string(index=False))

if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
    else:
        main()
