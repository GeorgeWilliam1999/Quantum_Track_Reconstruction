#!/usr/bin/env python3
"""The resource model of paper section 4.5: width, calls per shot, gates per
call, success probability, amplification and readout, for HHL, the one-bit
filter, and the three realisations of the degree-40 comb, on the figure-12
events (clean, fixed eps = 2 mrad, gamma = 3), T = 10..1000, rep 0.

Realisations of the comb p (degree d = 40, LCU register m = 6):
  LCU        linear combination of Chebyshev powers of the qubitization walk
             (the build the simulations run): width n_sys + 1 + m; the binary
             SELECT executes 2^m - 1 = 63 walk steps (d with an optimal
             SELECT); success ||p(X)u||^2 / ||c||_1^2 -- MEASURED in the store.
  phase      QSVT proper: d applications of the block encoding interleaved with
             phase rotations on one signal ancilla; the output's real part is q,
             recovered by the real-part trick (one ancilla, the two conjugate
             sequences); the comb has no definite parity, so even and odd parts
             run under one more ancilla.  Width n_sys + 4 (dilation, signal,
             real-part, parity); calls 4d (2d for a definite-parity design);
             success ||q(X)u||^2 / (alpha_e + alpha_o)^2 with q = p / max|p|
             -- COMPUTED matrix-free; the phases exist (01_phase_factors.py).
  QETU       the trigonometric comb over the one-bit filter's own evolution
             e^{-iAt} (the DSS route of section 11): width n_sys + 2, K = 40
             evolution calls, success ||p_trig(A)u||^2 -- COMPUTED.
Reference: HHL (n_c-bit clock: width n_sys + n_c + 1, 2(2^n_c - 1) calls,
success from the section-3 emulation) and the 1BQF (n_sys + 2, 2 calls,
success MEASURED in the store).

Gates per call, two prices: the Givens product for e^{-iAt} (DSS study,
measured to T = 400, power-law extrapolated beyond and flagged) and the
hit-priced block encoding (Block_Encoding cost model, to T = 1000).

Output: outputs/02_resource_model.csv, outputs/02_resource_fits.json
"""
from __future__ import annotations
import json, os, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from numpy.polynomial import chebyshev as C
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src", "/data/bfys/gscriven/Quantum_Track_Reconstruction/Block_Encoding"):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse

HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
ROOT = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction")
S, DEGREE, HW, DOMAIN = 4.0, 40, 0.18, (0.2, 7.8)
M_LCU = math.ceil(math.log2(DEGREE + 1))
NZ = dict(sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0); EPS = 0.002
TGRID = (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 550, 700, 1000)
T_TRIG = np.pi / 7.9; K_TRIG = 40
MAX_COMP = 3000


def cheb_apply(A, coef, lo, hi, u):
    c0, c1 = (lo + hi) / 2, (hi - lo) / 2
    T0 = u.copy(); T1 = (A @ u - c0 * u) / c1; y = coef[0] * T0 + coef[1] * T1
    for k in range(2, len(coef)):
        T2 = 2 * ((A @ T1 - c0 * T1) / c1) - T0; y += coef[k] * T2; T0, T1 = T1, T2
    return y

def trig_comb(K=K_TRIG, t=T_TRIG):
    """cosine series in theta = lambda t fitted to the comb target on the domain, scaled to max 1."""
    x = np.linspace(DOMAIN[0], DOMAIN[1], 6000)
    lines = [S - 2 * np.cos(k * np.pi / 5) for k in (1, 2, 3, 4)]
    y = np.zeros_like(x)
    for m in lines: y = np.maximum(y, np.exp(-((x - m) / HW) ** 2))
    y = y / np.clip(x, min(lines), None)
    B = np.cos(np.outer(x * t, np.arange(K + 1)))
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    f = lambda lam: np.cos(np.outer(np.asarray(lam, float) * t, np.arange(K + 1))) @ c
    mx = float(np.max(np.abs(f(x)))); c = c / mx
    f = lambda lam, c=c: np.cos(np.outer(np.asarray(lam, float) * t, np.arange(K + 1))) @ c
    return f, c

def modes(A, u):
    n = A.shape[0]
    Off = A.copy().tolil(); Off.setdiag(0); Off = Off.tocsr(); Off.eliminate_zeros()
    ncomp, lab = connected_components(Off, directed=False); sizes = np.bincount(lab)
    if sizes.max() > MAX_COMP: raise RuntimeError("component too large")
    order = np.argsort(lab, kind="stable"); bounds = np.searchsorted(lab[order], np.arange(ncomp + 1))
    iso = sizes[lab] == 1
    lam = [np.full(int(iso.sum()), S)]; b2 = [u[iso] ** 2]
    for c in np.flatnonzero(sizes > 1):
        idx = order[bounds[c]:bounds[c + 1]]
        w, V = eigh(A[np.ix_(idx, idx)].toarray()); lam.append(w); b2.append((V.T @ u[idx]) ** 2)
    return np.concatenate(lam), np.concatenate(b2)


def main():
    p = design_line_comb_inverse(degree=DEGREE, s=S, hw=HW, domain=DOMAIN)
    coef = np.asarray(p.coef, float); l1 = float(np.abs(coef).sum())
    xg = np.linspace(-1, 1, 20001); pmax = float(np.max(np.abs(C.chebval(xg, coef))))
    J = json.load(open(OUT / "01_phase_factors.json"))
    parity_pen = float(J["parity_penalty"])
    ftrig, ctrig = trig_comb()
    # measured inputs
    dsw = pd.read_csv(ROOT / "QSVT/Segment_level_studies/outputs/depth_and_qubits/degree_sweep.csv")
    P_lcu = dsw[dsw.degree == DEGREE].groupby("T")["P"].mean()
    P_lcu16 = dsw[dsw.degree == 16].groupby("T")["P"].mean()
    man = pd.read_csv(os.environ["QTRK_STORE"] + "/manifest/solutions.csv")
    hh = pd.read_csv(ROOT / "QSVT/Paper_planning/section3_onebit/outputs/01_hhl_metrics.csv")
    hh = hh[(hh.solver == "hhl") & (hh.gain == "textbook")]
    dss = pd.read_csv(ROOT / "QSVT/DSS/outputs/dss_gates.csv")
    # Givens per-call cost: measured to T=400, power law beyond
    lg = np.polyfit(np.log(dss["T"]), np.log(dss["call_CX"]), 1); giv_exp, giv_a = float(lg[0]), float(np.exp(lg[1]))
    be = pd.read_csv(ROOT / "Block_Encoding/outputs/06_resource_curves.csv")
    hit = be[be.method == "hit_oracle"].set_index("T")
    rows = []
    for T in TGRID:
        t0 = time.time()
        ev, _ = qp.ensure_event(n_trk=T, rep=0, **NZ)
        ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step", gamma=3.0, delta=1.0)
        A = sp.csr_matrix(ham.A); n = A.shape[0]; u = np.ones(n) / np.sqrt(n)
        n_sys = int(math.ceil(math.log2(n)))
        # success probabilities
        pu = cheb_apply(A, coef, *DOMAIN, u); P_phase = float(pu @ pu) / pmax ** 2 / parity_pen
        P_lcu_from_norm = float(pu @ pu) / l1 ** 2
        lam, b2 = modes(A, u); P_trig = float(np.sum(b2 * ftrig(lam) ** 2))
        q1 = man[(man.solver == "quantum") & (man.n_trk == T) & (man.rep == 0) & (man.gamma == 3.0) & (man.hit_ineff == 0.0) & (man.sigma_res == 0.0) & (man.epsilon == EPS)]
        P_1bqf = np.nan
        for k in q1.sol_key:
            if qp.solution_exists(k): P_1bqf = float(qp.load_solution(k)["P_anc"]); break
        P_hhl = {nc: float(hh[(hh.n_c == nc) & (hh["T"] == T)]["P_anc"].mean()) for nc in (6, 8)}
        # per-call gate prices
        giv = float(dss.loc[dss["T"] == T, "call_CX"].iloc[0]) if T in set(dss["T"]) else float(giv_a * T ** giv_exp)
        giv_measured = T in set(dss["T"])
        hit_cx = float(hit.loc[T, "cx_call"]) if T in hit.index else np.nan
        hit_anc = int(hit.loc[T, "anc"]) if T in hit.index else np.nan
        rec = dict(T=T, n_seg=n, n_sys=n_sys, nnz=int(A.nnz),
                   givens_cx_per_call=giv, givens_measured=giv_measured, hit_cx_per_call=hit_cx, hit_oracle_ancillas=hit_anc,
                   pmax=pmax, l1=l1, parity_penalty=parity_pen,
                   P_lcu_measured=float(P_lcu.get(T, np.nan)), P_lcu_from_norm=P_lcu_from_norm, P_lcu16_measured=float(P_lcu16.get(T, np.nan)),
                   P_phase=P_phase, P_qetu=P_trig, P_1bqf=P_1bqf, P_hhl6=P_hhl[6], P_hhl8=P_hhl[8])
        # widths
        rec.update(w_1bqf=n_sys + 2, w_lcu=n_sys + 1 + M_LCU, w_phase=n_sys + 4, w_phase_evencomb=n_sys + 3, w_qetu=n_sys + 2,
                   w_hhl6=n_sys + 7, w_hhl8=n_sys + 9)
        # calls per shot
        rec.update(calls_1bqf=2, calls_lcu=(1 << M_LCU) - 1, calls_lcu_optimal=DEGREE, calls_phase=4 * DEGREE, calls_phase_evencomb=2 * DEGREE,
                   calls_qetu=K_TRIG, calls_hhl6=2 * (2 ** 6 - 1), calls_hhl8=2 * (2 ** 8 - 1))
        # amplification rounds and totals (Givens price for evolution-based routes, hit price for block-encoding routes)
        def rounds(P): return float(np.pi / (4 * np.sqrt(P))) if P and P > 0 else np.nan
        rec.update(rounds_1bqf=rounds(P_1bqf), rounds_lcu=rounds(rec["P_lcu_measured"]), rounds_phase=rounds(P_phase), rounds_qetu=rounds(P_trig),
                   rounds_hhl8=rounds(P_hhl[8]))
        rec.update(total_calls_1bqf=2 * rec["rounds_1bqf"], total_calls_lcu=rec["calls_lcu"] * rec["rounds_lcu"],
                   total_calls_phase=rec["calls_phase"] * rec["rounds_phase"], total_calls_qetu=rec["calls_qetu"] * rec["rounds_qetu"],
                   total_calls_hhl8=rec["calls_hhl8"] * rec["rounds_hhl8"])
        M = 4 * T; readout = M * math.log(M)
        rec.update(readout_successes=readout)
        rows.append(rec)
        print(f"[ok] T={T:5d} n={n:8d} n_sys={n_sys:2d}  P: 1BQF {P_1bqf:.2e}  LCU {rec['P_lcu_measured']:.2e} (norm {P_lcu_from_norm:.2e})  "
              f"phase {P_phase:.2e}  QETU {P_trig:.2e}  HHL8 {P_hhl[8]:.2e}   Givens CX/call {giv:.3g}{'' if giv_measured else ' (extrap.)'}  hit {hit_cx:.3g}   {time.time()-t0:5.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "02_resource_model.csv", index=False)
    df = pd.DataFrame(rows)
    fits = {}
    for col in ("P_1bqf", "P_lcu_measured", "P_phase", "P_qetu", "P_hhl8", "givens_cx_per_call", "hit_cx_per_call"):
        d = df[["T", col]].dropna(); d = d[d[col] > 0]
        if len(d) >= 3:
            a, b = np.polyfit(np.log(d["T"]), np.log(d[col]), 1); fits[col] = dict(exponent=float(a), prefactor=float(np.exp(b)))
    fits["givens_extrapolation"] = dict(exponent=giv_exp, prefactor=giv_a, measured_T_max=int(dss["T"].max()))
    fits["comb"] = dict(degree=DEGREE, l1=l1, pmax=pmax, parity_penalty=parity_pen, trig_K=K_TRIG, trig_t=T_TRIG, trig_l1=float(np.abs(ctrig).sum()))
    (OUT / "02_resource_fits.json").write_text(json.dumps(fits, indent=1))
    print(json.dumps(fits, indent=1))

if __name__ == "__main__":
    main()
