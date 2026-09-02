#!/usr/bin/env python3
"""The dilation U acts as the reflection R(x) = [[x, s],[s, -x]] in each
eigenplane, not as W(x) = [[x, i s],[i s, x]]; phases found in the W convention
must be converted (Martyn, Rossi, Tan, Chuang 2021, eq. 17):
  phi_0^R = phi_0 + (2d-1) pi/4,  phi_k^R = phi_k - pi/2 (0<k<d),  phi_d^R = phi_d - pi/4.
Verify the scalar identity in the R convention and re-run the circuit check."""
import json, sys, os, numpy as np, scipy.sparse as sp
from numpy.polynomial import chebyshev as C
sys.path.insert(0, "."); exec(open("01_phase_factors.py").read().split("def main()")[0].replace("HERE = Path(__file__).resolve().parent", "HERE = Path('.').resolve()"))
J = json.load(open(OUT / "01_phase_factors.json"))
parts = comb_parts(DEGREE)

def convert(phis):
    d = len(phis) - 1; r = np.array(phis, float).copy()
    r[0] += (2 * d - 1) * np.pi / 4; r[1:d] -= np.pi / 2; r[d] -= np.pi / 4
    return r
def qsp_r00(phis, x):
    x = np.asarray(x, float); sq = np.sqrt(np.clip(1 - x * x, 0, None))
    U = np.zeros((len(x), 2, 2), complex); U[:, 0, 0] = np.exp(1j * phis[0]); U[:, 1, 1] = np.exp(-1j * phis[0])
    for ph in phis[1:]:
        R = np.zeros_like(U); R[:, 0, 0] = x; R[:, 1, 1] = -x; R[:, 0, 1] = sq; R[:, 1, 0] = sq
        E = np.zeros_like(U); E[:, 0, 0] = np.exp(1j * ph); E[:, 1, 1] = np.exp(-1j * ph)
        U = U @ R @ E
    return U[:, 0, 0]

xg = np.linspace(-1, 1, 4001)
import qtrk_pipeline as qp
ev, _ = qp.ensure_event(n_trk=3, rep=0, sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)
ham = qp.build_hamiltonian(ev, epsilon=0.002, kernel="step", gamma=3.0, delta=1.0)
A = sp.csr_matrix(ham.A).toarray(); n = A.shape[0]; lo, hi = DOMAIN
X = (2 * A - (lo + hi) * np.eye(n)) / (hi - lo); u = np.ones(n) / np.sqrt(n)
w, V = np.linalg.eigh(X)
res = {}
for name, coef in (("even", parts["ce"]), ("odd", parts["co"])):
    phW = np.array(J[name]["phases"]); phR = convert(phW)
    r00 = qsp_r00(phR, xg); q = C.chebval(xg, coef)
    e_real = float(np.max(np.abs(r00.real - q))); e_abs = float(np.max(np.abs(np.abs(r00) - np.abs(q))))
    psi = qsvt_circuit_small(X, u, phR); ref = ((V * C.chebval(w, coef)) @ V.T) @ u
    cosv = float(abs(np.vdot(psi, ref)) / np.linalg.norm(psi) / np.linalg.norm(ref))
    err = float(np.max(np.abs(psi.real - ref)))
    # also check against the global-phase-free comparison |psi| vs |ref|
    err_abs = float(np.max(np.abs(np.abs(psi) - np.abs(ref))))
    res[name] = dict(scalar_err_real=e_real, scalar_err_abs=e_abs, circuit_err_real=err, circuit_err_abs=err_abs, cos=cosv, phases_R=phR.tolist())
    print(f"[{name:4s}] R-convention scalar identity: max|Re r00 - q| = {e_real:.2e} (| |r00|-|q| | = {e_abs:.2e});  "
          f"circuit on T=3 event: max|Re psi - q(X)u| = {err:.2e}, max||psi|-|q(X)u|| = {err_abs:.2e}, cosine {cosv:.10f}")
J["reflection_convention"] = res
(OUT / "01_phase_factors.json").write_text(json.dumps(J)); print("[saved] outputs/01_phase_factors.json (reflection-convention phases added)")
