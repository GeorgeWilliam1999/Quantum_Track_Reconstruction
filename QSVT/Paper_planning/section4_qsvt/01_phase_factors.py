#!/usr/bin/env python3
"""Phase factors for the production comb: the QSVT proper, built and validated
at small size (paper section 4.2).

The literature's QSVT applies a real polynomial q of definite parity with
|q| <= 1 on [-1, 1] by d applications of the block encoding interleaved with
single-qubit phase rotations.  The production comb has no definite parity, so
it is split, p = p_even + p_odd, each part rescaled to peak at one and given
its own phase sequence, and the two are recombined with one further ancilla.

Angle finding: the symmetric-phase-factor formulation (Dong, Meng, Whaley, Lin,
Phys. Rev. A 103, 042419 (2021)) -- minimise the squared mismatch between the
real part of the QSP unitary's (0,0) element and q on Chebyshev nodes, with the
phases constrained symmetric.  L-BFGS with an analytic gradient; degree 40 is
seconds.  Convention: W(x) = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]],
U_Phi(x) = e^{i phi_0 Z} prod_{k=1..d} [ W(x) e^{i phi_k Z} ],
q(x) = Re <0| U_Phi(x) |0>.

Validation, two levels:
  1. the scalar identity on a fine grid, max |Re U_00(x) - q(x)|;
  2. the full circuit on a small event (T=3, 36 segments): the block encoding
     U of X = (A - c0)/c1 on the system + dilation registers, the projector-
     controlled phase rotations on a signal ancilla, d applications; compare
     the post-selected system state with q(X) u.

Outputs
  outputs/01_phase_factors.json     phases for both parity parts, validation numbers
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from numpy.polynomial import chebyshev as C
from scipy.optimize import minimize
for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse

HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)
S = 4.0; DOMAIN = (0.2, 7.8); DEGREE = 40; HW = 0.18


# ── the polynomial, its parity parts, and the scaling to |q| <= 1 ────────────
def comb_parts(degree=DEGREE):
    p = design_line_comb_inverse(degree=degree, s=S, hw=HW, domain=DOMAIN)
    c = np.asarray(p.coef, float)                      # Chebyshev coefficients on [-1,1] of x = (2 lam - (lo+hi))/(hi-lo)
    x = np.linspace(-1, 1, 20001)
    scale = float(np.max(np.abs(C.chebval(x, c))))     # rescale the whole comb to peak at 1: allowed, the metrics are scale-free
    c = c / scale
    ce = c.copy(); ce[1::2] = 0.0
    co = c.copy(); co[0::2] = 0.0
    ae = float(np.max(np.abs(C.chebval(x, ce)))); ao = float(np.max(np.abs(C.chebval(x, co))))
    return dict(c=c, ce=ce / ae, co=co / ao, alpha_e=ae, alpha_o=ao, scale=scale)


# ── QSP unitary and the objective ────────────────────────────────────────────
def qsp_u00(phis, x):
    """Re <0|U_Phi(x)|0> for arrays x (vectorised 2x2 products)."""
    x = np.asarray(x, float); sq = np.sqrt(np.clip(1 - x * x, 0, None))
    # W(x) e^{i phi Z}: entries
    U = np.zeros((len(x), 2, 2), complex)
    U[:, 0, 0] = np.exp(1j * phis[0]); U[:, 1, 1] = np.exp(-1j * phis[0])
    for ph in phis[1:]:
        W = np.zeros_like(U); W[:, 0, 0] = x; W[:, 1, 1] = x; W[:, 0, 1] = 1j * sq; W[:, 1, 0] = 1j * sq
        E = np.zeros_like(U); E[:, 0, 0] = np.exp(1j * ph); E[:, 1, 1] = np.exp(-1j * ph)
        U = U @ W @ E
    return U[:, 0, 0].real

def find_phases(q_coef, degree, nodes=None, tol=1e-12, seed=0):
    """Symmetric phase factors for the parity-definite Chebyshev series q_coef."""
    d = degree
    if nodes is None:
        nodes = np.cos(np.pi * (np.arange(2 * d + 2) + 0.5) / (2 * d + 2))    # Chebyshev nodes, more than d+1
    target = C.chebval(nodes, q_coef)
    nfree = d // 2 + 1                                       # symmetric: phi_k = phi_{d-k}
    def expand(theta):
        phis = np.zeros(d + 1)
        for k in range(nfree):
            phis[k] = theta[k]; phis[d - k] = theta[k]
        return phis
    def loss(theta):
        return float(np.sum((qsp_u00(expand(theta), nodes) - target) ** 2))
    # the standard initial guess: phi_0 = phi_d = pi/4, others 0
    theta0 = np.zeros(nfree); theta0[0] = np.pi / 4
    best = None
    for trial in range(6):
        th = theta0 if trial == 0 else theta0 + np.random.default_rng(seed + trial).normal(0, 0.3, nfree)
        res = minimize(loss, th, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-18, gtol=1e-14, maxcor=50))
        if best is None or res.fun < best.fun: best = res
        if best.fun < tol: break
    phis = expand(best.x)
    xg = np.linspace(-1, 1, 4001)
    err = float(np.max(np.abs(qsp_u00(phis, xg) - C.chebval(xg, q_coef))))
    return phis, err, float(best.fun)


# ── the full circuit on a small event, as dense linear algebra ───────────────
def qsvt_circuit_small(X, u, phis):
    """QSVT proper for a real symmetric X (||X|| <= 1): block encoding by dilation
    U = [[X, sqrt(I-X^2)], [sqrt(I-X^2), -X]] on (dilation, system); signal ancilla
    rotations e^{i phi (2 Pi - I)} with Pi the projector onto dilation=|0>.
    Returns the system state on the (ancilla=+, dilation=0) branch."""
    n = X.shape[0]
    w, V = np.linalg.eigh(X); w = np.clip(w, -1, 1)
    R = (V * np.sqrt(1 - w * w)) @ V.T
    U = np.block([[X, R], [R, -X]])
    Pi = np.diag(np.r_[np.ones(n), np.zeros(n)])           # dilation = |0>
    def rot(phi):                                          # e^{i phi (2 Pi - I)} on the 2n-dim space
        return np.diag(np.exp(1j * phi * (2 * np.diag(Pi) - 1)))
    d = len(phis) - 1
    psi = np.r_[u.astype(complex), np.zeros(n)]            # dilation |0>, system u
    psi = rot(phis[0]) @ psi
    for k in range(1, d + 1):
        psi = (U if (k % 2 == 1) else U.conj().T) @ psi      # alternate U, U^dagger (real symmetric U: identical)
        psi = rot(phis[k]) @ psi
    return psi[:n]                                          # dilation = |0> branch


def main():
    t0 = time.time()
    parts = comb_parts(DEGREE)
    out = dict(degree=DEGREE, domain=DOMAIN, hw=HW, scale_to_unit=parts["scale"],
               alpha_even=parts["alpha_e"], alpha_odd=parts["alpha_o"],
               parity_penalty=(parts["alpha_e"] + parts["alpha_o"]) ** 2)
    print(f"comb rescaled by 1/{parts['scale']:.4f} to peak at 1; even part peaks at {parts['alpha_e']:.4f}, odd at {parts['alpha_o']:.4f}; "
          f"recombination penalty (alpha_e+alpha_o)^2 = {out['parity_penalty']:.3f}")
    for name, coef, deg in (("even", parts["ce"], DEGREE), ("odd", parts["co"], DEGREE - 1)):
        t1 = time.time()
        phis, err, fun = find_phases(coef, deg)
        out[name] = dict(degree=deg, phases=phis.tolist(), max_abs_err_grid=err, final_loss=fun, seconds=time.time() - t1)
        print(f"[{name:4s}] degree {deg}: phases found in {time.time()-t1:.1f}s, max |Re U_00 - q| on a 4001-point grid = {err:.2e}")
    # circuit validation on a small event
    import qtrk_pipeline as qp
    ev, _ = qp.ensure_event(n_trk=3, rep=0, sigma_scatt=1e-4, sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)
    ham = qp.build_hamiltonian(ev, epsilon=0.002, kernel="step", gamma=3.0, delta=1.0)
    A = sp.csr_matrix(ham.A).toarray(); n = A.shape[0]
    lo, hi = DOMAIN; X = (2 * A - (lo + hi) * np.eye(n)) / (hi - lo)
    u = np.ones(n) / np.sqrt(n)
    val = {}
    for name, coef in (("even", parts["ce"]), ("odd", parts["co"])):
        phis = np.array(out[name]["phases"])
        psi = qsvt_circuit_small(X, u, phis)
        # exact polynomial of the matrix
        w, V = np.linalg.eigh(X); qX = (V * C.chebval(w, coef)) @ V.T; ref = qX @ u
        # the circuit's branch carries q(X)u on the real part (imaginary part is the complementary polynomial)
        err = float(np.max(np.abs(psi.real - ref))); cosv = float(abs(np.vdot(psi, ref)) / np.linalg.norm(psi) / np.linalg.norm(ref))
        val[name] = dict(max_abs_err=err, cos=cosv, norm_ref=float(np.linalg.norm(ref)), norm_branch=float(np.linalg.norm(psi)))
        print(f"[circuit] {name:4s} on T=3 event (n={n}): max|Re psi - q(X)u| = {err:.2e}, cosine {cosv:.10f}")
    out["circuit_validation"] = dict(T=3, n_seg=n, parts=val)
    out["seconds_total"] = time.time() - t0
    (OUT / "01_phase_factors.json").write_text(json.dumps(out))
    print(f"[saved] outputs/01_phase_factors.json  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
