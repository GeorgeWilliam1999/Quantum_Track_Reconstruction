"""
Why the 1-Bit Quantum Filter beats full HHL, made quantitative.

(a) The eigenvalue weight each applies.  Full HHL resolves lambda on n_time qubits
    and applies the true 1/lambda (a staircase that sharpens as n_time grows);
    the 1BQF uses ONE time qubit, so its weight is a single cos-notch -- which is
    all the bimodal track spectrum needs (false bulk on the notch, true off it).
(b) The resource cost.  Qubits: HHL = n_sys + n_time + 1 (needs an eigenvalue
    register); 1BQF = n_sys + 2.  Two-qubit gates: HHL's controlled e^{iAt} is a
    DENSE N x N unitary (~O(N^2) per time qubit) plus a 2^{n_time} rotation ladder;
    the 1BQF's e^{iAt} is sparse Givens, ~O(A_nnz) ~ O(n_seg).  1BQF's measured
    transpiled gate count is shown against the HHL dense-scaling reference.
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import matplotlib.pyplot as plt
import qtrk_pipeline as qp
from helpers import compute_epsilon

S = 4.0                      # gamma+delta
TVAL = np.pi / S


def _hhl_staircase(lam, n_time, gain=0.6):
    """1/lambda as HHL realises it: phase phi=lambda*t/2pi quantised to 2^n_time
    bins, then weight C/lambda_bin (clamped)."""
    t = TVAL
    phi = (lam * t) / (2 * np.pi)
    nb = 2 ** n_time
    phi_q = np.round(phi * nb) / nb                  # nearest phase bin
    lam_q = phi_q * 2 * np.pi / t
    w = np.where(lam_q > 1e-6, 1.0 / np.maximum(lam_q, 1e-6), 0.0)
    return np.clip(gain * w, 0, 1)


def fig_hhl_why_one_bit():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # ---- (a) the filters ----
    lam = np.linspace(0.6, 7.4, 700)
    ideal = 1.0 / lam; ideal /= ideal.max()
    axA.plot(lam, ideal, "k-", lw=2.2, label=r"ideal $1/\lambda$ (exact inverse)")
    for n_time, c in [(2, "#9ecae1"), (3, "#4292c6"), (5, "#08519c")]:
        axA.plot(lam, _hhl_staircase(lam, n_time), lw=1.8, color=c,
                 label=fr"HHL, $n_{{\rm time}}={n_time}$")
    axA.plot(lam, np.cos(lam * TVAL / 2) ** 2, color=cm.C["quantum"], lw=2.6,
             label="1BQF, 1 qubit (cos notch)")
    axA.axvline(S, ls=":", c="grey"); axA.text(S + 0.05, 0.92, r"notch $\lambda=\gamma+\delta$",
                                               color="grey", fontsize=9)
    axA.set_xlabel(r"eigenvalue $\lambda$"); axA.set_ylabel("applied weight (normalised)")
    axA.set_title("(a) HHL approximates $1/\\lambda$ with $n_{\\rm time}$ qubits;\n"
                  "1BQF uses one qubit = a single notch", fontsize=11)
    axA.legend(fontsize=8.5, loc="upper right"); axA.set_ylim(0, 1.08)

    # ---- (b) cost: qubits + two-qubit gate count vs system size ----
    T = np.array([10, 20, 50, 100, 200, 400, 700, 1000], float)
    n_seg = 4 * T ** 2
    n_sys = np.ceil(np.log2(n_seg))
    n_time_hhl = 5
    qb_hhl = n_sys + n_time_hhl + 1
    qb_1bqf = n_sys + 2

    # measure the 1BQF transpiled 2q-gate count at a few small T
    meas_T, meas_g = [], []
    try:
        from qiskit import transpile
        from lhcb_velo_toy.solvers.quantum.OneBQF import OneBQF
        eps = compute_epsilon(0.0, 1e-4)
        for Tm in (10, 20, 50):
            ev, _ = qp.ensure_event(n_trk=Tm, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step")
            o = OneBQF(ham.A, ham.b, num_time_qubits=1, readout="statevector"); o.build_circuit()
            qc = o.circuit.copy()
            qc.data = [d for d in qc.data if d.operation.name not in ("save_statevector", "measure")]
            tc = transpile(qc, basis_gates=["cx", "u3"], optimization_level=0)
            meas_T.append(Tm); meas_g.append(tc.count_ops().get("cx", 0))
            print(f"  1BQF T={Tm}: n_seg={int(ham.n_segments)} A_nnz={ham.A.nnz} cx={meas_g[-1]}")
    except Exception as e:
        print("1BQF gate measurement skipped:", e)

    ax2 = axB
    # gate-count scaling references (anchored to 1BQF measurement if available)
    if meas_g:
        anchor_ns = 4 * meas_T[0] ** 2
        c1 = meas_g[0] / anchor_ns                 # 1BQF ~ c1 * n_seg
        ax2.loglog(meas_T, meas_g, "s", color=cm.C["quantum"], ms=9, label="1BQF (measured CX)")
    else:
        c1 = 30.0
    ax2.loglog(T, c1 * n_seg, "-", color=cm.C["quantum"], lw=2,
               label=r"1BQF $\sim O(A_{\rm nnz})\!=\!O(n_{\rm seg})$")
    c2 = c1                                          # same anchor scale for fairness
    ax2.loglog(T, c2 * n_seg ** 2 / n_seg[0], "--", color=cm.C["classical"], lw=2,
               label=r"HHL dense $\sim O(N^2)\!=\!O(n_{\rm seg}^2)$")
    ax2.set_xlabel("Track multiplicity  $T$"); ax2.set_ylabel("two-qubit gates (CX)")
    ax2.set_title("(b) Gate cost: 1BQF sparse $O(n_{\\rm seg})$ vs HHL dense $O(n_{\\rm seg}^2)$",
                  fontsize=11)
    ax2.legend(fontsize=8.5, loc="upper left")
    # qubit-count inset
    axin = axB.inset_axes([0.62, 0.12, 0.35, 0.33])
    axin.plot(T, qb_hhl, "-", color=cm.C["classical"], lw=1.6, label="HHL")
    axin.plot(T, qb_1bqf, "-", color=cm.C["quantum"], lw=1.6, label="1BQF")
    axin.set_xscale("log"); axin.set_title("qubits", fontsize=8)
    axin.tick_params(labelsize=7); axin.legend(fontsize=6.5, loc="upper left")

    fig.suptitle("Why one bit: HHL needs an $n_{\\rm time}$-qubit eigenvalue register and a dense "
                 "$e^{iAt}$;\nthe 1BQF needs one qubit + a sparse $e^{iAt}$, and the single notch "
                 "already separates the bimodal track spectrum",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    cm.savefig(fig, "hhl_why_one_bit")


if __name__ == "__main__":
    print("== HHL why-one-bit figure ==")
    fig_hhl_why_one_bit()
    print("done.")
