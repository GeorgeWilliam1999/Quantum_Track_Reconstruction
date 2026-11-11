"""Shared pieces of the segment-pruning study.

Events come from the qtrk store (the same cached events as the paper corpus),
the matrix from the package builder, the prune from the package feature
(``SimpleHamiltonianFast.prune`` / ``construct_hamiltonian(prune=...)``), and
the three solvers from the package: classical inverse, the one-bit quantum
filter (the package circuit, matrix-free engine) and the QSVT comb (degree
40 on the base Hamiltonian).  The exact one-bit readout of the paper is kept
as a fourth column.

All metrics are computed in the FULL segment index space: a pruned segment
is absent from the solve and is given amplitude zero, so a pruned true
segment counts as a miss.  Every solver, classical included, is read at the
efficiency-first working point (threshold just below the 1 % quantile of the
true amplitudes, floored so structural zeros are never re-admitted).
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                   # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon           # noqa: E402
from lhcb_velo_toy.analysis.segment_metrics import segment_truth_mask   # noqa: E402
from lhcb_velo_toy.analysis.working_points import working_point_threshold, WP_TAU_FLOOR  # noqa: E402
from lhcb_velo_toy.solvers.hamiltonians import SimpleHamiltonianFast, coupling_pattern, qubits_for  # noqa: E402
from lhcb_velo_toy.solvers.quantum import QSVT, solve_one_bit_matrixfree, spectral_bounds  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FIG = HERE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
T_ONEBIT = np.pi / S
COMB_DEGREE = 40

# Noise configurations.  "paper" is the paper-1 corpus (George, 16 Sep 2026);
# the other two bracket it: no detector effects at all, and three times the
# paper's resolution and scattering.  phi_max = 0.2 everywhere.
CONFIGS = {
    "paper":    dict(sigma_scatt=1.5e-4, sigma_res=0.005, phi_max=0.2, hit_ineff=0.01),
    "no_noise": dict(sigma_scatt=1e-4,   sigma_res=0.0,   phi_max=0.2, hit_ineff=0.0),
    "wide":     dict(sigma_scatt=4.5e-4, sigma_res=0.015, phi_max=0.2, hit_ineff=0.01),
}
CONFIG_LABEL = {
    "paper": "5 µm, 0.15 mrad, 1 % hit loss (paper corpus)",
    "no_noise": "no resolution smearing, 0.1 mrad, no hit loss",
    "wide": "15 µm, 0.45 mrad, 1 % hit loss",
}
TS_DEFAULT = [10, 20, 50, 100, 200, 400, 700, 1000]


def epsilon_for(cfg: str) -> float:
    c = CONFIGS[cfg]
    return float(compute_epsilon(c["sigma_res"], c["sigma_scatt"]))


def load_event(cfg: str, T: int, rep: int):
    ev, ekey = qp.ensure_event(n_trk=int(T), rep=int(rep), **CONFIGS[cfg])
    return ev, ekey


def build_full(ev, eps: float) -> SimpleHamiltonianFast:
    """The full system, numpy arrays only (no Segment objects)."""
    ham = SimpleHamiltonianFast(epsilon=eps, gamma=GAMMA, delta=DELTA)
    ham.construct_segments(ev, materialize_segments=False)
    ham.construct_hamiltonian(ev)
    return ham


def pruned_copy(ham: SimpleHamiltonianFast, **prune_kw) -> SimpleHamiltonianFast:
    """A pruned view of a built Hamiltonian without rebuilding A.

    ``prune`` only reassigns attributes on the object (sliced copies of the
    per-segment arrays, a new A), so a shallow copy is enough and the
    original stays the full system.
    """
    hp = copy.copy(ham)
    hp.prune_info = {}
    hp.prune(**prune_kw)
    return hp


def component_sizes(A):
    """Connected-component label and size per segment of the coupling graph."""
    from scipy.sparse.csgraph import connected_components
    C = coupling_pattern(A)
    n_comp, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab, minlength=n_comp)
    deg = np.asarray(C.getnnz(axis=1)).ravel()
    return lab, sizes, deg


# ── solvers (all return amplitudes in the solver's own index space) ──────────
def solve_classical(ham):
    from scipy.sparse.linalg import minres, spsolve
    t0 = time.time()
    n = ham.A.shape[0]
    if n < 5000:
        x = spsolve(ham.A.tocsc(), ham.b)
    else:
        x, info = minres(ham.A, ham.b, rtol=1e-10, maxiter=50000)
    return np.asarray(x, float).ravel(), dict(t=time.time() - t0)


def solve_onebit_circuit(ham):
    """The package one-bit circuit, matrix-free engine (George's '1BQF')."""
    t0 = time.time()
    sol, P, n_sys = solve_one_bit_matrixfree(ham.A)
    return np.asarray(sol, float), dict(t=time.time() - t0, P=float(P), n_sys=int(n_sys),
                                        qubits=int(n_sys) + 2,
                                        statevector_bytes=16 * (1 << (int(n_sys) + 2)))


def solve_onebit_exact(ham):
    """Exact readout of the correctly wired one-bit filter (paper section 3):
    p_i = [cos^2(At/2) u]_i^2 + [sin(At) u / 2]_i^2, u uniform, t = pi/s."""
    from scipy.sparse.linalg import expm_multiply
    t0 = time.time()
    A = ham.A.tocsr()
    n = A.shape[0]
    u = np.ones(n)
    z = expm_multiply(-1j * T_ONEBIT * A.astype(np.complex128), u.astype(np.complex128))
    cosAt, sinAt = z.real, -z.imag
    y_cos2 = 0.5 * (u + cosAt)
    amp = np.sqrt(y_cos2 ** 2 + (0.5 * sinAt) ** 2)
    P = float(np.dot(u, y_cos2) / n)          # <u|cos^2(At/2)|u>, u normalised
    nrm = np.linalg.norm(amp)
    return amp / nrm if nrm else amp, dict(t=time.time() - t0, P=P)


def solve_qsvt(ham, bounds):
    """The QSVT (degree-40 line comb on the base Hamiltonian), exact statevector."""
    t0 = time.time()
    q = QSVT.comb(ham.A, np.asarray(ham.b, float), degree=COMB_DEGREE, spectral_bounds=bounds)
    sol, P = q.solve_statevector()
    r_lcu, r_ph = q.resources("lcu"), q.resources("phase")
    return np.asarray(sol, float), dict(
        t=time.time() - t0, P=float(P), n_sys=int(q.num_system_qubits),
        qubits=int(q.total_qubits), qubits_phase=int(r_ph["total_qubits"]),
        calls_lcu=int(r_lcu["calls_per_shot"]), calls_phase=int(r_ph["calls_per_shot"]),
        l1=float(q.l1), p_iso=float(q.poly(S)))


def bounds_for(ham):
    return spectral_bounds(ham.A, s=S, pad=0.10)


# ── metrics in the full index space ──────────────────────────────────────────
def metrics(x_full, truth, coupled=None, target_eff=0.99):
    """Efficiency-first working point and the eff-at-far<=1 % point.

    ``coupled`` (bool, optional) marks the segments the working point is set
    from: for the full matrix the segments with a compatible neighbour (the
    ones any segment-pair solver can find), for a pruned variant the segments
    that variant keeps.  ``eff_coupled_*`` is the efficiency over the true
    segments in that set; ``eff_e990`` is over all true segments.
    """
    x = np.abs(np.asarray(x_full, float).ravel())
    t = np.asarray(truth, bool)
    n_true = int(t.sum())
    out = {}
    # The threshold is set from the true segments that any segment-pair solver
    # can find (those with a compatible neighbour): a decoupled true segment
    # sits at the attractor delta/s in the full classical solve, and letting it
    # pull the 99 % threshold down to the attractor would admit every
    # decoupled false segment.  Efficiency is still reported over ALL true
    # segments, so the structural miss is visible in eff_e990.
    t_wp = (t & np.asarray(coupled, bool)) if coupled is not None else t
    tau = working_point_threshold(x, t_wp, target_eff)
    act = x > tau
    na = int(act.sum())
    out["tau_e990"] = float(tau)
    out["eff_e990"] = float((act & t).sum()) / max(n_true, 1)
    out["far_e990"] = float((act & ~t).sum()) / na if na else np.nan
    out["nfalse_e990"] = int((act & ~t).sum())
    if coupled is not None:
        nc = int((t & coupled).sum())
        out["n_true_coupled"] = nc
        out["eff_coupled_e990"] = float((act & t & coupled).sum()) / max(nc, 1)
    # sweep over the true amplitudes above the floor: efficiency at far <= 1 %, 0.1 %
    ts = np.unique(x[t_wp])
    ts = ts[ts > WP_TAU_FLOOR]
    if ts.size:
        xs, xt = np.sort(x), np.sort(x[t])
        taus = ts - 1e-12
        nact = len(x) - np.searchsorted(xs, taus, side="right")
        ntru = n_true - np.searchsorted(xt, taus, side="right")
        far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
        eff = ntru / max(n_true, 1)
        for ft, tag in ((0.01, "f010"), (0.001, "f001")):
            ok = far <= ft
            out[f"eff_{tag}"] = float(eff[ok].max()) if ok.any() else 0.0
    else:
        out["eff_f010"] = out["eff_f001"] = 0.0
    # AUC of the true-vs-false ordering
    xf = np.sort(x[~t])
    if n_true and xf.size:
        lo = np.searchsorted(xf, x[t], side="left")
        hi = np.searchsorted(xf, x[t], side="right")
        out["auc"] = float((lo + 0.5 * (hi - lo)).sum() / (n_true * xf.size))
    else:
        out["auc"] = np.nan
    return out


def truth_full(ham):
    return np.asarray(segment_truth_mask(ham, full_space=True), bool)
