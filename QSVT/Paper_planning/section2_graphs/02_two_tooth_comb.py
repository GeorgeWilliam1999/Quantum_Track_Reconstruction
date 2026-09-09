#!/usr/bin/env python3
"""Two-tooth versus four-tooth line comb.

The production comb puts a Gaussian pass at all four true-track lines
s - 2cos(k pi/5), k = 1..4.  But the bias b = delta*1 is invariant under the
chain's reflection, so it has exactly zero overlap with the two
reflection-antisymmetric modes: on a perfect P4 track the lines at k = 2
(3.382) and k = 4 (5.618) are DARK and carry no true amplitude at all.

Those two teeth therefore cannot pass any true weight on clean events -- they
can only admit false weight that happens to land there.

On clean events both designs already reach a false rate of exactly zero, so the
payoff is not false rate but DEGREE: a comb with half as many features to
resolve should stay realisable at lower polynomial degree, and degree is circuit
depth.  The experiment is therefore a downward degree scan -- the lowest degree
at which each design still separates true from false -- not a single-degree
comparison.

Under noise the production comb has no discriminating power at 99% efficiency
for ANY tooth pattern, because 0.3-0.5% of true segments sit in isolated (P1)
clusters that the comb nulls by design and 4-18% sit in tangles of five or more
segments whose eigenvalues are off the four true lines.  That ceiling is what
section 9 refits the coefficients to escape; it is not a property of the tooth
pattern.  The noisy configurations are therefore also read at lower matched
efficiency, where the comparison is not degenerate.

Outputs
  outputs/02_two_tooth.csv       one row per (config, T, rep, design, degree, hw)
  outputs/02_min_degree.csv      lowest degree reaching the far target, per design
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                      # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon              # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402
from lhcb_velo_toy.solvers.quantum.spectral_filter import (      # noqa: E402
    mode_system as _pkg_mode_system,
    apply_response as _pkg_apply_response,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)

GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
P4_LINES = tuple(S - 2.0 * np.cos(k * np.pi / 5.0) for k in (1, 2, 3, 4))
BRIGHT_LINES = (P4_LINES[0], P4_LINES[2])          # k = 1, 3
DARK_LINES = (P4_LINES[1], P4_LINES[3])            # k = 2, 4

CONFIGS = {
    "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
    "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    "heavy":    dict(sigma_scatt=1e-4, sigma_res=0.02, phi_max=0.2, hit_ineff=0.01),
}
TGRID = (100, 200, 400)
FAR_OK = 1e-3          # "still works": far at 99% efficiency below this
REPS = (0, 1, 2)
EFF_TARGETS = (0.90, 0.95, 0.99, 0.995)
MAX_COMP = 3000

# (design name, lines, half-width).  hw=0.18 is the production default; the
# two-tooth comb has half as many features to resolve at the same degree, so
# narrower teeth are also tried.
DESIGNS = [
    ("four teeth (production)", P4_LINES,     0.18),
    ("two teeth (bright only)", BRIGHT_LINES, 0.18),
    ("two teeth (bright only)", BRIGHT_LINES, 0.14),
    ("two teeth (bright only)", BRIGHT_LINES, 0.10),
    ("two teeth (bright only)", BRIGHT_LINES, 0.24),
    ("two teeth (dark only)",   DARK_LINES,   0.18),   # control: should be poor
]
DEGREES = (6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64)

LOG = open(OUT / "02_two_tooth.log", "w")
def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def mode_system(M, b, iso_lambda):
    """Per-component eigendecomposition -> sparse mode basis U, eigenvalues
    lam, overlaps w.

    The implementation is the package's (migrated 2026-09-09; verified
    bit-exact against the local copy it replaced, over 3 event sizes and the
    inverse / degree-40 comb / one-bit cosine responses).  This wrapper keeps
    the study's own MAX_COMP guard, which is tighter than the package default.

    NOTE the package returns the isolated-segment eigenvalues under the key
    ``iso_lam`` (an array), where the local version returned the scalar
    ``iso_lambda``.  Anything reading that key must be updated.
    """
    return _pkg_mode_system(M, b, iso_lambda=iso_lambda, max_comp=MAX_COMP)


def amplitudes(sysd, f):
    """Amplitudes |f(M) b| for a decomposed system -- the package's
    ``apply_response`` (verified bit-exact against the local copy)."""
    return _pkg_apply_response(sysd, f)


def matched_metrics(x, truth):
    """Efficiency-first / matched-efficiency reading: tau at the k-th largest
    true amplitude, so efficiency is fixed by construction and far is the price."""
    x = np.abs(np.asarray(x).ravel()); truth = np.asarray(truth, bool)
    n_true = int(truth.sum())
    ts = np.sort(x[truth])[::-1]
    out = {}
    for e in EFF_TARGETS:
        k = min(n_true, max(1, int(np.ceil(e * n_true))))
        tau = ts[k - 1]
        act = x >= tau
        na = int(act.sum())
        tag = f"e{int(round(e * 1000)):03d}"
        out[f"far_{tag}"] = (int((act & ~truth).sum()) / na) if na else np.nan
        out[f"eff_{tag}"] = float((act & truth).sum()) / n_true
        out[f"tau_{tag}"] = float(tau)
    # Separation margin.  On clean events the false rate saturates at zero for
    # every workable design, so it cannot rank them; the margin can.  >1 means
    # the lowest true amplitude still sits above the highest false one.
    xt, xf = x[truth], x[~truth]
    out["min_true"] = float(xt.min())
    out["max_false"] = float(xf.max())
    out["margin"] = float(xt.min() / xf.max()) if xf.max() > 0 else np.inf
    # margin at the 99% point: robust to the handful of worst true segments
    out["margin_p1"] = (float(np.quantile(xt, 0.01) / np.quantile(xf, 0.99999))
                        if xf.max() > 0 else np.inf)
    return out


def main():
    rows = []
    for config, nz in CONFIGS.items():
        eps = compute_epsilon(sigma_scatt=nz["sigma_scatt"], sigma_res=nz["sigma_res"])
        for T in TGRID:
            for rep in REPS:
                t0 = time.time()
                try:
                    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **nz)
                    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                               gamma=GAMMA, delta=DELTA)
                    truth = np.asarray(qp.truth_from_event(ev), bool)
                    A = sp.csr_matrix(ham.A)
                    b = DELTA * np.ones(A.shape[0])
                    sysd = mode_system(A, b, S)
                except Exception as exc:                          # noqa: BLE001
                    say(f"[skip] {config} T={T} rep={rep}: {type(exc).__name__}: {exc}")
                    continue
                lam_min, lam_max = float(sysd["lam"].min()), float(sysd["lam"].max())
                span = lam_max - lam_min
                pad = 0.05 * span + 0.2
                lo = min(lam_min, S - 2) - pad
                hi = max(lam_max, S + 2) + pad
                for deg in DEGREES:
                    for name, lines, hw in DESIGNS:
                        p = design_line_comb_inverse(degree=deg, s=S, lines=lines,
                                                     hw=hw, domain=(lo, hi))
                        x = amplitudes(sysd, lambda l, _p=p: np.abs(_p(l)))
                        m = matched_metrics(x, truth)
                        rows.append(dict(config=config, T=T, rep=rep, design=name,
                                         domain_lo=lo, domain_hi=hi,
                                         n_teeth=len(lines), degree=deg, hw=hw,
                                         n_seg=A.shape[0], n_true=int(truth.sum()),
                                         **m))
                say(f"[ok] {config:8s} T={T:5d} rep={rep}  "
                    f"n_seg={A.shape[0]:>8d}  {time.time()-t0:6.1f}s")
                pd.DataFrame(rows).to_csv(OUT / "02_two_tooth.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "02_two_tooth.csv", index=False)
    piv = (df[df.degree == 40]
             .groupby(["config", "T", "design", "hw"])["far_e990"]
             .agg(["mean", "std"]).reset_index())
    say("\nfalse rate at 99% efficiency, degree 40:\n" + piv.to_string(index=False))

    # lowest degree at which each design still separates, per (config, T, rep)
    ok = df[df.far_e990 <= FAR_OK]
    mind = (ok.groupby(["config", "T", "rep", "design", "hw"])["degree"]
              .min().reset_index().rename(columns={"degree": "min_degree"}))
    mind.to_csv(OUT / "02_min_degree.csv", index=False)
    say("\nlowest degree with far(99% eff) <= %g:\n" % FAR_OK
        + mind.groupby(["config", "T", "design", "hw"])["min_degree"]
              .agg(["mean", "min", "max"]).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
