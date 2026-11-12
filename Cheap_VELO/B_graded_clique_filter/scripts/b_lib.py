"""Experiment B, graded clique filter: shared per-event evaluation.

Plan: Notion 4155d544b9d98186919cd5024e93e8cb, section "B. Graded clique filter".

For one event and one vertex list this module builds the per-vertex hit operator C
(Vertexing/Focusing_scan/scripts/09_vertex_offset.operator_matrix, which is the
07_full_sweep.per_vertex_operator rule with the matrix returned: 4w distance form,
different planes only, each hit only under the vertex that claimed it), applies every
filter of FILTERS exactly with the package module lhcb_velo_toy.solvers.quantum.clique_filters
(per-component eigendecomposition, uniform probe; no circuit is simulated), prices it, reads
tracks from the surviving hits and scores them with the package EventValidator.

Pricing (the section 4.7 / 10_trotter_padding corrections):
    gates per solved event = calls x R_STEPS x gates per product step x max(1, pi / (4 sqrt P_padded))
    gates per product step = k_pairs x (2 (dH_mean - 1) + CX(MCRX with n_s controls)), the Givens
        price of 05_resources.py, hits indexed by plane then x
    R_STEPS = 4 product-formula steps per call (r_5% of the hit operator, section 10)
    calls: one-bit filter 2; step polynomial of degree d: d calls (one block-encoding call per
        degree; the shifted-domain variant also needs an identity term in the block encoding,
        not priced). Ideal filters are not circuits and get no gate count.
    qubits: ceil(log2 n_hits) + 2 (one-bit filter) or + 4 (polynomial), as in 05_resources.py.
This is a cost model, not a transpiled circuit.

Code locations: the Vertexing scripts are untracked files in the main Quantum_Track_Reconstruction
checkout and are imported read-only from there (FS below). The package is taken from the
claude/cheap-velo-B worktree (PKG below) so that clique_filters is importable; the main package
checkout is put on sys.path behind it so fs_lib does not push it in front.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

PKG = "/data/bfys/gscriven/worktrees/velo-toy-cheap-velo-B/src"
PKG_MAIN = "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"
FS = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Focusing_scan/scripts")
RE = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/Vertexing/Real_events/scripts")
for p in (PKG,):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
for p in (PKG_MAIN, str(FS), str(RE)):
    if p not in sys.path:
        sys.path.append(p)

import fs_lib as L  # noqa: E402
import lhcb_velo_toy  # noqa: E402
from lhcb_velo_toy.solvers.quantum import clique_filters as CF  # noqa: E402
from lhcb_velo_toy.solvers.quantum.spectral_filter import mode_system  # noqa: E402

assert lhcb_velo_toy.__file__.startswith(PKG), lhcb_velo_toy.__file__


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F9 = load("fs9", FS / "09_vertex_offset.py")
F7, MPV = F9.F7, F9.MPV

HERE = Path(__file__).resolve().parent.parent
OUT, FIG = HERE / "outputs", HERE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

R_STEPS = 4
# Readout: a hit survives when weight = |f(C) u| sqrt(n) (= |f(m - 1)| on a perfect m-hit clique) is at
# least tau. tau is per filter (the per-solver threshold rule): 0.5 for the ideal filters (f = 0 or 1),
# and the midpoint (|f(1)| + |f(2)|) / 2 between the pair line and the 3-hit line for the circuits.
READ_THRESHOLD_IDEAL = 0.5
DEGREES = (6, 10, 16)
METRICS = ("efficiency", "ghost_rate", "clone_fraction", "mean_purity", "hit_efficiency",
           "n_candidates", "n_ghosts", "n_clones", "n_reconstructible", "n_matched")
# filter key: (label, calls or None for ideal, ancillas)
FILTERS = {
    "ideal34": ("ideal projector, lines {3, 4}", None, 0),
    "step2": ("ideal step, integer lines >= 2", None, 0),
    "band": ("soft band, c > 1.5", None, 0),
    **{f"poly{d}": (f"step polynomial, degree {d}, domain [-alpha, alpha]", d, 4) for d in DEGREES},
    **{f"poly{d}s": (f"step polynomial, degree {d}, shifted domain [lambda_min, alpha]", d, 4) for d in DEGREES},
    "onebit": ("one-bit filter on 4I - C", 2, 2),
    "onebit8": ("one-bit filter on 8I - C", 2, 2),
}


def responses(alpha, lo):
    out = {"ideal34": CF.ideal_line_projector((3, 4)), "step2": CF.ideal_step(1.5),
           "band": CF.soft_band(1.5), "onebit": CF.one_bit_clique(4.0),
           "onebit8": CF.one_bit_clique(8.0)}
    for d in DEGREES:
        out[f"poly{d}"] = CF.design_step_polynomial(d, alpha)
        out[f"poly{d}s"] = CF.design_step_polynomial(d, alpha, lo=lo)
    return out


def register_order(ev):
    """Register index per hit: plane, then x (the order of the 05/07 gate count)."""
    n = len(ev.x)
    order = np.lexsort((ev.x, ev.module))
    idx = np.empty(n, int)
    idx[order] = np.arange(n)
    return idx


def rounds(P):
    return max(1.0, math.pi / (4 * math.sqrt(P))) if P > 0 else np.nan


def evaluate(ev, vz, tag):
    """All filters on the per-vertex operator at the vertices vz.

    Returns (row dict with keys f"{tag}_{filter}_{quantity}", {filter: set of matched truth ids},
    the operator summary dict).
    """
    C, i, j = F9.operator_matrix(ev, list(vz))
    n = len(ev.x)
    idx = register_order(ev)
    ns = max(1, math.ceil(math.log2(n)))
    k = len(i)
    dH = MPV.hamming_mean(idx[i], idx[j])
    cx_step = k * (2 * (dH - 1) + MPV.mcrx_cx(ns))
    u = np.full(n, 1.0 / math.sqrt(n))
    sysd = mode_system(C.tocsr().astype(float), u, iso_lambda=0.0)
    rho = float(np.max(np.abs(sysd["lam"]))) if sysd["lam"].size else 1.0
    alpha = max(rho, 2.0)
    lo = min(float(sysd["lam"].min()) if sysd["lam"].size else -1.0, -1.0)
    row = {f"{tag}_n": n, f"{tag}_ns": ns, f"{tag}_k": k, f"{tag}_dH": dH, f"{tag}_cx_step": cx_step,
           f"{tag}_rho": rho, f"{tag}_alpha": alpha, f"{tag}_lo": lo, f"{tag}_max_comp": sysd["max_comp"],
           f"{tag}_pad": n / 2 ** ns}
    matched = {}
    for key, f in responses(alpha, lo).items():
        r = CF.filter_hit_operator(C, f, sysd=sysd)
        _, calls, anc = FILTERS[key]
        p = f"{tag}_{key}"
        row[f"{p}_P"] = r["P"]
        row[f"{p}_P_padded"] = r["P_padded"]
        row[f"{p}_f0"] = float(np.asarray(f(np.array([0.0])))[0])
        row[f"{p}_f1"] = float(np.asarray(f(np.array([1.0])))[0])
        row[f"{p}_f2"] = float(np.asarray(f(np.array([2.0])))[0])
        if calls is not None:
            row[f"{p}_qubits"] = ns + anc
            row[f"{p}_calls"] = calls
            row[f"{p}_rounds"] = rounds(r["P_padded"])
            row[f"{p}_gates"] = calls * R_STEPS * cx_step * row[f"{p}_rounds"]
        tau = READ_THRESHOLD_IDEAL if calls is None else 0.5 * (abs(row[f"{p}_f1"]) + abs(row[f"{p}_f2"]))
        row[f"{p}_tau"] = tau
        groups = CF.tracks_from_filtered_hits(C, r["weight"], threshold=tau, min_hits=3)
        met, mt = F7.validate(ev, F7.make_tracks([list(g) for g in groups], ev))
        matched[key] = mt
        for m in METRICS:
            row[f"{p}_{m}"] = met[m]
    return row, matched

