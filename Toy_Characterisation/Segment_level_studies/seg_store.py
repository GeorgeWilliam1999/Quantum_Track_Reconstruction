"""seg_store.py — data layer for the Segment_level_studies project.

Reads the decoupled ``qtrk_store`` (events regenerated on demand, solution
vectors cached, metrics recomputed as a VIEW — never trusted from disk).
Everything here is restricted to the **fixed-epsilon** campaign
(``eps_provenance == 'set'``, ε = 2 mrad), which is the acceptance the original
solver segment-efficiency study used.  The fixed-ε campaign sweeps
γ ∈ {1, 2, 3}, hit_ineff ∈ {0, 0.01} at σ_scatt = 1e-4, σ_res = 0, φ_max = 0.2,
step kernel, n_trk ∈ {10 … 1000}.

Two access modes
----------------
* **metrics view** (`fixed_eps_metrics` / `agg_by_ntrk`): the recomputed
  ``manifest/metrics.csv`` rows — segment efficiency / false-rate / counts at the
  γ-aware absolute threshold τ = δ/(δ+γ) + 0.10.  No re-solve: this is what the
  2×2 efficiency figure reads.
* **raw vectors** (`solves_index` / `load_vectors` / `build_A`): per solve, the
  stored solution vector plus the regenerated event → truth mask, per-segment
  direction angle, and (optionally) the matrix A for spectral diagnostics.

Nothing is solved here.  The expensive solves already ran on Condor; this module
only consumes the store.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- locate the shared package + the toy-model lib -------------------------
_SHARED = Path("/data/bfys/gscriven/Quantum_Track_Reconstruction/"
               "Toy_Characterisation/_shared")
for _p in (str(_SHARED), "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp  # noqa: E402

STUDY = "Verify_new_results"
EPS_FIXED = 0.002          # 2 mrad — the fixed Hamiltonian acceptance
GAMMAS = (1.0, 2.0, 3.0)
DELTA = 1.0
HIT_INEFFS = (0.0, 0.01)


def threshold(gamma: float = 3.0, delta: float = DELTA) -> float:
    """γ-aware absolute segment-activation threshold τ = δ/(δ+γ) + 0.10."""
    return qp.threshold_for(float(gamma), float(delta))


# =========================================================================
# Metrics view  (recomputed — read, never re-solve)
# =========================================================================
def fixed_eps_metrics(study: str = STUDY) -> pd.DataFrame:
    """Fixed-ε rows of the recomputed metrics view (γ∈{1,2,3}, drop∈{0,0.01}).

    Returns an empty frame (with the expected columns) if the metrics rebuild
    has not yet written the ε=0.002 rows, so callers can flag-and-move-on.
    """
    df = qp.load_metrics(study=study)
    fx = df[(df["eps_provenance"] == "set") &
            (df["epsilon"].round(6) == EPS_FIXED)].copy()
    if len(fx):
        fx["eff_pct"] = fx["segment_efficiency"] * 100.0
        fx["far_pct"] = fx["segment_false_rate"] * 100.0
        fx["n_false_all"] = fx["n_seg"] - fx["n_true"]
        # wp99 high-efficiency working point (HEADLINE) — present once metrics.csv
        # is rebuilt with build_metrics.py >= 2026-06-14.  The fixed-τ eff/far
        # above are kept as the diagnostic comparison.
        if "segment_efficiency_wp" in fx.columns:
            fx["eff_wp_pct"] = fx["segment_efficiency_wp"] * 100.0
            fx["far_wp_pct"] = fx["segment_false_rate_wp"] * 100.0
    return fx


def _sem(a: np.ndarray) -> float:
    a = np.asarray(a, float)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def agg_by_ntrk(df: pd.DataFrame, solver: str, gamma: float,
                hit_ineff: float) -> dict:
    """Per-n_trk mean/sem arrays for the 2×2 panels, matching the original
    ``_agg_solver`` output (efficiency %, false-rate %, and the four counts).

    Returns ``{}`` when the slice is empty (e.g. rebuild not finished).
    """
    sub = df[(df["solver"] == solver) &
             (df["gamma"] == gamma) &
             (df["hit_ineff"] == hit_ineff)]
    if not len(sub):
        return {}
    has_wp = "eff_wp_pct" in sub.columns and "n_true_active_wp" in sub.columns
    tc, se_m, se_e, fr_m, fr_e = [], [], [], [], []
    se_wp_m, se_wp_e, fr_wp_m, fr_wp_e = [], [], [], []
    tam_wp, tase_wp, fam_wp, fase_wp = [], [], [], []
    ntm, ntse, nfm, nfse, tam, tase, fam, fase = ([] for _ in range(8))
    for n in sorted(sub["n_trk"].unique()):
        g = sub[sub["n_trk"] == n]
        tc.append(float(n))
        se_m.append(g["eff_pct"].mean());   se_e.append(_sem(g["eff_pct"]))
        fr_m.append(g["far_pct"].mean());   fr_e.append(_sem(g["far_pct"]))
        if has_wp:
            se_wp_m.append(g["eff_wp_pct"].mean()); se_wp_e.append(_sem(g["eff_wp_pct"]))
            fr_wp_m.append(g["far_wp_pct"].mean()); fr_wp_e.append(_sem(g["far_wp_pct"]))
            tam_wp.append(g["n_true_active_wp"].mean());  tase_wp.append(_sem(g["n_true_active_wp"]))
            fam_wp.append(g["n_false_active_wp"].mean()); fase_wp.append(_sem(g["n_false_active_wp"]))
        ntm.append(g["n_true"].mean());     ntse.append(_sem(g["n_true"]))
        nfm.append(g["n_false_all"].mean()); nfse.append(_sem(g["n_false_all"]))
        tam.append(g["n_true_active"].mean());  tase.append(_sem(g["n_true_active"]))
        fam.append(g["n_false_active"].mean()); fase.append(_sem(g["n_false_active"]))
    A = np.asarray
    out = dict(tc=A(tc), se_m=A(se_m), se_e=A(se_e), fr_m=A(fr_m), fr_e=A(fr_e),
               ntm=A(ntm), ntse=A(ntse), nfm=A(nfm), nfse=A(nfse),
               tam=A(tam), tase=A(tase), fam=A(fam), fase=A(fase),
               n_reps=int(sub.groupby("n_trk").size().min()))
    if has_wp:  # wp99 HEADLINE arrays (se_*/fr_*/tam/fam stay as the fixed-τ diagnostic)
        out.update(se_wp_m=A(se_wp_m), se_wp_e=A(se_wp_e),
                   fr_wp_m=A(fr_wp_m), fr_wp_e=A(fr_wp_e),
                   tam_wp=A(tam_wp), tase_wp=A(tase_wp),
                   fam_wp=A(fam_wp), fase_wp=A(fase_wp))
    return out


# =========================================================================
# Raw vectors  (regenerate event/A on demand; load stored solution)
# =========================================================================
def solves_index(solver: str | None = None, gamma: float | None = None,
                 hit_ineff: float | None = None,
                 existing_only: bool = True) -> pd.DataFrame:
    """Fixed-ε rows of ``manifest/solutions.csv`` (the solve inventory).

    ``existing_only`` (default) drops rows whose solution ``.npz`` is not yet on
    disk — the manifest can list solves that are mid-write or still queued, and
    loading those would crash.
    """
    s = pd.read_csv(qp.manifest_dir() / "solutions.csv")
    s = s[(s["eps_provenance"] == "set") & (s["epsilon"].round(6) == EPS_FIXED)]
    if solver is not None:
        s = s[s["solver"] == solver]
    if gamma is not None:
        s = s[s["gamma"] == gamma]
    if hit_ineff is not None:
        s = s[s["hit_ineff"] == hit_ineff]
    if existing_only:
        s = s[s["sol_key"].map(qp.solution_exists)]
    return s.reset_index(drop=True)


def _event_of(row) -> object:
    return qp.ensure_event(
        n_trk=int(row["n_trk"]), rep=int(row["rep"]),
        sigma_scatt=float(row["sigma_scatt"]), sigma_res=float(row["sigma_res"]),
        phi_max=float(row["phi_max"]), hit_ineff=float(row["hit_ineff"]))[0]


def build_A(row, gamma: float | None = None):
    """Regenerate the Hamiltonian (with ``.A``) for one solve row.

    A is never persisted — this rebuilds it from the stored event with the
    row's own (γ, δ, ε, kernel).  Exposes ``._segment_vectors`` for angles and
    ``.A`` (csc) for the spectrum.
    """
    ev = _event_of(row)
    return qp.build_hamiltonian(
        ev, epsilon=float(row["epsilon"]),
        kernel=str(row.get("kernel", "step")),
        gamma=float(gamma if gamma is not None else row["gamma"]),
        delta=float(row.get("delta", DELTA)))


def segment_angles_mrad(ham) -> np.ndarray:
    """Per-segment polar angle from the beam (z) axis, in mrad.

    Each segment has a unit direction vector; the polar angle θ = arccos(v_z)
    measures how far the segment tilts off-beam.  True (on-track) segments are
    collimated within the generation cone; cross-track false segments spread to
    larger θ.
    """
    v = np.asarray(ham._segment_vectors, float)
    vz = np.clip(np.abs(v[:, 2]), -1.0, 1.0)
    return np.arccos(vz) * 1e3


def load_vectors(row, classical_partner: pd.DataFrame | None = None) -> dict:
    """Stored solution vector + geometry for one solve row.

    Returns ``sol`` (for quantum, **rescaled** to the matching classical L2
    norm so amplitudes are comparable), ``truth`` (bool mask), ``theta_mrad``
    (per-segment angle), ``n_seg`` and the solve params.  ``classical_partner``
    is the fixed-ε classical solves frame; needed to rescale a quantum solve.
    """
    sol = np.asarray(qp.load_solution(row["sol_key"])["sol"], float)
    ham = build_A(row)
    truth = np.asarray(qp.truth_from_event(_event_of(row)), bool)
    theta = segment_angles_mrad(ham)
    out = dict(solver=str(row["solver"]), gamma=float(row["gamma"]),
               n_trk=int(row["n_trk"]), rep=int(row["rep"]),
               hit_ineff=float(row["hit_ineff"]), n_seg=int(ham.n_segments),
               truth=truth, theta_mrad=theta, sol=sol,
               tau=threshold(float(row["gamma"]), float(row.get("delta", DELTA))))
    if str(row["solver"]) != "classical" and classical_partner is not None:
        c = classical_partner[(classical_partner["event_key"] == row["event_key"]) &
                              (classical_partner["ham_key"] == row["ham_key"])]
        if len(c):
            sol_C = np.asarray(qp.load_solution(c.iloc[0]["sol_key"])["sol"], float)
            # signal-support rescale (the corrected convention — NOT full ||sol_C||,
            # which is false-bulk dominated and inflates with multiplicity)
            out["sol"] = qp.rescale_to_signal(sol, sol_C, out["tau"])
            out["sol_raw"] = sol
    return out
