"""qtrk_view — store-backed data layer for the Verify_new_results notebooks.

Replaces the old per-event ``pickle.load(outputs/**/pickles/*.pkl)`` + live-solve
path with the decoupled ``qtrk_store`` (see the ``qtrk-data-pipeline`` skill).

Everything here is a recomputed VIEW: metrics are never trusted from disk, they
are recomputed from the stored solution vectors with the gamma-aware absolute
threshold (tau = delta/(delta+gamma) + 0.10 = 0.35 at gamma=3).  Nothing is
solved here — high-T quantum statevector runs come from the Condor GPU pipeline.

Usage in a notebook
-------------------
    import sys
    sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/"
                       "Toy_Characterisation/_shared")
    from qtrk_view import (load_view, paired, aggregate, get_vectors,
                           vector_records, threshold)
    view   = load_view(study="Verify_new_results")     # tidy metrics view
    agg_sv = aggregate(paired(view))                   # statevector aggregate
    recs   = vector_records(view)                      # raw sol_C/sol_Q/truth per event
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.stats import sem as _sem
except Exception:                                       # pragma: no cover
    def _sem(x):
        x = np.asarray(x, float)
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0

import qtrk_pipeline as qp

GAMMA, DELTA = 3.0, 1.0


def threshold(gamma: float = GAMMA, delta: float = DELTA) -> float:
    """gamma-aware absolute activation threshold (= 0.35 at gamma=3)."""
    return qp.threshold_for(gamma, delta)


# ---------------------------------------------------------------- view -------
def load_view(study: str | None = None, readout: str = "statevector") -> pd.DataFrame:
    """Tidy metrics view from the store (recomputed, fixed metrics).

    ``study`` filters on the shared ``studies`` tag (an event may serve several
    studies).  Quantum rows are restricted to ``readout`` (default statevector).
    """
    df = qp.load_metrics()
    if study is not None:
        df = df[df["studies"].fillna("").str.contains(study, regex=False)].copy()
    # keep all classical + only the requested quantum readout
    df = df[(df["solver"] == "classical") |
            ((df["solver"] == "quantum") & (df["readout"] == readout))].copy()
    return df


# --------------------------------------------------- pair classical+quantum --
def paired(view: pd.DataFrame) -> pd.DataFrame:
    """Pair the classical and quantum solve of each event (same ``event_key``).

    Returns one row per event with both solvers' metrics, ready to aggregate.
    """
    cols = ["event_key", "ham_key", "n_trk", "rep", "sigma_scatt", "sigma_res",
            "phi_max", "hit_ineff", "epsilon", "segment_efficiency",
            "segment_purity", "segment_false_rate", "n_active", "n_true_active",
            "n_false_active", "n_seg", "n_true", "t_solve", "sol_key"]
    # wp99 high-efficiency working-point columns (present once metrics.csv is
    # rebuilt with build_metrics.py >= 2026-06-14); carried through if available.
    WP = ["segment_efficiency_wp", "segment_false_rate_wp", "segment_purity_wp",
          "tau_wp", "n_active_wp", "n_true_active_wp", "n_false_active_wp"]
    wp = [c for c in WP if c in view.columns]
    C = view[view.solver == "classical"][cols + wp]
    Q = view[view.solver == "quantum"][cols + wp + ["cos_QC", "P_anc"]]
    # pair on the SAME (event, matrix): classical & quantum solve of the same A
    m = Q.merge(C, on=["event_key", "ham_key"], suffixes=("_Q", "_C"))
    # convenience short names used by the plotting cells
    m["n_trk"] = m["n_trk_Q"]
    m["eff_C"] = m["segment_efficiency_C"]; m["eff_Q"] = m["segment_efficiency_Q"]
    m["pur_C"] = m["segment_purity_C"];     m["pur_Q"] = m["segment_purity_Q"]
    m["far_C"] = m["segment_false_rate_C"]; m["far_Q"] = m["segment_false_rate_Q"]
    m["cos"]   = m["cos_QC"]
    if "segment_efficiency_wp" in wp:       # the HEADLINE working-point short names
        m["eff_C_wp"] = m["segment_efficiency_wp_C"]
        m["eff_Q_wp"] = m["segment_efficiency_wp_Q"]
        m["far_C_wp"] = m["segment_false_rate_wp_C"]
        m["far_Q_wp"] = m["segment_false_rate_wp_Q"]
        m["pur_C_wp"] = m["segment_purity_wp_C"]
        m["pur_Q_wp"] = m["segment_purity_wp_Q"]
        m["tau_wp_C"] = m["tau_wp_C"]; m["tau_wp_Q"] = m["tau_wp_Q"]
    return m


def aggregate(m: pd.DataFrame) -> pd.DataFrame:
    """Per-n_trk mean/sem for every metric the plotting cells reference.

    Produces columns ``<metric>_mean`` / ``<metric>_sem`` for metric in
    {eff_C, eff_Q, pur_C, pur_Q, far_C, far_Q, cos}, plus ``n_reps``.
    """
    if m is None or len(m) == 0:
        return pd.DataFrame(columns=["n_trk", "n_reps"])
    g = m.groupby("n_trk")
    out = pd.DataFrame({"n_trk": sorted(m["n_trk"].unique())}).set_index("n_trk")
    out["n_reps"] = g.size()
    base = ["eff_C", "eff_Q", "pur_C", "pur_Q", "far_C", "far_Q", "cos"]
    # add the wp99 headline columns when present
    wp = [c for c in ["eff_C_wp", "eff_Q_wp", "far_C_wp", "far_Q_wp",
                      "pur_C_wp", "pur_Q_wp", "tau_wp_Q"] if c in m.columns]
    for col in base + wp:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_sem"]  = g[col].apply(lambda x: float(_sem(x)) if len(x) > 1 else 0.0)
    return out.reset_index()


# ----------------------------------------------------------- raw vectors -----
def _event_for(row) -> object:
    return qp.ensure_event(
        n_trk=int(row["n_trk"]), rep=int(row["rep"]),
        sigma_scatt=float(row["sigma_scatt"]), sigma_res=float(row["sigma_res"]),
        phi_max=float(row["phi_max"]), hit_ineff=float(row["hit_ineff"]))[0]


def get_vectors(view: pd.DataFrame, event_key: str, ham_key: str | None = None):
    """Raw solution vectors + truth for one (event, matrix) (no metrics baked in).

    Returns dict with sol_C, sol_Q_raw, sol_Q_scaled (rescaled to ||sol_C||),
    truth (bool mask), and the event's physics params.  This is what the
    amplitude-distribution / leakage cells need.  ``ham_key`` selects the matrix
    so classical and quantum come from the SAME A.
    """
    sub = view[view.event_key == event_key]
    if ham_key is not None:
        sub = sub[sub.ham_key == ham_key]
    crow = sub[sub.solver == "classical"].iloc[0]
    sol_C = np.asarray(qp.load_solution(crow.sol_key)["sol"], float)
    ev = _event_for(crow)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    nt = int(truth.sum())
    out = dict(event_key=event_key, ham_key=str(crow.ham_key),
               n_trk=int(crow.n_trk), rep=int(crow.rep),
               sigma_res=float(crow.sigma_res), sigma_scatt=float(crow.sigma_scatt),
               phi_max=float(crow.phi_max), hit_ineff=float(crow.hit_ineff),
               sol_C=sol_C, truth=truth, seg_truth=truth,          # 'seg_truth' alias
               n_true=nt, n_true_clean=nt, n_true_all=nt,
               n_seg=int(sol_C.size), drop_rate=float(crow.hit_ineff))
    # recomputed metric-count dicts, drop-in compatible with the OLD pkl schema
    tau = threshold()
    mC = qp.metrics_at(sol_C, truth, threshold=tau)
    out["seg_C"] = {**mC, "n_true_clean": mC["n_true_all"]}
    q = sub[sub.solver == "quantum"]
    if len(q):
        qrow = q.iloc[0]
        sol_Q_raw = np.asarray(qp.load_solution(qrow.sol_key)["sol"], float)
        sol_Q_scaled = qp.rescale_to(sol_Q_raw, sol_C)
        out["sol_Q_raw"] = sol_Q_raw
        out["sol_Q_scaled"] = sol_Q_scaled
        mQ = qp.quantum_metrics(sol_Q_raw, sol_C, truth, tau)
        out["seg_Q"] = {**mQ, "n_true_clean": mQ["n_true_all"]}
        out["cos_QC"] = float(mQ["cos_QC"])
        out["success_prob"] = float(qrow.P_anc)
        aC, aQ = sol_C > tau, sol_Q_scaled > tau
        inter = int(np.logical_and(aC, aQ).sum()); union = int(np.logical_or(aC, aQ).sum())
        nrm = float(np.linalg.norm(sol_C))
        out["fidelity"] = dict(
            cos=float(mQ["cos_QC"]),
            rel_l2=float(np.linalg.norm(sol_Q_scaled - sol_C) / nrm) if nrm else np.nan,
            jaccard=float(inter / union) if union else np.nan,
            n_active_C=int(aC.sum()), n_active_Q=int(aQ.sum()))
    return out


def get_one(view: pd.DataFrame, n_trk: int, rep: int = 0, require_quantum: bool = True):
    """Single store record for (n_trk, rep) or None — for the specific-(T,rep) loads.

    With ``require_quantum`` (default), returns None unless a quantum solve exists
    for that point (so a sampling-filtered view -> None, since the campaign is
    statevector-only and has no sampling quantum solves).
    """
    sub = view[(view.n_trk == n_trk) & (view.rep == rep)]
    q = sub[sub.solver == "quantum"]
    if len(q):
        r = q.iloc[0]
        return get_vectors(view, r.event_key, r.ham_key)
    if require_quantum:
        return None
    c = sub[sub.solver == "classical"]
    if len(c):
        r = c.iloc[0]
        return get_vectors(view, r.event_key, r.ham_key)
    return None


def vector_records(view: pd.DataFrame):
    """List of get_vectors(...) dicts for every event that has a quantum solve.

    Drop-in replacement for the old ``records = [pickle.load(p) for p in pkls]``
    loops, with keys sol_C / sol_Q_scaled / sol_Q_raw / truth / n_trk / rep.
    """
    q = view[view.solver == "quantum"][["event_key", "ham_key"]].drop_duplicates()
    return [get_vectors(view, ek, hk) for ek, hk in zip(q.event_key, q.ham_key)]


# ------------------------------------- per-event records DataFrame (shim) ----
# Columns the §5/§6 ingestion of the quantum notebook expects.  Metrics come
# from the recomputed view; jaccard/rel_l2 need the raw vectors; track-level
# (trk_*) is NOT in the store (no downstream tracking) -> NaN, gracefully.
_TRK_COLS = [f"trk_{s}_{k}" for s in ("Ccc", "Qcc", "Clay", "Qlay")
             for k in ("efficiency", "ghost_rate", "clone_fraction",
                       "mean_purity", "hit_efficiency", "n_rec_tracks")]


def records_df(view: pd.DataFrame, with_vectors: bool = True) -> pd.DataFrame:
    """Per-event metrics table matching the notebook's old ``load_results`` output.

    eff/pur/far/cos/P_anc/timing/n_seg from the recomputed view; jaccard & rel_l2
    computed from the rescaled vectors (``with_vectors``); trk_* = NaN (the store
    has no downstream tracking — recompute separately if a tracking figure needs it).
    """
    m = paired(view)
    if len(m) == 0:
        return pd.DataFrame(columns=["n_trk", "rep"])
    tau = threshold()
    rows = []
    for _, r in m.iterrows():
        d = dict(
            n_trk=int(r["n_trk"]), rep=int(r["rep_Q"]), shots=0,
            readout="statevector", n_hits=np.nan, n_seg=float(r["n_seg_Q"]),
            eff_C=r["eff_C"], eff_Q=r["eff_Q"], pur_C=r["pur_C"], pur_Q=r["pur_Q"],
            far_C=r["far_C"], far_Q=r["far_Q"],
            cosine=float(r["cos"]), P_anc=float(r["P_anc"]),
            n_active_C=float(r["n_active_C"]), n_active_Q=float(r["n_active_Q"]),
            t_classical=float(r["t_solve_C"]), t_quantum=float(r["t_solve_Q"]),
            wall_total=float(r["t_solve_C"]) + float(r["t_solve_Q"]),
            rel_l2=np.nan, jaccard=np.nan,
        )
        if with_vectors:
            v = get_vectors(view, r["event_key"], r["ham_key"])
            sC, sQ = v["sol_C"], v.get("sol_Q_scaled")
            if sQ is not None:
                aC, aQ = sC > tau, sQ > tau
                inter = np.logical_and(aC, aQ).sum()
                union = np.logical_or(aC, aQ).sum()
                d["jaccard"] = float(inter / union) if union else np.nan
                nrm = np.linalg.norm(sC)
                d["rel_l2"] = float(np.linalg.norm(sQ - sC) / nrm) if nrm else np.nan
        for c in _TRK_COLS:
            d[c] = np.nan
        rows.append(d)
    return pd.DataFrame(rows).sort_values(["n_trk", "rep"]).reset_index(drop=True)
