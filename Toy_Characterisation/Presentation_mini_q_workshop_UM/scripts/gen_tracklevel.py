"""
Track-level LHCb metrics vs multiplicity (classical vs 1BQF) — the segments->tracks
closure the deck was missing.

For each clean-cell event the store solved (Epsilon_study_2, step, formula eps,
sigma_res=0, gamma=3), we re-solve classically and load the stored quantum vector,
threshold each at x>0.35, group active segments into track candidates (connected
components of the shared-hit graph — the LHCb connected-components tracker), and
match to truth with EventValidator (purity>=0.7, >=3 hits).  Aggregate by T.

Produces figures/track_metrics_vs_T.png : efficiency, ghost rate, clone fraction,
mean purity vs T, classical vs quantum.
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import qtrk_pipeline as qp
from lhcb_velo_toy.generation.entities.track import Track
from lhcb_velo_toy.analysis import EventValidator

THR = 0.35


def tracks_from_solution(ham, sol, threshold=THR):
    """Group active segments into track candidates (connected components of the
    shared-hit graph), returning Track objects with their hit_ids."""
    sh = np.asarray(ham._segment_to_hit_ids)          # (n_seg, 2)
    active = np.where(np.asarray(sol) > threshold)[0]
    if active.size == 0:
        return []
    r = sh[active, 0]; c = sh[active, 1]
    nh = int(sh.max()) + 1
    A = sp.coo_matrix((np.ones(active.size), (r, c)), shape=(nh, nh))
    A = (A + A.T).tocsr()
    _, lab = connected_components(A, directed=False)
    used = np.unique(np.concatenate([r, c]))
    comp = {}
    for h in used:
        comp.setdefault(int(lab[h]), []).append(int(h))
    return [Track(track_id=k, hit_ids=sorted(hs)) for k, hs in enumerate(comp.values())]


def metrics_for(ham, sol, ev):
    trks = tracks_from_solution(ham, sol)
    _, met = EventValidator(ev, trks).match_tracks(purity_min=0.7, min_rec_hits=3)
    met = dict(met); met["n_reco"] = len(trks)
    return met


EPS_WIDE = 0.002      # fixed 2 mrad — the deck's canonical segment-level point
EPS_TIGHT = 0.00042   # noise-matched formula eps (tight) — reference
TGRID = [10, 20, 50, 100, 200, 400, 700, 1000]


def _row(T, rep, solver, eps_tag, m):
    return dict(T=int(T), rep=int(rep), solver=solver, eps=eps_tag, **m)


def collect_classical(eps, eps_tag, reps=3):
    rows = []
    for T in TGRID:
        for rep in range(reps):
            ev, _ = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=1e-4, sigma_res=0.0)
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step")
            solC, _ = qp.solve_classical(ham)
            rows.append(_row(T, rep, "classical", eps_tag, metrics_for(ham, solC, ev)))
            print(f"  [C {eps_tag}] T={T:4d} r{rep} eff={rows[-1]['efficiency']:.2f} "
                  f"ghost={rows[-1]['ghost_rate']:.3f} nreco={rows[-1]['n_reco']}", flush=True)
    return rows


def collect_quantum_wide(reps_max=3):
    """fixed-eps (2 mrad) stored quantum solves + track-level (reaches T=400)."""
    M = pd.read_csv(cm.METRICS_CSV)
    q = M[(M.solver == "quantum") & (M.kernel == "step") &
          (M.eps_provenance == "set") & (np.isclose(M.epsilon, EPS_WIDE, atol=1e-4)) &
          (np.isclose(M.sigma_scatt, 1e-4)) & (M.sigma_res == 0.0) & (M.hit_ineff == 0.0)]
    if "gamma" in q.columns:
        q = q[np.isclose(q.gamma, 3.0)]
    rows = []
    for T in sorted(q.n_trk.unique()):
        for k, (_, r) in enumerate(q[q.n_trk == T].head(reps_max).iterrows()):
            ev = qp.load_event(qp.event_path(r.event_key))
            ham = qp.build_hamiltonian(ev, epsilon=float(r.epsilon), kernel="step", gamma=float(r.gamma))
            solC, _ = qp.solve_classical(ham)
            solQ = qp.rescale_to_signal(np.asarray(qp.load_solution(r.sol_key)["sol"], float), solC)
            rows.append(_row(T, k, "quantum", "2mrad", metrics_for(ham, solQ, ev)))
            print(f"  [Q 2mrad] T={int(T):4d} r{k} eff={rows[-1]['efficiency']:.2f} "
                  f"ghost={rows[-1]['ghost_rate']:.3f} nreco={rows[-1]['n_reco']}", flush=True)
    return rows


def _agg(df, col):
    g = df.groupby("T")[col].agg(["mean", "count", "std"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g.sort_values("T")


def fig_track_metrics(df):
    df = df.copy()
    df["merge_ratio"] = df["n_reco"] / df["T"]      # reco tracks per truth track
    panels = [("efficiency", "Reconstruction efficiency", (0, 1.05)),
              ("merge_ratio", r"reco tracks / truth ($n_{\rm reco}/T$)", (0, 1.1)),
              ("ghost_rate", "Ghost rate", None),
              ("mean_purity", "Mean track purity", (0, 1.05))]
    series = [("classical", "2mrad", cm.C["classical"], "o", "-", "classical (ε=2 mrad)"),
              ("quantum", "2mrad", cm.C["quantum"], "s", "-", "1BQF (ε=2 mrad)"),
              ("classical", "tight", "0.5", "^", "--", "classical (tight ε, reference)")]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    for ax, (col, title, ylim) in zip(axes.ravel(), panels):
        for solver, eps, c, mk, ls, lab in series:
            sub = df[(df.solver == solver) & (df.eps == eps)]
            if sub.empty:
                continue
            a = _agg(sub, col)
            ax.errorbar(a["T"], a["mean"], yerr=a["sem"], marker=mk, ls=ls, capsize=3,
                        color=c, label=lab)
        ax.set_xscale("log"); ax.set_xlabel("Track multiplicity  $T$")
        ax.set_ylabel(title); ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8.5)
    fig.suptitle("Track-level LHCb metrics: segment false-positive bridges merge tracks at high $T$\n"
                 "(wide ε=2 mrad collapses efficiency; tight ε stays perfect — same events)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cm.savefig(fig, "track_metrics_vs_T")


if __name__ == "__main__":
    print("== track-level metrics ==")
    rows = []
    rows += collect_classical(EPS_WIDE, "2mrad")
    rows += collect_classical(EPS_TIGHT, "tight")
    rows += collect_quantum_wide()
    df = pd.DataFrame(rows)
    df.to_csv(cm.ASSETS / "track_metrics.csv", index=False)
    print("metric columns:", [c for c in df.columns])
    fig_track_metrics(df)
    print("done.")
