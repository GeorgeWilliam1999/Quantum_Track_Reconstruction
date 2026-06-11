"""
Run the Hough tracker on the SAME qtrk_store events as the classical / 1BQF / QSVT
benchmark (Verify_new_results clean config) — Hough as a first-class 4th solver.

The segment solvers report SEGMENT-level metrics; Hough builds tracks directly, so
it produces TRACK-level efficiency / ghost / clone — exactly the track-level numbers
the segment store currently lacks (the §15-17 gap). We therefore report Hough's own
track metrics on the shared events, and extend them to T=700/1000 where the 1BQF
statevector did not scale.

Config selected (the clean Verify point): sigma_scatt=1e-4, sigma_res=0,
hit_ineff=0, ghost_rate=0, phi_max=0.2.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "Toy_Characterisation", "_shared"))

import qtrk_pipeline as qp  # noqa: E402
from hough_prototype import (hit_directions, hough_accumulate, extract_tracks,
                             evaluate)  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def select_clean(df):
    return df[(df.sigma_scatt == 1e-4) & (df.sigma_res == 0.0) &
              (df.hit_ineff == 0.0) & (df.ghost_rate == 0.0) &
              (df.phi_max == 0.2)].copy()


def run_event(event_key):
    ev = qp.load_event(qp.store.event_path(event_key))
    D, planes, truth = hit_directions(ev)
    t0 = time.time()
    counts, flat = hough_accumulate(D)
    cands = extract_tracks(D, planes, flat)
    dt = time.time() - t0
    res = evaluate(cands, truth, ev)
    res.update(t_solve=dt, n_hits=len(D))
    return res


if __name__ == "__main__":
    df = pd.read_csv(qp.store.manifest_dir() / "events.csv")
    sel = select_clean(df)
    print(f"selected {len(sel)} clean events; T = {sorted(sel.n_trk.unique())}")
    rows = []
    for _, r in sel.iterrows():
        p = qp.store.event_path(r.event_key)
        if not os.path.exists(p):
            continue
        res = run_event(r.event_key)
        res.update(n_trk=int(r.n_trk), rep=int(r.rep))
        rows.append(res)
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, "hough_store_per_event.csv"), index=False)

    agg = (per.groupby("n_trk")
              .agg(n_events=("eff", "size"),
                   eff=("eff", "mean"), eff_sem=("eff", "sem"),
                   ghost=("ghost_rate", "mean"), ghost_sem=("ghost_rate", "sem"),
                   clone=("clone_rate", "mean"),
                   t_ms=("t_solve", lambda s: 1e3 * s.mean()))
              .reset_index())
    agg.to_csv(os.path.join(OUT, "hough_store_summary.csv"), index=False)
    print(agg.to_string(index=False))

    # figure: Hough track-level eff/ghost vs T over the shared events
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].errorbar(agg.n_trk, agg.eff, yerr=agg.eff_sem, fmt="o-", label="efficiency")
    ax[0].errorbar(agg.n_trk, agg.ghost, yerr=agg.ghost_sem, fmt="s--",
                   color="crimson", label="ghost rate")
    ax[0].set(xscale="log", xlabel="n_tracks T", ylabel="track-level rate",
              ylim=(0, 1.05),
              title="Hough on the shared qtrk_store events (clean Verify config)")
    ax[0].legend()
    ax[1].loglog(agg.n_trk, agg.t_ms, "o-")
    ax[1].set(xlabel="n_tracks T", ylabel="solve [ms]",
              title="Hough wall time on store events (1 core)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hough_store_benchmark.png"), dpi=130)
    print("saved hough_store_summary.csv + hough_store_benchmark.png")
