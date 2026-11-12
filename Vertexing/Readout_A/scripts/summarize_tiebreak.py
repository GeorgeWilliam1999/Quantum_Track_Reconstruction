#!/usr/bin/env python3
"""Old vs new numbers of experiment A after the deterministic tie-break (run_tiebreak.py).

Pooled as summarize.py: efficiency = sum matched / sum reconstructible, ghost rate =
sum ghosts / sum candidates. Toy-like = prompt, not an electron, >= 4 hits, p > 2 GeV.
Checks:
  old == A published   the "old" variant (sig_digits = None) reproduces outputs/real_readouts_events.csv
                       (reading I, readout ccu) and real_control_events.csv (score deg) per event;
  noisy == new         1e-14 relative noise on the scores gives the same track list on every event
                       (hash of the sorted hit-index groups).
Output: outputs/summary_tiebreak.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "outputs"
SOLV = ("classical", "1bqf", "qsvt", "control")
L = []
P = L.append


def pooled(g):
    return pd.Series(dict(efficiency=g.n_matched.sum() / max(g.n_reconstructible.sum(), 1),
                          ghost_rate=g.n_ghosts.sum() / max(g.n_candidates.sum(), 1)))


for mode in ("real", "toy"):
    e = pd.read_csv(OUT / f"tiebreak_{mode}_events.csv")
    t = pd.read_csv(OUT / f"tiebreak_{mode}_tracks.csv")
    w = e.pivot_table(index=["event", "solver"], columns="variant", values=["efficiency", "groups_key"],
                      aggfunc="first")
    P(f"\n=== {'Run 3, 103 events' if mode == 'real' else 'toy, T = 1000, 3 events per point'} ===")
    # invariance and change counts
    for s in SOLV:
        ws = w.xs(s, level="solver")
        inv = int((ws[("groups_key", "noisy")] == ws[("groups_key", "new")]).sum())
        chg_tracks = int((ws[("groups_key", "old")] != ws[("groups_key", "new")]).sum())
        chg_eff = int((np.abs(ws[("efficiency", "old")] - ws[("efficiency", "new")]) > 1e-12).sum())
        d = ws[("efficiency", "new")] - ws[("efficiency", "old")]
        P(f"{s:9s}: noisy == new on {inv}/{len(ws)} events; tracks changed old -> new on {chg_tracks} events, "
          f"efficiency changed on {chg_eff} (per event {d.min():+.3f} ... {d.max():+.3f}, mean {d.mean():+.4f})")
    # reproduction of A's published numbers by the old ranking
    if mode == "real":
        a = pd.read_csv(OUT / "real_readouts_events.csv")
        a = a[(a.reading == "I") & (a.readout == "ccu")].set_index(["event", "solver"]).efficiency
        c = pd.read_csv(OUT / "real_control_events.csv")
        c = c[c.score == "deg"].set_index("event").efficiency
    else:
        a = pd.read_csv(OUT / "toy_readouts_events.csv")
        a = a[(a.reading == "I") & (a.readout == "ccu")].set_index(["event", "solver"]).efficiency
        c = pd.read_csv(OUT / "toy_control_events.csv")
        c = c[c.score == "deg"].set_index("event").efficiency
    for s in SOLV:
        old = w.xs(s, level="solver")[("efficiency", "old")]
        ref = c.reindex(old.index) if s == "control" else a.reindex(pd.MultiIndex.from_arrays([old.index, [s] * len(old)])).values
        dd = np.abs(old.values - np.asarray(ref, float))
        P(f"{s:9s}: old ranking vs A published: max |diff| {np.nanmax(dd):.1e} on {int(np.sum(~np.isnan(dd)))} events")
    # tables
    if mode == "real":
        P("\nsolver     variant  efficiency  ghost  toy-like  3-hit  secondaries  p<1GeV")
        cats = {"toy-like": t.prompt & ~t.electron & (t.n_hits >= 4) & (t.p > 2000), "3-hit": t.n_hits == 3,
                "secondaries": ~t.prompt, "p<1GeV": t.p < 1000}
        for s in SOLV:
            for v in ("old", "new"):
                g = e[(e.solver == s) & (e.variant == v)]
                pl = pooled(g)
                cv = [t[m][f"found_{s}_{v}"].mean() for m in cats.values()]
                P(f"{s:9s}  {v:6s}  {pl.efficiency:10.4f}  {pl.ghost_rate:.4f}  {cv[0]:8.4f}  {cv[1]:.4f}  {cv[2]:11.4f}  {cv[3]:.4f}")
    else:
        P("\nsolver     variant  efficiency (1 / 5 / 10 / 20 PVs)        ghost rate")
        for s in SOLV:
            for v in ("old", "new"):
                g = e[(e.solver == s) & (e.variant == v)]
                pl = g.groupby("n_pv").apply(pooled)
                P(f"{s:9s}  {v:6s}  " + " / ".join(f"{x:.4f}" for x in pl.efficiency) + "    "
                  + " / ".join(f"{x:.4f}" for x in pl.ghost_rate))
txt = "\n".join(L)
print(txt)
(OUT / "summary_tiebreak.txt").write_text(txt.lstrip() + "\n")
