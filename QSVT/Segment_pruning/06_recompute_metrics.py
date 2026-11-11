#!/usr/bin/env python3
"""Recompute every metric column of outputs/<config>_solves.csv from the stored
amplitudes, with each variant's working point set from the true segments that
variant keeps (fix of 2026-10-13: the first pass set the components-under-3
reading from the decoupled-prune set, which includes the true pairs that
variant zeroes).

The pruned solves equal the full solve on the kept segments (checked per
event, maxdiff_vs_full), and the metrics are scale-invariant, so the pruned
amplitudes are the stored full amplitudes masked to the kept set.  Non-metric
columns (P, timings, qubits, maxdiff) are left untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from common import CONFIGS, OUT, build_full, epsilon_for, load_event, metrics, pruned_copy
from lhcb_velo_toy.solvers.hamiltonians import coupling_pattern

METRIC_COLS = ["tau_e990", "eff_e990", "far_e990", "nfalse_e990", "n_true_coupled",
               "eff_coupled_e990", "eff_f010", "eff_f001", "auc"]


def comp3_mask(cfg, T, rep, n_full):
    ev, _ = load_event(cfg, T, rep)
    ham = build_full(ev, epsilon_for(cfg))
    hp = pruned_copy(ham)
    C = coupling_pattern(hp.A)
    ncomp, lab = connected_components(C, directed=False)
    m = np.zeros(n_full, bool)
    m[hp.kept_index[np.bincount(lab, minlength=ncomp)[lab] >= 3]] = True
    return m


def main():
    for cfg in CONFIGS:
        path = OUT / f"{cfg}_solves.csv"
        if not path.exists():
            continue
        so = pd.read_csv(path)
        for (T, rep), idx in so.groupby(["T", "rep"]).groups.items():
            f = OUT / "amplitudes" / f"{cfg}_T{T}_rep{rep}.npz"
            if not f.exists():
                print("missing", f)
                continue
            d = np.load(f)
            truth, coupled = d["truth"].astype(bool), d["coupled"].astype(bool)
            masks = {"full": (None, coupled), "deg1": (coupled, coupled)}
            c3 = comp3_mask(cfg, T, rep, len(truth))
            masks["comp3"] = (c3, c3)
            for i in idx:
                r = so.loc[i]
                keep, wp = masks[r.variant]
                x = np.abs(d[r.solver].astype(float))
                if keep is not None:
                    x = np.where(keep, x, 0.0)
                m = metrics(x, truth, wp)
                for c in METRIC_COLS:
                    so.loc[i, c] = m.get(c, np.nan)
            print(cfg, T, rep, "recomputed", flush=True)
        so.to_csv(path, index=False)


if __name__ == "__main__":
    main()
