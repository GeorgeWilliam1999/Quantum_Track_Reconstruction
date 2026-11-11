#!/usr/bin/env python3
"""Segment pruning study: matrix size, register width and the three solvers
with and without the decoupled-segment prune.

Per event (configuration, T, rep):
  scaling   n_segments full / kept (degree >= 1) / kept (component >= 3, 4),
            true segments pruned, component census, nnz, qubits, build time
  solves    classical, one-bit circuit (package, matrix-free), exact one-bit
            readout, QSVT comb (degree 40) on three variants of the SAME
            matrix: full, deg1 (decoupled segments dropped), comp3
            (components of fewer than three segments dropped).  Metrics in the
            full index space at the efficiency-first working point; success
            probability; time; statevector size; and the exactness check
            (max |amplitude difference| on the kept segments against the full
            solve, each renormalised over the kept set).

Usage
-----
  python 01_run_study.py --config paper --reps 10
  python 01_run_study.py --config wide --reps 3 --ts 10,100,1000 --no-solve
Outputs are appended per event to outputs/<config>_scaling.csv and
outputs/<config>_solves.csv; an event already present is skipped.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from common import (CONFIGS, OUT, TS_DEFAULT, bounds_for, build_full, component_sizes,
                    epsilon_for, load_event, metrics, pruned_copy, qubits_for,
                    solve_classical, solve_onebit_circuit, solve_onebit_exact, solve_qsvt,
                    truth_full)

SOLVERS = {
    "classical": lambda h, bd: solve_classical(h),
    "onebit_circuit": lambda h, bd: solve_onebit_circuit(h),
    "onebit_exact": lambda h, bd: solve_onebit_exact(h),
    "qsvt": lambda h, bd: solve_qsvt(h, bd),
}
VARIANTS = {
    "full": None,
    "deg1": dict(min_degree=1),
    "comp3": dict(min_degree=1, min_component_size=3),
}


def say(log, *a):
    print(*a, flush=True)
    print(*a, file=log, flush=True)


def append(path, row):
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def done(path, T, rep):
    if not path.exists():
        return False
    df = pd.read_csv(path, usecols=["T", "rep"])
    return bool(((df["T"] == T) & (df["rep"] == rep)).any())


def run_event(cfg, T, rep, eps, solve, log):
    t_ev = time.time()
    ev, ekey = load_event(cfg, T, rep)
    t0 = time.time()
    ham = build_full(ev, eps)
    t_build = time.time() - t0
    n = int(ham.n_segments)
    lab, sizes, deg = component_sizes(ham.A)
    truth = truth_full(ham)
    coupled = deg >= 1
    csz = sizes[lab]
    row = dict(config=cfg, T=T, rep=rep, event_key=ekey, eps=eps, n_hits=int(ev.n_hits),
               n_full=n, nnz=int(ham.A.nnz), n_true=int(truth.sum()),
               n_true_isolated=int((truth & ~coupled).sum()),
               n_true_in_pairs=int((truth & (csz == 2)).sum()),
               n_kept_deg1=int(coupled.sum()), n_kept_c3=int((csz >= 3).sum()),
               n_kept_c4=int((csz >= 4).sum()),
               n_false_kept_deg1=int((coupled & ~truth).sum()),
               n_components=int(len(sizes)), max_component=int(sizes.max()),
               n_comp_1=int((sizes == 1).sum()), n_comp_2=int((sizes == 2).sum()),
               n_comp_3=int((sizes == 3).sum()), n_comp_4=int((sizes == 4).sum()),
               n_comp_5plus=int((sizes >= 5).sum()),
               qubits_full=qubits_for(n), qubits_deg1=qubits_for(int(coupled.sum())),
               qubits_c3=qubits_for(int((csz >= 3).sum())), t_build=t_build)
    append(OUT / f"{cfg}_scaling.csv", row)
    say(log, f"[{cfg}] T={T} rep={rep} n={n} kept deg1={row['n_kept_deg1']} c3={row['n_kept_c3']} "
             f"qubits {row['qubits_full']}->{row['qubits_deg1']} true isolated={row['n_true_isolated']} "
             f"build {t_build:.1f}s")
    if not solve:
        return

    bd = bounds_for(ham)
    full_sols = {}
    amp_dir = OUT / "amplitudes"
    amp_dir.mkdir(exist_ok=True)
    for vname, vkw in VARIANTS.items():
        h = ham if vkw is None else pruned_copy(ham, **vkw)
        kept = None if vkw is None else h.kept_index
        # the working point of a variant is set from the true segments THAT
        # VARIANT keeps (for the full matrix: the coupled ones, the segments
        # any segment-pair solver can find)
        wp_mask = coupled
        if kept is not None:
            wp_mask = np.zeros(n, bool)
            wp_mask[kept] = True
        for sname, fn in SOLVERS.items():
            x, info = fn(h, bd)
            x_full = h.expand(x, fill=0.0)
            m = metrics(x_full, truth, wp_mask)
            r = dict(config=cfg, T=T, rep=rep, variant=vname, solver=sname,
                     n=int(h.A.shape[0]), n_full=n, t_solve=info.get("t"),
                     P=info.get("P", np.nan), n_sys=info.get("n_sys", qubits_for(int(h.A.shape[0]))),
                     qubits=info.get("qubits", np.nan), qubits_phase=info.get("qubits_phase", np.nan),
                     statevector_bytes=info.get("statevector_bytes", np.nan),
                     calls_lcu=info.get("calls_lcu", np.nan), calls_phase=info.get("calls_phase", np.nan),
                     p_iso=info.get("p_iso", np.nan))
            if vkw is None:
                full_sols[sname] = x
                r["maxdiff_vs_full"] = 0.0
            else:
                xf = np.abs(full_sols[sname][kept])
                xp = np.abs(x)
                nf, npn = np.linalg.norm(xf), np.linalg.norm(xp)
                r["maxdiff_vs_full"] = float(np.max(np.abs(xf / nf - xp / npn))) if nf and npn else np.nan
            r.update(m)
            append(OUT / f"{cfg}_solves.csv", r)
        if vkw is None:
            # full-space amplitudes of the full solve (float32); the pruned
            # variants are identical on the kept segments (maxdiff_vs_full)
            np.savez_compressed(amp_dir / f"{cfg}_T{T}_rep{rep}.npz", truth=truth, coupled=coupled,
                                **{k: v.astype(np.float32) for k, v in full_sols.items()})
            say(log, f"    {vname:5s} {sname:15s} n={r['n']:8d} P={r['P']:.4g} eff99={m['eff_e990']:.4f} "
                     f"far99={m['far_e990']:.4f} eff@1%={m['eff_f010']:.4f} dmax={r['maxdiff_vs_full']:.1e} "
                     f"t={r['t_solve']:.1f}s")
    say(log, f"    event done in {time.time() - t_ev:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="paper", choices=sorted(CONFIGS))
    ap.add_argument("--ts", default=",".join(map(str, TS_DEFAULT)))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-solve", action="store_true")
    a = ap.parse_args()
    ts = [int(t) for t in a.ts.split(",")]
    eps = epsilon_for(a.config)
    log = open(OUT / f"{a.config}.log", "a")
    say(log, f"=== {time.strftime('%Y-%m-%d %H:%M')} config={a.config} eps={eps * 1e3:.4f} mrad "
             f"T={ts} reps={a.reps} solve={not a.no_solve}")
    target = OUT / (f"{a.config}_scaling.csv" if a.no_solve else f"{a.config}_solves.csv")
    for T in ts:
        for rep in range(a.reps):
            if done(target, T, rep):
                continue
            run_event(a.config, T, rep, eps, not a.no_solve, log)
    say(log, "=== finished")


if __name__ == "__main__":
    main()
