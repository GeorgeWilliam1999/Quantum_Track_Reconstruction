#!/usr/bin/env python3
"""Stage 3b — the L1-L3 loss budget (the experiment's central deliverable).

Converts "the filter loses 2%" into "1.2% is twins, 0.5% is ripple, 0.3% is
threshold overlap", and so says which knob (if any) can pay each part back.
Scope ruling (George, 2026-08-25): quantum-only — three channels, no gate/
slot-contest L4.

Channels, per config at its far<=1% operating point (tau at the k-th true
amplitude, the matched-eff convention):

  L1  twin/fragment floor -- IRREDUCIBLE for any p(A) on that A.
      Floor theorem: for a permutation P, f(P A P^T) b = P f(A) b for EVERY
      f, so a true segment sitting in a component isomorphic to a false
      segment's component (uniform b) receives an IDENTICAL amplitude under
      every response.  Operational test = a multi-response FINGERPRINT match:
      a true segment is a twin if some false segment matches it to relative
      tolerance simultaneously under K independent probe responses
      (classical 1/lam, comb d40, comb d44, the highest-degree fit).  One
      probe could coincide by accident; K in agreement is isomorphism.
  L2  response ripple -- missed at this degree but RECOVERED by another
      degree of the same family (each at its own far<=1% tau).  Fixable by
      degree choice; this is exactly the BE-05 non-monotonicity.
  L3  threshold overlap -- the remainder: not a twin, no degree of the family
      recovers it; it sits under tau because false amplitudes overlap from
      above.  Fixable only by widening the true/false gap (operator or a
      better-shaped response).

By construction L1 + L2 + L3 = 1 - eff@far<=1%.

Reads the amplitude caches written by 02 (moderate/clean) and 04 (heavy);
no solving.  Also reports twin_frac = the response-independent floor size.

Outputs: outputs/05_loss_budget.csv
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CACHE = OUT / "cache"

FAR_TARGET = 0.01
PROBE_PREFIXES = ("classical_invA_d", "comb_prod_d40", "comb_prod_d44",
                  "fitted_moment_d160", "fitted_moment_d80")
TOL = 1e-7                      # relative, on max-normalised amplitudes


def op_point(x, truth, far_target=FAR_TARGET):
    """tau at the largest matched-eff working point with far <= target."""
    x = np.abs(np.asarray(x, float))
    n_true = int(truth.sum())
    ts = np.sort(x[truth])[::-1]
    xs, xt = np.sort(x), np.sort(x[truth])
    nact = len(x) - np.searchsorted(xs, ts, side="left")
    ntru = n_true - np.searchsorted(xt, ts, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    ok = np.flatnonzero(far <= far_target)
    if not len(ok):
        return np.inf, 0.0
    k = ok[np.argmax(ntru[ok])]
    return float(ts[k]), float(ntru[k] / n_true)


def twin_mask(z, truth):
    """True segments that are multi-response fingerprint twins of a false one."""
    probes = [k for k in z.files
              if any(k.startswith(p) for p in PROBE_PREFIXES)]
    probes = sorted(probes)[:5]
    if len(probes) < 2:
        return np.zeros(truth.shape, bool), probes
    cols = []
    for k in probes:
        v = np.abs(np.asarray(z[k], float))
        m = v.max()
        cols.append(np.round(v / (m if m > 0 else 1.0) / TOL).astype(np.int64))
    F = np.stack(cols, 1)
    false_keys = set(map(tuple, F[~truth]))
    tw = np.zeros(truth.shape, bool)
    idx = np.flatnonzero(truth)
    tw[idx] = [tuple(r) in false_keys for r in F[idx]]
    return tw, probes


def budget_for_cache(path, regime_default):
    z = np.load(path)
    truth = np.asarray(z["truth"], bool)
    n_true = int(truth.sum())
    stem = Path(path).stem
    m = re.match(r"0[24]_amps_(.+)_rep(\d+)$", stem)
    setout, rep = m.group(1), int(m.group(2))
    regime = "heavy" if stem.startswith("04_") else regime_default
    tw, probes = twin_mask(z, truth)

    fams = {}
    for k in z.files:
        if k in ("truth", "sol_C") or "_d" not in k:
            continue
        fam, dg = k.rsplit("_d", 1)
        if fam == "classical_invA":
            continue                       # reference, not a QSVT config
        fams.setdefault(fam, []).append((k, dg))

    rows = []
    for fam, entries in fams.items():
        # recovered-set per degree, for the L2 (ripple) test
        rec = {}
        for k, dg in entries:
            tau, _ = op_point(z[k], truth)
            rec[dg] = (np.abs(np.asarray(z[k], float)) >= tau) & truth
        for k, dg in entries:
            tau, eff = op_point(z[k], truth)
            got = rec[dg]
            missed = truth & ~got
            other = np.zeros(truth.shape, bool)
            for dg2, r2 in rec.items():
                if dg2 != dg:
                    other |= r2
            L1 = int((missed & tw).sum())
            L2 = int((missed & ~tw & other).sum())
            L3 = int((missed & ~tw & ~other).sum())
            rows.append(dict(
                config=f"{regime}/{setout}/{fam}_d{dg}", regime=regime,
                setout=setout, family=fam, degree=float(dg), rep=rep,
                n_true=n_true, eff=eff, tau=tau,
                L1=L1 / n_true, L2=L2 / n_true, L3=L3 / n_true,
                twin_frac=int(tw.sum()) / n_true, n_probes=len(probes)))
    return rows


def main():
    pats = sys.argv[1:] or ["02_amps_*.npz", "04_amps_*.npz"]
    files = sorted(f for p in pats for f in glob.glob(str(CACHE / p)))
    if not files:
        print("no caches found — run 02/04 first")
        return
    rows = []
    for f in files:
        # 02 caches: moderate set-outs, except the clean fork one
        rd = "clean" if "fork" in Path(f).stem else "moderate"
        rows += budget_for_cache(f, rd)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "05_loss_budget.csv", index=False)
    print(f"{len(df)} config-rep rows from {len(files)} caches "
          f"-> outputs/05_loss_budget.csv")

    chk = (df.L1 + df.L2 + df.L3 + df.eff - 1.0).abs().max()
    print(f"identity check  max|L1+L2+L3+eff-1| = {chk:.2e}")

    print("\n== response-independent floor (twin fraction of trues) ==")
    print(df.groupby(["regime", "setout"]).twin_frac.median().round(4).to_string())

    print("\n== median loss budget at far<=1%, best degree per "
          "(regime, setout, family) ==")
    g = (df.groupby(["regime", "setout", "family", "degree"])
           [["eff", "L1", "L2", "L3"]].median().reset_index())
    best = g.loc[g.groupby(["regime", "setout", "family"]).eff.idxmax()]
    for _, r in best.sort_values(["regime", "setout", "eff"],
                                 ascending=[True, True, False]).iterrows():
        print(f"  {r.regime:8s} {r.setout:10s} {r.family:20s} d={r.degree:5.0f}: "
              f"eff {r.eff:.4f} | L1 {r.L1:.4f}  L2 {r.L2:.4f}  L3 {r.L3:.4f}")


if __name__ == "__main__":
    main()
