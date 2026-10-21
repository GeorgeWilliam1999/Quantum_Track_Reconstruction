#!/usr/bin/env python3
"""Which comb gives the best QSVT result on the paper corpus: four teeth, or two?

Corpus: 5 um, 0.15 mrad, 1 % hit loss (outputs_5um events), T = 100, 200, 400, 700, 1000,
10 events each. Designs (design_line_comb_inverse, base Hamiltonian, lambda_c = 4):
    four teeth   passes on all four lines of a complete track, lambda_c - 2cos(k pi/5), k=1..4
    two bright   passes on the two lines a complete track actually carries weight on (k = 1, 3)
    two dark     control: passes on the two dark lines only (k = 2, 4)
  degrees 24, 40, 60 and pass half-widths 0.12, 0.18, 0.24, 0.30 for four and two bright;
  the dark control at degree 40, half-width 0.18.
Readings per event: efficiency at false rate <= 1 % (the paper's rule) and <= 5 %; false rate at
97 % and 99 % efficiency.
Selection is held out: the best design is chosen on events 0-4 (mean over T of the median
efficiency at <= 1 % false) and reported on events 5-9.
Check: for the production design (four teeth, degree 40, half-width 0.18) the amplitudes must
reproduce the checked corpus files (outputs_5um/amplitudes) to 1e-10, else the run stops.
Output: outputs_teeth/teeth_scan.csv, outputs_teeth/summary.txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp                                          # noqa: E402
import dp_terms                                                     # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon                  # noqa: E402
from lhcb_velo_toy.solvers.quantum import design_line_comb_inverse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs_teeth")
os.makedirs(OUT, exist_ok=True)
NOISE = dict(sigma_scatt=1.5e-4, sigma_res=0.005, phi_max=0.2, hit_ineff=0.01)
ALPHA, BETA = 3.0, 1.0
LC = ALPHA + BETA
LINES = {k: LC - 2 * np.cos(k * np.pi / 5) for k in (1, 2, 3, 4)}
TEETH = {"four": (LINES[1], LINES[2], LINES[3], LINES[4]),
         "two bright": (LINES[1], LINES[3]),
         "two dark": (LINES[2], LINES[4])}
DESIGNS = [(t, d, hw) for t in ("four", "two bright") for d in (24, 40, 60) for hw in (0.12, 0.18, 0.24, 0.30)]
DESIGNS += [("two dark", 40, 0.18)]
TS = (100, 200, 400, 700, 1000)
REPS = range(10)


def event_groups(T, rep):
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **NOISE)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=ALPHA, delta=BETA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, _ = dp_terms.dp_system(ham, beta=0.0, eps_B=None, alpha=0.0, gamma=ALPHA, delta=BETA)
    A = A.tocsr()
    C = (-A).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    nc, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.cumsum(sizes)]
    by_size = {}
    for c in np.nonzero(sizes > 1)[0]:
        idx = order[starts[c]:starts[c + 1]]
        by_size.setdefault(len(idx), []).append(idx)
    stacks, lo, hi = [], LC, LC
    for m, idxs in by_size.items():
        I = np.array(idxs)
        blocks = np.stack([A[i][:, i].toarray() for i in I])
        w, V = np.linalg.eigh(blocks)
        proj = np.einsum("gji,gj->gi", V, np.ones((len(I), m)))
        stacks.append((I, w, V, proj))
        lo, hi = min(lo, w.min()), max(hi, w.max())
    iso = sizes[lab] == 1
    return truth, iso, stacks, lo, hi


def amplitudes(poly, n, iso, stacks):
    x = np.empty(n)
    x[iso] = abs(BETA * poly(LC))
    for I, w, V, proj in stacks:
        x[I] = np.abs(np.einsum("gij,gj->gi", V, BETA * poly(w) * proj))
    return x


def readings(x, t):
    o = np.argsort(-x, kind="stable")
    xs, ts = x[o], t[o]
    tp, fp = np.cumsum(ts), np.cumsum(~ts)
    last = np.r_[xs[1:] != xs[:-1], True]
    eff, far = tp[last] / t.sum(), fp[last] / (tp[last] + fp[last])
    r = {}
    for e in (0.97, 0.99):
        r[f"far_at_eff{e}"] = far[min(np.searchsorted(eff, e), len(eff) - 1)]
    for fm in (0.01, 0.05):
        ok = far <= fm
        r[f"eff_at_far{fm}"] = eff[ok].max() if ok.any() else 0.0
    return r


def main():
    rows = []
    checked = False
    for T in TS:
        for rep in REPS:
            t0 = time.time()
            truth, iso, stacks, lo, hi = event_groups(T, rep)
            n = len(truth)
            pad = 0.05 * (hi - lo) + 0.2
            dom = (lo - pad, hi + pad)
            for teeth, deg, hw in DESIGNS:
                poly = design_line_comb_inverse(degree=deg, s=LC, lines=TEETH[teeth], hw=hw, domain=dom)
                x = amplitudes(poly, n, iso, stacks)
                if (teeth, deg, hw) == ("four", 40, 0.18) and not checked:
                    z = np.load(os.path.join(HERE, "outputs_5um", "amplitudes", f"T{T}_rep{rep}.npz"))
                    diff = float(np.max(np.abs(x[z["seg_index"]] - z["x_qsvt"])))
                    diff = max(diff, abs(float(x[iso][0]) - float(z["iso_qsvt"])))
                    print(f"[check] T{T} rep{rep}: production design vs corpus {diff:.1e}", flush=True)
                    if diff > 1e-10:
                        sys.exit("check failed: scan does not reproduce the corpus")
                    checked = True
                rows.append(dict(T=T, rep=rep, teeth=teeth, degree=deg, half_width=hw, **readings(x, truth)))
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "teeth_scan.csv"), index=False)
            print(f"T{T} rep{rep} done ({time.time() - t0:.0f}s)", flush=True)

    d = pd.DataFrame(rows)
    key = ["teeth", "degree", "half_width"]
    val = d[d.rep < 5].groupby(key + ["T"])["eff_at_far0.01"].median().groupby(key).mean().sort_values(ascending=False)
    best = val.index[0]
    test = d[d.rep >= 5]
    cols = ["eff_at_far0.01", "eff_at_far0.05", "far_at_eff0.97", "far_at_eff0.99"]
    lines = ["=== selection on events 0-4 (mean over T of median efficiency at <= 1 % false), top 10",
             val.head(10).round(4).to_string(),
             f"\nbest on events 0-4: {best}",
             "\n=== events 5-9, medians by T"]
    for des in [best, ("four", 40, 0.18), ("two bright", 40, 0.18), ("two bright", 40, 0.24), ("two dark", 40, 0.18)]:
        sel = test[(test.teeth == des[0]) & (test.degree == des[1]) & (test.half_width == des[2])]
        lines.append(f"\n{des}")
        lines.append(sel.groupby("T")[cols].median().round(3).to_string())
    text = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
