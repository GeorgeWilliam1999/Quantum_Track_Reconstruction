#!/usr/bin/env python3
"""Paper 1, Figure 3: what the one-bit filter leaves on the table, by error type.

Rows: the seven error types of the corpus (complete track, broken track, contaminated
track, isolated false segment, false pair, false triple or bridge, hub), ordered as in
the plan. Columns: classical activation and one-bit activation (and, with --qsvt, the
QSVT), each drawn as a log-count histogram per row, true (green, upper lane) and false
(red, lower lane), with that solver's working point:
    classical, 1BQF   efficiency-first: lowest threshold keeping 99 % of true segments
    QSVT              lowest threshold at which the false rate is at most 1 %
Printed at the right of each row: false segments kept / false segments of that type.
Pooled over all repetitions at one T. Every count is read from the corpus npz files.

usage: 12_fig_error_atlas.py [--corpus outputs] [--T 400] [--qsvt]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
import numpy as np

HERE = Path(__file__).resolve().parent
TYPES = ["complete track", "broken track", "contaminated track",
         "isolated false segment", "false pair", "false triple or bridge", "hub"]
GREEN, RED = "#2e7d32", "#c62828"


def threshold_eff(x, t, e=0.99):
    ts = np.sort(x[t])[::-1]
    return ts[min(len(ts), int(np.ceil(e * len(ts)))) - 1]


def threshold_far(x, t, f=0.01):
    o = np.argsort(-x, kind="stable")
    xs, ts = x[o], t[o]
    tp, fp = np.cumsum(ts), np.cumsum(~ts)
    last = np.r_[xs[1:] != xs[:-1], True]
    far = fp[last] / (tp[last] + fp[last])
    ok = np.nonzero(far <= f)[0]
    return xs[last][ok[-1]] if len(ok) else np.inf


def load(corpus, T):
    per = {k: [] for k in ("classical", "1BQF", "QSVT")}
    truth, etype = [], []
    for f in sorted(glob.glob(str(HERE / corpus / "amplitudes" / f"T{T}_rep*.npz"))):
        z = np.load(f, allow_pickle=True)
        ni_t, ni_f = int(z["n_isolated_true"]), int(z["n_isolated_false"])
        truth.append(np.r_[z["truth"], np.ones(ni_t, bool), np.zeros(ni_f, bool)])
        # isolated true segments are single-segment fragments of broken tracks
        etype.append(np.r_[z["error_type"], np.full(ni_t, 1), np.full(ni_f, 3)])
        for name, key, iso in (("classical", "x_classical", "iso_classical"),
                               ("1BQF", "x_1bqf", "iso_1bqf"), ("QSVT", "x_qsvt", "iso_qsvt")):
            per[name].append(np.r_[z[key], np.full(ni_t + ni_f, float(z[iso]))])
    return ({k: np.concatenate(v) for k, v in per.items()},
            np.concatenate(truth), np.concatenate(etype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="outputs")
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--qsvt", action="store_true")
    a = ap.parse_args()
    x, truth, et = load(a.corpus, a.T)
    cols = ["classical", "1BQF"] + (["QSVT"] if a.qsvt else [])
    fig, axs = plt.subplots(len(cols), 1, figsize=(6.6, 2.75 * len(cols)), sharex=True)
    lines = []
    XMAX = 3.0
    for ax, name in zip(axs, cols):
        v = x[name]
        tau = threshold_far(v, truth) if name == "QSVT" else threshold_eff(v, truth)
        vn = v / tau
        edges = np.linspace(0, XMAX, 121)
        for r, ty in enumerate(TYPES):
            y0 = len(TYPES) - 1 - r
            for lane, (mask, col) in enumerate(((truth, GREEN), (~truth, RED))):
                sel = (et == r) & mask
                if not sel.any():
                    continue
                h, _ = np.histogram(np.clip(vn[sel], 0, XMAX * 0.999), edges)
                hh = np.log10(h + 1) / np.log10(max(h.max(), 1) + 1) * 0.36
                base = y0 + (0.04 if lane == 0 else -0.40)
                ax.bar(edges[:-1], hh, width=np.diff(edges), bottom=base, align="edge",
                       color=col, alpha=0.85, lw=0)
            nf = int(((et == r) & ~truth).sum())
            kept = int(((et == r) & ~truth & (v >= tau)).sum())
            nt = int(((et == r) & truth).sum())
            keptt = int(((et == r) & truth & (v >= tau)).sum())
            txt = []
            if nt:
                txt.append((f"{keptt:,} of {nt:,}", GREEN))
            if nf:
                txt.append((f"{kept:,} of {nf:,}", RED))
            for q, (tx, col) in enumerate(txt):
                ax.text(XMAX + 0.12, y0 + (0.14 if len(txt) == 2 and q == 0 else -0.2 if len(txt) == 2 else 0),
                        tx, fontsize=7, va="center", color=col)
            lines.append((name, ty, nt, keptt, nf, kept))
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.axvline(XMAX + 0.05, color="0.6", lw=0.6)
        eff = float((truth & (v >= tau)).sum() / truth.sum())
        far = float((~truth & (v >= tau)).sum() / max((v >= tau).sum(), 1))
        rule = "false rate at most 1 %" if name == "QSVT" else "keeps 99 % of true segments"
        ax.set_title(f"{name}: threshold {rule}; efficiency {eff:.3f}, false rate {far:.3f}",
                     fontsize=8, loc="left")
        ax.set_yticks(range(len(TYPES)))
        ax.set_yticklabels(TYPES[::-1], fontsize=7.5)
        ax.set_ylim(-0.55, len(TYPES) - 0.45)
        ax.set_xlim(0, XMAX + 1.55)
        ax.set_xticks([0, 1, 2, 3])
        
        for y in range(len(TYPES)):
            ax.axhline(y + 0.5, color="0.85", lw=0.6)
    
    axs[-1].set_xlabel("activation / threshold (kept if right of the dashed line)")
    fig.tight_layout()
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    stem = f"fig3_error_atlas_T{a.T}" + ("_with_qsvt" if a.qsvt else "")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    with open(out / f"{stem}_counts.csv", "w") as f:
        f.write("solver,error_type,n_true,true_kept,n_false,false_kept\n")
        for row in lines:
            f.write(",".join(map(str, row)) + "\n")
    print("wrote", stem)


if __name__ == "__main__":
    main()
