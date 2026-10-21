#!/usr/bin/env python3
"""Paper 1, figure 6: segment efficiency against false rate for the three solvers at T = 200,
400, 700 and 1000 tracks, pooled over the 10 events of the corpus, with the two reading rules
marked (the threshold that keeps 99 % of true segments; the lowest threshold with a false rate
of at most 1 %).

Data: outputs_5um/amplitudes/T*_rep*.npz (01_build_corpus.py). The files hold every coupled
segment and summarise the isolated ones by their count and their common value; the isolated
segments are put back with those counts before the curve is drawn. Nothing is typed in.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C = "outputs_5um"
FIG = HERE / "figures" / C
AMP = HERE / C / "amplitudes"
STYLE = {"classical": ("classical solver", "k"), "1bqf": ("one-bit filter", "#1f77b4"), "qsvt": ("polynomial, degree 40", "#d95f02")}
TS = (200, 400, 700, 1000)


def pooled(T, solver):
    v, t, w = [], [], []
    for rep in range(10):
        z = np.load(AMP / f"T{T}_rep{rep}.npz")
        x = z[f"x_{solver}"]; tr = z["truth"]
        v.append(x); t.append(tr); w.append(np.ones(len(x)))
        iso = float(z[f"iso_{solver}"]); nt, nf = int(z["n_isolated_true"]), int(z["n_isolated_false"])
        v.append(np.array([iso, iso])); t.append(np.array([True, False])); w.append(np.array([nt, nf], float))
    return np.concatenate(v), np.concatenate(t), np.concatenate(w)


def curve(v, t, w):
    o = np.argsort(-v, kind="stable"); v, t, w = v[o], t[o], w[o]
    tp = np.cumsum(w * t); fp = np.cumsum(w * ~t)
    last = np.r_[v[1:] != v[:-1], True]            # last index of each distinct value: threshold >= v
    tp, fp = tp[last], fp[last]
    eff = tp / (w * t).sum(); far = fp / np.maximum(tp + fp, 1)
    return eff, far


fig, axs = plt.subplots(1, 4, figsize=(8.6, 2.7), sharey=True)
rows = []
for ax, T in zip(axs, TS):
    for solver, (lab, c) in STYLE.items():
        eff, far = curve(*pooled(T, solver))
        ax.plot(far, eff, color=c, lw=1.4, label=lab)
        i99 = np.argmax(eff >= 0.99); ok = np.where(far <= 0.01)[0]
        i1 = ok[-1] if len(ok) else None            # lowest threshold (largest index) with far <= 1 %
        ax.plot(far[i99], eff[i99], "o", color=c, ms=4, mfc="w")
        if i1 is not None: ax.plot(far[i1], eff[i1], "s", color=c, ms=4)
        rows.append(dict(T=T, solver=solver, far_at_eff99_pooled=far[i99], eff_at_far1_pooled=(eff[i1] if i1 is not None else np.nan)))
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.02)
    ax.axvline(0.01, color="0.7", lw=0.6, ls=":"); ax.axhline(0.99, color="0.7", lw=0.6, ls=":")
    ax.set_title(f"$T={T}$", loc="left", fontsize=9); ax.set_xlabel("segment false rate"); ax.grid(alpha=0.25)
axs[0].set_ylabel("segment efficiency")
axs[0].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(w_pad=1.0)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig6_curves.{ext}", dpi=200)
pd.DataFrame(rows).to_csv(FIG / "fig6_numbers.csv", index=False)
print("wrote", FIG / "fig6_curves.pdf"); print(pd.DataFrame(rows).to_string())
