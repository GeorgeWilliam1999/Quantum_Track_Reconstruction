#!/usr/bin/env python3
"""Paper 1, figure 5: results against track multiplicity on the paper corpus.

(a) efficiency at the lowest threshold with a false rate of at most 1 %, against T, for the
    classical solver, the one-bit filter and the polynomial (degree 40, four passes, half-width
    0.18); median over the 10 events as a line, every event as a faint dot (dots are offset slightly in T per solver); the degree-60 variant
    (four passes, half-width 0.12, held-out events 5-9 of the teeth scan) as a dashed line
(b) false rate at the threshold that keeps 99 % of true segments, against T, same solvers

Data: outputs_5um/summary_by_event.csv (01_build_corpus.py), outputs_teeth/teeth_scan.csv
(05_teeth_scan.py). Nothing is typed in.
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
FIG.mkdir(parents=True, exist_ok=True)
STYLE = {"classical": ("classical solver", "k", "-"), "1BQF": ("one-bit filter", "#1f77b4", "-"),
         "QSVT": ("polynomial, degree 40", "#d95f02", "-")}

s = pd.read_csv(HERE / C / "summary_by_event.csv")
ts = pd.read_csv(HERE / "outputs_teeth" / "teeth_scan.csv")
held = ts[(ts.rep >= 5) & (ts.teeth == "four") & (ts.degree == 60) & np.isclose(ts.half_width, 0.12)]
d60 = held.groupby("T")["eff_at_far0.01"].agg(["median", "min", "max"]).reset_index()

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
for col, ax, ylab, title in (("eff_at_far1", axs[0], "segment efficiency", "(a) efficiency at a false rate of at most 1 %"),
                             ("far99", axs[1], "segment false rate", "(b) false rate at 99 % efficiency")):
    for solver, (lab, c, ls) in STYLE.items():
        g = s[s.solver == solver].groupby("T")[col].agg(["median", "min", "max"]).reset_index()
        ev = s[s.solver == solver]
        ax.plot(ev["T"] * (1 + 0.04 * (list(STYLE).index(solver) - 1)), ev[col], ".", color=c, ms=3, alpha=0.35, zorder=1)
        ax.plot(g["T"], g["median"], color=c, ls=ls, lw=1.6, marker="o", ms=3, label=lab, zorder=3)
    if col == "eff_at_far1":
        ax.plot(d60["T"], d60["median"], color=STYLE["QSVT"][1], ls="--", lw=1.2, marker="s", ms=3,
                label="polynomial, degree 60, half-width 0.12 (held-out events)")
    ax.set_xscale("log"); ax.set_xlabel("tracks per event $T$"); ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xticks([10, 20, 50, 100, 200, 400, 700, 1000]); ax.set_xticklabels([10, 20, 50, 100, 200, 400, 700, 1000], fontsize=7)
    ax.set_xlim(8, 1400); ax.minorticks_off()
    ax.grid(alpha=0.25)
axs[0].set_ylim(0, 1.02); axs[1].set_ylim(0, 1.02)
axs[0].legend(fontsize=7, loc="lower left")
fig.tight_layout(w_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig5_multiplicity.{ext}", dpi=200)

# numbers for the text
out = {}
for solver in STYLE:
    g = s[s.solver == solver].groupby("T")
    for T, grp in g:
        out[f"eff_at_far1 median {solver} T={T}"] = grp.eff_at_far1.median()
        out[f"eff_at_far1 min {solver} T={T}"] = grp.eff_at_far1.min()
        out[f"far99 median {solver} T={T}"] = grp.far99.median()
        out[f"far99 max {solver} T={T}"] = grp.far99.max()
# the largest T at which the classical median at <=1 % is within 0.02 of the polynomial's
cq = s[s.solver == "QSVT"].groupby("T").eff_at_far1.median(); cc = s[s.solver == "classical"].groupby("T").eff_at_far1.median()
close = [T for T in cq.index if abs(cq[T] - cc[T]) <= 0.02]
out["classical close to polynomial up to T"] = max(close) if close else np.nan
first_drop = [T for T in cq.index if cc[T] < cq[T] - 0.05]
out["classical first T more than 0.05 below polynomial"] = min(first_drop) if first_drop else np.nan
for T, r in d60.iterrows(): pass
for _, r in d60.iterrows():
    out[f"degree60 eff_at_far1 median T={int(r['T'])}"] = r["median"]
pd.Series(out, name="value").rename_axis("quantity").to_csv(FIG / "fig5_numbers.csv")
print("wrote", FIG / "fig5_multiplicity.pdf"); print(pd.Series(out).to_string())
