#!/usr/bin/env python3
"""Generated tables of paper 1's supplement (2026-11-07).

  supp_design_table.tex  the design scan (05_teeth_scan.py, outputs_teeth/teeth_scan.csv):
                         per design, the selection score on events 0-4 (mean over T of the
                         median efficiency at a false rate of at most 1 %) and the held-out
                         median efficiency at <= 1 % false on events 5-9 at 400, 700, 1000 tracks.
  supp_P_table.tex       post-selection probability P and amplification rounds against T
                         (18_fig_cost.py, figures/outputs_5um/fig7_cost_table.csv).
Writes into figures/outputs_5um/ and copies to ../1BQF_to_spectral_filter/figures/.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "outputs_5um"
PAPER_FIG = HERE.parent / "1BQF_to_spectral_filter" / "figures"
K = "eff_at_far0.01"

# ---- design scan ----
ts = pd.read_csv(HERE / "outputs_teeth" / "teeth_scan.csv")
key = ["teeth", "degree", "half_width"]
sel = ts[ts.rep <= 4].groupby(key + ["T"])[K].median().groupby(level=key).mean().rename("sel")
held = ts[ts.rep >= 5].groupby(key + ["T"])[K].median().unstack("T")
d = pd.concat([sel, held[[400, 700, 1000]]], axis=1).reset_index()
d = d.sort_values("sel", ascending=False, kind="stable")
best = d.iloc[0][key].tolist()
PASSES = {"four": "four", "two bright": "two bright", "two dark": "two dark"}
lines = [r"\begin{tabular}{lccrccc}\toprule",
         r"Passes & degree & half-width & selection, events 0--4 & \multicolumn{3}{c}{held out, events 5--9}\\",
         r"\cmidrule(lr){5-7}",
         r" & & & & 400 & 700 & 1000\\\midrule"]
for _, r in d.iterrows():
    mark = r"$^\ast$" if r[key].tolist() == best else ""
    if r[key].tolist() == ["four", 40, 0.18]:
        mark += r"$^\dagger$"  # the design of the main text
    lines.append(f"{PASSES[r.teeth]}{mark} & {int(r.degree)} & {r.half_width:.2f} & {r.sel:.3f} & "
                 f"{r[400]:.3f} & {r[700]:.3f} & {r[1000]:.3f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(FIG / "supp_design_table.tex").write_text("\n".join(lines) + "\n")

# ---- post-selection probability against T ----
c = pd.read_csv(FIG / "fig7_cost_table.csv")
P = c.pivot(index="T", columns="route", values="P")
R = c.pivot(index="T", columns="route", values="rounds")
lines = [r"\begin{tabular}{rcccccc}\toprule",
         r" & \multicolumn{2}{c}{One-bit filter} & \multicolumn{2}{c}{QSVT, phase sequence} & \multicolumn{2}{c}{QSVT, Chebyshev walk}\\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"$T$ & $P$ & rounds & $P$ & rounds & $P$ & rounds\\\midrule"]
def sci(x):
    m, e = f"{float(x):.2e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
for T in P.index:
    lines.append(f"{T} & " + " & ".join(f"{sci(P.loc[T, k])} & {R.loc[T, k]:.0f}"
                                        for k in ("1bqf", "phase", "lcu")) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(FIG / "supp_P_table.tex").write_text("\n".join(lines) + "\n")

for f in ("supp_design_table.tex", "supp_P_table.tex"):
    shutil.copy(FIG / f, PAPER_FIG / f)
print((FIG / "supp_design_table.tex").read_text()); print((FIG / "supp_P_table.tex").read_text())
