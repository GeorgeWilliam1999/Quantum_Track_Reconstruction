#!/usr/bin/env python3
"""Paper 1, segment-level metrics of each solver against T (the view of figure 2 of the 1BQF paper).

Standard definitions (Notion, Data & Metrics): a segment is active when its value exceeds the
solver's threshold; efficiency = true active / true; false rate = false active / active. No fixed
efficiency or false-rate level is imposed. Thresholds (George, 22 Oct 2026):
  classical  tau = delta/(delta+gamma) + 0.10 = 0.35, the gamma-aware cut of the 1BQF paper
  one-bit    a tiny amount above zero, the level the filter sends the isolated segments to
  QSVT       the same rule: a tiny amount above the level of the isolated segments
"A tiny amount" = 1e-3 of the largest amplitude on the event.
(a) segment efficiency  (b) segment false rate  (c) candidate segments, true and false
(d) false segments kept, with the number of true segments as a reference.
Points: mean over the 10 events; error bars: standard error of the mean.
Also writes seg_metrics_table.tex (means over events) and fig8_numbers.csv.

Data: outputs_5um/amplitudes/T*_rep*.npz (01_build_corpus.py; isolated segments are stored by
count with their common value). Nothing is typed in.
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
TS = [10, 20, 50, 100, 200, 400, 700, 1000]
REPS = range(10)
TAU_CLASSICAL = 0.35
TINY = 1e-3
STYLE = {"classical": ("classical solver", "k"), "1BQF": ("one-bit filter", "#1f77b4"),
         "QSVT": ("QSVT, degree 40", "#d95f02")}

rows = []
for T in TS:
    for r in REPS:
        z = np.load(HERE / C / "amplitudes" / f"T{T}_rep{r}.npz")
        tr = z["truth"].astype(bool)
        n_iso_true, n_iso_false = int(z["n_isolated_true"]), int(z["n_isolated_false"])
        n_true = int(tr.sum()) + n_iso_true
        n_false = int((~tr).sum()) + n_iso_false
        for solver, key, iso_key in (("classical", "x_classical", "iso_classical"),
                                     ("1BQF", "x_1bqf", "iso_1bqf"), ("QSVT", "x_qsvt", "iso_qsvt")):
            x = z[key]
            iso = float(z[iso_key])
            tau = TAU_CLASSICAL if solver == "classical" else iso + TINY * x.max()
            assert tau > iso, (T, r, solver)       # the isolated segments are never active
            act = x > tau
            n_ta, n_fa = int((act & tr).sum()), int((act & ~tr).sum())
            n_act = n_ta + n_fa
            rows.append(dict(T=T, rep=r, solver=solver, tau=tau, n_true=n_true, n_false=n_false,
                             eff=n_ta / n_true, far=n_fa / n_act if n_act else 0.0,
                             true_kept=n_ta, false_kept=n_fa))
d = pd.DataFrame(rows)
d["eff_pct"] = 100 * d.eff

fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.2))
(ax_e, ax_f), (ax_c, ax_k) = axs


def band(ax, g, col, c, ls="-", lab=None, marker="o"):
    # mean over the events with the standard error of the mean, std (ddof=1) / sqrt(n)
    q = g.groupby("T")[col].agg(["mean", "std", "count"]).reset_index()
    q["sem"] = q["std"].fillna(0.0) / np.sqrt(q["count"])
    ax.errorbar(q["T"], q["mean"], yerr=q["sem"], color=c, ls=ls, lw=1.3, marker=marker, ms=3,
                capsize=2.5, elinewidth=1.0, label=lab)
    return q


for solver, (lab, c) in STYLE.items():
    g = d[d.solver == solver]
    band(ax_e, g, "eff_pct", c, lab=lab)
    band(ax_f, g, "far", c, lab=lab)
    band(ax_k, g, "false_kept", c, lab=lab, marker="s")
g = d[d.solver == "classical"]
band(ax_c, g, "n_true", "#1b9e77", lab="true segments")
band(ax_c, g, "n_false", "#7570b3", "--", "false candidates", marker="s")
band(ax_k, g, "n_true", "#1b9e77", ":", "true segments in the event (reference)")

for ax in axs.flat:
    ax.set_xscale("log"); ax.set_xlim(8, 1250); ax.set_xlabel("tracks per event $T$")
    ax.grid(alpha=0.3, which="both", lw=0.4)
ax_e.set_ylabel("segment efficiency (%)"); ax_e.set_title("(a) segment efficiency", loc="left", fontsize=8.5)
ax_e.set_ylim(0, 100)
ax_f.set_ylabel("segment false rate"); ax_f.set_title("(b) segment false rate", loc="left", fontsize=8.5)
ax_f.set_ylim(0, 1)
ax_c.set_yscale("log"); ax_c.set_ylabel("segments"); ax_c.set_title("(c) candidate segments", loc="left", fontsize=8.5)
_m = d.groupby(["solver", "T"])["false_kept"].agg(["mean", "sem"])
ax_k.set_ylim(0, 1.1 * (_m["mean"] + _m["sem"].fillna(0)).max())
ax_k.set_ylabel("false segments kept"); ax_k.set_title("(d) false segments kept", loc="left", fontsize=8.5)
ax_e.legend(fontsize=7, loc="lower left"); ax_c.legend(fontsize=7)
ax_f.legend(fontsize=7, loc="upper left"); ax_k.legend(fontsize=6, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig8_segment_metrics.{ext}", dpi=300)

num = d.groupby(["T", "solver"])[["eff", "far", "true_kept", "false_kept", "tau"]].agg(["mean", "sem"])
num.columns = ["_".join(c) for c in num.columns]
num = num.reset_index()
num.to_csv(FIG / "fig8_numbers.csv", index=False)


def pm(m, s, nd=3):
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def fint(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


lines = [r"\begin{tabular}{r" + "rrr" * 3 + "}", r"\toprule",
         r" & \multicolumn{3}{c}{Classical} & \multicolumn{3}{c}{One-bit filter} & \multicolumn{3}{c}{QSVT, degree 40}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         r"$T$ & $\varepsilon_{\rm seg}$ & $f_{\rm seg}$ & false kept & $\varepsilon_{\rm seg}$ & $f_{\rm seg}$ & false kept & $\varepsilon_{\rm seg}$ & $f_{\rm seg}$ & false kept\\\midrule"]
for T in TS:
    cells = [f"{T}"]
    for solver in ("classical", "1BQF", "QSVT"):
        r = num[(num["T"] == T) & (num.solver == solver)].iloc[0]
        cells += [pm(r.eff_mean, r.eff_sem), pm(r.far_mean, r.far_sem), fint(r.false_kept_mean)]
    lines.append(" & ".join(cells) + r"\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(FIG / "seg_metrics_table.tex").write_text("\n".join(lines) + "\n")
print(num.round(4).to_string())
