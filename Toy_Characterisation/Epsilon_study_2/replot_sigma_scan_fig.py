#!/usr/bin/env python3
"""Regenerate sigma_scan_formula_eps.png from epsilon_sensitivity_scan.json
(same plotting block as gen_epsilon_sensitivity_scan.py Fig 4, with the
unclipped y-limits that keep the 1BQF ~0.75 plateau in view)."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

out = json.loads((HERE / "outputs" / "epsilon_sensitivity_scan.json")
                 .read_text())
scans = out["scans"]
T, NREP, TAU = out["T"], out["nrep"], out["tau"]
SR_FIX, SS_FIX = 0.01, 1e-4
DZ = 33.0
P_DEFAULT = float(np.exp(-9.0))


def eff_analytic(p):
    q = 1.0 - np.asarray(p, dtype=float)
    return q**2 * (1.0 + 0.5 * (1.0 - q))


def agg(rows, xkey, ykey):
    xs = sorted(set(r[xkey] for r in rows))
    m, e = [], []
    for x in xs:
        v = [r[ykey] for r in rows if r[xkey] == x]
        m.append(np.mean(v))
        e.append(np.std(v) / np.sqrt(len(v)))
    return np.array(xs), np.array(m), np.array(e)


def fit_quadratic(eps, far):
    keep = (far > 0) & (far < 0.5)
    if keep.sum() < 2:
        return float("nan")
    return float(np.sum(far[keep] * eps[keep] ** 2)
                 / np.sum(eps[keep] ** 4))


fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharey="row")
for j, (axis, xlab, fixed) in enumerate((
    ("ss", r"$\sigma_{\rm scatt}$ [rad]",
     rf"$\sigma_{{\rm res}}={SR_FIX}$ mm fixed"),
    ("sr", r"$\sigma_{\rm res}$ [mm]",
     rf"$\sigma_{{\rm scatt}}={SS_FIX:g}$ fixed"),
)):
    rows = scans[axis]
    xkey = "ss" if axis == "ss" else "sr"
    for i, met in enumerate(("eff", "far")):
        ax = axes[i, j]
        for solver, ls, mk, col in (("C", "-", "o", "tab:blue"),
                                    ("Q", "--", "s", "tab:red")):
            x, m, e = agg(rows, xkey, f"{met}_{solver}")
            ax.errorbar(x, m, yerr=e, fmt=mk, ls=ls, ms=5, capsize=3,
                        color=col,
                        label="classical" if solver == "C" else "1BQF")
        if met == "eff":
            ax.axhline(eff_analytic(P_DEFAULT), color="k", ls=":", lw=1.5,
                       label=r"analytic $(1-p)^2(1+p/2)$ at $p=e^{-9}$")
            ax.set_ylim(0.55, 1.04)
        else:
            xs = np.array(sorted(set(r[xkey] for r in rows)))
            eps_f = np.array([float(compute_epsilon(
                (SR_FIX if axis == "ss" else v),
                (v if axis == "ss" else SS_FIX))) for v in xs])
            _, mC, _ = agg(rows, xkey, "far_C")
            _, mQ, _ = agg(rows, xkey, "far_Q")
            c = fit_quadratic(eps_f, mC)
            print(f"refit c (scan_{axis}) = {c:.4g}")
            ax.plot(xs, np.clip(c * eps_f**2, 0, 1.0), "k:", lw=1.5,
                    label=r"$c\,\varepsilon^2(\sigma)\ \propto\ \sigma_p^2$")
            ax.set_ylim(-0.02, max(0.45, 1.3 * mC.max(), 1.3 * mQ.max()))
        if axis == "ss":
            ax.set_xscale("log")
            xstar = np.sqrt(6.0) * np.arctan(SR_FIX / DZ)
            ax.axvline(xstar, color="gray", lw=1.2, alpha=0.7)
            if i == 0:
                ax.text(xstar * 1.07, 0.58,
                        r"$\sigma^*_{\rm scatt}=\sqrt{6}\,"
                        r"\arctan(\sigma_r/\Delta z)$",
                        fontsize=8, color="gray", rotation=90, va="bottom")
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(fixed, fontsize=10.5)
axes[0, 0].set_ylabel("segment efficiency")
axes[1, 0].set_ylabel("segment false rate")
axes[0, 0].legend(fontsize=8, loc="lower left")
axes[1, 0].legend(fontsize=8, loc="upper left")
fig.suptitle(rf"Noise scans at the formula $\varepsilon$ "
             rf"($p_{{\rm miss}}=e^{{-9}}$) — $T={T}$, {NREP} reps, "
             rf"$\tau={TAU}$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE / "figures" / "epsilon_sensitivity"
            / "sigma_scan_formula_eps.png", dpi=160)
print("[ok] sigma_scan_formula_eps.png regenerated")
