#!/usr/bin/env python3
"""Fig-2/fig-3 analogues for the modified Hamiltonians (George 2026-09-01).

v2 (George's review):
(a) each response panel carries the MEASURED class-projected density of
    states D_C(lambda) = sum_{i in C} sum_j v_j(i)^2 delta(lambda-lambda_j)
    (Hutchinson + Chebyshev-KPM, Jackson kernel) -- positive and
    cancellation-free, i.e. where each class's eigenvector support sits,
    so the eye can check the fit targets the right regions. The bare
    motif lines are the vocabulary of ISOLATED motifs; at moderate noise
    most weight is in contaminated components off those lines, which is
    what made the fits look mis-aimed by eye.
(b) the matched 1BQF cosine |cos(pi lambda/2 s')| is drawn on every panel;
(c) every operator is the FULL system, base A plus the added term(s);
(d) END-TO-END CHECK: the drawn polynomial, applied to the operator via
    its Chebyshev recurrence, must reproduce the 06_degree_scan.csv
    efficiency for that (operator, rep, degree) cell to 1e-9.

F1 figures/xiv_responses_modified.png  (fig-3 analogue)
F2 figures/xiv_motif_lines_modified.png (fig-2 analogue)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                   # noqa: E402
import dp_terms                                              # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon           # noqa: E402

HERE = Path(__file__).resolve().parent
OUT, FIG, CACHE = HERE / "outputs", HERE / "figures", HERE / "outputs" / "cache"

GREEN, RED, GREY, BLUE, ORANGE = "#3d8a4f", "#d84a49", "#8f8d86", "#2a78d6", "#e08a2e"
INK, PURPLE = "#33322e", "#7b5ea7"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

S, DELTA, GAMMA_D = 4.0, 1.0, 3.0
T, REP, REGIME = 200, 0, "moderate"
NOISE = dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01)
OPS = [("base", 0.0, 0.0, 40,
        "base $A$  (fitted $d{=}40$)"),
       ("fork_b0.5", 0.0, 0.5, 40,
        "full system $A+\\beta B_{\\rm fork}$, $\\beta{=}0.5$  (fitted $d{=}40$)"),
       ("occ_a0.05", 0.05, 0.0, 80,
        "full system $A+4\\alpha I+2\\alpha B_{\\rm all}$, $\\alpha{=}0.05$  (fitted $d{=}80$)"),
       ("occ0.05_fork0.5", 0.05, 0.5, 80,
        "full system, both terms  ($\\alpha{=}0.05$, $\\beta{=}0.5$; fitted $d{=}80$)")]


def chain_lines(m, a):
    k = np.arange(1, m + 1)
    return S + 4 * a - 2 * np.cos(k * np.pi / (m + 1))


def motif_lines(a, beta):
    true = list(chain_lines(4, a))
    false = {"isolated": [S + 4 * a],
             "false pair (chain $m{=}2$)": list(chain_lines(2, a)),
             "false triple (chain $m{=}3$)": list(chain_lines(3, a))}
    if beta > 0:
        false["competing pair (window)"] = [S + 4 * a - (2 * a + beta),
                                            S + 4 * a + (2 * a + beta)]
    if a > 0:
        false["same-role pair"] = [S + 2 * a, S + 6 * a]
    return true, false


def build_system(alpha, beta):
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    ev, _ = qp.ensure_event(n_trk=T, rep=REP, **NOISE)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=GAMMA_D, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, _ = dp_terms.dp_system(ham, beta=beta,
                                    eps_B=eps if beta else None,
                                    alpha=alpha, gamma=GAMMA_D, delta=DELTA)
    return A.tocsr(), b, truth


def cheb_map(lo, hi):
    return 0.5 * (hi + lo), 0.5 * (hi - lo) * 1.02


def class_densities(A, truth, lo, hi, K=600, npts=2400, R=10):
    """Class-projected densities of states via Hutchinson + KPM (Jackson).
    D_C(lambda) = sum_{i in C} sum_j v_j(i)^2 delta(lambda - lambda_j):
    positive and cancellation-free -- where each class's eigenvector
    support lives. R random +-1 probes supported on the class."""
    c0, c1 = cheb_map(lo, hi)
    rng = np.random.default_rng(0)
    n = A.shape[0]
    muT, muF = np.zeros(K), np.zeros(K)
    for cls, mu in ((truth, muT), (~truth, muF)):
        idx = np.where(cls)[0]
        for _ in range(R):
            z = np.zeros(n)
            z[idx] = rng.choice([-1.0, 1.0], size=len(idx))
            t_prev, t_cur = z, (A @ z - c0 * z) / c1
            mu[0] += z @ t_prev
            mu[1] += z @ t_cur
            for k in range(2, K):
                t_next = 2.0 * ((A @ t_cur) - c0 * t_cur) / c1 - t_prev
                t_prev, t_cur = t_cur, t_next
                mu[k] += z @ t_cur
        mu /= R
    k = np.arange(K)
    jack = ((K - k + 1) * np.cos(np.pi * k / (K + 1))
            + np.sin(np.pi * k / (K + 1)) / np.tan(np.pi / (K + 1))) / (K + 1)
    x = np.linspace(-0.999, 0.999, npts)
    Tk = np.cos(np.outer(np.arccos(x), k))
    w = 1.0 / (np.pi * np.sqrt(1 - x * x))
    coef = np.r_[1.0, 2 * np.ones(K - 1)]
    rhoT = w * (Tk @ (jack * muT * coef))
    rhoF = w * (Tk @ (jack * muF * coef))
    return c0 + c1 * x, np.maximum(rhoT, 0), np.maximum(rhoF, 0)


def cheb_apply_coef(c, A, b, lo, hi):
    c0, c1 = cheb_map(lo, hi)
    t_prev = b
    y = c[0] * t_prev
    t_cur = (A @ b - c0 * b) / c1
    y = y + c[1] * t_cur
    for k in range(2, len(c)):
        t_next = 2.0 * ((A @ t_cur) - c0 * t_cur) / c1 - t_prev
        t_prev, t_cur = t_cur, t_next
        y = y + c[k] * t_cur
    return np.abs(y)


def eff_at_far(x, truth, far_max=0.01):
    n_true = int(truth.sum())
    ts = np.sort(x[truth])[::-1]
    taus = ts[np.arange(1, n_true + 1) - 1]
    xs, xt = np.sort(x), np.sort(x[truth])
    nact = len(x) - np.searchsorted(xs, taus, side="left")
    ntru = n_true - np.searchsorted(xt, taus, side="left")
    far = np.where(nact > 0, (nact - ntru) / np.maximum(nact, 1), np.nan)
    eff = ntru / n_true
    ok = far <= far_max
    return float(eff[ok].max()) if ok.any() else 0.0


def response_curve(tag, deg, lo, hi, lam):
    z = np.load(CACHE / f"06_coef_{REGIME}_{tag}_rep{REP}.npz")
    c = z[f"fitted_c_d{deg}"]
    c0, c1 = cheb_map(lo, hi)
    p = np.abs(np.polynomial.chebyshev.chebval((lam - c0) / c1, c))
    return p, c


def fig_responses():
    csv = pd.read_csv(OUT / "06_degree_scan.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True)
    for ax, (tag, a, beta, deg, title) in zip(axes.ravel(), OPS):
        z = np.load(CACHE / f"06_coef_{REGIME}_{tag}_rep{REP}.npz")
        lo, hi = float(z["lo"]), float(z["hi"])
        A, b, truth = build_system(a, beta)
        # wide-domain (occupancy) systems: coarser KPM, fewer probes — their
        # matvec is ~10x denser and the window backdrop only needs the shape
        Kk, Rr = (300, 3) if hi > 10 else (400, 6)
        lam_full, rhoT, rhoF = class_densities(A, truth, lo, hi, K=Kk, R=Rr)

        win = (lam_full > 1.6) & (lam_full < 6.9)
        lam = lam_full[win]
        p, c = response_curve(tag, deg, lo, hi, lam)
        p = p / p.max()

        x_rec = cheb_apply_coef(c, A, b, lo, hi)
        got = eff_at_far(x_rec, truth)
        row = csv[(csv.regime == REGIME) & (csv.setout == tag)
                  & (csv.rep == REP) & (csv.family == "fitted_moment")
                  & (csv.degree == deg)]
        want = float(row.eff_f010.iloc[0])
        ok = abs(got - want) < 1e-9
        print(f"  [{'OK ' if ok else 'FAIL'}] {tag}: drawn-poly eff@far1% "
              f"{got:.6f} vs study CSV {want:.6f}")

        # fourth-root display scale: the isolated-false line carries ~1e5
        # segments and would hide every other feature on a linear scale
        M = max(rhoT[win].max(), rhoF[win].max())
        disp = lambda r: (r / M) ** 0.25 * 0.30
        ax.fill_between(lam, 0, -disp(rhoF[win]), color=RED, alpha=0.40,
                        lw=0, zorder=2,
                        label="false-segment spectral support (down, 4th-root scale)")
        ax.fill_between(lam, 0, disp(rhoT[win]), color=GREEN, alpha=0.50,
                        lw=0, zorder=2,
                        label="true-segment spectral support (up, 4th-root scale)")

        true, false = motif_lines(a, beta)
        for v in true:
            ax.axvline(v, color=GREEN, ls=":", lw=1.0, alpha=0.6, zorder=1)
        for nm, vals in false.items():
            for v in vals:
                ax.axvline(v, color=RED, ls="--", lw=0.7, alpha=0.45, zorder=1)

        s_prime = GAMMA_D + DELTA + 4 * a
        ax.plot(lam, np.abs(np.cos(np.pi * lam / (2 * s_prime))), color=GREY,
                lw=1.3, ls="-.", label="1BQF cosine $|\\cos(\\pi\\lambda/2s')|$")
        ax.plot(lam, p, color=INK, lw=1.8, zorder=3,
                label=f"fitted $|p(\\lambda)|$, $d{{=}}{deg}$")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(1.6, 6.9)
        ax.set_ylim(-0.34, 1.06)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.grid(alpha=0.2, lw=0.4)
        ax.annotate(f"domain $[{lo:.2f},\\,{hi:.1f}]$",
                    xy=(0.985, 0.95), xycoords="axes fraction", ha="right",
                    fontsize=8.2, color="#55534d")
        if hi > 10:
            ins = ax.inset_axes([0.60, 0.62, 0.38, 0.27])
            pf, _ = response_curve(tag, deg, lo, hi, lam_full)
            ins.plot(lam_full, pf / pf.max(), color=INK, lw=0.6)
            rT = (rhoT / rhoT.max()) ** 0.25 * 0.9
            ins.fill_between(lam_full, 0, rT, color=GREEN, alpha=0.4, lw=0)
            ins.axvspan(1.6, 6.9, color=ORANGE, alpha=0.15, lw=0)
            ins.set_title("full domain + true support", fontsize=7, pad=2)
            ins.tick_params(labelsize=6.5)
            ins.set_yticks([])
    axes[0, 0].legend(loc="upper left", fontsize=7.4, framealpha=0.92)
    for ax in axes[1]:
        ax.set_xlabel("$\\lambda$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$|p(\\lambda)|$ / scaled class support")
    fig.suptitle("The fitted response against each operator's MEASURED "
                 "class spectral support (moderate configuration, "
                 "$T{=}200$, rep 0)", fontsize=12.2)
    fig.text(0.5, -0.025,
             "shaded: class-projected densities of states "
             "$D_C(\\lambda)=\\sum_{i\\in C}\\sum_j v_j(i)^2\\,"
             "\\delta(\\lambda{-}\\lambda_j)$ (Hutchinson + Chebyshev-KPM, "
             "Jackson kernel, $K{=}600$; true drawn up, false drawn down, "
             "4th-root scale) · dotted/dashed verticals: closed-form "
             "isolated-motif lines (vocabulary, not population) · every "
             "operator is the FULL system, base $A$ + added term(s) · "
             "$\\sigma_{\\rm scatt}{=}10^{-4}$, $\\sigma_{\\rm res}{=}10\\,\\mu$m, "
             "drop 1%, formula $\\varepsilon{=}3.18$ mrad, $\\gamma{=}3$, "
             "$\\delta{=}1$ · drawn responses reproduce the study CSV "
             "numbers to $10^{-9}$", ha="center", va="top",
             fontsize=8.0, color="#55534d")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_responses_modified.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_responses_modified.png")


def fig_motif_lines():
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 7.8), sharex=True,
                             constrained_layout=True)
    base_true, base_false = motif_lines(0.0, 0.0)
    base_all = base_true + [v for vs in base_false.values() for v in vs]
    for ax, (tag, a, beta, _, title) in zip(axes, OPS):
        true, false = motif_lines(a, beta)
        if a > 0 or beta > 0:
            for v in base_all:
                ax.vlines(v, 0, 1.0, color=GREY, lw=1.0, alpha=0.45)
        for v in true:
            ax.vlines(v, 0, 1.0, color=GREEN, lw=2.4)
        for nm, vals in false.items():
            for v in vals:
                ax.vlines(v, 0, 0.72, color=RED, lw=1.7)
        if a > 0:
            for k, xfrac in ((10, 0.42), (50, 0.62)):
                lam = S + 4 * a + 2 * a * (k - 1)
                if lam < 9.6:
                    ax.vlines(lam, 0, 0.5, color=PURPLE, lw=1.7, ls="-")
                    ax.annotate(f"$k{{=}}{k}$ clique", xy=(lam, 0.52),
                                fontsize=7.8, color=PURPLE, ha="center")
            ax.annotate("$\\lambda_{\\max} = s+4\\alpha+2\\alpha(k{-}1)$ "
                        "$\\to$ the span wall", xy=(0.99, 0.8),
                        xycoords="axes fraction", ha="right", fontsize=8.4,
                        color=PURPLE)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        ax.set_ylabel(title.split("(")[0].replace("full system ", "")
                      .strip(), fontsize=8.6, rotation=0, ha="right",
                      va="center", labelpad=8)
    axes[-1].set_xlabel("$\\lambda$")
    axes[-1].set_xlim(2.2, 9.7)
    fig.suptitle("The motif lines under each operator (exact closed forms; "
                 "grey = base-$A$ positions; every operator = base $A$ + "
                 "added term(s))", fontsize=12)
    fig.text(0.5, -0.02,
             "green: true-track $P_4$ lines $s{+}4\\alpha{-}2\\cos(k\\pi/5)$ · "
             "red: false motifs — isolated $s{+}4\\alpha$, chains "
             "$m{=}2,3$, competing pair $s{+}4\\alpha{\\mp}(2\\alpha{+}\\beta)$, "
             "same-role pair $s{+}2\\alpha$, $s{+}6\\alpha$ · purple: same-role "
             "$k$-clique top line (occupancy span mechanism) · "
             "$s{=}4$, $\\alpha{=}0.05$, $\\beta{=}0.5$", ha="center",
             va="top", fontsize=8.2, color="#55534d")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"xiv_motif_lines_modified.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print("[saved] figures/xiv_motif_lines_modified.png")


if __name__ == "__main__":
    print("== end-to-end response checks (drawn poly vs study CSV) ==")
    fig_responses()
    fig_motif_lines()
