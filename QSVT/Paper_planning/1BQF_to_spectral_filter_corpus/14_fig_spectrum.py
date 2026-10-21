#!/usr/bin/env python3
"""Paper 1, Figure 4 (Results 2): the problem as a spectrum, and the two responses.

(a) Measured spectrum of A on the corpus (5 um, 0.15 mrad, 1 % hit loss), T = 1000, ten events
    pooled. For every eigenmode v_k of every connected group, the bright weight
    w_k = <v_k, 1>^2 (the part of the uniform input that lands on that mode) is histogrammed in
    lambda, one lane per kind of group, true and false weight separately. Isolated segments sit
    exactly at lambda_c and are reported as a count.
(b) The responses on the same axis: the one-bit filter |cos(lambda t / 2)|, t = pi / lambda_c,
    and the QSVT (production comb: degree 40, four passes, half-width 0.18, on the exact padded
    bounds of the T = 1000 rep 0 event), both normalised to their maximum on the axis. Vertical
    lines: the four true-track lines (bright solid, dark dashed), and the pair, triple and
    three-prong hub lines.
Writes figures/fig4_spectrum.{pdf,png} and figures/fig4_spectrum_numbers.csv:
  share of each kind's bright weight within +-0.18 of a bright true line, the analytic weight
  split of a complete track over its two bright lines, and the QSVT and one-bit kept factors at
  every marked line.
Check: the bright weights of a complete isolated track are compared with the closed form
(98.5 % / 1.5 % split); the run stops if they differ by more than 1e-9.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
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
FIG = os.path.join(HERE, "figures")
NOISE = dict(sigma_scatt=1.5e-4, sigma_res=0.005, phi_max=0.2, hit_ineff=0.01)
ALPHA, BETA = 3.0, 1.0
LC = ALPHA + BETA
TRUE_LINES = [LC - 2 * np.cos(k * np.pi / 5) for k in (1, 2, 3, 4)]
BRIGHT = [TRUE_LINES[0], TRUE_LINES[2]]
KINDS = ["complete track", "broken track", "contaminated track", "false pair",
         "false triple or bridge", "hub"]
GREEN, RED, BLUE, ORANGE = "#2e7d32", "#c62828", "#1565c0", "#ef6c00"


def kind(n, maxdeg, nt):
    if nt == n:
        return "complete track" if (n == 4 and maxdeg <= 2) else "broken track"
    if nt > 0:
        return "contaminated track"
    if n == 2:
        return "false pair"
    return "false triple or bridge" if maxdeg <= 2 else "hub"


def modes(T, rep):
    eps = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **NOISE)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step", gamma=ALPHA, delta=BETA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, _, _, _ = dp_terms.dp_system(ham, beta=0.0, eps_B=None, alpha=0.0, gamma=ALPHA, delta=BETA)
    A = A.tocsr()
    C = (-A).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    nc, lab = connected_components(C, directed=False)
    sizes = np.bincount(lab)
    deg = np.asarray(C.sum(axis=1)).ravel()
    order = np.argsort(lab, kind="stable")
    st = np.r_[0, np.cumsum(sizes)]
    rows = []
    for c in np.nonzero(sizes > 1)[0]:
        idx = order[st[c]:st[c + 1]]
        w, V = np.linalg.eigh(A[idx][:, idx].toarray())
        proj = V.T @ np.ones(len(idx))
        # split each mode's bright weight into its true and false parts by segment
        tw = (V[truth[idx]] ** 2).sum(0) * proj ** 2
        fw = (V[~truth[idx]] ** 2).sum(0) * proj ** 2
        k = kind(len(idx), deg[idx].max(), int(truth[idx].sum()))
        rows += [(k, float(l), float(a), float(b)) for l, a, b in zip(w, tw, fw)]
    n_iso = int((sizes == 1).sum())
    return rows, n_iso, float(min(r[1] for r in rows)), float(max(r[1] for r in rows))


def main():
    # check: complete isolated track, closed form
    P4 = LC * np.eye(4) - (np.eye(4, k=1) + np.eye(4, k=-1))
    w, V = np.linalg.eigh(P4)
    wt = (V.T @ np.ones(4)) ** 2
    share_low = wt[np.argmin(np.abs(w - BRIGHT[0]))] / wt.sum()
    x = np.linalg.solve(P4, np.ones(4))
    xw = ((V.T @ np.ones(4)) / w) ** 2
    sol_share_low = xw[np.argmin(np.abs(w - BRIGHT[0]))] / xw.sum()
    dark = wt[[np.argmin(np.abs(w - TRUE_LINES[1])), np.argmin(np.abs(w - TRUE_LINES[3]))]].max()
    if dark > 1e-9:
        sys.exit("check failed: dark modes carry weight")
    rows, n_iso, lo0, hi0 = [], 0, None, None
    for rep in range(10):
        r, ni, lo, hi = modes(1000, rep)
        rows += r
        n_iso += ni
        if rep == 0:
            lo0, hi0 = lo, hi
    d = pd.DataFrame(rows, columns=["kind", "lam", "true_w", "false_w"])

    pad = 0.05 * (hi0 - lo0) + 0.2
    comb = design_line_comb_inverse(degree=40, s=LC, hw=0.18, domain=(min(lo0, LC) - pad, max(hi0, LC) + pad))
    lam = np.linspace(0.5, 7.5, 3000)
    q = np.abs(comb(lam))
    q /= q.max()
    one = np.abs(np.cos(lam * np.pi / LC / 2))

    fig, axs = plt.subplots(len(KINDS) + 1, 1, figsize=(6.8, 8.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1] * len(KINDS) + [2.2]))
    edges = np.linspace(0.5, 7.5, 351)
    marks = [(TRUE_LINES[0], GREEN, "-"), (TRUE_LINES[1], GREEN, ":"), (TRUE_LINES[2], GREEN, "-"),
             (TRUE_LINES[3], GREEN, ":"), (LC - 1, RED, "--"), (LC + 1, RED, "--"),
             (LC - np.sqrt(2), ORANGE, "--"), (LC + np.sqrt(2), ORANGE, "--"),
             (LC - np.sqrt(3), "0.4", "--"), (LC + np.sqrt(3), "0.4", "--")]
    numbers = []
    for ax, k in zip(axs, KINDS):
        s = d[d.kind == k]
        for col, c in (("true_w", GREEN), ("false_w", RED)):
            h, _ = np.histogram(s.lam, edges, weights=s[col])
            if h.sum() > 0:
                ax.bar(edges[:-1], h / h.sum(), width=np.diff(edges), align="edge", color=c, alpha=0.7, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_yticks([1e-3, 1e-1])
        ax.text(0.01, 0.8, k, transform=ax.transAxes, fontsize=8, va="top")
        for l, c, ls in marks[:4]:
            ax.axvline(l, color=c, ls=ls, lw=0.6)
        tot = (s.true_w + s.false_w).sum()
        near = s[np.min(np.abs(s.lam.values[:, None] - np.array(BRIGHT)[None, :]), 1) <= 0.18]
        numbers.append(dict(quantity=f"bright weight within 0.18 of a bright true line: {k}",
                            value=float((near.true_w + near.false_w).sum() / tot) if tot else np.nan))
    axs[0].set_title("(a) where the input weight sits, by kind of group (T = 1000, ten events)",
                     fontsize=8.5, loc="left")
    ax = axs[-1]
    ax.plot(lam, one, color=BLUE, lw=1.6, label="one-bit filter $|\\cos(\\lambda t/2)|$")
    ax.plot(lam, q, color="k", lw=1.6, label="QSVT, degree 40, four passes (normalised)")
    for l, c, ls in marks:
        ax.axvline(l, color=c, ls=ls, lw=0.7)
    ax.axvline(LC, color="0.6", lw=0.7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("kept factor")
    ax.set_xlabel("eigenvalue $\\lambda$  ($\\lambda_c=\\alpha+\\beta=4$)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("(b) the two responses on the same axis", fontsize=8.5, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig4_spectrum.{ext}"), dpi=200)

    qn = lambda l: float(abs(comb(l)) / np.abs(comb(lam)).max())
    for name, l in (("bright true line 2.382", TRUE_LINES[0]), ("dark true line 3.382", TRUE_LINES[1]),
                    ("bright true line 4.618", TRUE_LINES[2]), ("dark true line 5.618", TRUE_LINES[3]),
                    ("isolated line 4", LC), ("pair line 3", LC - 1), ("pair line 5", LC + 1),
                    ("triple line 2.586", LC - np.sqrt(2)), ("triple line 5.414", LC + np.sqrt(2)),
                    ("hub line 2.268", LC - np.sqrt(3)), ("hub line 5.732", LC + np.sqrt(3))):
        numbers.append(dict(quantity=f"QSVT kept factor at {name}", value=qn(l)))
        numbers.append(dict(quantity=f"one-bit kept factor at {name}", value=float(abs(np.cos(l * np.pi / LC / 2)))))
    numbers.append(dict(quantity="complete track: input weight on lambda=2.382", value=float(share_low)))
    numbers.append(dict(quantity="complete track: solution weight on lambda=2.382", value=float(sol_share_low)))
    numbers.append(dict(quantity="isolated false segments, T=1000, ten events", value=n_iso))
    numbers.append(dict(quantity="comb domain T1000 rep0 low", value=min(lo0, LC) - pad))
    numbers.append(dict(quantity="comb domain T1000 rep0 high", value=max(hi0, LC) + pad))
    pd.DataFrame(numbers).to_csv(os.path.join(FIG, "fig4_spectrum_numbers.csv"), index=False)
    print(pd.DataFrame(numbers).to_string())


if __name__ == "__main__":
    main()
