#!/usr/bin/env python3
"""PROOF figure: occupancy does not help a notch-equipped solver (George, 2026-07-06).

Claim to prove without doubt: the per-hit occupancy term is useless-to-harmful
for the 1BQF, because the notch already removes every C-isolated false segment
exactly, and occupancy revives them by coupling them. Four independent lines of
evidence, one figure:

(a) MECHANISM — per-segment 1BQF amplitudes, base vs occ (same event, same
    segments): the co-hit false bulk moves from ~1e-19 (notched dead) onto the
    true amplitude band. Scatter, class-coloured.
(b) NO THRESHOLD ESCAPE — full tau sweep eff vs far for 1BQF base/fork/occ and
    classical base/occ on the same event: no operating point on the occ-1BQF
    curve beats any point of the base-1BQF curve; rank AUC drops to ~chance.
    (Classical curves show the SAME term helping the notch-less solver — the
    claim is solver-specific, not "occupancy is useless".)
(c) NO GAMMA ESCAPE — from dp_gamma_validation.csv: 1BQF far@wp99 and co-hit
    bulk admission vs gamma (fork 3->8, occ 3->236). Distance-to-notch is
    gamma-invariant; the curves are flat/worse.
(d) WHERE THE READOUT GOES — post-selected probability mass (P_anc share) by
    truth, base vs fork vs occ vs both: occupancy diverts the ancilla=1 mass
    almost entirely to false segments.

Event: heavy T=200 rep0 (occupancy's BEST classical showing — fairest test);
faint overlays in (b): heavy T=100 reps 0-2 (not a one-event fluke).
Outputs: figures/dp_occupancy_proof.png + results/dp_occupancy_proof.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import minres

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_matrix_characterisation as dmc  # noqa: E402
import dp_terms  # noqa: E402
import bif  # noqa: E402

HERE = Path(__file__).resolve().parent
NOISE = "heavy"
CONFIGS = [("base", 0.0, 0.0), ("fork", 1.0, 0.0),
           ("occ", 0.0, 0.3), ("both", 1.0, 0.3)]
CCOL = dmc.CCOL
KCOL = {"base": "#33322e", "fork": "#eb6834", "occ": "#e34948", "both": "#8e6fad"}

FOOT = (r"Occupancy-vs-notch proof · heavy noise ($\sigma_{res}$=20 µm, drop 1%, "
        r"$\sigma_{scatt}$=1e-4), formula $\varepsilon$, step, $\gamma$=3 $\delta$=1 · "
        r"A($\beta,\alpha$) = A0 + $\beta B_{fork}$ + $4\alpha I$ + $2\alpha B_{all}$ · "
        r"1BQF exact filter readout $x_Q$ = |(e$^{iAt}$u + u)/2|, t = $\pi$/s', "
        r"s' = $\gamma$+$\delta$+4$\alpha$ (notch auto-tracks the diagonal)"
        "\n"
        r"(a) heavy T=200 rep 0, per-segment amplitudes (base x-axis floored at 1e-19) · "
        r"(b) full $\tau$ sweep, far = N$_{false\,act}$/N$_{act}$; bold T=200, faint "
        r"T=100 reps 0-2; dots = wp99 · (c) dp_gamma_validation.csv, T=200 · "
        r"(d) $\Sigma|x_Q|^2$ split by truth after ancilla=1 post-selection · "
        r"commit-provenance: dp_occupancy_proof.py")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def solve_all(T, rep):
    """(cls, truth, {config: dict(x, xq, p_anc)}) for one event."""
    ham, truth, eps = dmc.get_ham(T, NOISE, rep=rep)
    cls, _ = dmc.classify(ham, truth, eps)
    out = {}
    for name, beta, alpha in CONFIGS:
        A, b, _, _ = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                        gamma=dmc.GAMMA, delta=dmc.DELTA)
        x, _ = minres(A, b, rtol=1e-8, maxiter=8000)
        xq, p_anc, s_p, t_emu = dmc.emulate_1bqf(A, ham.n_segments)
        out[name] = dict(x=np.abs(np.asarray(x).ravel()), xq=xq, p_anc=p_anc)
        print(f"  [{NOISE} T={T} rep{rep}] {name}: P_anc={p_anc:.3e} "
              f"[emu {t_emu:.0f}s]", flush=True)
    return cls, truth, out


def sweep_eff_far(x, truth, n=400):
    """(eff, far) along a tau sweep over amplitude quantiles (far = FA/act)."""
    taus = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n)))[::-1]
    eff, far = [], []
    nt = max(int(truth.sum()), 1)
    for t in taus:
        act = x > t
        na = int(act.sum())
        eff.append((act & truth).sum() / nt)
        far.append((act & ~truth).sum() / max(na, 1))
    return np.asarray(eff), np.asarray(far)


def load_or_solve():
    """Pickle-cache the solves so figure iteration is cheap."""
    import pickle
    cache = HERE / "results" / "dp_occupancy_proof_cache.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    payload = dict(main=solve_all(200, 0),
                   faint=[solve_all(100, r) for r in (0, 1, 2)])
    with open(cache, "wb") as fh:
        pickle.dump(payload, fh)
    return payload


def main():
    t0 = time.time()
    payload = load_or_solve()
    cls, truth, main_ev = payload["main"]
    faint = payload["faint"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))

    # ---- (a) mechanism: base vs occ per-segment 1BQF amplitudes -------------
    ax = axes[0, 0]
    FLOOR = 1e-17                                # notch kill level
    rng = np.random.default_rng(0)
    xb = main_ev["base"]["xq"].copy()
    xo = main_ev["occ"]["xq"].copy()
    # spread the exactly-notched population across a visible band
    dead = xb < FLOOR
    xb[dead] = 10 ** rng.uniform(-19.7, -17.0, int(dead.sum()))
    ax.axvspan(2e-20, FLOOR, color="#9a9890", alpha=0.10)
    for c in ["cohit-true", "cohit-false", "C-true", "TRUE-coupled"]:
        m = np.where(cls == c)[0]
        if not len(m):
            continue
        if len(m) > 20000:                       # thin the bulk for rendering
            m = rng.choice(m, 20000, replace=False)
        ax.loglog(xb[m], xo[m], ".", ms=3.0 if c.startswith("TRUE") else 1.5,
                  color=CCOL[c], alpha=0.9 if c.startswith("TRUE") else 0.20,
                  label=f"{c} (n={int((cls == c).sum()):,})", rasterized=True)
    tq = float(np.median(xo[truth]))
    ax.axhline(tq, color="#33322e", lw=0.8, ls="--")
    ax.text(1.5e-8, tq * 1.06, "TRUE median (occ system)", fontsize=7,
            color="#33322e", va="bottom")
    bulk = cls == "cohit-true"
    q99b = float(np.quantile(main_ev["base"]["xq"][bulk], 0.99))
    q99o = float(np.quantile(xo[bulk], 0.99))
    ax.annotate("base: 99% of the co-hit bulk\nis EXACTLY notched "
                f"(q99 = {q99b:.0e})", xy=(3e-19, 1.1e-3), xytext=(1e-13, 4e-4),
                fontsize=7.5, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.9))
    ax.annotate("+occupancy: the SAME segments land\nON the true band "
                f"(q99 = {q99o:.1e} vs true med {tq:.1e})",
                xy=(4e-19, 1.5e-3), xytext=(1e-13, 2.3e-3), fontsize=7.5,
                color="#a83232",
                arrowprops=dict(arrowstyle="->", color="#a83232", lw=0.9))
    ax.text(4e-19, 1.05e-4, "notched by the base 1BQF\n(x-jittered below "
            "10⁻¹⁷ for display)", fontsize=6.5, color="#79776f")
    ax.set_xlim(2e-20, 2e-2), ax.set_ylim(1e-4, 3e-3)
    ax.set_xlabel("1BQF amplitude, BASE Hamiltonian")
    ax.set_ylabel("1BQF amplitude, + occupancy (α=0.3)")
    leg = ax.legend(fontsize=7, loc="center left", markerscale=6)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.set_title("(a) the same 156k co-hit false segments: notched dead "
                 "(~10⁻¹⁹) → revived ONTO the true band", loc="left", fontsize=9)

    # ---- (b) no threshold escape: full tau sweep ----------------------------
    ax = axes[0, 1]
    aucs = {}
    for name in ("base", "fork", "occ"):
        e, f = sweep_eff_far(main_ev[name]["xq"], truth)
        aucs[name] = bif.auc(main_ev[name]["xq"], truth)
        ax.plot(np.maximum(f, 5e-4), e, "-", color=KCOL[name], lw=1.8,
                label=f"1BQF {name} (AUC {aucs[name]:.3f})")
        tau = dmc.qp.working_point_threshold(main_ev[name]["xq"], truth)
        act = main_ev[name]["xq"] > tau
        ax.plot(max((act & ~truth).sum() / max(act.sum(), 1), 5e-4),
                (act & truth).sum() / truth.sum(), "o", color=KCOL[name], ms=6)
    for name, ls in [("base", (0, (2, 2))), ("occ", (0, (5, 2)))]:
        e, f = sweep_eff_far(main_ev[name]["x"], truth)
        a = bif.auc(main_ev[name]["x"], truth)
        ax.plot(np.maximum(f, 5e-4), e, ls=ls, color=KCOL[name], lw=1.2,
                alpha=0.65, label=f"classical {name} (AUC {a:.3f})")
    for _, tr, ev in faint:
        for name in ("base", "occ"):
            e, f = sweep_eff_far(ev[name]["xq"], tr)
            ax.plot(np.maximum(f, 5e-4), e, "-", color=KCOL[name], lw=0.6,
                    alpha=0.30)
    ax.set_xscale("log")
    ax.set_xlim(5e-4, 1.05), ax.set_ylim(0, 1.02)
    ax.set_xlabel("false rate = N_false_act / N_act  (τ swept over full range)")
    ax.set_ylabel("segment efficiency")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(b) no threshold exists: every occ-1BQF operating point is "
                 "dominated; classical (dashed) shows the same term HELPING",
                 loc="left", fontsize=9)

    # ---- (c) no gamma escape ------------------------------------------------
    ax = axes[1, 0]
    gv = pd.read_csv(HERE / "results" / "dp_gamma_validation.csv")
    for cfg, col, lab in [("fork b=1", KCOL["fork"], "fork β=1"),
                          ("occ a=0.3", KCOL["occ"], "occ α=0.3"),
                          ("base", KCOL["base"], "base")]:
        d = gv[(gv.config == cfg) & (gv.solver == "1bqf_emu")
               & (gv.cls == "ALL")].sort_values("gamma")
        ax.plot(d.gamma, d.far_wp, "o-", color=col, lw=1.6, ms=5,
                label=f"1BQF far@wp99 · {lab}")
        dc = gv[(gv.config == cfg) & (gv.solver == "1bqf_emu")
                & (gv.cls == "cohit-true")].sort_values("gamma")
        ax.plot(dc.gamma, dc.frac_wp, "s--", color=col, lw=1.0, ms=4, alpha=0.6,
                label=f"co-hit bulk admitted · {lab}")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.axvline(3, color="#79776f", lw=0.7, ls=":")
    ax.text(3.1, 0.02, "γ=3", fontsize=7, color="#79776f")
    ax.axvline(236, color="#79776f", lw=0.7, ls=":")
    ax.text(150, 0.02, "γ_win(α=0.3)=236", fontsize=7, color="#79776f")
    ax.set_xlabel("γ  (heavy T=200; occ point at its window-restoring γ=236)")
    ax.set_ylabel("1BQF false rate / bulk admission at wp99")
    ax.legend(fontsize=7, loc="center left")
    ax.set_title("(c) no γ escape: distance-to-notch is γ-invariant — "
                 "fork frozen γ=3→8, occ WORSE at γ=236", loc="left", fontsize=9)

    # ---- (d) where the post-selected mass goes ------------------------------
    ax = axes[1, 1]
    xs = np.arange(len(CONFIGS))
    ft, ff = [], []
    for name, _, _ in CONFIGS:
        w = main_ev[name]["xq"] ** 2
        tot = w.sum()
        ft.append(w[truth].sum() / tot)
        ff.append(w[~truth].sum() / tot)
    ax.bar(xs, ft, 0.55, color="#79b465", label="mass on TRUE segments")
    ax.bar(xs, ff, 0.55, bottom=ft, color="#e34948", alpha=0.85,
           label="mass on FALSE segments")
    for i, (name, _, _) in enumerate(CONFIGS):
        ax.text(i, 1.03, f"P_anc\n{main_ev[name]['p_anc']:.1e}", ha="center",
                fontsize=7.5, color="#33322e")
        ax.text(i, ft[i] / 2 if ft[i] > 0.08 else ft[i] + 0.05,
                f"{100 * ft[i]:.1f}%", ha="center", fontsize=8,
                color="#1e4620" if ft[i] > 0.08 else "#1e4620")
    ax.set_xticks(xs, [n for n, _, _ in CONFIGS])
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("share of post-selected probability mass Σ|x_Q|²")
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("(d) occupancy diverts the ancilla=1 readout to false "
                 "segments (heavy T=200)", loc="left", fontsize=9)

    fig.suptitle("PROOF: the occupancy term cannot help a notch-equipped solver "
                 "— the notch already kills the co-hit bulk exactly; coupling it "
                 "back in revives it", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    fig.text(0.01, 0.004, FOOT, fontsize=5.8, color="#52514e", va="bottom")
    fig.savefig(HERE / "figures" / "dp_occupancy_proof.png", bbox_inches="tight")
    plt.close(fig)

    # ---- summary CSV --------------------------------------------------------
    rows = []
    for name, _, _ in CONFIGS:
        for solver, key in [("1bqf_emu", "xq"), ("classical", "x")]:
            v = main_ev[name][key]
            tau = dmc.qp.working_point_threshold(v, truth)
            act = v > tau
            bulk = cls == "cohit-true"
            rows.append(dict(
                noise=NOISE, T=200, rep=0, config=name, solver=solver,
                auc_all=bif.auc(v, truth),
                auc_bulk_vs_true=bif.auc(
                    np.r_[v[bulk], v[truth]],
                    np.r_[np.zeros(int(bulk.sum()), bool),
                          np.ones(int(truth.sum()), bool)]),
                eff_wp=(act & truth).sum() / truth.sum(),
                far_wp=(act & ~truth).sum() / max(act.sum(), 1),
                bulk_admitted=float((v[bulk] > tau).mean()),
                bulk_q99=float(np.quantile(v[bulk], 0.99)),
                true_med=float(np.median(v[truth])),
                p_anc=main_ev[name]["p_anc"] if key == "xq" else np.nan,
                mass_on_false=float((v[~truth] ** 2).sum() / (v ** 2).sum()),
            ))
    pd.DataFrame(rows).to_csv(HERE / "results" / "dp_occupancy_proof.csv",
                              index=False)
    print(pd.DataFrame(rows)[["config", "solver", "auc_all", "auc_bulk_vs_true",
                              "eff_wp", "far_wp", "bulk_admitted",
                              "mass_on_false"]].round(4).to_string(index=False),
          flush=True)
    print(f"[done] {time.time() - t0:.0f}s -> figures/dp_occupancy_proof.png",
          flush=True)


if __name__ == "__main__":
    main()
