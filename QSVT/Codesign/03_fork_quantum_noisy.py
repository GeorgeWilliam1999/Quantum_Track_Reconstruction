#!/usr/bin/env python3
"""Fork penalty on NOISY events, quantum side — todo 3965d544-b9d9-810a.

The DP spectral atlas (QSVT IX) proved the classical half: on A'(beta=0.5)
the fork migrates ~2/3 of contaminated-true spectral weight into the comb
bands at <=1.8% false re-admission, and the emulated three-knob composition
(comb + occupancy gate) reached ~99.1% eff / ~0.9% far.  This study runs the
REAL quantum solvers on the fork system at the operating point where B_fork
is actually populated — heavy noise, T=200 — with the two retunes the theory
demands:

  * gamma raised to gamma*(beta) = the window/positive-definiteness bound
    measured from the spectral extremes at gamma_ref=3 (DP matrix write-up
    Sec 2): gamma_pd = 3 - lam_min_ref, gamma_win = lam_max_ref - 3 - 2*delta.
    The 1BQF notch AUTO-TRACKS the diagonal (t = pi/s', s' = gamma+delta).
  * the QSVT comb designed on the MEASURED shifted spectrum (bounds by eigsh,
    degree scaled with the domain d ~ 2W/0.30, comb centre s' = gamma*+delta).

Configs (heavy = sigma_scatt 1e-4, sigma_res 20 um, drop 1%, formula eps;
eps_B = eps unless stated):
  beta0        beta=0,   gamma=3          baseline, reps 0-2
  b0.5-g3      beta=0.5, gamma=3          WINDOW-VIOLATING control, rep 0
  b0.5-gstar   beta=0.5, gamma=gamma*     the atlas pick, reps 0-2
  b1-gstar     beta=1.0, gamma=gamma*     reps 0-2
  b0.5-wide    beta=0.5, eps_B=2eps, gamma=gamma*   rep 0

Solvers: classical MINRES + comb-QSVT (statevector, store-registered under
study=QSVT_fork_noisy with fork provenance in the ham_key) + the EXACT 1BQF
readout (expm_multiply, notch at the tracked diagonal) + the REAL OneBQF
statevector circuit on rep 0 of the three main configs (Trotter-on-A' check,
mirroring today's erf exact-evolution arbiter).

Composition: the comb actives are then passed through the hit-uniqueness
post-selection gate (qtrk_pipeline.hit_uniqueness_filter, greedy) — the
three-knob validation against the atlas's emulated ~99.1/0.9 target.

Outputs: outputs/fork_noisy/fork_quantum_noisy.{csv,png} + store rows.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
os.environ.setdefault("AER_MAX_MEM_MB", "16000")

import qtrk_pipeline as qp  # noqa: E402
from qtrk_pipeline import hit_uniqueness_filter  # noqa: E402
from qtrk_pipeline.metrics import rescale_to_signal  # noqa: E402
import bif  # noqa: E402
from dp_matrix_characterisation import emulate_1bqf, NOISES  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "fork_noisy"
OUT.mkdir(parents=True, exist_ok=True)
STUDY = "QSVT_fork_noisy"
GAMMA_REF, DELTA = 3.0, 1.0
T = 200
NOISE = "heavy"
TAU_COMB = 0.18                      # nb04 per-solver comb cut (rescaled scale)
TARGET_EFF = 0.99
REGISTER = os.environ.get("FORK_REGISTER", "1") == "1"

CONFIGS = [
    # (label, beta, epsB_mult, gamma_mode, reps, circuit_rep0)
    ("beta0",      0.0, 1.0, "ref",   (0, 1, 2), True),
    ("b0.5-g3",    0.5, 1.0, "ref",   (0,),      False),
    ("b0.5-gstar", 0.5, 1.0, "gstar", (0, 1, 2), True),
    ("b1-gstar",   1.0, 1.0, "gstar", (0, 1, 2), True),
    ("b0.5-wide",  0.5, 2.0, "gstar", (0,),      False),
]

LOG = open(OUT / "fork_quantum_noisy.log", "w")


def say(*a):
    print(*a, flush=True)
    print(*a, file=LOG, flush=True)


def extremes(A):
    hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                     maxiter=4000, tol=1e-4)[0])
    lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                     maxiter=4000, tol=1e-4)[0])
    return lo, hi


def wp99(v, truth):
    tau = qp.working_point_threshold(np.abs(v), truth)
    act = np.abs(v) > tau
    return (float((act & truth).sum() / max(truth.sum(), 1)),
            float((act & ~truth).sum() / max(int(act.sum()), 1)), float(tau))


def composed_gate(v, truth, seg_hits, tau):
    """comb/classical actives at tau -> uniqueness greedy gate -> (eff, far)."""
    act = hit_uniqueness_filter(np.abs(v), seg_hits, np.abs(v) > tau,
                                mode="greedy")
    return (float((act & truth).sum() / max(truth.sum(), 1)),
            float((act & ~truth).sum() / max(int(act.sum()), 1)))


def composed_best(v, truth, seg_hits, n_tau=40):
    """efficiency-first working point of threshold->gate on |v|."""
    x = np.abs(np.asarray(v).ravel())
    lo = float(np.quantile(x[truth], 0.001)) * 0.25
    taus = np.unique(np.quantile(x, np.linspace(0, 1, 300)))
    taus = taus[(taus >= lo) & (taus < x.max())]
    if len(taus) > n_tau:
        taus = taus[np.linspace(0, len(taus) - 1, n_tau).astype(int)]
    best = None
    for t in taus:
        e, f = composed_gate(x, truth, seg_hits, t)
        cand = (e >= TARGET_EFF, e if e < TARGET_EFF else -f)
        if best is None or cand > best[0]:
            best = (cand, t, e, f)
    _, t, e, f = best
    return float(t), float(e), float(f)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.abs(a) @ np.abs(b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    t00 = time.time()
    nz = NOISES[NOISE]
    eps = float(compute_epsilon(nz["sigma_res"], nz["sigma_scatt"]))
    say(f"[setup] heavy formula eps = {eps*1e3:.3f} mrad; T={T}")

    rows, panels, new_solrows = [], {}, []
    for label, beta, epsB_mult, gmode, reps, circ in CONFIGS:
        eps_B = eps * epsB_mult
        for rep in reps:
            tt = time.time()
            ev, ekey = qp.ensure_event(n_trk=T, rep=rep, **nz)
            ham0 = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                        gamma=GAMMA_REF, delta=DELTA)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            seg_hits = np.asarray(ham0._segment_to_hit_ids)
            n = int(ham0.n_segments)
            A_ref = ham0.A.tocsr()
            if beta:
                B = bif.fork_graph_eps(ham0._segment_to_hit_ids,
                                       ham0._segment_vectors, eps_B)
                A_ref = (A_ref + beta * B).tocsr()
                nnzB = int(B.nnz)
            else:
                nnzB = 0

            # gamma requirement from measured extremes at gamma_ref
            lo_r, hi_r = extremes(A_ref)
            g_pd = GAMMA_REF - lo_r
            g_win = hi_r - GAMMA_REF - 2 * DELTA
            g_star = max(GAMMA_REF, g_pd + 0.10, g_win + 0.10)
            g_use = GAMMA_REF if gmode == "ref" else g_star
            A = (A_ref + (g_use - GAMMA_REF) *
                 sp.identity(n, format="csr")).tocsr()
            s_p = g_use + DELTA
            lo, hi = lo_r + (g_use - GAMMA_REF), hi_r + (g_use - GAMMA_REF)
            tau_abs = qp.threshold_for(g_use, DELTA)
            say(f"[{label} rep{rep}] nnzB={nnzB:,} lam(ref)=[{lo_r:.2f},{hi_r:.2f}] "
                f"g_pd={g_pd:.2f} g_win={g_win:.2f} -> gamma={g_use:.2f} "
                f"s'={s_p:.2f} lam=[{lo:.2f},{hi:.2f}] tau={tau_abs:.3f}")

            ham = SimpleNamespace(A=A, b=DELTA * np.ones(n), n_segments=n,
                                  gamma=g_use, delta=DELTA)
            hkey = qp.ham_key(epsilon=eps, kernel="step", gamma=g_use,
                              delta=DELTA, eps_provenance="formula",
                              fork_beta=beta, fork_eps=(eps_B if beta else None))
            common = dict(event_key=ekey, ham_key=hkey, study=STUDY,
                          studies=STUDY, n_trk=T, rep=rep, **nz,
                          ghost_rate=0.0, epsilon=eps,
                          eps_provenance="formula", kernel="step",
                          erf_sigma=0.0, gamma=g_use, delta=DELTA,
                          fork_beta=beta,
                          fork_eps=(eps_B if beta else 0.0), shots=0)

            # --- classical on A' ---
            xC, _ = minres(A.tocsc(), ham.b, rtol=1e-8, maxiter=8000)
            xC = np.abs(np.asarray(xC).ravel())
            kC = qp.sol_key(ekey, hkey, "classical")
            if REGISTER and not qp.solution_exists(kC):
                qp.save_solution(kC, xC, event_key_=ekey, ham_key_=hkey,
                                 solver="classical", n_seg=n,
                                 n_true=int(truth.sum()), epsilon=eps,
                                 kernel="step", gamma=g_use, delta=DELTA,
                                 fork_beta=beta, fork_eps=(eps_B if beta else 0.0),
                                 B_nnz=nnzB, A_nnz=int(A.nnz))
                new_solrows.append(dict(common, sol_key=kC, solver="classical",
                                        device="na", readout="na"))

            # --- comb QSVT on the measured shifted spectrum ---
            half = max(s_p - lo, hi - s_p) + 0.05 * (hi - lo) + 0.2
            dom = (s_p - half, s_p + half)
            deg = int(min(127, max(40, np.ceil(2 * half / 0.30))))
            res = qp.solve_qsvt(ham, degree=deg, spectral_bounds=(lo, hi),
                                domain=dom)
            xQs = np.abs(np.asarray(res["sol"], np.float64))
            kQ = qp.sol_key(ekey, hkey, "qsvt", readout="statevector")
            if REGISTER and not qp.solution_exists(kQ):
                qp.save_solution(kQ, res["sol"], event_key_=ekey, ham_key_=hkey,
                                 solver="qsvt", n_seg=n,
                                 n_true=int(truth.sum()), P_anc=res["P_anc"],
                                 n_sys=res["n_sys"], n_qubits=res["n_qubits"],
                                 degree=res["degree"],
                                 filter_design=res["filter_design"],
                                 epsilon=eps, kernel="step", gamma=g_use,
                                 delta=DELTA, fork_beta=beta,
                                 fork_eps=(eps_B if beta else 0.0),
                                 B_nnz=nnzB, A_nnz=int(A.nnz))
                new_solrows.append(dict(common, sol_key=kQ, solver="qsvt",
                                        device="na", readout="statevector"))

            # --- exact 1BQF readout (notch auto-tracks s') ---
            xq, p_anc, s_chk, t_emu = emulate_1bqf(A, n)
            assert abs(s_chk - s_p) < 1e-9

            # --- real OneBQF circuit (rep 0 of the main configs) ---
            xq_circ = None
            if circ and rep == 0:
                kB = qp.sol_key(ekey, hkey, "quantum", "CPU", "statevector")
                if qp.solution_exists(kB):
                    xq_circ = np.abs(np.asarray(qp.load_solution(kB)["sol"],
                                                np.float64)[:n])
                else:
                    qd = qp.solve_quantum(ham, device="CPU",
                                          readout="statevector")
                    xq_circ = np.abs(np.asarray(qd["sol"], np.float64)[:n])
                    if REGISTER:
                        qp.save_solution(kB, qd["sol"], event_key_=ekey,
                                         ham_key_=hkey, solver="quantum",
                                         n_seg=n, n_true=int(truth.sum()),
                                         P_anc=qd["P_anc"], n_sys=qd["n_sys"],
                                         n_qubits=qd["n_qubits"],
                                         t_solve=qd["t_solve"], epsilon=eps,
                                         kernel="step", gamma=g_use,
                                         delta=DELTA, fork_beta=beta,
                                         fork_eps=(eps_B if beta else 0.0),
                                         B_nnz=nnzB, A_nnz=int(A.nnz))
                        new_solrows.append(dict(common, sol_key=kB,
                                                solver="quantum", device="CPU",
                                                readout="statevector"))

            # --- metrics ---
            row = dict(config=label, beta=beta, eps_B=eps_B, gamma=g_use,
                       rep=rep, n_seg=n, n_true=int(truth.sum()), B_nnz=nnzB,
                       lam_lo=lo, lam_hi=hi, s_prime=s_p, degree=deg,
                       tau_abs=tau_abs, P_anc_comb=res["P_anc"],
                       P_anc_1bqf=p_anc)
            eC, fC, _ = wp99(xC, truth)
            row.update(cls_eff_wp=eC, cls_far_wp=fC)
            mC = qp.metrics_at(xC, truth, tau_abs)
            row.update(cls_eff_abs=mC["segment_efficiency"],
                       cls_far_abs=mC["segment_false_rate"])
            xq_r = rescale_to_signal(xq, xC, tau_abs)
            eQ, fQ, _ = wp99(xq_r, truth)
            row.update(bqf_eff_wp=eQ, bqf_far_wp=fQ)
            xs_r = rescale_to_signal(xQs, xC, tau_abs)
            eS, fS, _ = wp99(xs_r, truth)
            row.update(comb_eff_wp=eS, comb_far_wp=fS)
            act = xs_r > TAU_COMB
            row.update(comb_eff_tau=float((act & truth).sum() / truth.sum()),
                       comb_far_tau=float((act & ~truth).sum()
                                          / max(int(act.sum()), 1)))
            # three-knob: comb -> uniqueness gate
            eg, fg = composed_gate(xs_r, truth, seg_hits, TAU_COMB)
            row.update(comb_gate_eff=eg, comb_gate_far=fg)
            tb, eb, fb = composed_best(xs_r, truth, seg_hits)
            row.update(comb_gate_best_tau=tb, comb_gate_best_eff=eb,
                       comb_gate_best_far=fb)
            # classical + gate for the fair classical-fork comparison
            tbc, ebc, fbc = composed_best(xC, truth, seg_hits)
            row.update(cls_gate_best_eff=ebc, cls_gate_best_far=fbc)
            if xq_circ is not None:
                xc_r = rescale_to_signal(xq_circ, xC, tau_abs)
                eB, fB, _ = wp99(xc_r, truth)
                row.update(circ_eff_wp=eB, circ_far_wp=fB,
                           cos_emu_circ=cosine(xq, xq_circ))
            rows.append(row)
            if rep == 0:
                panels[label] = dict(xC=xC, xq=xq_r, comb=xs_r, truth=truth,
                                     seg_hits=seg_hits)
            say(f"  [{label} rep{rep}] cls wp {eC:.4f}/{fC:.4f} | 1bqf-exact wp "
                f"{eQ:.4f}/{fQ:.4f} | comb wp {eS:.4f}/{fS:.4f} tau18 "
                f"{row['comb_eff_tau']:.4f}/{row['comb_far_tau']:.4f} | "
                f"comb+gate {eg:.4f}/{fg:.4f} best {eb:.4f}/{fb:.4f} | "
                f"deg={deg} [{time.time()-tt:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fork_quantum_noisy.csv", index=False)

    # --- register new solution rows + rebuild study metrics ----------------
    if REGISTER and new_solrows:
        import shutil
        md = qp.manifest_dir()
        sols = pd.read_csv(md / "solutions.csv")
        merged = pd.concat([sols, pd.DataFrame(new_solrows)],
                           ignore_index=True).drop_duplicates(
                               subset="sol_key", keep="first")
        shutil.copy(md / "solutions.csv", md / "solutions.csv.bak")
        merged.to_csv(md / "solutions.csv", index=False)
        say(f"[store] solutions.csv {len(sols)} -> {len(merged)} rows")
        from qtrk_pipeline import build_metrics as bm
        df_new = bm.build(STUDY)
        mpath = md / "metrics.csv"
        old = pd.read_csv(mpath)
        out = pd.concat([old[old.study != STUDY], df_new], ignore_index=True)
        shutil.copy(mpath, md / "metrics.csv.bak")
        out.to_csv(mpath, index=False)
        say(f"[store] metrics.csv {len(old)} -> {len(out)} rows "
            f"({STUDY}: {df_new.groupby('solver').size().to_dict()})")

    # ------------------------------ figure ---------------------------------
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
    order = [c[0] for c in CONFIGS]
    agg = df.groupby("config").mean(numeric_only=True).reindex(order)

    ax = axes[0, 0]
    xs = np.arange(len(order))
    for k, (col, lab, ecol, fcol) in enumerate([
            ("#33322e", "classical", "cls_eff_wp", "cls_far_wp"),
            ("#e34948", "1BQF exact (retuned notch)", "bqf_eff_wp", "bqf_far_wp"),
            ("#2a78d6", "comb (measured spectrum)", "comb_eff_wp", "comb_far_wp")]):
        ax.bar(xs + (k - 1) * 0.26, agg[fcol], 0.24, color=col, label=lab)
        for xi, cfg in enumerate(order):
            v, e = agg.loc[cfg, fcol], agg.loc[cfg, ecol]
            if np.isfinite(v):
                ax.text(xi + (k - 1) * 0.26, v + 0.015, f"{e:.3f}", ha="center",
                        fontsize=6, color=col, rotation=90)
    ax.set_xticks(xs, order, fontsize=8)
    ax.set_ylabel("false rate at wp99 (labels = efficiency)")
    ax.legend(fontsize=7.5)
    ax.set_title("(a) the fork on real solvers at heavy T=200 (rep-mean)",
                 loc="left", fontsize=9)

    ax = axes[0, 1]
    for col, lab, ecol, fcol, mk in [
            ("#33322e", "classical + gate (best)", "cls_gate_best_eff",
             "cls_gate_best_far", "o"),
            ("#2a78d6", "comb@0.18 + gate", "comb_gate_eff", "comb_gate_far", "s"),
            ("#1e9e6a", "comb + gate (best τ)", "comb_gate_best_eff",
             "comb_gate_best_far", "*")]:
        ax.plot(np.maximum(agg[fcol], 2e-4), agg[ecol], mk, ms=9, color=col,
                label=lab)
        for cfg in order:
            ax.annotate(cfg, (max(agg.loc[cfg, fcol], 2e-4),
                              agg.loc[cfg, ecol]), fontsize=5.5,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(TARGET_EFF, color="#79776f", lw=0.7, ls=":")
    ax.plot(0.009, 0.991, "P", ms=11, color="#eb6834",
            label="atlas emulated target (~99.1/0.9)")
    ax.set_xscale("log")
    ax.set_xlabel("false rate")
    ax.set_ylabel("segment efficiency")
    ax.legend(fontsize=7)
    ax.set_title("(b) three-knob composition: comb + uniqueness gate vs the "
                 "atlas target", loc="left", fontsize=9)

    ax = axes[1, 0]
    p0, p5 = panels.get("beta0"), panels.get("b0.5-gstar")
    if p0 is not None and p5 is not None:
        tr = p0["truth"]
        lim = 1e-6
        ax.loglog(np.maximum(p0["comb"][~tr], lim), np.maximum(p5["comb"][~tr], lim),
                  ".", ms=2, alpha=0.2, color="#eb6834", label="false",
                  rasterized=True)
        ax.loglog(np.maximum(p0["comb"][tr], lim), np.maximum(p5["comb"][tr], lim),
                  ".", ms=2.5, alpha=0.5, color="#33322e", label="true",
                  rasterized=True)
        ax.axhline(TAU_COMB, color="#2a78d6", lw=0.8, ls="--")
        ax.axvline(TAU_COMB, color="#2a78d6", lw=0.8, ls="--")
        ax.set_xlabel("comb amplitude, beta=0 (gamma=3)")
        ax.set_ylabel("comb amplitude, beta=0.5 (gamma*)")
        ax.legend(fontsize=7.5, markerscale=6)
    ax.set_title("(c) per-segment comb readout: what the fork migrates "
                 "(rep 0)", loc="left", fontsize=9)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(xs, agg["lam_hi"] - agg["lam_lo"], "o-", color="#33322e",
            label="spectral span")
    ax.plot(xs, agg["degree"], "s-", color="#2a78d6", label="comb degree")
    ax2.semilogy(xs, agg["P_anc_comb"], "^-", color="#1e9e6a",
                 label="P_anc comb")
    d0 = df[df.rep == 0].set_index("config").reindex(order)
    if "cos_emu_circ" in d0:
        ax2.semilogy(xs, d0["cos_emu_circ"], "D--", color="#e34948",
                     label="cos(exact, circuit)")
    ax.set_xticks(xs, order, fontsize=8)
    ax.set_ylabel("span / degree")
    ax2.set_ylabel("P_anc · cos(exact,circuit)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax.set_title("(d) costs of the fork: span, degree, success, Trotter "
                 "fidelity", loc="left", fontsize=9)

    fig.suptitle("Fork penalty at heavy noise, REAL quantum solvers: retuned "
                 "notch + measured-spectrum comb + uniqueness gate",
                 y=0.995, fontsize=11)
    foot = (r"heavy = ($\sigma_{scatt}$ 1e-4, $\sigma_{res}$ 20 µm, drop 1%), "
            r"formula $\varepsilon$, step kernel, $\delta$=1, T=200 · "
            r"A' = A0($\gamma$) + $\beta B_{fork}(\varepsilon_B)$, "
            r"$\gamma^*$ = max(3, $\gamma_{pd}$, $\gamma_{win}$)+0.1 from "
            r"measured extremes · notch t=$\pi$/s', s'=$\gamma$+$\delta$ "
            r"(auto-tracked) · comb = line-comb inverse at s'-2cos(k$\pi$/5), "
            r"bounds by eigsh, deg=min(127, max(40, 2W/0.30)) · gate = "
            r"hit-uniqueness greedy (same-role slots) · wp99 = 1% true "
            r"quantile (WP_TAU_FLOOR) · far = N$_{false,act}$/N$_{act}$ · "
            "store study QSVT_fork_noisy · provenance: "
            "QSVT/Codesign/03_fork_quantum_noisy.py")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.text(0.01, 0.004, foot, fontsize=5.6, color="#52514e", va="bottom")
    fig.savefig(OUT / "fork_quantum_noisy.png", bbox_inches="tight")
    plt.close(fig)

    say(df[["config", "rep", "gamma", "cls_far_wp", "bqf_far_wp",
            "comb_far_wp", "comb_gate_best_eff", "comb_gate_best_far"]]
        .round(4).to_string(index=False))
    say(f"[done] {time.time()-t00:.0f}s -> outputs/fork_noisy/")


if __name__ == "__main__":
    main()
