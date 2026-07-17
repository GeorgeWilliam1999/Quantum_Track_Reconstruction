#!/usr/bin/env python3
"""Is the erf wp99 reversal Trotter-dominated?  The exact-evolution arbiter.

Open question on todo 3925d544-b9d9-8149 (supervisor point 1 -> 2 -> 3 order,
step 2): on the heavy-pair T=200 step2x/kink cells the STORED 1BQF circuit
(OneBQF = per-pair first-order product formula) shows far_wp99 ~ 0.55, while
the ideal-notch emulation of the ERF spectral study predicted ~ 0.05.  Two
hypotheses:
  (H-trotter)  the reversal is implementation error — the product formula's
               O(t^2) phase error scatters the readout; the notch physics is
               fine.  Then an EXACT evolution readout must reproduce the
               ideal-notch numbers.
  (H-physics)  the ideal-notch emulation (Lanczos response rows) was the
               approximation, and the notch genuinely fails on the erf
               spectrum.  Then exact evolution must reproduce the circuit.

Arbiter: x_Q = |(e^{iAt}u + u)/2| with u uniform, t = pi/s, computed by
scipy expm_multiply on the sparse A — mathematically the exact one-bit QPE
readout, no Trotter, no Lanczos, no eigh (dp_matrix_characterisation.
emulate_1bqf, validated against dense eigh at T=50).  Same events, same
ham_keys, same wp99 evaluation as the store.

Cells: pairs moderate (3e-4, 10 um) & heavy (5e-4, 20 um), T=200 rep 0,
kernels step2x (erf theta_d=1e-6, the doubled-coupling hard edge) and kink
(theta_d = eps/3, the kink-matched width).  Data via qtrk_pipeline only.
Outputs: results/erf_exact_evolution_check.csv +
         figures/erf_exact_evolution_check.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp  # noqa: E402
from qtrk_pipeline.metrics import rescale_to_signal  # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon  # noqa: E402
from dp_matrix_characterisation import emulate_1bqf  # noqa: E402  (expm_multiply)

HERE = Path(__file__).resolve().parent
GAMMA, DELTA = 3.0, 1.0
TAU = qp.threshold_for(GAMMA, DELTA)
PAIRS = [(3e-4, 0.01, "moderate"), (5e-4, 0.02, "heavy")]
CELLS = [("step2x", lambda eps: 1e-6), ("kink", lambda eps: eps / 3.0)]


def wp99(x, truth):
    tau = float(np.quantile(x[truth], 0.01)) - 1e-12
    act = x > tau
    eff = float((act & truth).sum() / max(truth.sum(), 1))
    far = float((act & ~truth).sum() / max(int(act.sum()), 1))
    return tau, eff, far


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    t0 = time.time()
    rows, panels = [], {}
    for ss, sr, pair in PAIRS:
        eps = float(compute_epsilon(sr, ss))
        ev, ekey = qp.ensure_event(n_trk=200, rep=0, sigma_scatt=ss, sigma_res=sr,
                                   phi_max=0.2, hit_ineff=0.0)
        truth = np.asarray(qp.truth_from_event(ev), bool)
        for label, td_of in CELLS:
            td = float(td_of(eps))
            tt = time.time()
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="erf",
                                       erf_sigma=td, gamma=GAMMA, delta=DELTA)
            A, n = ham.A, int(ham.n_segments)
            x, _ = minres(A.tocsc(), np.asarray(ham.b, float).ravel(),
                          rtol=1e-8, maxiter=8000)
            x = np.abs(np.asarray(x).ravel())

            # exact evolution — the arbiter (no Trotter)
            xq_ex, p_anc_ex, s_p, t_emu = emulate_1bqf(A, n)

            # stored circuit solve (per-pair product formula)
            hkey = qp.ham_key(eps, "erf", td, GAMMA, DELTA, "formula")
            skey = qp.sol_key(ekey, hkey, "quantum", "CPU", "statevector")
            back = qp.load_solution(skey)
            xq_st = np.abs(np.asarray(back["sol"], np.float64)[:n])
            p_anc_st = float(back.get("P_anc", np.nan))

            xr_ex = rescale_to_signal(xq_ex, x, TAU)
            xr_st = rescale_to_signal(xq_st, x, TAU)
            _, eff_ex, far_ex = wp99(xr_ex, truth)
            _, eff_st, far_st = wp99(xr_st, truth)
            _, eff_c, far_c = wp99(x, truth)
            mex = qp.metrics_at(xr_ex, truth, TAU)
            mst = qp.metrics_at(xr_st, truth, TAU)
            row = dict(pair=pair, cell=label, T=200, rep=0, epsilon=eps,
                       theta_d=td, n_seg=n, n_true=int(truth.sum()),
                       A_nnz=int(A.nnz), s_prime=s_p,
                       eff_wp99_exact=eff_ex, far_wp99_exact=far_ex,
                       eff_wp99_circuit=eff_st, far_wp99_circuit=far_st,
                       eff_wp99_classical=eff_c, far_wp99_classical=far_c,
                       eff_abs_exact=mex["segment_efficiency"],
                       far_abs_exact=mex["segment_false_rate"],
                       eff_abs_circuit=mst["segment_efficiency"],
                       far_abs_circuit=mst["segment_false_rate"],
                       cos_exact_circuit=cosine(xq_ex, xq_st),
                       cos_exact_classical=cosine(xq_ex, x),
                       cos_circuit_classical=cosine(xq_st, x),
                       P_anc_exact=p_anc_ex, P_anc_circuit=p_anc_st,
                       t_emu=t_emu)
            rows.append(row)
            panels[(pair, label)] = dict(x=x, xq_ex=xr_ex, xq_st=xr_st,
                                         truth=truth)
            print(f"[{pair} {label}] eps={eps:.5f} td={td:.2e} nnz={A.nnz:,} "
                  f"| far_wp99: exact={far_ex:.3f} circuit={far_st:.3f} "
                  f"classical={far_c:.3f} | cos(exact,circuit)={row['cos_exact_circuit']:.4f} "
                  f"[emu {t_emu:.0f}s, cell {time.time()-tt:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "erf_exact_evolution_check.csv", index=False)

    # verdict logic, printed and stamped on the figure
    verdicts = []
    for _, r in df.iterrows():
        gap_ideal = abs(r.far_wp99_exact - r.far_wp99_circuit)
        trotter = (r.far_wp99_circuit - r.far_wp99_exact) > 0.5 * max(
            r.far_wp99_circuit, 0.02)
        verdicts.append("TROTTER-DOMINATED" if trotter else
                        ("consistent (no large gap)" if gap_ideal < 0.05
                         else "physics?"))
    df["verdict"] = verdicts
    print(df[["pair", "cell", "far_wp99_exact", "far_wp99_circuit",
              "cos_exact_circuit", "verdict"]].round(4).to_string(index=False),
          flush=True)

    # ------------------------------- figure ---------------------------------
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))

    ax = axes[0, 0]
    xs = np.arange(len(df))
    ax.bar(xs - 0.22, df.far_wp99_exact, 0.4, color="#2a78d6",
           label="EXACT evolution (no Trotter)")
    ax.bar(xs + 0.22, df.far_wp99_circuit, 0.4, color="#e34948",
           label="stored circuit (product formula)")
    for i, r in df.iterrows():
        ax.plot([i - 0.42, i + 0.42], [r.far_wp99_classical] * 2,
                color="#33322e", lw=1.4, ls="--")
        ax.text(i, max(r.far_wp99_circuit, r.far_wp99_exact) + 0.03,
                r.verdict, ha="center", fontsize=6.5, rotation=0)
    ax.set_xticks(xs, [f"{r.pair}\n{r.cell}" for _, r in df.iterrows()],
                  fontsize=8)
    ax.set_ylabel("false rate at wp99")
    ax.plot([], [], color="#33322e", ls="--", label="classical (same event)")
    ax.legend(fontsize=7.5)
    ax.set_title("(a) the arbiter: exact e^{iAt} readout vs the implemented "
                 "circuit", loc="left", fontsize=9)

    ax = axes[0, 1]
    ax.bar(xs - 0.22, df.eff_wp99_exact, 0.4, color="#2a78d6")
    ax.bar(xs + 0.22, df.eff_wp99_circuit, 0.4, color="#e34948")
    ax.set_xticks(xs, [f"{r.pair}\n{r.cell}" for _, r in df.iterrows()],
                  fontsize=8)
    ax.set_ylim(0.9, 1.005)
    ax.set_ylabel("efficiency at wp99")
    ax.set_title("(b) efficiency is pinned by construction (wp99) — the far "
                 "moves", loc="left", fontsize=9)

    ax = axes[1, 0]
    p = panels[("heavy", "kink")]
    tr = p["truth"]
    lim = (1e-6, max(p["xq_st"].max(), p["xq_ex"].max()) * 1.6)
    ax.loglog(np.maximum(p["xq_ex"][~tr], lim[0]),
              np.maximum(p["xq_st"][~tr], lim[0]), ".", ms=2, alpha=0.25,
              color="#eb6834", label="false", rasterized=True)
    ax.loglog(np.maximum(p["xq_ex"][tr], lim[0]),
              np.maximum(p["xq_st"][tr], lim[0]), ".", ms=2.4, alpha=0.5,
              color="#33322e", label="true", rasterized=True)
    ax.plot(lim, lim, color="#c3c2b7", lw=0.8, ls=":")
    ax.set_xlabel("exact-evolution amplitude")
    ax.set_ylabel("circuit amplitude")
    ax.legend(fontsize=7.5, markerscale=6)
    ax.set_title("(c) heavy/kink per-segment: where the product formula "
                 "scatters the readout", loc="left", fontsize=9)

    ax = axes[1, 1]
    w = 0.26
    for k, (col, lab, key) in enumerate([
            ("#2a78d6", "cos(exact, circuit)", "cos_exact_circuit"),
            ("#33322e", "cos(exact, classical)", "cos_exact_classical"),
            ("#9a9890", "cos(circuit, classical)", "cos_circuit_classical")]):
        ax.bar(xs + (k - 1) * w, df[key], w, color=col, label=lab)
    ax.set_xticks(xs, [f"{r.pair}\n{r.cell}" for _, r in df.iterrows()],
                  fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine similarity")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(d) fidelity decomposition of the implementation gap",
                 loc="left", fontsize=9)

    foot = (r"T=200 rep 0 · matched events, formula $\varepsilon$: moderate "
            r"($\sigma_{scatt}$=3e-4, $\sigma_{res}$=10 µm) $\varepsilon$="
            f"{compute_epsilon(0.01, 3e-4)*1e3:.2f} mrad · heavy (5e-4, 20 µm) "
            f"$\\varepsilon$={compute_epsilon(0.02, 5e-4)*1e3:.2f} mrad · "
            r"erf kernel, step2x: $\theta_d$=1e-6 (doubled-coupling hard edge), "
            r"kink: $\theta_d$=$\varepsilon$/3 · $\gamma$=3 $\delta$=1, notch "
            r"t=$\pi$/s from the diagonal · exact readout = expm_multiply "
            r"|(e$^{iAt}$u+u)/2| (no Trotter, no Lanczos) · circuit = stored "
            r"OneBQF statevector solve (per-pair product formula) · far = "
            r"N$_{false\,act}$/N$_{act}$ at wp99 (1% true quantile) · "
            "provenance: ERF/erf_exact_evolution_check.py")
    fig.suptitle("Exact-evolution arbiter: is the erf wp99 reversal Trotter "
                 "error or notch physics?", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    fig.text(0.01, 0.004, foot, fontsize=5.8, color="#52514e", va="bottom")
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "erf_exact_evolution_check.png",
                bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {time.time()-t0:.0f}s -> figures/erf_exact_evolution_check.png",
          flush=True)


if __name__ == "__main__":
    main()
