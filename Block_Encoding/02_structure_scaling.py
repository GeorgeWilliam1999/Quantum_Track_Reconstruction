"""
02 -- structure metrics of real segment Hamiltonians vs T + per-method resource
scaling (subnormalization, rotations/CX per block-encoding call, QSVT degree
multiplier), including the compression-vs-error curves of the FABLE family.

Outputs: outputs/02_structure.csv, outputs/02_method_resources.csv,
         outputs/02_compression_curves.csv, outputs/fig02_*.png
"""

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared")
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import be_lib as bl
import qtrk_pipeline as qp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
T_GRID = [5, 10, 20, 50, 100, 200, 400]
NOISY = dict(sigma_scatt=3e-4, sigma_res=0.01)
CLEAN = dict(sigma_scatt=1e-4, sigma_res=0.0)
HAH_NMAX = 4096          # compute ||HAH||_max up to this padded dim
COMPRESS_T = [10, 20]    # compression curves at these T (padded dim <= 4096)

# fixed method colors (dataviz categorical order; color follows the method)
MCOL = dict(fable="#2a78d6", sfable="#008300", lsfable="#e87ba4",
            camps="#eda100", dictionary="#1baf7a", szegedy="#eb6834",
            dilation="#4a3aa7", hit_oracle="#e34948", onebqf="#52514e")


def build(T, kernel, noise):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, phi_max=0.2, hit_ineff=0.0, **noise)
    kw = dict(epsilon=EPS, gamma=GAMMA, delta=DELTA, kernel=kernel)
    if kernel == "erf":
        kw["erf_sigma"] = EPS / 3.0
    ham = qp.build_hamiltonian(ev, **kw)
    return ham.A.tocsr()


def main():
    rows = []
    res_rows = []
    for kernel in ("step", "erf"):
        for label, noise in (("clean", CLEAN), ("noisy", NOISY)):
            for T in T_GRID:
                if label == "noisy" and T not in (100, 400):
                    continue
                t0 = time.time()
                A = build(T, kernel, noise)
                m = bl.structure_metrics(A)
                m.update(T=T, kernel=kernel, noise=label,
                         build_s=round(time.time() - t0, 2))
                # Hadamard-domain density (the S-FABLE axis)
                if m["N"] <= HAH_NMAX:
                    HAH = bl._hadamard_conj(A.toarray(), m["N"])
                    m["hah_max"] = float(np.abs(HAH).max())
                    m["hah_frac_gt1e3"] = float((np.abs(HAH) > 1e-3).mean())
                classes_pairs = m.pop("classes_pairs")
                rows.append(m)
                print(f"{kernel}/{label} T={T}: d={m['d']} nnz={m['nnz_off']} "
                      f"Δ={m['delta_max']} χ'={m['chi_greedy']} vals={m['distinct_vals']} "
                      f"xor={m['distinct_xor']} λ=[{m['lam_min']:.2f},{m['lam_max']:.2f}]",
                      flush=True)

                # ---- per-method resources at this matrix ----
                n, N, nnz = m["n"], m["N"], m["nnz_off"]
                span_A = m["lam_max"] - m["lam_min"]
                a_dil = span_A / 2.0
                # C-target span: spec(C) = s - spec(A) reversed; use bounds
                s_diag = GAMMA + DELTA
                lamC_max = s_diag - m["lam_min"]
                lamC_min = s_diag - m["lam_max"]
                a_dil_C = (lamC_max - lamC_min) / 2.0

                def add(method, alpha, target, a_ref, rot=None, cx=None, anc=None,
                        note=""):
                    res_rows.append(dict(T=T, kernel=kernel, noise=label,
                                         method=method, alpha=alpha,
                                         target=target,
                                         degree_mult=alpha / a_ref if a_ref else np.nan,
                                         rot=rot, cx=cx, anc=anc, note=note))

                mx = m["max_abs"]
                add("fable", N * mx, "A", a_dil, rot=N * N, cx=N * N + 3 * n,
                    anc=n + 1)
                if "hah_max" in m:
                    add("sfable", N * m["hah_max"], "A", a_dil,
                        rot=np.nan, cx=np.nan, anc=n + 1,
                        note="rot from compression curve")
                add("lsfable", N, "A", a_dil, rot=nnz + m["d"] + 1,
                    cx=nnz + m["d"] + 3 * n, anc=n + 1, note="fixed approx err")
                s_pad = 1 << max(1, int(np.ceil(np.log2(max(2, m["chi_greedy"] + 1)))))
                add("camps", s_pad * mx, "A", a_dil, anc=int(np.log2(s_pad)) + 1,
                    cx=bl.model_transposition_oracle(nnz // 2, n)["cx"],
                    note="cx = one transposition-compiled O_c pass")
                s_padC = 1 << max(1, int(np.ceil(np.log2(max(2, m["chi_greedy"])))))
                add("camps_C", s_padC * 1.0 if kernel == "step" else s_padC * 1.0,
                    "C", a_dil_C, anc=int(np.log2(s_padC)) + 1,
                    cx=bl.model_transposition_oracle(nnz // 2, n)["cx"])
                # dictionary: values are per-class; step -> chi' classes of value 1;
                # erf -> every unordered pair its own class -> sum|a_ij|/2
                alpha_dict_A = s_diag + (m["chi_greedy"] if kernel == "step"
                                         else m["sum_abs_vals"] / 2.0)
                alpha_dict_C = (m["chi_greedy"] if kernel == "step"
                                else m["sum_abs_vals"] / 2.0)
                dmod = bl.model_dictionary(n, m["chi_greedy"] + 1, classes_pairs)
                add("dictionary", alpha_dict_A, "A", a_dil, rot=dmod["rot"],
                    cx=dmod["cx"], anc=dmod["anc"],
                    note="erf: classes degenerate to per-entry")
                add("dictionary_C", alpha_dict_C, "C", a_dil_C, rot=dmod["rot"],
                    cx=dmod["cx"], anc=dmod["anc"])
                add("szegedy", 1.0, "Dnorm", 1.0, cx=np.nan, anc=n,
                    note="alpha=1 by construction; operator changed")
                add("dilation", a_dil, "A", a_dil, cx=np.nan, anc=1,
                    note=f"dense 4^{n} sim-only")
                hmod = bl.model_hit_oracle(T, n, w_pad=s_padC)
                add("hit_oracle", float(s_padC), "C", a_dil_C, cx=hmod["cx"],
                    anc=hmod["anc"], note="proposed; QROM over 5T hits")
                add("onebqf", np.nan, "exp(-iAt)", np.nan,
                    cx=nnz // 2 * (2 * n + 6 * max(0, 12 * n - 36)),
                    anc=1, note="native Givens/Trotter per exp(-iAt) call")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "02_structure.csv"), index=False)
    dr = pd.DataFrame(res_rows)
    dr.to_csv(os.path.join(OUT, "02_method_resources.csv"), index=False)

    # ---- compression curves (classical emulator; step + erf + random ctrl) ----
    comp_rows = []
    thrs = np.geomspace(1e-6, 2.0, 25)
    for T in COMPRESS_T:
        mats = {"step": build(T, "step", CLEAN), "erf": build(T, "erf", CLEAN)}
        d = mats["step"].shape[0]
        n, N = bl.npad(d)
        rng = np.random.default_rng(3)
        R = rng.uniform(-1, 1, (d, d))
        R = sp.csr_matrix(np.where(rng.random((d, d)) < mats["step"].nnz / d ** 2,
                                   (R + R.T) / 2, 0.0))
        mats["random_ctrl"] = R
        for name, M in mats.items():
            Ad = M.toarray()
            nrmA = np.linalg.norm(Ad)
            for thr in thrs:
                encS, keptS = bl.sfable_classical(Ad, thr)
                encF, keptF = bl.fable_classical(Ad, thr)
                comp_rows.append(dict(T=T, matrix=name, thr=thr,
                                      sfable_err=float(np.linalg.norm(encS - Ad) / nrmA),
                                      sfable_rot=keptS,
                                      fable_err=float(np.linalg.norm(encF - Ad) / nrmA),
                                      fable_rot=keptF))
            encL, keptL = bl.lsfable_classical(Ad)
            comp_rows.append(dict(T=T, matrix=name, thr=np.nan,
                                  sfable_err=np.nan, sfable_rot=np.nan,
                                  fable_err=float(np.linalg.norm(encL - Ad) / nrmA),
                                  fable_rot=keptL, lsfable=True))
            print(f"compression T={T} {name}: LS-FABLE err "
                  f"{np.linalg.norm(encL - Ad)/nrmA:.3f} rot {keptL}", flush=True)
    dc = pd.DataFrame(comp_rows)
    dc.to_csv(os.path.join(OUT, "02_compression_curves.csv"), index=False)

    make_figures(df, dr, dc)
    print("done")


def make_figures(df, dr, dc):
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": "#c3c2b7", "axes.grid": True,
                         "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
                         "font.size": 11, "axes.titlesize": 12})

    # fig 1: subnormalization vs T (step, clean)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, kernel in zip(axes, ("step", "erf")):
        sub = dr[(dr.kernel == kernel) & (dr.noise == "clean")]
        for meth, col in [("fable", "fable"), ("sfable", "sfable"),
                          ("lsfable", "lsfable"), ("camps_C", "camps"),
                          ("dictionary_C", "dictionary"), ("szegedy", "szegedy"),
                          ("dilation", "dilation"), ("hit_oracle", "hit_oracle")]:
            g = sub[sub.method == meth].sort_values("T")
            if g.alpha.notna().any():
                ax.plot(g["T"], g["alpha"], "-o", color=MCOL[col], lw=2, ms=5,
                        label=meth)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tracks T"); ax.set_title(f"{kernel} kernel")
    axes[0].set_ylabel("subnormalization α (lower = better)")
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("Subnormalization of each block encoding on real events (clean, rep 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig02_alpha_vs_T.png"), dpi=160)
    plt.close(fig)

    # fig 2: per-call CX cost vs T (step clean; models)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sub = dr[(dr.kernel == "step") & (dr.noise == "clean")]
    for meth, col in [("fable", "fable"), ("lsfable", "lsfable"),
                      ("camps_C", "camps"), ("dictionary_C", "dictionary"),
                      ("hit_oracle", "hit_oracle"), ("onebqf", "onebqf")]:
        g = sub[sub.method == meth].sort_values("T")
        if g.cx.notna().any():
            ax.plot(g["T"], g["cx"], "-o", color=MCOL[col], lw=2, ms=5, label=meth)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tracks T"); ax.set_ylabel("CX per block-encoding call (model)")
    ax.set_title("Per-call 2-qubit cost, step kernel (clean)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig02_cx_per_call.png"), dpi=160)
    plt.close(fig)

    # fig 3: comb-degree multiplier at T=200 step (bar)
    sub = dr[(dr.kernel == "step") & (dr.noise == "clean") & (dr["T"] == 200)]
    order = ["dilation", "szegedy", "dictionary_C", "camps_C", "hit_oracle",
             "dictionary", "camps", "lsfable", "sfable", "fable"]
    sub = sub.set_index("method").reindex([o for o in order if o in set(sub.method)])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    cols = [MCOL.get(i.replace("_C", ""), "#52514e") for i in sub.index]
    ax.bar(range(len(sub)), sub.degree_mult, color=cols, width=0.62)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.axhline(1.0, color="#52514e", lw=1, ls="--")
    ax.set_ylabel("QSVT comb degree multiplier α / α_dilation")
    ax.set_title("What each encoding costs in polynomial degree (T=200, step, clean)")
    for i, v in enumerate(sub.degree_mult):
        if np.isfinite(v):
            ax.text(i, v * 1.15, f"×{v:.3g}", ha="center", fontsize=8,
                    color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig02_degree_multiplier.png"), dpi=160)
    plt.close(fig)

    # fig 4: compression curves (rot vs err) at T=20
    sub = dc[(dc["T"] == dc["T"].max()) & dc.thr.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for ax, mat in zip(axes, ("step", "erf", "random_ctrl")):
        g = sub[sub.matrix == mat].sort_values("sfable_rot")
        ax.plot(g.fable_rot, g.fable_err, "-o", color=MCOL["fable"], lw=2, ms=4,
                label="FABLE (compressed)")
        ax.plot(g.sfable_rot, g.sfable_err, "-o", color=MCOL["sfable"], lw=2, ms=4,
                label="S-FABLE (compressed)")
        ls = dc[(dc["T"] == dc["T"].max()) & dc.thr.isna() & (dc.matrix == mat)]
        if len(ls):
            ax.plot(ls.fable_rot, ls.fable_err, "*", color=MCOL["lsfable"], ms=14,
                    label="LS-FABLE (fixed)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("rotation gates kept")
        ax.set_title(mat)
    axes[0].set_ylabel("relative encoding error ‖Â−A‖/‖A‖")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"FABLE-family compression on real segment matrices "
                 f"(T={int(dc['T'].max())}) vs the paper's favorable random ensemble")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig02_compression_curves.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
