"""
06 -- write-up resource plots requested by George (2026-07-20):

  1. qubits required vs tracks T        -- separate QSVT / 1BQF figures
  2. gates (CX) required vs tracks T    -- separate QSVT / 1BQF figures
  3. encoding error vs proportion of original gates (FABLE-family compression;
     oracle encodings are exact by construction -- their accuracy axis is the
     polynomial degree, fig05_degree_curves)

Conventions: step kernel, clean events, base Hamiltonian. 1BQF = ONE call of
the encoding (the degree-1 / e^{-iAt} member); QSVT = a full comb solve at the
metric-validated degree d* from 05 (dilation 12->20, C/alpha 16->20,
normalized 20->28 between T=200 and T=400; held at the T=400 value beyond).
nnz/Delta/spectra measured (02_structure + 04_hit_oracle_window CSVs).

Outputs: outputs/fig06_qubits_vs_T_{qsvt,1bqf}.png,
         outputs/fig06_gates_vs_T_{qsvt,1bqf}.png,
         outputs/fig06_error_vs_gate_fraction.png, outputs/06_resource_curves.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import be_lib as bl

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

MCOL = dict(fable="#2a78d6", sfable="#008300", lsfable="#e87ba4",
            camps="#eda100", dictionary="#1baf7a", szegedy="#eb6834",
            dilation="#4a3aa7", hit_oracle="#e34948", native="#52514e")

T_GRID = [5, 10, 20, 50, 100, 200, 400, 700, 1000]


def measured_structure():
    """nnz_off and Delta per T (step clean), from 02 (T<=400) + 04 (700/1000)."""
    st = pd.read_csv(os.path.join(OUT, "02_structure.csv"))
    st = st[(st.kernel == "step") & (st.noise == "clean")]
    hw = pd.read_csv(os.path.join(OUT, "04_hit_oracle_window.csv"))
    hw = hw[hw.noise == "clean"]
    out = {}
    for T in T_GRID:
        r = st[st["T"] == T]
        if len(r):
            out[T] = dict(nnz=int(r.nnz_off.iloc[0]), Delta=int(r.delta_max.iloc[0]),
                          lam_min=float(r.lam_min.iloc[0]), lam_max=float(r.lam_max.iloc[0]))
        else:
            r = hw[hw["T"] == T]
            out[T] = dict(nnz=int(r.nnz_off.iloc[0]), Delta=int(r.delta_max.iloc[0]),
                          lam_min=2.15, lam_max=5.85)  # conservative wide span
    return out


def dstar(T, domain):
    """Metric-validated degree from 05 (held at the T=400 value beyond)."""
    table = {"dilation": [(200, 12), (400, 20)],
             "C_alpha": [(200, 16), (400, 20)],
             "normalized": [(200, 20), (400, 28)]}[domain]
    if T <= 200:
        return table[0][1]
    return table[1][1]


def resource_rows():
    S = measured_structure()
    rows = []
    for T in T_GRID:
        n = int(np.ceil(np.log2(4 * T * T)))
        N = 1 << n
        nnz, Delta = S[T]["nnz"], S[T]["Delta"]
        span = S[T]["lam_max"] - S[T]["lam_min"]
        a_dil = span / 2.0
        s_pad = 1 << max(1, int(np.ceil(np.log2(max(2, Delta)))))
        m = int(np.log2(s_pad))
        d_dil, d_ca, d_nm = dstar(T, "dilation"), dstar(T, "C_alpha"), dstar(T, "normalized")
        # FABLE-family required degree: d_dil scaled by alpha ratio
        d_fab = int(d_dil * (N * 4.0) / a_dil)
        d_lsf = int(d_dil * N / a_dil)

        # per-call CX models
        cx_native = nnz // 2 * (2 * n + 6 * max(0, 12 * n - 36))
        cx_fable = N * N + 3 * n
        cx_lsf = nnz + 4 * T * T - nnz + 3 * n  # nnz + diagonal entries + wiring
        cx_camps = bl.model_transposition_oracle(nnz // 2, n)["cx"]
        cx_dict = bl.model_dictionary(n, Delta + 1,
                                      [nnz // 2 // max(1, Delta)] * max(1, Delta))["cx"]
        cx_hit = bl.model_hit_oracle(T, n, w_pad=s_pad)["cx"]
        cx_sz = 2 * cx_hit + n

        # ancillas (encoding only); QSVT adds ceil(log2(d*+1)) LCU qubits
        entries = [
            # tag,            anc,       cx/call,   d* (QSVT), sim_only
            ("native",        1,         cx_native, None,      False),
            ("dilation",      1,         None,      d_dil,     True),
            ("fable",         n + 1,     cx_fable,  d_fab,     False),
            ("sfable",        n + 1,     cx_fable,  d_fab,     False),
            ("lsfable",       n + 1,     cx_lsf,    d_lsf,     False),
            ("camps",         m + 1,     cx_camps,  d_ca,      False),
            ("dictionary",    m + 2,     cx_dict,   d_ca,      False),
            ("hit_oracle",    45,        cx_hit,    d_ca,      False),
            ("szegedy",       n + 45,    cx_sz,     d_nm,      False),
        ]
        for tag, anc, cx, dq, sim in entries:
            q_1bqf = n + anc
            q_qsvt = (n + anc + int(np.ceil(np.log2(dq + 1)))) if dq else None
            rows.append(dict(T=T, method=tag, n_sys=n, anc=anc,
                             qubits_1bqf=q_1bqf, qubits_qsvt=q_qsvt,
                             cx_call=cx, degree=dq,
                             cx_qsvt=(dq * cx if (dq and cx) else np.nan),
                             sim_only=sim))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "06_resource_curves.csv"), index=False)
    return df


LABELS = dict(native="native 1BQF (Givens exp(-iAt))",
              dilation="dilation (simulation-only)",
              fable="FABLE", sfable="S-FABLE", lsfable="LS-FABLE (broken acc.)",
              camps="Camps O_c/O_A (C/α)", dictionary="dictionary (C/α)",
              hit_oracle="hit-level oracle (C/α)", szegedy="szegedy walk (α=1, hit-prep)")


def style(ax, ylab, title):
    ax.set_xscale("log")
    ax.set_xlabel("tracks T")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, ncol=2)


def figures(df):
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.grid": True, "grid.color": "#e8e7e2",
                         "grid.linewidth": 0.6, "font.size": 11})

    # ---- 1. qubits vs T -------------------------------------------------
    for algo, col_q, extra in (("1bqf", "qubits_1bqf", ["native"]),
                               ("qsvt", "qubits_qsvt", ["dilation"])):
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        order = (["native"] if algo == "1bqf" else ["dilation"]) + \
                ["camps", "dictionary", "hit_oracle", "szegedy",
                 "fable", "sfable", "lsfable"]
        for tag in order:
            g = df[df.method == tag].sort_values("T")
            if not g[col_q].notna().any():
                continue
            c = MCOL.get(tag, "#52514e")
            ls = "--" if tag in ("dilation", "lsfable") else "-"
            ax.plot(g["T"], g[col_q], ls + "o", color=c, lw=2, ms=4.5,
                    label=LABELS[tag])
        nm = "1BQF (one call of the encoding)" if algo == "1bqf" else \
             "QSVT comb solve (at metric-validated degree d*)"
        style(ax, "qubits (system + encoding ancillas" +
              (" + LCU register)" if algo == "qsvt" else ")"),
              f"Qubits vs tracks — {nm}\nstep kernel, base Hamiltonian")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig06_qubits_vs_T_{algo}.png"), dpi=160)
        plt.close(fig)

    # ---- 2. gates vs T --------------------------------------------------
    for algo, col_g in (("1bqf", "cx_call"), ("qsvt", "cx_qsvt")):
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        order = ["fable", "sfable", "lsfable", "dictionary", "camps",
                 "native", "szegedy", "hit_oracle"]
        for tag in order:
            if algo == "qsvt" and tag == "native":
                continue
            g = df[df.method == tag].sort_values("T")
            vals = g[col_g]
            if not vals.notna().any():
                continue
            c = MCOL.get(tag, "#52514e")
            ls = "--" if tag == "lsfable" else "-"
            mk = "s" if tag == "native" else "o"
            ax.plot(g["T"], vals, ls + mk, color=c, lw=2, ms=4.5,
                    label=LABELS[tag])
        if algo == "qsvt":
            gg = df[df.method == "native"].sort_values("T")
            ax.plot(gg["T"], gg.cx_call, ":s", color=MCOL["native"], lw=1.8,
                    ms=4.5, label="one native 1BQF call (reference)")
        ax.set_yscale("log")
        nm = "1BQF (CX per call of the encoding)" if algo == "1bqf" else \
             "QSVT comb solve (CX total = d* × per-call)"
        style(ax, "CX gates", f"Gates vs tracks — {nm}\nstep, base H; dilation omitted (sim-only); Camps ≡ native Givens (both transposition-priced)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"fig06_gates_vs_T_{algo}.png"), dpi=160)
        plt.close(fig)

    # ---- 3. error vs proportion of original gates ------------------------
    pa = pd.read_csv(os.path.join(OUT, "05_fable_pareto.csv"))
    d = 1600
    n, N = bl.npad(d)
    full = float(N * N)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), sharey=True)
    for ax, kernel in zip(axes, ("step", "erf")):
        sub = pa[pa.kernel == kernel]
        for enc, col in (("fable", MCOL["fable"]), ("sfable", MCOL["sfable"])):
            g = sub[sub.enc == enc].groupby("thr")[["rot", "enc_err"]].mean()
            g = g.sort_values("rot")
            ax.plot(g.rot / full, np.maximum(g.enc_err, 1e-16), "-o", color=col,
                    lw=2, ms=4.5, label=enc.upper())
        ls = sub[sub.enc == "lsfable"][["rot", "enc_err"]].mean()
        ax.plot(ls.rot / full, ls.enc_err, "*", color=MCOL["lsfable"], ms=15,
                label="LS-FABLE (fixed)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("proportion of original rotation gates kept")
        ax.set_title(f"{kernel} kernel")
        ax.legend(fontsize=8.5, frameon=False)
    axes[0].set_ylabel("encoding error ‖Â−A‖/‖A‖")
    fig.suptitle("Encoding error vs proportion of original gates — FABLE family, T=20\n(oracle encodings are exact; their accuracy axis is the degree, fig05)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig06_error_vs_gate_fraction.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    df = resource_rows()
    figures(df)
    print(df[df["T"].isin([200, 1000])][["T", "method", "qubits_1bqf", "qubits_qsvt",
                                         "cx_call", "degree", "cx_qsvt"]].to_string(index=False))
    print("done")
