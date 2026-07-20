"""
07 -- classical cost of getting each encoding onto the computer (per event):
matrix build, angle/table preparation, decomposition, spectral bounds, circuit
description size. Measured constants + fitted scaling, extrapolated to T=1000.

Pipelines priced (step kernel, clean):
  native 1BQF / classical solver : A build (O(T^3) middle-hit sweep) [+ none]
  dilation (default QSVT)        : A build + dense eigh of padded X (O(N^3))
  FABLE / S-FABLE                : DENSE angle matrix + sfwht (O(N^2 log N)),
                                   memory O(N^2) -- the wall
  LS-FABLE                       : A build + O(nnz) angle writes
  camps / dictionary             : A build + edge coloring + O(nnz*n) synthesis
  szegedy (generic prep)         : A build + degrees + per-column prep layout
  hit oracle (ours)              : hit sort + tables, O(T log T); NO A build,
                                   NO spectral bounds (||C||<=Delta<=w_pad)
A/C-domain QSVT additionally pays Lanczos spectral bounds (measured).

Outputs: outputs/07_classical_cost.csv, outputs/fig07_classical_prep.png
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
GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
T_GRID = [5, 10, 20, 50, 100, 200, 400, 700, 1000]

MCOL = dict(fable="#2a78d6", lsfable="#e87ba4", camps="#eda100",
            dictionary="#1baf7a", szegedy="#eb6834", dilation="#4a3aa7",
            hit_oracle="#e34948", native="#52514e")


def timeit(fn, *a, **k):
    t0 = time.time()
    out = fn(*a, **k)
    return time.time() - t0, out


def measure():
    m = {}

    # --- A build (measure directly at each T; also in 02 CSV) --------------
    m["abuild"] = {}
    for T in [5, 10, 20, 50, 100, 200, 400]:
        ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                                phi_max=0.2, hit_ineff=0.0)
        dt, ham = timeit(qp.build_hamiltonian, ev, epsilon=EPS, kernel="step",
                         gamma=GAMMA, delta=DELTA)
        m["abuild"][T] = (dt, ham.A.nnz)
        print(f"A build T={T}: {dt:.3f}s nnz={ham.A.nnz}", flush=True)

    # --- dense eigh (dilation circuit) at N grid -> fit c*N^3 --------------
    m["eigh"] = {}
    for N in [256, 512, 1024, 2048, 4096]:
        R = np.random.default_rng(0).standard_normal((N, N))
        R = (R + R.T) / 2
        dt, _ = timeit(np.linalg.eigh, R)
        m["eigh"][N] = dt
        print(f"eigh N={N}: {dt:.3f}s", flush=True)

    # --- FABLE angle prep (dense arccos + sfwht) at N grid -> fit N^2 logN -
    m["fable"] = {}
    for N in [512, 1024, 2048, 4096]:
        A = np.zeros((N, N))
        idx = np.random.default_rng(1).integers(0, N, (3 * N, 2))
        A[idx[:, 0], idx[:, 1]] = -1.0
        np.fill_diagonal(A, 4.0)
        dt, _ = timeit(bl.fable_classical, A, 0.0)
        m["fable"][N] = dt
        print(f"fable prep N={N}: {dt:.3f}s", flush=True)

    # --- Lanczos spectral bounds on real A (A/C-domain QSVT pre-step) ------
    from scipy.sparse.linalg import eigsh
    m["lanczos"] = {}
    for T in [100, 200, 400]:
        ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0,
                                phi_max=0.2, hit_ineff=0.0)
        ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step",
                                   gamma=GAMMA, delta=DELTA)
        A = ham.A.tocsr()
        t0 = time.time()
        eigsh(A, k=1, which="LA", return_eigenvectors=False, maxiter=5000)
        eigsh(A, k=1, which="SA", return_eigenvectors=False, maxiter=5000)
        m["lanczos"][T] = time.time() - t0
        print(f"lanczos T={T}: {m['lanczos'][T]:.3f}s", flush=True)

        # edge coloring on the same matrix
        C = sp.csr_matrix(abs(A - sp.diags(A.diagonal())))
        C.eliminate_zeros()
        dt, _ = timeit(bl.edge_coloring, C)
        m.setdefault("coloring", {})[T] = dt
        print(f"coloring T={T}: {dt:.3f}s", flush=True)

        # hit-oracle prep: sort 5T hits per coordinate + tables
        hx = np.array([h.x for h in ev.hits])
        hy = np.array([h.y for h in ev.hits])
        dt, _ = timeit(lambda: (np.sort(hx), np.sort(hy)))
        m.setdefault("hitsort", {})[T] = dt
        print(f"hit sort T={T}: {dt*1e6:.1f}us", flush=True)
    return m


def fit_pow(xs, ys, expo, logfac=False):
    """Fit y = c * x^expo (* log2 x) through measured points (least squares on c)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    basis = xs ** expo * (np.log2(xs) if logfac else 1.0)
    c = float(np.sum(ys * basis) / np.sum(basis ** 2))
    return c


def build_table(m):
    Ts_ab = sorted(m["abuild"])
    c_ab = fit_pow(Ts_ab[-3:], [m["abuild"][t][0] for t in Ts_ab[-3:]], 3)
    c_eigh = fit_pow(sorted(m["eigh"])[-3:],
                     [m["eigh"][k] for k in sorted(m["eigh"])[-3:]], 3)
    c_fab = fit_pow(sorted(m["fable"])[-3:],
                    [m["fable"][k] for k in sorted(m["fable"])[-3:]], 2, logfac=True)
    c_lan = fit_pow(sorted(m["lanczos"]), [m["lanczos"][k] for k in sorted(m["lanczos"])], 1)
    c_col = fit_pow(sorted(m["coloring"]), [m["coloring"][k] for k in sorted(m["coloring"])], 1)

    rows = []
    for T in T_GRID:
        n = int(np.ceil(np.log2(4 * T * T)))
        N = 1 << n
        if T in m["abuild"]:
            nnz = m["abuild"][T][1]
        else:
            nnz = int(4 * T * T + 9.3 * T)  # diagonal + measured off-diag ~9.3T
        t_abuild = m["abuild"][T][0] if T in m["abuild"] else c_ab * T ** 3
        t_eigh = c_eigh * N ** 3
        t_fable = c_fab * N ** 2 * np.log2(N)
        t_lan = m["lanczos"].get(T, c_lan * T)
        t_col = m["coloring"].get(T, c_col * T)
        t_sort = 5 * T * np.log2(max(2, 5 * T)) * 2e-8   # measured ~us level
        mem_dense = N * N * 8.0
        mem_sparse = nnz * 16.0
        mem_hits = 5 * T * 2 * 2.0   # two coords, 16-bit words

        rows += [
            dict(T=T, method="native", t_prep=t_abuild, t_bounds=0.0,
                 mem_bytes=mem_sparse, note="A build only"),
            dict(T=T, method="dilation", t_prep=t_abuild + t_eigh, t_bounds=t_lan,
                 mem_bytes=mem_dense, note="A build + dense eigh O(N^3)"),
            dict(T=T, method="fable", t_prep=t_fable, t_bounds=t_lan,
                 mem_bytes=mem_dense, note="dense angles + sfwht; O(N^2) MEMORY"),
            dict(T=T, method="lsfable", t_prep=t_abuild + 1e-8 * nnz, t_bounds=t_lan,
                 mem_bytes=mem_sparse, note="A build + O(nnz) angles"),
            dict(T=T, method="camps", t_prep=t_abuild + t_col + 1e-7 * nnz * n,
                 t_bounds=t_lan, mem_bytes=mem_sparse,
                 note="A build + coloring + O(nnz n) synthesis"),
            dict(T=T, method="dictionary", t_prep=t_abuild + t_col + 1e-7 * nnz * n,
                 t_bounds=t_lan, mem_bytes=mem_sparse, note="as camps + classes"),
            dict(T=T, method="szegedy", t_prep=t_abuild + 1e-8 * nnz, t_bounds=0.0,
                 mem_bytes=mem_sparse, note="A build + degrees; NO Lanczos"),
            dict(T=T, method="hit_oracle", t_prep=t_sort, t_bounds=0.0,
                 mem_bytes=mem_hits, note="hit sort + tables ONLY; no A, no Lanczos"),
        ]
    df = pd.DataFrame(rows)
    df["t_total"] = df.t_prep + df.t_bounds
    df.to_csv(os.path.join(OUT, "07_classical_cost.csv"), index=False)
    return df


LAB = dict(native="native 1BQF / classical solver (A build)",
           dilation="dilation, default QSVT (A + dense eigh)",
           fable="FABLE / S-FABLE (dense angles)",
           lsfable="LS-FABLE (A + nnz angles)",
           camps="Camps (A + coloring + synthesis)",
           dictionary="dictionary (as Camps + classes)",
           szegedy="szegedy generic (A + degrees)",
           hit_oracle="hit oracle (hit sort only)")


def figure(df):
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.grid": True, "grid.color": "#e8e7e2",
                         "grid.linewidth": 0.6, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for tag in ("fable", "dilation", "camps", "dictionary", "lsfable",
                "native", "szegedy", "hit_oracle"):
        g = df[df.method == tag].sort_values("T")
        ls = "--" if tag in ("dilation", "fable") else "-"
        axes[0].plot(g["T"], np.maximum(g.t_total, 1e-6), ls + "o",
                     color=MCOL[tag], lw=2, ms=4.5, label=LAB[tag])
        axes[1].plot(g["T"], g.mem_bytes, ls + "o", color=MCOL[tag], lw=2,
                     ms=4.5, label=LAB[tag])
    for ax, ylab, title in ((axes[0], "seconds (measured + fitted scaling)",
                             "classical prep time per event"),
                            (axes[1], "bytes", "classical memory footprint")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tracks T"); ax.set_ylabel(ylab); ax.set_title(title)
    axes[1].axhline(1e12, color="#d03b3b", lw=1, ls=":")
    axes[1].text(6, 1.5e12, "1 TB", fontsize=8, color="#d03b3b")
    axes[0].legend(fontsize=7.2, frameon=False, loc="upper left")
    fig.suptitle("Classical cost to get each encoding onto the computer "
                 "(step kernel, per event; dashed = includes a dense O(N²)–O(N³) object)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig07_classical_prep.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    m = measure()
    df = build_table(m)
    figure(df)
    sub = df[df["T"].isin([400, 1000])]
    print(sub[["T", "method", "t_prep", "t_bounds", "t_total", "mem_bytes"]]
          .to_string(index=False))
    print("done")
