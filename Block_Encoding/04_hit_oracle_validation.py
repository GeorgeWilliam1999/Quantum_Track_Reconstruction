"""
04 -- hit-level geometry-oracle validation (step 1 of HIT_ORACLE_DESIGN.md).

Claim under test: for a segment (a,b), every coupled continuation (b,c) has hit
c inside a narrow window around the straight-line extrapolation of (a,b) into
the next layer.  If true, the oracle's slot register only needs
w = (hits in window) slots -> alpha = 2^ceil(log2 w), and the index oracle is
sorted-position arithmetic + an O(n_hits) QROM instead of anything per-nonzero.

Measures on real store events (step kernel, fixed eps=2 mrad):
  - extrapolation residual dx, dr of every accepted coupling,
  - window occupancy w for a global per-event radius (x-band, sorted-x oracle),
  - coverage (must be 1.0),
  - comparison with the in-matrix degree Delta and greedy chi'.

Output: outputs/04_hit_oracle_window.csv, outputs/fig04_window_vs_T.png
"""

import os
import sys

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
CLEAN = dict(sigma_scatt=1e-4, sigma_res=0.0)
NOISY = dict(sigma_scatt=3e-4, sigma_res=0.01)


def analyse(T, noise, label):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, phi_max=0.2, hit_ineff=0.0, **noise)
    ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step",
                               gamma=GAMMA, delta=DELTA)
    A = ham.A.tocsr()
    d = A.shape[0]
    C = sp.csr_matrix(A - sp.diags(A.diagonal()))
    C.eliminate_zeros()

    hits = ev.hits
    hx = np.array([h.x for h in hits])
    hy = np.array([h.y for h in hits])
    hz = np.array([h.z for h in hits])
    hmod = np.array([h.module_id for h in hits])
    s2h = np.asarray(ham._segment_to_hit_ids)

    # coupled pairs (s1 -> s2 continuation: end hit of s1 == start hit of s2)
    coo = sp.triu(C, k=1).tocoo()
    dxs, drs = [], []
    pred_cache = {}
    for i, j in zip(coo.row, coo.col):
        a, b = s2h[i]
        b2, c = s2h[j]
        if b != b2:                      # orientation: allow either order
            a, b = s2h[j]
            b2, c = s2h[i]
            if b != b2:
                continue
        z_ratio = (hz[c] - hz[b]) / (hz[b] - hz[a])
        xp = hx[b] + (hx[b] - hx[a]) * z_ratio
        yp = hy[b] + (hy[b] - hy[a]) * z_ratio
        pred_cache[(a, b)] = (xp, yp, hmod[c])
        dxs.append(abs(hx[c] - xp))
        drs.append(np.hypot(hx[c] - xp, hy[c] - yp))
    dxs = np.array(dxs) if dxs else np.zeros(1)
    drs = np.array(drs) if drs else np.zeros(1)
    r_x = float(dxs.max()) * 1.05 + 1e-12     # sorted-x band radius
    r_r = float(drs.max()) * 1.05 + 1e-12     # disc radius (2D oracle)

    # window occupancy for EVERY segment with a possible continuation
    xs_by_mod = {m: np.sort(hx[hmod == m]) for m in np.unique(hmod)}
    w_band, w_disc = [], []
    for si in range(d):
        a, b = s2h[si]
        gm = hmod[b]
        if gm + 1 not in xs_by_mod:
            continue
        z_next = hz[hmod == gm + 1][0]
        z_ratio = (z_next - hz[b]) / (hz[b] - hz[a])
        xp = hx[b] + (hx[b] - hx[a]) * z_ratio
        yp = hy[b] + (hy[b] - hy[a]) * z_ratio
        xs = xs_by_mod[gm + 1]
        w_band.append(int(np.searchsorted(xs, xp + r_x, "right")
                          - np.searchsorted(xs, xp - r_x, "left")))
        mask = hmod == gm + 1
        w_disc.append(int((np.hypot(hx[mask] - xp, hy[mask] - yp) <= r_r).sum()))
    w_band = np.array(w_band) if w_band else np.zeros(1, int)
    w_disc = np.array(w_disc) if w_disc else np.zeros(1, int)

    # coverage check: every accepted continuation inside the x band
    cov = 1.0
    n_pairs = len(coo.row)
    if n_pairs:
        bad = 0
        for i, j in zip(coo.row, coo.col):
            a, b = s2h[i]
            b2, c = s2h[j]
            if b != b2:
                a, b = s2h[j]
                b2, c = s2h[i]
                if b != b2:
                    continue
            xp, yp, _ = pred_cache[(a, b)]
            if abs(hx[c] - xp) > r_x:
                bad += 1
        cov = 1.0 - bad / n_pairs

    m = bl.structure_metrics(A)
    row = dict(T=T, noise=label, d=d, nnz_off=int(C.nnz), n_pairs=n_pairs,
               r_x_mm=r_x, r_disc_mm=r_r,
               w_band_max=int(w_band.max()), w_band_p95=float(np.percentile(w_band, 95)),
               w_disc_max=int(w_disc.max()), w_disc_p95=float(np.percentile(w_disc, 95)),
               coverage=cov, delta_max=m["delta_max"], chi_greedy=m["chi_greedy"],
               alpha_hit_band=1 << int(np.ceil(np.log2(max(2, w_band.max())))),
               alpha_hit_disc=1 << int(np.ceil(np.log2(max(2, w_disc.max())))))
    print(row, flush=True)
    return row


def main():
    rows = []
    for T in (5, 10, 20, 50, 100, 200, 400, 700, 1000):
        rows.append(analyse(T, CLEAN, "clean"))
        if T in (100, 400):
            rows.append(analyse(T, NOISY, "noisy"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "04_hit_oracle_window.csv"), index=False)

    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.grid": True, "grid.color": "#e8e7e2",
                         "grid.linewidth": 0.6, "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    g = df[df.noise == "clean"]
    ax.plot(g["T"], g.w_disc_max, "-o", color="#e34948", lw=2, ms=5,
            label="disc window max (2D oracle)")
    ax.plot(g["T"], g.w_band_max, "-o", color="#eb6834", lw=2, ms=5,
            label="x-band window max (1D sort oracle)")
    ax.plot(g["T"], g.delta_max, "-o", color="#1baf7a", lw=2, ms=5,
            label="in-matrix degree Δ (lower bound)")
    gn = df[df.noise == "noisy"]
    if len(gn):
        ax.plot(gn["T"], gn.w_band_max, "s", color="#eb6834", ms=8, mfc="none",
                label="x-band, noisy")
    ax.set_xscale("log")
    ax.set_xlabel("tracks T")
    ax.set_ylabel("window occupancy w  (→ α = 2^⌈log₂ w⌉)")
    ax.set_title("Hit-oracle slot budget on real events (step, ε=2 mrad, rep 0)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig04_window_vs_T.png"), dpi=160)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
