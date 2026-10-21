#!/usr/bin/env python3
"""Paper 1, Figure 1: from hits to a Hamiltonian, on the paper corpus configuration.

(a) a toy event (T = 30 for legibility, same configuration as the corpus) in the x-z
    projection, with an inset of hit residuals from straight-line fits (micrometres)
(b) the acceptance window: kink distribution of true continuations against the Rayleigh
    law with sigma_p, and of false head-to-tail candidates, with the step window at epsilon
    (T = 400, reps 0-9 pooled)
(c) graph and matrix: real connected groups cut from the T = 400 rep 0 event (a complete
    track, a false pair, a hub, an isolated false segment) drawn as graphs and as blocks of
    A = sI - C
(d) classical activations on the T = 400 rep 0 event, true and false, log count, with the
    efficiency-first threshold (99 % of true segments kept)

Every number drawn is read from the event or from outputs/metrics.csv; nothing is typed in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
import numpy as np
from matplotlib.patches import Rectangle
from scipy.sparse.csgraph import connected_components

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
NOISE = dict(sigma_scatt=float(os.environ.get("CORPUS_SIGMA_SCATT", "1.5e-4")),
             sigma_res=float(os.environ.get("CORPUS_SIGMA_RES", "0.005")), phi_max=0.2, hit_ineff=0.01)
CORPUS = os.environ.get("CORPUS_OUT", "outputs_5um")
FIG = HERE / "figures" / CORPUS
FIG.mkdir(parents=True, exist_ok=True)
GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA
EPS = float(compute_epsilon(NOISE["sigma_res"], NOISE["sigma_scatt"]))
SIGMA_P = float(np.sqrt(NOISE["sigma_scatt"] ** 2 + 6 * np.arctan(NOISE["sigma_res"] / 33.0) ** 2))
GREEN, RED, GREY = "#2e7d32", "#c62828", "#9e9e9e"


def build(T, rep):
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **NOISE)
    ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step", gamma=GAMMA, delta=DELTA)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    A, b, _, _ = dp_terms.dp_system(ham, beta=0.0, eps_B=None, alpha=0.0, gamma=GAMMA, delta=DELTA)
    A = A.tocsr()
    C = (-A).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    return ev, ham, truth, A, C


def kinks(ham, truth, C, n_false_sample=400_000, seed=0):
    hid = np.asarray(ham._segment_to_hit_ids).astype(np.int64)
    vec = np.asarray(ham._segment_vectors, float)
    tid = np.asarray(ham._segment_track_ids)
    tr = np.nonzero(truth)[0]
    t_start = {int(hid[i, 0]): i for i in tr}
    ti, tj = [], []
    for i in tr:
        j = t_start.get(int(hid[i, 1]))
        if j is not None and tid[j, 0] == tid[i, 0]:
            ti.append(i)
            tj.append(j)
    ti, tj = np.asarray(ti), np.asarray(tj)

    def k(i, j):
        return np.arccos(np.clip(np.einsum("ij,ij->i", vec[i], vec[j]), -1, 1))

    n = len(hid)
    srt = np.argsort(hid[:, 0], kind="stable")
    ss = hid[srt, 0]
    lo = np.searchsorted(ss, hid[:, 1], "left")
    hi = np.searchsorted(ss, hid[:, 1], "right")
    nout = hi - lo
    rng = np.random.default_rng(seed)
    cand = np.nonzero(nout > 0)[0]
    ii = rng.choice(cand, size=n_false_sample, replace=True)
    jj = srt[lo[ii] + (rng.random(len(ii)) * nout[ii]).astype(np.int64)]
    false_pair = ~(truth[ii] & truth[jj] & (tid[ii, 0] == tid[jj, 0]))
    return k(ti, tj), k(ii[false_pair], jj[false_pair])


def panel_event(ax):
    ev, ham, truth, A, C = build(30, 0)
    zs = sorted({h.z for h in ev.hits})
    for z in zs:
        ax.axvline(z, color="0.85", lw=6, zorder=0)
    by_track = {}
    for h in ev.hits:
        by_track.setdefault(h.track_id, []).append(h)
    for t, hs in by_track.items():
        hs = sorted(hs, key=lambda h: h.z)
        z = np.array([h.z for h in hs])
        x = np.array([h.x for h in hs])
        ax.plot(np.r_[0.0, z], np.r_[0.0, x], color=GREY, lw=0.6, zorder=1)
        ax.plot(z, x, "o", ms=2.2, color="k", zorder=2)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_xlim(-5, 175)
    ax.set_title(f"(a) an event: T = 30 tracks, five planes\n"
                 f"$\\sigma_{{\\rm res}}$ = {NOISE['sigma_res']*1e3:.0f} µm, $\\sigma_{{\\rm scatt}}$ = {NOISE['sigma_scatt']*1e3:.2f} mrad, 1 % hits lost",
                 fontsize=8.5, loc="left")
    # inset: residuals from a straight-line fit, micrometres, all tracks with five hits
    ev4, ham4, truth4, A4, C4 = build(400, 0)
    res = []
    by4 = {}
    for h in ev4.hits:
        by4.setdefault(h.track_id, []).append(h)
    for t, hs in by4.items():
        if len(hs) < 5:
            continue
        z = np.array([h.z for h in hs])
        for coord in ("x", "y"):
            v = np.array([getattr(h, coord) for h in hs])
            p = np.polyfit(z, v, 1)
            res.append((v - np.polyval(p, z)) * 1e3)
    res = np.concatenate(res)
    ins = ax.inset_axes([0.06, 0.66, 0.30, 0.25])
    ins.hist(res, bins=60, range=(-25, 25), color="0.35")
    ins.set_xlabel("residual from a straight line [µm]", fontsize=6.5)
    ins.tick_params(labelsize=6)
    ins.set_yticks([])
    ins.set_title(f"T = 400: rms {np.std(res):.1f} µm", fontsize=6.5)
    return float(np.std(res))


def panel_window(ax):
    kt, kf = [], []
    for rep in range(10):
        ev, ham, truth, A, C = build(400, rep)
        a, b = kinks(ham, truth, C, n_false_sample=100_000, seed=rep)
        kt.append(a)
        kf.append(b)
    kt, kf = np.concatenate(kt) * 1e3, np.concatenate(kf) * 1e3
    edges = np.logspace(-2, np.log10(600), 71)
    dlog = np.diff(np.log10(edges))
    ht, _ = np.histogram(kt, edges)
    hf, _ = np.histogram(kf, edges)
    ctr = np.sqrt(edges[1:] * edges[:-1])
    ax.step(ctr, ht / (len(kt) * dlog), where="mid", color=GREEN, lw=1.4,
            label=f"true continuations ({len(kt):,})")
    ax.step(ctr, hf / (len(kf) * dlog), where="mid", color=RED, lw=1.4,
            label=f"false candidates ({len(kf):,} sampled)")
    th = np.logspace(-2, np.log10(600), 600)
    sp = SIGMA_P * 1e3
    ray = th / sp ** 2 * np.exp(-th ** 2 / (2 * sp ** 2))
    ax.plot(th, ray * th * np.log(10), color=GREEN, ls="--", lw=1,
            label=f"Rayleigh, $\\sigma_p$ = {sp:.2f} mrad")
    ax.axvspan(1e-2, EPS * 1e3, color="#fff3c4", zorder=0)
    ax.axvline(EPS * 1e3, color="k", lw=1)
    frac_out = float(np.mean(kt >= EPS * 1e3))
    frac_f_in = float(np.mean(kf < EPS * 1e3))
    ax.text(EPS * 1e3 * 1.15, 1.6,
            f"$\\theta_{{\\max}}$ = {EPS * 1e3:.2f} mrad = $3\\sqrt{{2}}\\,\\sigma_p$\n"
            f"true continuations outside: {frac_out:.1e}\n"
            f"false candidates inside: {frac_f_in:.1e}", fontsize=7.5, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 600)
    ax.set_ylim(0, 2.1)
    ax.set_xlabel("kink between two segments sharing a hit [mrad]")
    ax.set_ylabel("fraction per decade")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(b) the window: $\\cos\\theta \\geq 1-\\varepsilon$, i.e. $\\theta<\\theta_{\\max}$ (shaded)",
                 fontsize=8.5, loc="left")
    return frac_out, frac_f_in, len(kt)


def _groups(T, rep):
    ev, ham, truth, A, C = build(T, rep)
    ncomp, lab = connected_components(C, directed=False)
    return truth, A, C, lab, np.bincount(lab)


def panel_blocks(ax):
    picks = []
    truth, A, C, lab, sizes = _groups(400, 0)
    deg = np.asarray(C.sum(axis=1)).ravel()
    gt = np.bincount(lab, weights=truth.astype(float))
    md = np.zeros(len(sizes))
    np.maximum.at(md, lab, deg)
    for name, cond in (("complete track", (sizes == 4) & (gt == 4) & (md <= 2)),
                       ("false pair", (sizes == 2) & (gt == 0)),
                       ("isolated false segment", (sizes == 1) & (gt == 0))):
        c = int(np.nonzero(cond)[0][0])
        picks.append((name, truth, A, C, np.nonzero(lab == c)[0]))
    hub = None
    for T, rep in [(1000, r) for r in range(10)]:
        truth_h, A_h, C_h, lab_h, sizes_h = _groups(T, rep)
        deg_h = np.asarray(C_h.sum(axis=1)).ravel()
        gt_h = np.bincount(lab_h, weights=truth_h.astype(float))
        md_h = np.zeros(len(sizes_h))
        np.maximum.at(md_h, lab_h, deg_h)
        cand = np.nonzero((gt_h == 0) & (md_h >= 3) & (sizes_h <= 5))[0]
        if len(cand):
            hub = (f"hub (T = {T})", truth_h, A_h, C_h, np.nonzero(lab_h == int(cand[0]))[0])
            break
    if hub is not None:
        picks.insert(2, hub)

    blocks, offset = [], 0
    for name, tr, AA, CC, idx in picks:
        sub = CC[idx][:, idx].toarray()
        start = int(np.argmin(sub.sum(1)))
        order, seen = [start], {start}
        while len(order) < len(idx):
            nxt = [j for i in order for j in np.nonzero(sub[i])[0] if j not in seen] or \
                  [j for j in range(len(idx)) if j not in seen][:1]
            for j in nxt:
                if j not in seen:
                    seen.add(j)
                    order.append(j)
        idx = idx[order]
        blocks.append((name, AA[idx][:, idx].toarray(), tr[idx], CC[idx][:, idx].toarray(), offset))
        offset += len(idx)
    m = offset
    M = np.full((m, m), np.nan)
    tv = []
    for name, B, t, G, o in blocks:
        M[o:o + len(t), o:o + len(t)] = np.where(B == 0, np.nan, B)
        tv.extend(t.tolist())
    ax.imshow(M, cmap="coolwarm_r", vmin=-4, vmax=4, extent=(-0.5, m - 0.5, m - 0.5, -0.5))
    for name, B, t, G, o in blocks:
        L = len(t)
        ax.add_patch(Rectangle((o - 0.5, o - 0.5), L, L, fill=False, lw=1.2, color="k"))
        for i in range(L):
            for j in range(L):
                if B[i, j] != 0:
                    ax.text(o + j, o + i, f"{B[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="w" if B[i, j] > 0 else "k")
        # graph drawing to the right of the block
        gx0, gy = m + 1.0, o + (L - 1) / 2
        if G.sum(1).max() >= 3:
            c = int(np.argmax(G.sum(1)))
            others = [k for k in range(L) if k != c]
            pos = {c: (gx0 + 0.6, gy)}
            for k, nb in enumerate(others):
                ang = np.pi * (k + 1) / (len(others) + 1) - np.pi / 2
                pos[nb] = (gx0 + 0.6 + 1.1 * np.cos(ang), gy + 1.1 * np.sin(ang))
        else:
            pos = {k: (gx0 + 0.9 * k, gy) for k in range(L)}
        for i in range(L):
            for j in range(i + 1, L):
                if G[i, j]:
                    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="0.4", lw=1)
        for k in range(L):
            ax.plot(*pos[k], "o", ms=7, color=GREEN if t[k] else RED, mec="k", mew=0.5)
        ax.text(gx0 + 4.2, gy, name, fontsize=7.5, va="center")
    ax.set_xticks(range(m))
    ax.set_xticklabels(["T" if v else "F" for v in tv], fontsize=6.5)
    ax.set_yticks([])
    ax.set_xlim(-0.5, m + 8.5)
    ax.set_ylim(m - 0.3, -0.7)
    ax.set_title("(c) $A = (\\alpha+\\beta)I - C$ is block-diagonal: one block per group\n"
                 "of compatible segments, with its graph (green true, red false)",
                 fontsize=8.5, loc="left")


def panel_classical(ax):
    import pandas as pd
    ev, ham, truth, A, C = build(400, 0)
    z = np.load(HERE / CORPUS / "amplitudes" / "T400_rep0.npz", allow_pickle=True)
    n = A.shape[0]
    x = np.full(n, float(z["iso_classical"]))
    x[z["seg_index"]] = z["x_classical"]
    met = pd.read_csv(HERE / CORPUS / "metrics.csv")
    row = met[(met["T"] == 400) & (met.rep == 0) & (met.solver == "classical")
              & (met.convention == "all true")].iloc[0]
    edges = np.linspace(0, 0.7, 141)
    ax.hist(x[~truth], bins=edges, color=RED, alpha=0.75, label=f"false ({(~truth).sum():,})")
    ax.hist(x[truth], bins=edges, color=GREEN, alpha=0.85, label=f"true ({truth.sum():,})")
    ax.axvline(row.tau_e990, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("classical activation $x_i$")
    ax.set_ylabel("segments")
    for v, lab_ in ((0.25, "1/4 isolated"), (1 / 3, "1/3 pairs"), (4 / 11, "4/11"), (5 / 11, "5/11")):
        ax.text(v, 2e5 if v == 0.25 else 3e3, lab_, fontsize=6.5, rotation=90, ha="right", va="top")
    ax.text(row.tau_e990 - 0.008, 2e5, ha="right", s=
            f"threshold keeping\n99 % of true;\nfalse rate {row.far_e990:.2f}", fontsize=7.5)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("(d) the classical answer $A\\mathbf{x}=\\mathbf{b}$ on a T = 400 event",
                 fontsize=8.5, loc="left")
    return float(row.tau_e990), float(row.far_e990)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 7.0))
    rms = panel_event(axs[0, 0])
    frac_out, frac_f_in, n_true_cont = panel_window(axs[0, 1])
    panel_blocks(axs[1, 0])
    tau, far = panel_classical(axs[1, 1])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_hits_to_hamiltonian.{ext}", dpi=200)
    with open(FIG / "fig1_numbers.txt", "w") as f:
        f.write(f"epsilon_mrad={EPS * 1e3:.4f}\nsigma_p_mrad={SIGMA_P * 1e3:.4f}\n"
                f"residual_rms_um_T400={rms:.2f}\ntrue_continuations_T400_10reps={n_true_cont}\n"
                f"true_outside_window={frac_out:.3e}\nfalse_candidates_inside_window={frac_f_in:.3e}\n"
                f"classical_tau_e990_T400_rep0={tau:.4f}\nclassical_far_e990_T400_rep0={far:.4f}\n")
    print(open(FIG / "fig1_numbers.txt").read())


if __name__ == "__main__":
    main()
