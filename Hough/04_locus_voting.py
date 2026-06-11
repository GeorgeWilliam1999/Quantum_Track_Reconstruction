"""
Deep dive part 4 — LOCUS VOTING: the exact Hough transform dismantles the smear.

First principles: the point vote d = (x/z, y/z) ASSUMES z_pv = 0.  But a hit at
(x, y, z) is consistent with any vertex position zeta on the beam line, tracing
the 1-D locus
        d(zeta) = (x, y) / (z - zeta) ,   zeta in [-3.5, 3.5] mm  (3.5 sigma_z)
-- a short RADIAL segment in vote space.  All 5 loci of one track intersect
EXACTLY at t when zeta = z_pv: voting along the locus restores the concurrency
that the reference's hit->line picture promises, and the vertex smear (part 1's
dominant term, ~18x scattering) disappears BY CONSTRUCTION.  What remains is
scattering (~1e-4) -- two orders below the 256-grid bin width.

Predictions to test:
  (P-a) Fine grids stop splitting: the locus claim is segment-distance based, so
        the 1024/2048-point-vote M2 losses (fig12b) must vanish.
  (P-b) The density wall recedes: r0 ~ 2.5 bins still, but bins can now shrink
        to the SCATTERING floor instead of the vertex floor -> at 1024-locus the
        merge radius is ~4x smaller than 256-point -> ~16x fewer merges at fixed T.
  (P-c) Cost: x n_zeta votes (29) and a duplicate-free accumulator; register for
        a quantum implementation unchanged (the vote ORACLE changes, not the
        bin register) -> 20 qubits at 1024^2, still constant in T.

Figures: fig14_locus_explained (the geometry), fig15_wall_dismantled (the result)
Data:    locus_results.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hough_study_lib import (load_clean_store_events, event_arrays, run_point,
                             run_locus, locus_curves, accumulate_locus,
                             truth_tables, eval_candidates, per_track_outcome,
                             find_peaks, bin_width)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "deep_dive")
os.makedirs(OUT, exist_ok=True)
REPS = list(range(10))

CONFIGS = [
    ("point 256",  dict(mode="point", n_bins=256),  "steelblue"),
    ("point 1024", dict(mode="point", n_bins=1024), "darkorange"),
    ("locus 1024", dict(mode="locus", n_bins=1024), "crimson"),
    ("locus 2048", dict(mode="locus", n_bins=2048), "purple"),
]


def solve(mode, n_bins, arrays):
    xs, ys, zs, planes, truth = arrays
    D = np.column_stack([xs / zs, ys / zs])
    t0 = time.time()
    if mode == "point":
        cands, _, _ = run_point(D, planes, n_bins, 1.0)
    else:
        cands, _, _ = run_locus((xs, ys, zs, planes), n_bins=n_bins)
    dt = time.time() - t0
    return cands, dt


def fig14():
    """The locus geometry, drawn for one real T=50 store event region."""
    sel, qp = load_clean_store_events()
    # pick the T=50 event with the LARGEST |z_pv| (the smear is per-event)
    best, best_z = None, -1.0
    for _, rr in sel[sel.n_trk == 50].iterrows():
        e = qp.load_event(qp.store.event_path(rr.event_key))
        az = abs(e.primary_vertices[0].z)
        if az > best_z:
            best, best_z = e, az
    ev = best
    print(f"fig14 event |z_pv| = {best_z:.2f} mm")
    xs, ys, zs, planes, truth = event_arrays(ev)
    D = np.column_stack([xs / zs, ys / zs])
    # pick the track with the LARGEST point-vote smear (worst case for point vote)
    df_truth = truth_tables(ev, D, planes, truth)
    worst = df_truth.sort_values("spread_max").iloc[-1]
    m = np.where(truth == worst.tid)[0]
    Dz, segA, segB = locus_curves(xs[m], ys[m], zs[m])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(m)))
    for i, c in enumerate(cols):
        ax[0].scatter(*D[m[i]], color=c, s=70, zorder=5)
        ax[1].plot(Dz[i, :, 0], Dz[i, :, 1], "-", color=c, lw=2,
                   label=f"hit on z={zs[m[i]]:.0f}")
        ax[1].scatter(*D[m[i]], color=c, s=55, zorder=5)
    for a in ax:
        a.scatter([worst.cx], [worst.cy], marker="+", s=160, c="k", zorder=6,
                  label="point-vote centroid" if a is ax[0] else None)
    w256 = bin_width(256); w1024 = bin_width(1024)
    from matplotlib.patches import Rectangle
    for a, w, lab in [(ax[0], w256, "256 bin"), (ax[1], w1024, "1024 bin")]:
        a.add_patch(Rectangle((worst.cx - w / 2, worst.cy - w / 2), w, w,
                              fill=False, ec="crimson", lw=1.6))
        a.annotate(lab, (worst.cx + w / 2, worst.cy - w / 2), fontsize=9,
                   color="crimson")
    pad = 1.1 * worst.spread_max
    for a in ax:
        a.set(xlim=(worst.cx - pad, worst.cx + pad),
              ylim=(worst.cy - pad, worst.cy + pad),
              xlabel="$t_x$", ylabel="$t_y$")
    ax[0].set_title("(a) POINT votes of the worst-smeared T=50 track:\n"
                    "5 votes scattered over many fine-grid bins (the $z_{pv}$ smear)")
    ax[1].set_title("(b) LOCUS votes of the SAME hits: each hit votes along\n"
                    "$d(\\zeta)=(x,y)/(z-\\zeta)$ — all 5 segments cross at one point")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig14_locus_explained.png"), dpi=130)
    plt.close(fig)
    print(f"fig14 done | worst track spread={worst.spread_max:.2e} "
          f"({worst.spread_max/w1024:.0f} fine bins)")


def sweep():
    sel, qp = load_clean_store_events()
    sel = sel[sel.rep.isin(REPS)]
    rows = []
    for _, r in sel.iterrows():
        ev = qp.load_event(qp.store.event_path(r.event_key))
        arrays = event_arrays(ev)
        truth = arrays[4]
        for name, cfg, _ in CONFIGS:
            cands, dt = solve(cfg["mode"], cfg["n_bins"], arrays)
            df_c, matched = eval_candidates(cands, truth)
            D = np.column_stack([arrays[0] / arrays[2], arrays[1] / arrays[2]])
            df_t = per_track_outcome(
                truth_tables(ev, D, arrays[3], truth), matched)
            nm = matched  # noqa
            clones = sum(v - 1 for v in matched.values())
            rows.append(dict(config=name, T=int(r.n_trk), rep=int(r.rep),
                             eff=df_t.matched.mean(),
                             ghost=(df_c.purity < 0.70).sum() / max(1, len(df_c)),
                             clone=clones / max(1, len(matched)),
                             t_ms=1e3 * dt))
    return pd.DataFrame(rows)


def fig15(df):
    g = (df.groupby(["config", "T"])
         .agg(eff=("eff", "mean"), eff_sem=("eff", "sem"),
              ghost=("ghost", "mean"), clone=("clone", "mean"),
              t_ms=("t_ms", "mean")).reset_index())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, cfg, col in CONFIGS:
        d = g[g.config == name]
        ax[0].errorbar(d["T"], d["eff"], yerr=d["eff_sem"], fmt="o-", color=col,
                       lw=2, ms=5, label=name)
        ax[1].plot(d["T"], d["ghost"], "s--", color=col, lw=1.5, ms=5, label=name)
    ax[0].axhline(0.99, color="gray", lw=0.7, ls=":")
    ax[0].set(xscale="log", xlabel="T", ylabel="track efficiency", ylim=(0.55, 1.02),
              title="(a) The wall dismantled: locus removes the smear,\nfine bins "
                    "remove the density wall\n(point-1024 scatter = $z_{pv}$ sampling, fig16)")
    ax[1].set(xscale="log", yscale="log", xlabel="T", ylabel="ghost rate",
              title="(b) Ghost rate falls with the merge probability")
    ax[0].legend(fontsize=9); ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig15_wall_dismantled.png"), dpi=130)
    plt.close(fig)
    print("fig15 done")
    print(g.pivot_table(index="T", columns="config", values="eff").to_string())


if __name__ == "__main__":
    fig14()
    cache = os.path.join(OUT, "locus_results.csv")
    if os.path.exists(cache):
        print("loading cached sweep (delete locus_results.csv to recompute)")
        df = pd.read_csv(cache)
    else:
        df = sweep()
        df.to_csv(cache, index=False)
    fig15(df)
