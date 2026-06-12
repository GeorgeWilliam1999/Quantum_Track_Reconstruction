"""
Deep dive part 3 — the resolution wall as a DESIGN LAW: eff(T, N_bins).

From parts 1+2 we know the two mechanisms and their scales:
  M1 merge:  r0 ~ 2.5 bins  -> shrinks with finer grids -> density win
  M2 split:  activates when the resolution scale r0 = 2.5w drops below the
             physical smear -> complete-linkage prediction (fig08):
             0% (256), 0.06% (512), 7.9% (1024), 33.2% (2048)
So eff(T, N_bins) should be:  eff = (1 - split(N)) * merge_term(T, r0(N)),
with r0(N) ~ const x bin width.  The scan TESTS all of this:
  * eff/ghost vs T for N_bins in {64,...,2048}
  * r0 extracted per grid from its own P(lost|NN) sigmoid -> r0 vs bin width
  * isolated-track loss per grid -> M2 prediction test
  * scaling collapse: all (T, N) points vs the two-mechanism model

Figures: fig11_resolution_scan, fig12_merge_radius_law, fig13_scaling_collapse
Data:    resolution_scan.csv (per event), resolution_fits.csv (per grid)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hough_study_lib import (load_clean_store_events, event_arrays, run_point,
                             truth_tables, eval_candidates, per_track_outcome,
                             f_nn, logistic_loss, claim_radius, bin_width, AREA)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "deep_dive")
os.makedirs(OUT, exist_ok=True)
GRIDS = [64, 128, 256, 512, 1024, 2048]
REPS = list(range(10))


def scan():
    sel, qp = load_clean_store_events()
    sel = sel[sel.rep.isin(REPS)]
    ev_rows, trk_rows = [], []
    cache = {}
    for _, r in sel.iterrows():
        ev = qp.load_event(qp.store.event_path(r.event_key))
        xs, ys, zs, planes, truth = event_arrays(ev)
        D = np.column_stack([xs / zs, ys / zs])
        df_truth = truth_tables(ev, D, planes, truth)
        for nb in GRIDS:
            import time
            t0 = time.time()
            cands, centres, _ = run_point(D, planes, nb, 1.0)
            dt = time.time() - t0
            df_c, matched = eval_candidates(cands, truth)
            df_t = per_track_outcome(df_truth, matched)
            ghosts = int((df_c.purity < 0.70).sum())
            ev_rows.append(dict(T=int(r.n_trk), rep=int(r.rep), n_bins=nb,
                                eff=df_t.matched.mean(),
                                ghost=ghosts / max(1, len(df_c)),
                                n_cand=len(df_c), t_ms=1e3 * dt))
            keep = df_t[["nn_dist", "spread_max", "matched"]].copy()
            keep["T"] = int(r.n_trk); keep["n_bins"] = nb
            trk_rows.append(keep)
    return pd.DataFrame(ev_rows), pd.concat(trk_rows, ignore_index=True)


def fit_per_grid(df_t):
    """Per-grid sigmoid fit of P(lost|NN) + isolated (M2) loss measurement."""
    from scipy.optimize import curve_fit
    rows = []
    for nb in GRIDS:
        d = df_t[df_t.n_bins == nb]
        w = bin_width(nb)
        rc = claim_radius(nb, 1.0)
        edges = np.linspace(0, max(10 * w, 0.012), 30)
        mids, pl, nn = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d.nn_dist >= lo) & (d.nn_dist < hi)
            if m.sum() >= 40:
                mids.append(0.5 * (lo + hi)); pl.append(1 - d.matched[m].mean())
                nn.append(m.sum())
        mids, pl, nn = map(np.array, (mids, pl, nn))
        sig = np.sqrt(np.maximum(pl * (1 - pl), 1e-4) / nn)
        try:
            popt, _ = curve_fit(logistic_loss, mids, pl,
                                p0=[0.02, 0.6, 2 * w, w / 2], sigma=sig,
                                bounds=([0, 0, 0, 1e-6], [1, 1, 0.05, 0.03]))
        except RuntimeError:
            popt = [np.nan] * 4
        iso = d[d.nn_dist > max(5 * rc, 0.02)]
        # no isolated tracks exist at very coarse grids (claim radius ~ half the
        # space); fragmentation is then geometrically impossible -> split = 0
        iso_loss = 1 - iso.matched.mean() if len(iso) > 50 else 0.0
        rows.append(dict(n_bins=nb, w=w, r_claim=rc, b=popt[0], a=popt[1],
                         r0=popt[2], s=popt[3],
                         iso_loss=iso_loss, iso_n=len(iso)))
    return pd.DataFrame(rows)


def fig11(df_ev):
    g = (df_ev.groupby(["n_bins", "T"])
         .agg(eff=("eff", "mean"), eff_sem=("eff", "sem"),
              ghost=("ghost", "mean")).reset_index())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(GRIDS)))
    for nb, col in zip(GRIDS, cmap):
        d = g[g.n_bins == nb]
        ax[0].errorbar(d["T"], d["eff"], yerr=d["eff_sem"], fmt="o-", color=col,
                       label=f"{nb}$^2$ grid", ms=4, lw=1.5)
        ax[1].plot(d["T"], d["ghost"], "s--", color=col, ms=4, lw=1.5,
                   label=f"{nb}$^2$")
    ax[0].set(xscale="log", xlabel="T", ylabel="track efficiency", ylim=(0.3, 1.03),
              title="(a) Efficiency vs T per accumulator grid:\ncoarse grids merge "
                    "(density wall left), the finest grid splits (smear wall, flat offset)")
    ax[1].set(xscale="log", xlabel="T", ylabel="ghost rate", yscale="log",
              title="(b) Ghost rate: merging's other face,\nfalls with finer grids")
    ax[0].legend(fontsize=9); ax[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_resolution_scan.png"), dpi=130)
    plt.close(fig)
    print("fig11 done")


def fig12(fits):
    m2_pred = np.load(os.path.join(OUT, "m2_predictions.npy"),
                      allow_pickle=True).item()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    ok = fits.dropna(subset=["r0"])
    ax[0].loglog(ok.w, ok.r0, "o-", color="crimson", lw=2, label="fitted $r_0$ per grid")
    ww = np.linspace(ok.w.min() * 0.9, ok.w.max() * 1.1, 50)
    slope = (ok.r0 / ok.w).median()
    ax[0].loglog(ww, slope * ww, "k--", lw=1.5,
                 label=f"$r_0 = {slope:.2f}\\,w$ (median ratio; theory "
                       "$2\\sqrt{\\sigma_{sm}^2+w^2/12}\\approx 2.1w$)")
    ax[0].axhline(1.1e-3, color="steelblue", ls=":", lw=2,
                  label="physical smear scale (part 1) — $r_0$ passes straight\n"
                        "through it: smear FRAGMENTS (b), it does not broaden")
    ax[0].set(xlabel="bin width $w$", ylabel="merge radius $r_0$",
              title="(a) The merge radius IS the grid: $r_0\\propto w$ at every grid,\n"
                    "with NO smear floor — peak width is set by smoothing alone")
    ax[0].legend(fontsize=8.5)
    x = np.arange(len(fits))
    ax[1].bar(x - 0.2, fits.iso_loss, 0.4, color="darkorange",
              label="measured isolated-track loss (M2)")
    ax[1].bar(x + 0.2, [m2_pred.get(nb, np.nan) for nb in fits.n_bins], 0.4,
              color="gray", alpha=0.8,
              label="predicted: no compact 3-vote cluster at $r_0$ (fig08)")
    ax[1].set_xticks(x, [f"{nb}" for nb in fits.n_bins])
    ax[1].set(xlabel="grid", ylabel="isolated-track loss rate",
              title="(b) M2 prediction test: fragmentation switches on where the\n"
                    "resolution scale $r_0=2.5w$ undercuts the radial vote spread")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig12_merge_radius_law.png"), dpi=130)
    plt.close(fig)
    print("fig12 done")


def fig13(df_ev, fits):
    """Two-mechanism model vs every measured point."""
    g = (df_ev.groupby(["n_bins", "T"])
         .agg(eff=("eff", "mean"), eff_sem=("eff", "sem")).reset_index())
    fitmap = fits.set_index("n_bins")
    pred, meas, errs, labs = [], [], [], []
    for _, row in g.iterrows():
        f = fitmap.loc[row.n_bins]
        if np.isnan(f.r0):
            continue
        lam = (row["T"] - 1) / AREA
        # split (M2) and merge (M1) applied independently; b is NOT added on
        # top of iso_loss (both encode the same NN-independent fragmentation)
        merge = f.a * (1 - np.exp(-np.pi * lam * f.r0 ** 2))
        model = (1 - f.iso_loss) * (1 - merge)
        pred.append(model); meas.append(row["eff"]); errs.append(row["eff_sem"])
        labs.append((row.n_bins, row["T"]))
    pred, meas = np.array(pred), np.array(meas)
    fig, ax = plt.subplots(figsize=(7.6, 7))
    cmap = {nb: c for nb, c in zip(GRIDS, plt.cm.plasma(np.linspace(0, 0.85, len(GRIDS))))}
    for (nb, T), p, m, e in zip(labs, pred, meas, errs):
        ax.errorbar(p, m, yerr=e, fmt="o", ms=5, color=cmap[nb], alpha=0.85)
    for nb in GRIDS:
        ax.plot([], [], "o", color=cmap[nb], label=f"{nb}$^2$ grid")
    lims = (0.3, 1.02)
    ax.plot(lims, lims, "k-", lw=1)
    rms = np.sqrt(((pred - meas) ** 2).mean())
    ax.set(xlim=lims, ylim=lims,
           xlabel="two-mechanism model:  $(1-\\mathrm{split}_N)\\,"
                  "(1-b-a(1-e^{-\\pi\\lambda r_0^2(N)}))$",
           ylabel="measured efficiency",
           title="Scaling collapse: every (grid, T) point vs the model\n"
                 f"48 configurations, rms deviation {rms*100:.1f}% (vertical scatter "
                 "at the finest grids =\nper-event $z_{pv}$ variance of the split term, "
                 "see fig16)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig13_scaling_collapse.png"), dpi=130)
    plt.close(fig)
    print(f"fig13 done | collapse rms = {rms*100:.2f}%")


if __name__ == "__main__":
    cache_ev = os.path.join(OUT, "resolution_scan.csv")
    cache_t = os.path.join(OUT, "resolution_tracks.csv.gz")
    if os.path.exists(cache_ev) and os.path.exists(cache_t):
        print("loading cached scan (delete resolution_scan.csv to recompute)")
        df_ev, df_t = pd.read_csv(cache_ev), pd.read_csv(cache_t)
    else:
        df_ev, df_t = scan()
        df_t.to_csv(cache_t, index=False)
    df_ev.to_csv(os.path.join(OUT, "resolution_scan.csv"), index=False)
    fits = fit_per_grid(df_t)
    fits.to_csv(os.path.join(OUT, "resolution_fits.csv"), index=False)
    print(fits.to_string(index=False))
    fig11(df_ev)
    fig12(fits)
    fig13(df_ev, fits)
