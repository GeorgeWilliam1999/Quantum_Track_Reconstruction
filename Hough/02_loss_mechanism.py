"""
Deep dive part 2 — WHERE the efficiency goes, track by track (store events).

Logic of the investigation:
  (1) Validate the density model: measured nearest-neighbour (NN) distances in
      vote space vs the Poisson prediction f(r) = 2 pi lam r exp(-pi lam r^2).
  (2) Measure the MECHANISM: P(lost | NN distance).  If merging is the loss
      mechanism, this curve is a property of the PEAK FINDER, not of T -> all
      T must collapse onto one universal curve.  Fit  b + a/(1+exp((r-r0)/s)).
      Predictions to check:  a ~ 0.5 (of a merged pair, ONE candidate usually
      survives with majority purity -> one track lost, not two);
      r0 ~ 2 bins (the 3x3 maximum_filter + sigma=1 smoothing scale);
      b ~ the isolated (smear) loss rate, small.
  (3) The efficiency LAW with no per-T fitting:
          eff(T) = 1 - integral p_loss(r) f(r;lam) dr
      must reproduce the measured eff(T) curve - the decisive test that we
      understand every lost track.
  (4) Census: classify every lost track (merge / smear-split / other).
  (5) Ghost anatomy: candidate purity, ghost composition (merged-pair products
      vs accidental froth).
  (6) Visual proof: accumulator zooms of a merged pair and a resolved pair.

Figures: fig04..fig09 + loss_census.csv.
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
                             f_nn, logistic_loss, eff_model, claim_radius,
                             bin_width, AREA, hit_directions)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "deep_dive")
os.makedirs(OUT, exist_ok=True)
N_BINS, SMOOTH = 256, 1.0
W = bin_width(N_BINS)
R_CLAIM = claim_radius(N_BINS, SMOOTH)
T_COLORS = plt.cm.viridis(np.linspace(0, 1, 8))


def harvest():
    """Run the baseline tracker on all 160 clean store events; keep per-track
    and per-candidate tables."""
    sel, qp = load_clean_store_events()
    tracks, cands_all, ev_rows = [], [], []
    for _, r in sel.iterrows():
        ev = qp.load_event(qp.store.event_path(r.event_key))
        xs, ys, zs, planes, truth = event_arrays(ev)
        D = np.column_stack([xs / zs, ys / zs])
        cands, centres, _ = run_point(D, planes, N_BINS, SMOOTH)
        df_c, matched = eval_candidates(cands, truth)
        df_t = per_track_outcome(truth_tables(ev, D, planes, truth), matched)
        df_t["T"] = int(r.n_trk); df_t["rep"] = int(r.rep)
        df_c["T"] = int(r.n_trk); df_c["rep"] = int(r.rep)
        tracks.append(df_t); cands_all.append(df_c)
        ev_rows.append(dict(T=int(r.n_trk), rep=int(r.rep),
                            eff=df_t.matched.mean(),
                            n_cand=len(df_c),
                            n_ghost=int((df_c.purity < 0.70).sum())))
    return (pd.concat(tracks, ignore_index=True),
            pd.concat(cands_all, ignore_index=True),
            pd.DataFrame(ev_rows))


def fig04_nn(df_t):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for a, T in zip(ax, [50, 200, 1000]):
        d = df_t[df_t["T"] == T].nn_dist
        bins = np.linspace(0, np.quantile(d, 0.995), 40)
        a.hist(d, bins=bins, density=True, alpha=0.6, color="steelblue",
               label="measured NN distance")
        rr = np.linspace(0, bins[-1], 300)
        a.plot(rr, f_nn(rr, T), "k-", lw=2,
               label="Poisson  $2\\pi\\lambda r e^{-\\pi\\lambda r^2}$\n"
                     f"$\\lambda=(T{{-}}1)/{AREA:.2f}$")
        a.axvline(2 * W, color="crimson", ls="--", lw=1.5, label="2 bins")
        a.set(xlabel="NN distance in vote space", title=f"T = {T}")
        if T == 50:
            a.set_ylabel("density")
        a.legend(fontsize=8)
    fig.suptitle("Track density model: nearest-neighbour distances are Poisson "
                 "(tracks iid uniform on $[-0.2,0.2]^2$)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig04_nn_distance.png"), dpi=130)
    plt.close(fig)
    print("fig04 done")


def fit_loss_curve(df_t):
    """Pooled P(lost | NN) and its logistic fit."""
    from scipy.optimize import curve_fit
    edges = np.concatenate([np.linspace(0, 0.012, 25), [0.016, 0.022, 0.03, 0.05]])
    mids, ploss, nn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df_t.nn_dist >= lo) & (df_t.nn_dist < hi)
        if m.sum() < 30:
            continue
        mids.append(0.5 * (lo + hi)); ploss.append(1 - df_t.matched[m].mean())
        nn.append(m.sum())
    mids, ploss, nn = map(np.array, (mids, ploss, nn))
    sig = np.sqrt(np.maximum(ploss * (1 - ploss), 1e-4) / nn)
    popt, _ = curve_fit(logistic_loss, mids, ploss, p0=[0.02, 0.5, 2 * W, W / 2],
                        sigma=sig, bounds=([0, 0, 0, 1e-5], [0.3, 1, 0.03, 0.02]))
    return mids, ploss, nn, popt


def fig05_loss_vs_nn(df_t, popt):
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    Ts = sorted(df_t["T"].unique())
    for T, col in zip(Ts, T_COLORS):
        d = df_t[df_t["T"] == T]
        edges = np.linspace(0, 0.015, 16)
        mids, pl = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d.nn_dist >= lo) & (d.nn_dist < hi)
            if m.sum() >= 25:
                mids.append(0.5 * (lo + hi)); pl.append(1 - d.matched[m].mean())
        ax.plot(mids, pl, "o-", ms=4, lw=1, color=col, alpha=0.8, label=f"T={T}")
    rr = np.linspace(0, 0.015, 400)
    b, a, r0, s = popt
    ax.plot(rr, logistic_loss(rr, *popt), "k-", lw=2.5,
            label=f"pooled fit  $b+a/(1+e^{{(r-r_0)/s}})$\n"
                  f"$a$={a:.2f}, $r_0$={r0*1e3:.2f}e-3 ({r0/W:.1f} bins), $b$={b:.3f}")
    ax.axvline(W, color="gray", ls=":", lw=1.5); ax.text(W, 0.62, " 1 bin", fontsize=9)
    ax.axvline(2 * W, color="gray", ls="--", lw=1.5); ax.text(2 * W, 0.62, " 2 bins", fontsize=9)
    ax.axvline(R_CLAIM, color="crimson", ls="--", lw=1.5)
    ax.text(R_CLAIM, 0.55, " claim radius", fontsize=9, color="crimson")
    ax.set(xlabel="distance to nearest other track in vote space",
           ylabel="P(track lost | NN distance)", ylim=(-0.02, 0.75),
           title="THE loss mechanism: a universal, T-independent merge curve\n"
                 f"amplitude $a$={a:.2f}: one of a merged pair USUALLY survives "
                 "(both die when the per-plane hit split is ~50/50 → ghost); "
                 f"midpoint $r_0$={r0/W:.1f} bins = the peak-finder resolution")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig05_loss_vs_nn.png"), dpi=130)
    plt.close(fig)
    print(f"fig05 done | fit: b={b:.4f} a={a:.3f} r0={r0:.4e} ({r0/W:.2f} bins) s={s:.2e}")


def fig06_eff_law(df_ev, popt):
    g = df_ev.groupby("T").agg(eff=("eff", "mean"), eff_sem=("eff", "sem")).reset_index()
    Ts = np.array(sorted(g["T"]))
    model = [eff_model(T, popt) for T in Ts]
    Tfine = np.logspace(1, 3.1, 60)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.errorbar(g["T"], g["eff"], yerr=g["eff_sem"], fmt="o", ms=7, color="steelblue",
                capsize=3, label="measured efficiency (20 events/point)", zorder=5)
    ax.plot(Tfine, [eff_model(T, popt) for T in Tfine], "k-", lw=2,
            label="parameter-free law\n$1-\\int p_{loss}(r)\\,f_{NN}(r;\\lambda)\\,dr$\n"
                  "($p_{loss}$ = pooled fit of fig05, $f_{NN}$ = Poisson)")
    b, a, r0, s = popt
    ax.plot(Tfine, 1 - b - a * (1 - np.exp(-np.pi * (Tfine - 1) / AREA * r0 ** 2)),
            "--", color="gray", lw=1.5,
            label="hard-disc approximation\n$1-b-a(1-e^{-\\pi\\lambda r_0^2})$")
    for T, m, e in zip(Ts, model, g["eff"]):
        ax.annotate(f"{(e-m)*100:+.1f}%", (T, e), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="steelblue")
    ax.set(xscale="log", xlabel="track multiplicity T", ylabel="track efficiency",
           ylim=(0.7, 1.02),
           title="The efficiency law: density merging explains the whole curve\n"
                 "(annotations: measured $-$ model residual)")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig06_efficiency_law.png"), dpi=130)
    plt.close(fig)
    print("fig06 done | residuals:",
          ", ".join(f"T{T}:{(e-m)*100:+.1f}%" for T, m, e in zip(Ts, model, g["eff"])))


def fig07_census(df_t, popt):
    b, a, r0, s = popt
    lost = df_t[~df_t.matched].copy()
    lost["mech"] = "M3 other"
    lost.loc[lost.spread_max > R_CLAIM, "mech"] = "M2 smear-split"
    lost.loc[lost.nn_dist < r0 + 2 * s, "mech"] = "M1 merge"   # merge takes priority
    census = (lost.groupby(["T", "mech"]).size().unstack(fill_value=0)
              .reindex(columns=["M1 merge", "M2 smear-split", "M3 other"], fill_value=0))
    totals = df_t.groupby("T").size()
    census_frac = census.div(totals, axis=0)
    census.to_csv(os.path.join(OUT, "loss_census.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    census.plot.bar(stacked=True, ax=ax[0],
                    color=["crimson", "darkorange", "gray"], width=0.75)
    ax[0].set(xlabel="T", ylabel="lost tracks (20 events)", yscale="log",
              title="(a) Census of every lost track")
    census_frac.plot.bar(stacked=True, ax=ax[1],
                         color=["crimson", "darkorange", "gray"], width=0.75)
    ax[1].set(xlabel="T", ylabel="fraction of all tracks lost",
              title="(b) ... as a fraction of all tracks: merging is EVERYTHING\n"
                    "(M2 smear-split $\\equiv$ 0 at this grid; even the rare low-T "
                    "losses are close pairs)")
    for a_ in ax:
        a_.legend(fontsize=9)
        a_.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig07_loss_census.png"), dpi=130)
    plt.close(fig)
    print("fig07 done\n", census)


def fig08_isolated(df_t):
    """The smear-split (M2) NULL at the baseline + the fine-grid PREDICTION.

    Measured: isolated tracks (NN > 5 r_claim) have loss rate EXACTLY 0 at the
    256 grid -- the claim radius (6 bins = 1.17e-2) covers the entire physical
    smear.  But the claim radius shrinks linearly with the grid: this figure
    computes, per track, the survival-relevant scale d3 = 3rd-smallest
    |d_k - c| (a track survives iff >=3 votes are claimable) and overlays the
    claim radii of finer grids -> a PREDICTION of where M2 switches on,
    testable in the resolution scan (part 3)."""
    iso = df_t[df_t.nn_dist > 5 * R_CLAIM]
    # FRAGMENTATION criterion: a track survives iff >=3 of its votes form a
    # COMPACT cluster (complete linkage: ALL pairwise distances <= r0=2.5w) --
    # a compact blob makes ONE smoothed maximum; a single-linkage CHAIN of
    # votes spaced ~r0 still smooths into separate maxima and fragments.
    # (Validated: complete@2.5w predicts 0.1/7.9/33.2% vs measured 1.7/8.8/32.7%
    # at 512/1024/2048; single-linkage underpredicts by ~10x.)
    sel, qp = load_clean_store_events()
    vote_sets = []
    for _, r in sel.iterrows():
        ev = qp.load_event(qp.store.event_path(r.event_key))
        xs, ys, zs, planes, truth = event_arrays(ev)
        D = np.column_stack([xs / zs, ys / zs])
        for t in ev.tracks:
            m = np.where(truth == t.track_id)[0]
            if len(m) >= 3:
                vote_sets.append(D[m])
    grids = [128, 256, 512, 1024, 2048]

    def frag_prob(r_link):
        """Fraction of tracks with NO compact >=3-vote cluster at scale r_link."""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        lost = 0
        for V in vote_sets:
            if len(V) < 3:
                lost += 1
                continue
            lab = fcluster(linkage(pdist(V), "complete"), r_link, criterion="distance")
            if np.bincount(lab).max() < 3:
                lost += 1
        return lost / len(vote_sets)

    preds = {nb: frag_prob(2.5 * bin_width(nb)) for nb in grids}
    # the spread observable, for the visual: largest gap closed at each scale
    span3 = []
    for V in vote_sets:
        dd = np.sort(np.linalg.norm(V - np.median(V, axis=0), axis=1))
        span3.append(dd[min(2, len(dd) - 1)])
    span3 = np.array(span3)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bins = np.logspace(-5.2, -1.9, 70)
    ax.hist(np.clip(span3, 1e-5, None), bins=bins, color="steelblue", alpha=0.75,
            label="per-track $d_3$ (3rd-closest vote to the median vote),\n"
                  "pooled over all T (49.6k tracks)")
    for nb, col in zip(grids, ["seagreen", "crimson", "darkorange", "purple", "brown"]):
        rl = 2.5 * bin_width(nb)
        ax.axvline(rl, color=col, ls="--", lw=1.8)
        ax.text(rl, ax.get_ylim()[1] * 0.93, f" {nb}\n {preds[nb]*100:.1f}%",
                color=col, fontsize=9)
    ax.set(xscale="log", xlabel="vote-cluster scale  [direction units]",
           ylabel="tracks / bin",
           title="The fragmentation (M2) mechanism — ABSENT at the 256 grid "
                 f"(measured isolated loss = {1-iso.matched.mean():.4f}, n={len(iso)})\n"
                 "predicted split losses (labels) = P(no COMPACT 3-vote cluster at "
                 "$r_0=2.5w$, complete linkage); dashed = each grid's $r_0$")
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig08_isolated_losses.png"), dpi=130)
    plt.close(fig)
    np.save(os.path.join(OUT, "m2_predictions.npy"), preds, allow_pickle=True)
    print("fig08 done | isolated loss rate:", f"{1 - iso.matched.mean():.4f}",
          f"({len(iso)} tracks) | predicted M2 split:",
          {k: f"{v*100:.2f}%" for k, v in preds.items()})


def fig09_ghosts(df_c, df_ev):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    # (a) purity distributions
    for T, col in zip([100, 400, 1000], ["steelblue", "darkorange", "crimson"]):
        d = df_c[df_c["T"] == T]
        ax[0].hist(d.purity, bins=np.linspace(0.2, 1.0, 33), histtype="step",
                   lw=2, color=col, label=f"T={T}", density=True)
    ax[0].axvline(0.70, color="k", ls="--", lw=1.5, label="purity cut 0.70")
    ax[0].set(xlabel="candidate purity (majority-hit fraction)", ylabel="density",
              yscale="log", title="(a) Candidate purity: a clean 1.0 spike +\n"
                                  "a merged-pair shoulder at ~0.4-0.6")
    ax[0].legend(fontsize=9)
    # (b) ghost composition: merged-pair product vs froth
    ghosts = df_c[df_c.purity < 0.70].copy()
    ghosts["kind"] = np.where((ghosts.n_major >= 2) & (ghosts.n_second >= 2),
                              "merged-pair product", "froth (>=3 truths)")
    comp = ghosts.groupby(["T", "kind"]).size().unstack(fill_value=0)
    comp.plot.bar(stacked=True, ax=ax[1], color=["crimson", "gray"], width=0.7)
    ax[1].set(xlabel="T", ylabel="ghost candidates (20 events)",
              title="(b) What a ghost IS here: two real tracks\n"
                    "merged into one impure candidate (not froth)")
    ax[1].tick_params(axis="x", rotation=0)
    ax[1].legend(fontsize=9)
    # (c) ghost rate vs T with the merge prediction shape
    g = df_ev.groupby("T").agg(gr=("n_ghost", "sum"), nc=("n_cand", "sum")).reset_index()
    ax[2].plot(g["T"], g.gr / g.nc, "o-", color="crimson", label="measured ghost rate")
    ax[2].set(xscale="log", xlabel="T", ylabel="ghosts / candidates",
              title="(c) Ghost rate: the other face of merging\n"
                    "(grows with the same $\\pi\\lambda r_0^2$ pair probability)")
    ax[2].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig09_ghost_anatomy.png"), dpi=130)
    plt.close(fig)
    print("fig09 done")


def fig10_zoom():
    """Visual proof: a merged pair and a resolved pair at T=1000."""
    from scipy.ndimage import gaussian_filter
    sel, qp = load_clean_store_events()
    r = sel[(sel.n_trk == 1000)].iloc[0]
    ev = qp.load_event(qp.store.event_path(r.event_key))
    xs, ys, zs, planes, truth = event_arrays(ev)
    D = np.column_stack([xs / zs, ys / zs])
    cands, centres, sm = run_point(D, planes, N_BINS, SMOOTH)
    df_c, matched = eval_candidates(cands, truth)
    df_t = per_track_outcome(truth_tables(ev, D, planes, truth), matched)
    lost = df_t[(~df_t.matched) & (df_t.nn_dist < 2.5 * W)]
    res = df_t[(df_t.matched) & (df_t.nn_dist > 3 * W) & (df_t.nn_dist < 6 * W)]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for a, row, ttl in [
        (ax[0], lost.iloc[0], "(a) MERGED pair: two tracks one peak\n"
                              f"NN = {lost.iloc[0].nn_dist/W:.1f} bins -> one track lost"),
        (ax[1], res.iloc[0], "(b) RESOLVED pair: "
                             f"NN = {res.iloc[0].nn_dist/W:.1f} bins -> two peaks, both kept")]:
        cx, cy = row.cx, row.cy
        half = 8 * W
        a.imshow(sm.T, origin="lower",
                 extent=[-0.25, 0.25, -0.25, 0.25], cmap="magma",
                 interpolation="nearest")
        m = (np.abs(df_t.cx - cx) < half) & (np.abs(df_t.cy - cy) < half)
        a.scatter(df_t[m].cx, df_t[m].cy, marker="x", s=90, c="cyan",
                  label="truth track centroids")
        pm = (np.abs(centres[:, 0] - cx) < half) & (np.abs(centres[:, 1] - cy) < half)
        a.scatter(centres[pm, 0], centres[pm, 1], marker="o", s=120,
                  facecolors="none", edgecolors="lime", label="found peaks")
        mh = (np.abs(D[:, 0] - cx) < half) & (np.abs(D[:, 1] - cy) < half)
        a.scatter(D[mh, 0], D[mh, 1], s=8, c="white", alpha=0.7, label="hit votes")
        a.set(xlim=(cx - half, cx + half), ylim=(cy - half, cy + half),
              xlabel="$t_x$", ylabel="$t_y$", title=ttl)
        a.legend(fontsize=8, loc="upper right")
    fig.suptitle("T = 1000 accumulator zooms — the merge mechanism caught in the act", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_merge_zoom.png"), dpi=130)
    plt.close(fig)
    print("fig10 done")


if __name__ == "__main__":
    print(f"baseline: {N_BINS} bins, w={W:.2e}, claim radius={R_CLAIM:.2e}")
    df_t, df_c, df_ev = harvest()
    df_t.to_csv(os.path.join(OUT, "per_track.csv.gz"), index=False)
    print(f"harvest: {len(df_t)} tracks, {len(df_c)} candidates, "
          f"overall eff={df_t.matched.mean():.4f}")
    fig04_nn(df_t)
    mids, ploss, nn, popt = fit_loss_curve(df_t)
    fig05_loss_vs_nn(df_t, popt)
    fig06_eff_law(df_ev, popt)
    fig07_census(df_t, popt)
    fig08_isolated(df_t)
    fig09_ghosts(df_c, df_ev)
    fig10_zoom()
    np.save(os.path.join(OUT, "loss_fit_popt.npy"), popt)
