"""make_error_rates_vs_size.py — error rates per false-segment category vs event
size, on the Run-3 events, at fixed epsilon.  Writes
  outputs/error_rates_per_event.csv   (per-event classical breakdown)
  outputs/error_rates_vs_size.{png,pdf}

Run from Run3_Verification/ with the toy-model + qtrk_pipeline on PYTHONPATH
(see README).  Reads outputs/quantum_vs_classical.csv (from the notebook) for the
1BQF overlay; skip it if absent.
"""
import glob, os, json, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import run3_loader as R
import qtrk_pipeline as qp

EPS, G, D = 0.005, 3.0, 1.0
TAU = qp.threshold_for(G, D); S = G + D
N = 400                                    # events (natural order -> full size range)
EVDIR = "events_bsphiphi_mag_down_run3"
files = sorted(glob.glob(EVDIR + "/velo_event_*.json"))


def build_table(n=N):
    rows = []; t0 = time.time()
    for f in files[:n]:
        evt, meta = R.load_event(f)
        ham = qp.build_hamiltonian(evt, epsilon=EPS, gamma=G, delta=D, assert_sparse=False)
        truth = np.asarray(qp.truth_from_event(evt), bool); nseg = ham.n_segments
        solC, _ = qp.solve_classical(ham); A = ham.A.tocsr()
        C = (S * sp.identity(nseg, format="csr") - A); C.setdiag(0); C.eliminate_zeros()
        C = (abs(C) > 1e-9).astype(np.int8)
        deg = np.asarray(C.sum(1)).ravel(); _, lab = connected_components(C, directed=False)
        csz = np.bincount(lab)[lab]; maxdeg = pd.Series(deg).groupby(lab).transform("max").values
        cls = np.where(truth, "TRUE", np.where(csz == 1, "iso", np.where(csz == 2, "pair",
              np.where(maxdeg >= 3, "hub", "chain"))))
        act = solC > TAU; nact = int(act.sum())
        d = dict(event=os.path.basename(f), n_hits=meta["n_hits"], n_seg=nseg,
                 n_true=int(truth.sum()), n_active=nact,
                 eff=qp.metrics_at(solC, truth, TAU)["segment_efficiency"])
        for c in ["iso", "pair", "chain", "hub"]:
            m = cls == c
            d[f"n_{c}"] = int(m.sum()); d[f"fp_{c}"] = int((m & act).sum())
            d[f"far_{c}"] = (m & act).sum() / max(nact, 1)
        d["far"] = ((~truth) & act).sum() / max(nact, 1)
        rows.append(d)
    df = pd.DataFrame(rows); df["ineff"] = 1 - df["eff"]
    df.to_csv("outputs/error_rates_per_event.csv", index=False)
    print(f"{len(df)} events in {time.time()-t0:.0f}s; n_hits [{df.n_hits.min()}-{df.n_hits.max()}]")
    return df


def binmed(x, y, nb=9):
    b = np.linspace(x.min(), x.max(), nb + 1); c = 0.5 * (b[1:] + b[:-1])
    idx = np.clip(np.digitize(x, b) - 1, 0, nb - 1)
    return c, np.array([np.median(y[idx == k]) if (idx == k).any() else np.nan for k in range(nb)])


def make_figure(df):
    try:
        q = pd.read_csv("outputs/quantum_vs_classical.csv")
        q["n_hits"] = [len(json.load(open(EVDIR + "/" + e))["x"]) for e in q["event"]]
    except Exception:
        q = None
    COL = {"iso": "#999999", "pair": "#7fb0d0", "chain": "#d6604d", "hub": "#7a0177"}
    x = df.n_hits.values
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for c in ["iso", "pair", "chain", "hub"]:
        ax[0, 0].scatter(x, df[f"n_{c}"], s=10, alpha=0.35, color=COL[c], ec="none", label=c)
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("event size  (n_hits)")
    ax[0, 0].set_ylabel("# false segments of this type")
    ax[0, 0].set_title("(a) Population of each false-segment type vs event size", fontweight="bold"); ax[0, 0].legend(fontsize=9)
    for c, lab in [("chain", "chain$\\geq$3"), ("hub", "hub")]:
        ax[0, 1].scatter(x, df[f"far_{c}"] * 100, s=9, alpha=0.3, color=COL[c], ec="none")
        cx, cy = binmed(x, df[f"far_{c}"].values * 100); ax[0, 1].plot(cx, cy, "-o", color=COL[c], lw=2, ms=4, label=lab + " (median)")
    cx, cy = binmed(x, df["far"].values * 100); ax[0, 1].plot(cx, cy, "-s", color="k", lw=2, ms=4, label="total far (median)")
    if q is not None:
        ax[0, 1].scatter(q.n_hits, q.farQ * 100, s=70, marker="D", color="#1b7837", ec="k", zorder=5, label="1BQF total far")
    ax[0, 1].set_xlabel("event size  (n_hits)"); ax[0, 1].set_ylabel("false rate contribution (%)")
    ax[0, 1].set_title("(b) Error RATE by source vs event size (classical; quantum overlay)", fontweight="bold"); ax[0, 1].legend(fontsize=8.5)
    for c, lab in [("chain", "chain$\\geq$3"), ("hub", "hub")]:
        ax[1, 0].scatter(x, df[f"fp_{c}"], s=10, alpha=0.4, color=COL[c], ec="none", label=lab)
    ax[1, 0].set_xlabel("event size  (n_hits)"); ax[1, 0].set_ylabel("# false positives of this type")
    ax[1, 0].set_title("(c) Absolute false-positive count by source vs size", fontweight="bold"); ax[1, 0].legend(fontsize=9)
    cx, cy = binmed(x, df["far"].values * 100); ax[1, 1].plot(cx, cy, "-s", color="#d6604d", lw=2, ms=5, label="false rate (FP)")
    ax[1, 1].scatter(x, df["far"] * 100, s=8, alpha=0.2, color="#d6604d", ec="none")
    cx, cy = binmed(x, df["ineff"].values * 100); ax[1, 1].plot(cx, cy, "-o", color="#2166ac", lw=2, ms=5, label="inefficiency 1$-$eff (FN)")
    ax[1, 1].scatter(x, df["ineff"] * 100, s=8, alpha=0.2, color="#2166ac", ec="none")
    if q is not None:
        ax[1, 1].scatter(q.n_hits, q.farQ * 100, s=60, marker="D", color="#d6604d", ec="k", zorder=5, label="1BQF false rate")
        ax[1, 1].scatter(q.n_hits, (1 - q.effQ) * 100, s=60, marker="o", color="#2166ac", ec="k", zorder=5, label="1BQF inefficiency")
    ax[1, 1].set_xlabel("event size  (n_hits)"); ax[1, 1].set_ylabel("error rate (%)")
    ax[1, 1].set_title("(d) The two error rates vs size: false positives & missed true", fontweight="bold"); ax[1, 1].legend(fontsize=8.5)
    for a in ax.flat:
        a.grid(alpha=0.25); a.tick_params(direction="in")
    fig.suptitle("Run-3 error rates vs event size (fixed ε=5 mrad): false positives grow with multiplicity, driven by hubs & bridges",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    for e, dp in (("pdf", 600), ("png", 300)):
        fig.savefig(f"outputs/error_rates_vs_size.{e}", dpi=dp, bbox_inches="tight", facecolor="white")
    print("saved outputs/error_rates_vs_size")


if __name__ == "__main__":
    import os
    if os.path.exists("outputs/error_rates_per_event.csv"):
        df = pd.read_csv("outputs/error_rates_per_event.csv")
        if "ineff" not in df: df["ineff"] = 1 - df["eff"]
    else:
        df = build_table()
    make_figure(df)
