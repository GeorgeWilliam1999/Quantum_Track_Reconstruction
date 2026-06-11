"""
Deep dive part 5 — fragmentation is an EVENT-level property: eff(|z_pv|).

Discovery (made while checking the part-4 sweep): the point-vote fine-grid
efficiency is NON-MONOTONE in T at 10 reps.  Cause: every track in an event
shares ONE primary vertex, and the vote smear is delta_k = -(z_pv/z_k) t --
so fragmentation strikes whole events in proportion to their |z_pv| (measured
corr(eff, |z_pv|) = -0.80 for point-1024).  The per-T averages at few reps
inherit z_pv sampling luck.

Closed form (no fit): a track fragments at grid w iff its tightest plane
triple (3,4,5; max mutual Delta(1/z) = 1/99 - 1/165 = 4.04e-3/mm) is wider
than the resolution scale:
        |t| |z_pv| * 4.04e-3 > 2.5 w   <=>   |t| > t_crit = 618.6 w / |z_pv|
so the event-level efficiency drop is  P(|t| > t_crit)  with t uniform on
[-0.2,0.2]^2 -- drawn below with zero free parameters.

The locus vote has NO z_pv term by construction -> its eff(|z_pv|) must be flat.
Figure: fig16_zpv_dependence.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hough_study_lib import (load_clean_store_events, bin_width, INVZ)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "deep_dive")
DINV_345 = INVZ[2] - INVZ[4]          # 1/99 - 1/165 = 4.04e-3 per mm


def p_t_above(c, n=400000, rng=np.random.default_rng(3)):
    """P(|t| > c) for t uniform on [-0.2, 0.2]^2 (vectorised over c)."""
    t = rng.uniform(-0.2, 0.2, size=(n, 2))
    mag = np.hypot(t[:, 0], t[:, 1])
    c = np.atleast_1d(c)
    return np.array([(mag > ci).mean() for ci in c])


def main():
    res = pd.read_csv(os.path.join(OUT, "locus_results.csv"))
    sel, qp = load_clean_store_events()
    zrows = []
    for _, r in sel[sel.rep < 10].iterrows():
        ev = qp.load_event(qp.store.event_path(r.event_key))
        zrows.append(dict(T=int(r.n_trk), rep=int(r.rep),
                          azpv=abs(ev.primary_vertices[0].z)))
    m = res.merge(pd.DataFrame(zrows), on=["T", "rep"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    zz = np.linspace(0.05, 3.0, 120)
    for a, cfg_pt, cfg_lc, nb in [(ax[0], "point 1024", "locus 1024", 1024),
                                  (ax[1], "point 256", "locus 2048", None)]:
        d = m[m.config == cfg_pt]
        h = a.scatter(d.azpv, d.eff, c=np.log10(d["T"]), cmap="viridis", s=40,
                      label=f"{cfg_pt} (events, colour = log T)")
        dl = m[m.config == cfg_lc]
        a.scatter(dl.azpv, dl.eff, marker="^", s=34, facecolors="none",
                  edgecolors="crimson", label=f"{cfg_lc} (same events)")
        if nb:
            t_crit = 2.5 * bin_width(nb) / (DINV_345 * zz)
            a.plot(zz, 1 - p_t_above(t_crit), "k-", lw=2,
                   label="closed form  $1-P(|t|>2.5w/(\\Delta_{345}|z_{pv}|))$\n"
                         "(zero fitted parameters)")
        a.set(xlabel="event $|z_{pv}|$  [mm]", ylim=(0.42, 1.03))
        a.legend(fontsize=8.5, loc="lower left")
    ax[0].set_ylabel("event track efficiency")
    ax[0].set_title("(a) Fragmentation is an event property: point-1024 follows the\n"
                    "closed-form $|z_{pv}|$ curve; the locus vote is FLAT by construction")
    ax[1].set_title("(b) point-256 is $z_{pv}$-blind (claim radius covers the smear);\n"
                    "locus-2048 keeps the flatness at 8× finer resolution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig16_zpv_dependence.png"), dpi=130)
    plt.close(fig)
    c1024 = np.corrcoef(m[m.config == "point 1024"].eff,
                        m[m.config == "point 1024"].azpv)[0, 1]
    cl1024 = np.corrcoef(m[m.config == "locus 1024"].eff,
                         m[m.config == "locus 1024"].azpv)[0, 1]
    print(f"fig16 done | corr(eff,|zpv|): point-1024 {c1024:+.2f}, "
          f"locus-1024 {cl1024:+.2f}")


if __name__ == "__main__":
    main()
