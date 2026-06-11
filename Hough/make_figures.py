"""Figures for the Hough prototype: (1) the accumulator + peaks for one event,
(2) efficiency / ghost vs T."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from hough_prototype import (make_geometry, safe_generate, hit_directions,
                             hough_accumulate, extract_tracks, evaluate,
                             TX_RANGE, TY_RANGE, N_BINS, run_one)
from scipy.ndimage import gaussian_filter

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ---- Figure 1: accumulator + recovered peaks for one T=50 event --------------
geom = make_geometry()
ev = safe_generate(50, 2024, geom, measurement_error=0.0, collision_noise=1e-4)
D, planes, truth = hit_directions(ev)
counts, flat = hough_accumulate(D)
cands = extract_tracks(D, planes, flat)
sm = gaussian_filter(counts.astype(float), 1.0, mode="constant")

fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
im = ax[0].imshow(sm.T, origin="lower", extent=[*TX_RANGE, *TY_RANGE],
                  aspect="auto", cmap="magma")
ax[0].scatter([c["tx"] for c in cands], [c["ty"] for c in cands],
              s=40, facecolors="none", edgecolors="cyan", linewidths=1.2,
              label=f"{len(cands)} peaks")
ax[0].set(xlabel=r"$t_x = x/z$", ylabel=r"$t_y = y/z$",
          title="Smoothed Hough accumulator (T=50)\nhits vote, tracks = peaks")
ax[0].legend(loc="upper right", framealpha=0.9)
fig.colorbar(im, ax=ax[0], label="votes (smoothed)")

# event display (z,x) with hits coloured by truth and reco lines overlaid
xs = np.array([h.x for h in ev.hits]); zs = np.array([h.z for h in ev.hits])
ax[1].scatter(zs, xs, c=truth % 20, cmap="tab20", s=14, alpha=0.7)
zline = np.array([0, 170])
for c in cands:
    ax[1].plot(zline, c["tx"] * zline, color="k", lw=0.4, alpha=0.4)
ax[1].set(xlabel="z [mm]", ylabel="x [mm]",
          title="Event (z,x): hits by truth track + reco directions")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hough_accumulator.png"), dpi=130)
print("saved hough_accumulator.png")

# ---- Figure 2: efficiency / ghost vs T (with a density comparison note) -------
Ts = [10, 25, 50, 100, 200, 400]
reps = 5
eff, gh, tms = [], [], []
for T in Ts:
    a = [run_one(T, 3000 + r, geom) for r in range(reps)]
    eff.append(np.mean([x["eff"] for x in a]))
    gh.append(np.mean([x["ghost_rate"] for x in a]))
    tms.append(np.mean([x["t_solve"] for x in a]) * 1e3)
    print(f"T={T}: eff={eff[-1]:.3f} ghost={gh[-1]:.3f} t={tms[-1]:.0f}ms")

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.6))
ax2[0].plot(Ts, eff, "o-", label="efficiency (Hough)")
ax2[0].plot(Ts, gh, "s--", color="crimson", label="ghost rate (Hough)")
ax2[0].axhline(1.0, color="gray", lw=0.5)
ax2[0].set(xlabel="n_tracks T", ylabel="rate", ylim=(0, 1.05),
           title="Hough tracking on the toy (5 planes, $\\sigma_{scatt}=10^{-4}$)")
ax2[0].legend()
ax2[1].loglog(Ts, tms, "o-")
ax2[1].set(xlabel="n_tracks T", ylabel="classical solve [ms]",
           title="Wall time (accumulate + peak-find), single core")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT, "hough_efficiency.png"), dpi=130)
print("saved hough_efficiency.png")
