"""
Four presentation animations (GIF, embeddable via raw GitHub URL in Notion):

  event_3d.gif              rotating 3D toy VELO event
  hopfield_relaxation.gif   classical activations relaxing onto 0.25/0.375 fixed points
  spectrum_step_to_erf.gif  eigenspectrum of A morphing as theta_d sweeps step->erf
  hhl_vs_1bqf_filter.gif     the 1BQF single-bit filter notch sliding onto the bulk

All built fresh from the package / analytic single-bit response.
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import qtrk_pipeline as qp
from helpers import compute_epsilon
from gen_per_event import draw_event_3d, ZPLANES

EPS = compute_epsilon(0.0, 1e-4)


def _save(anim, name, fps=15):
    path = cm.ANIM / (name if name.endswith(".gif") else name + ".gif")
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close("all")
    print(f"  saved {path.relative_to(cm.PRES)}")


# ---------------------------------------------------------------------------
def anim_event_3d(T=10, nframes=72):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    fig = plt.figure(figsize=(8.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_event_3d(ax, ev, T)
    ax.set_title(f"Toy VELO event — {T} tracks, 5 planes")

    def update(i):
        ax.view_init(elev=18 + 6 * np.sin(2 * np.pi * i / nframes),
                     azim=-90 + 360 * i / nframes)
        return ()
    anim = FuncAnimation(fig, update, frames=nframes, interval=70, blit=False)
    _save(anim, "event_3d", fps=15)


# ---------------------------------------------------------------------------
def anim_hopfield(T=20, nframes=70):
    """Gradient flow x_{k+1}=x_k-eta(Ax_k-b) -> A^{-1}b, the Hopfield relaxation."""
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    truth = np.asarray(qp.truth_from_event(ev), bool)
    ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step")
    A = ham.A.tocsr(); b = np.asarray(ham.b, float)
    n = A.shape[0]
    lam_max = 4.0 + 2.0          # (gamma+delta) + max coupling spread
    eta = 1.0 / lam_max
    x = np.zeros(n)
    traj = [x.copy()]
    for _ in range(nframes - 1):
        x = x - eta * (A @ x - b)
        traj.append(x.copy())

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    idx = np.arange(n)
    for y, lab, c in [(0.25, "false attractor 0.25", "#555"),
                      (0.375, "true plateau ~0.375", cm.C["true"]),
                      (0.35, r"$\tau=0.35$", cm.C["quantum"])]:
        ax.axhline(y, ls="--" if y != 0.35 else "-", color=c, lw=1.5, alpha=0.85)
    scF = ax.scatter([], [], s=8, color=cm.C["false"], label="false segments")
    scT = ax.scatter([], [], s=12, color=cm.C["true"], label="true segments")
    ax.set_xlim(-n*0.02, n*1.02); ax.set_ylim(-0.02, 0.48)
    ax.set_xlabel("segment (fixed order, false then true)")
    ax.set_ylabel("activation  $x_i$")
    ax.legend(loc="center right")
    title = ax.set_title("")
    # fixed plotting order: false first, then true
    order = np.concatenate([idx[~truth], idx[truth]])
    posF = np.where(~truth[order])[0]; posT = np.where(truth[order])[0]

    def update(k):
        xv = traj[k][order]
        scF.set_offsets(np.c_[posF, xv[posF]])
        scT.set_offsets(np.c_[posT, xv[posT]])
        title.set_text(f"Hopfield relaxation: iteration {k}/{nframes-1}  "
                       f"($T={T}$, classical gradient flow to $A^{{-1}}b$)")
        return scF, scT, title
    anim = FuncAnimation(fig, update, frames=nframes, interval=80, blit=False)
    _save(anim, "hopfield_relaxation", fps=14)


# ---------------------------------------------------------------------------
def anim_spectrum_step_to_erf(T=10, nframes=48):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    tds = np.logspace(-6, np.log10(3e-3), nframes)
    specs, kappas = [], []
    for td in tds:
        ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="erf", erf_sigma=float(td))
        A = ham.A.toarray().astype(float); A = 0.5 * (A + A.T)
        w = np.sort(np.linalg.eigvalsh(A))
        specs.append(w)
        wp = w[w > 1e-9]; kappas.append(w.max() / wp.min())
    specs = np.array(specs)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.set_xlim(0, len(specs[0])); ax.set_ylim(specs.min() - 0.3, specs.max() + 0.3)
    ax.axhline(4.0, ls=":", color="grey"); ax.text(2, 4.05, r"$\gamma+\delta=4$", color="grey", fontsize=9)
    line, = ax.plot([], [], color=cm.C["erf"], lw=2.2)
    ax.set_xlabel("eigenvalue index (sorted)"); ax.set_ylabel(r"eigenvalue $\lambda$")
    title = ax.set_title("")

    def update(i):
        line.set_data(np.arange(len(specs[i])), specs[i])
        title.set_text(fr"Spectrum of $A$ vs erf width:  $\theta_d={tds[i]:.1e}$,  "
                       fr"$\kappa={kappas[i]:.1f}$   (step $\to$ wide erf)")
        return line, title
    anim = FuncAnimation(fig, update, frames=nframes, interval=110, blit=False)
    _save(anim, "spectrum_step_to_erf", fps=10)


# ---------------------------------------------------------------------------
def anim_hhl_vs_1bqf_filter(nframes=60):
    """Slide the 1BQF evolution time t; the single-bit notch cos^2(lambda t/2)
    slides until t=pi/(gamma+delta) lands it exactly on the false-segment bulk."""
    gd = 4.0
    lam = np.linspace(0.3, 8, 500)
    true_band = [1.0, 1.6, 6.4, 7.1]
    t_target = np.pi / gd
    ts = np.linspace(0.25 * t_target, 1.55 * t_target, nframes)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axvline(gd, color="grey", lw=9, alpha=0.25)
    ax.text(gd, 1.07, r"bulk $\lambda=\gamma+\delta$", color="grey", ha="center", fontsize=9)
    for lc in true_band:
        ax.axvline(lc, color=cm.C["true"], lw=2, alpha=0.7)
    hhl = (1.0 / lam); hhl /= hhl.max()
    ax.plot(lam, hhl, color=cm.C["classical"], lw=2, alpha=0.8, label=r"HHL $\propto1/\lambda$")
    fline, = ax.plot([], [], color=cm.C["quantum"], lw=2.6, label="1BQF single-bit filter")
    vmark = ax.axvline(np.nan, color=cm.C["quantum"], ls=":", lw=1.5)
    ax.set_xlabel(r"eigenvalue $\lambda$ of $A$"); ax.set_ylabel("filter amplitude")
    ax.set_ylim(0, 1.15); ax.legend(loc="upper right")
    title = ax.set_title("")

    def update(i):
        t = ts[i]
        acc = np.cos(lam * t / 2.0) ** 2
        fline.set_data(lam, acc)
        notch = np.pi / t           # first lambda where cos^2=0
        vmark.set_xdata([notch, notch])
        locked = abs(t - t_target) < (ts[1] - ts[0])
        title.set_text(("LOCKED: $t=\\pi/(\\gamma+\\delta)$ — notch on the bulk, true band kept"
                        if locked else fr"sliding the 1BQF notch:  $t={t:.2f}$  (target $\pi/(\gamma+\delta)$)"))
        title.set_color(cm.C["quantum"] if locked else "k")
        return fline, vmark, title
    anim = FuncAnimation(fig, update, frames=nframes, interval=90, blit=False)
    _save(anim, "hhl_vs_1bqf_filter", fps=12)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["all"])
    a = ap.parse_args()
    w = set(a.which)
    print("== animations ==")
    if w & {"all", "filter"}:   anim_hhl_vs_1bqf_filter()
    if w & {"all", "spectrum"}: anim_spectrum_step_to_erf()
    if w & {"all", "hopfield"}: anim_hopfield()
    if w & {"all", "event"}:    anim_event_3d()
    print("done.")
