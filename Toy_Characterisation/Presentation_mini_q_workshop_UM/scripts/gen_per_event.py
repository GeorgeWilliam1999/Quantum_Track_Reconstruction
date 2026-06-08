"""
Per-event / theory figures, built fresh from the package on stored events.

Covers the reconstruction-problem and spectrum story:
  event_display_3d         3D VELO event: 5 planes, hits coloured by track
  kernel_C_theta           compatibility kernel C(theta): step vs erf
  activation_hopfield      classical activations -> 0.25/0.375 fixed points, tau=0.35
  angle_distribution       true vs false triplet kink angles, with epsilon cut
  eigenspectrum_step_erf   eigenvalues of A: step vs erf (spectral reshaping)
  kappa_cosqc_vs_thetad    kappa(A) and quantum cos_QC as theta_d sweeps step->erf

All deterministic (seed from event_key).  Quantum pieces use T=10 (fast).
"""
from __future__ import annotations
import sys; sys.path.insert(0, ".")
import _common as cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import qtrk_pipeline as qp
from helpers import compute_epsilon, collect_segment_pair_angles

EPS = compute_epsilon(0.0, 1e-4)
ZPLANES = [33.0, 66.0, 99.0, 132.0, 165.0]
HALF = 40.0


# ---------------------------------------------------------------------------
def draw_event_3d(ax, ev, T, elev=20, azim=-72, half=HALF):
    """Render an event onto a 3D axis with axes ordered (z, x, y).  Shared by
    the static figure and the rotating animation."""
    # detector planes: constant z_plane, spanning x,y in [-half, half]
    for z in ZPLANES:
        verts = [[(z, -half, -half), (z, half, -half), (z, half, half), (z, -half, half)]]
        pc = Poly3DCollection(verts, alpha=0.08, facecolor="#4477aa", edgecolor="#335577")
        ax.add_collection3d(pc)
    cmap = plt.cm.turbo
    ntr = max(ev.n_tracks, 1)
    for k, trk in enumerate(ev.tracks):
        hs = sorted(ev.get_hits_by_ids(trk.hit_ids), key=lambda h: h.z)
        zs = [h.z for h in hs]; xs = [h.x for h in hs]; ys = [h.y for h in hs]
        col = cmap(k / ntr)
        ax.plot(zs, xs, ys, "-", color=col, lw=1.5, alpha=0.9)
        ax.scatter(zs, xs, ys, color=col, s=20, depthshade=True)
    ax.set_xlabel("z (mm) — beam", labelpad=10)
    ax.set_ylabel("x (mm)", labelpad=6); ax.set_zlabel("y (mm)", labelpad=2)
    ax.set_xlim(0, 175); ax.set_ylim(-half, half); ax.set_zlim(-half, half)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((2.6, 1, 1))


def fig_event_display_3d(T=10, save="event_display_3d", elev=20, azim=-72):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    fig = plt.figure(figsize=(9.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    draw_event_3d(ax, ev, T, elev=elev, azim=azim)
    ax.set_title(f"Toy VELO event: {T} tracks through 5 planes  ($\\phi_{{\\max}}=0.2$)")
    cm.savefig(fig, save)


# ---------------------------------------------------------------------------
def fig_kernel_C_theta():
    """The ACTUAL kernels used in the code: step in {0,1}; erf = 1+erf(...) in [0,2].
    The erf peak coupling is 2x the step's -> stronger off-diagonal attraction."""
    from scipy.special import erf as _erf
    th = np.linspace(0, 2.2 * EPS, 800)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(th / EPS, (th < EPS).astype(float), color=cm.C["step"], lw=2.8,
            label=r"step:  $C=1$ if $\theta<\varepsilon$, else $0$")
    for td, a in [(2e-4, 0.95), (5e-4, 0.7), (1e-3, 0.5)]:
        C = 1.0 + _erf((EPS - th) / (td * np.sqrt(2)))
        ax.plot(th / EPS, C, lw=2.4, alpha=a, color=cm.C["erf"],
                label=fr"erf $=1+\mathrm{{erf}}(\cdot)$:  $\theta_d={td:g}$")
    ax.axvline(1.0, ls=":", c="grey"); ax.text(1.02, 0.08, r"$\theta=\varepsilon$", color="grey")
    ax.axhline(1.0, ls=":", c="grey", lw=1)
    ax.set_xlabel(r"kink angle  $\theta/\varepsilon$")
    ax.set_ylabel(r"compatibility  $C(\theta)$  =  $-A_{ij}$")
    ax.set_ylim(-0.05, 2.15)
    ax.set_title("Compatibility kernel: hard step ($\\in\\{0,1\\}$) vs smooth erf ($\\in[0,2]$)")
    ax.legend(); cm.savefig(fig, "kernel_C_theta")


# ---------------------------------------------------------------------------
def fig_activation_hopfield(T=20):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    truth = qp.truth_from_event(ev)
    ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel="step")
    solC, _ = qp.solve_classical(ham)
    x = np.asarray(solC); tr = np.asarray(truth, bool)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.7),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))
    # sorted activations
    order = np.argsort(x)
    xs = x[order]; trs = tr[order]
    idx = np.arange(len(xs))
    axL.scatter(idx[~trs], xs[~trs], s=8, color=cm.C["false"], label="false segments")
    axL.scatter(idx[trs], xs[trs], s=10, color=cm.C["true"], label="true segments")
    for y, lab, c in [(0.25, r"false attractor $\delta/(\delta+\gamma)=0.25$", "#555"),
                      (0.375, r"true plateau $\approx0.375$", cm.C["true"]),
                      (0.35, r"threshold $\tau=0.35$", cm.C["quantum"])]:
        axL.axhline(y, ls="--" if y != 0.35 else "-", color=c, lw=1.6, alpha=0.9)
        axL.text(2, y + 0.006, lab, fontsize=9, color=c)
    axL.set_xlabel("segment (sorted by activation)")
    axL.set_ylabel("activation  $x_i$")
    axL.set_title(f"Activations $\\to$ Hopfield fixed points ($T={T}$)")
    axL.legend(loc="upper left")
    # histogram
    bins = np.linspace(0, max(0.45, x.max() * 1.05), 60)
    axR.hist(x[~tr], bins=bins, color=cm.C["false"], alpha=0.8, label="false")
    axR.hist(x[tr], bins=bins, color=cm.C["true"], alpha=0.85, label="true")
    axR.axvline(0.35, color=cm.C["quantum"], lw=2, label=r"$\tau=0.35$")
    axR.set_yscale("log"); axR.set_xlabel("activation  $x_i$"); axR.set_ylabel("count")
    axR.set_title(r"Bimodal: $\tau=0.35$ separates"); axR.legend()
    fig.tight_layout()
    cm.savefig(fig, "activation_hopfield")


# ---------------------------------------------------------------------------
def fig_angle_distribution(T=50):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    true_ang, false_ang = collect_segment_pair_angles(ev)
    true_ang = np.asarray(true_ang); false_ang = np.asarray(false_ang)
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    hi = np.percentile(np.concatenate([true_ang, false_ang[false_ang < 0.05]]), 99) if len(false_ang) else 3 * EPS
    bins = np.linspace(0, max(hi, 3 * EPS), 80)
    ax.hist(false_ang, bins=bins, color=cm.C["false"], alpha=0.7, label=f"false triplets (n={len(false_ang)})")
    ax.hist(true_ang, bins=bins, color=cm.C["true"], alpha=0.8, label=f"true triplets (n={len(true_ang)})")
    ax.axvline(EPS, color=cm.C["quantum"], lw=2.2, label=fr"$\varepsilon={EPS:.2e}$ rad")
    ax.set_yscale("log"); ax.set_xlabel(r"triplet kink angle  $\theta$ (rad)")
    ax.set_ylabel("count")
    ax.set_title(f"Kink-angle separation of true vs false triplets ($T={T}$, clean)")
    ax.legend(); cm.savefig(fig, "angle_distribution")


# ---------------------------------------------------------------------------
def _spectrum(ev, kernel, erf_sigma=1e-3):
    ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel=kernel, erf_sigma=erf_sigma)
    A = ham.A.toarray().astype(float)
    A = 0.5 * (A + A.T)
    w = np.linalg.eigvalsh(A)
    return w, ham


def fig_eigenspectrum_step_erf(T=10):
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    w_s, _ = _spectrum(ev, "step")
    w_e, _ = _spectrum(ev, "erf", 1e-3)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    axL.plot(np.sort(w_s), color=cm.C["step"], lw=2, label="step")
    axL.plot(np.sort(w_e), color=cm.C["erf"], lw=2, label=r"erf ($\theta_d=10^{-3}$)")
    axL.set_xlabel("eigenvalue index (sorted)"); axL.set_ylabel(r"eigenvalue $\lambda$")
    axL.set_title("Sorted spectrum of $A$"); axL.legend()
    lo = min(w_s.min(), w_e.min()); hi = max(w_s.max(), w_e.max())
    bins = np.linspace(lo - 0.1, hi + 0.1, 70)
    axR.hist(w_s, bins=bins, color=cm.C["step"], alpha=0.6, label="step")
    axR.hist(w_e, bins=bins, color=cm.C["erf"], alpha=0.6, label="erf")
    axR.axvline(4.0, ls=":", c="grey"); axR.text(4.02, axR.get_ylim()[1]*0.6, r"$\gamma+\delta=4$", color="grey", fontsize=9, rotation=90)
    axR.set_yscale("log"); axR.set_xlabel(r"eigenvalue $\lambda$"); axR.set_ylabel("count")
    ks = w_s.max() / max(w_s[w_s > 1e-9].min(), 1e-9)
    ke = w_e.max() / max(w_e[w_e > 1e-9].min(), 1e-9)
    axR.set_title(fr"Density of eigenvalues  ($\kappa_{{\rm step}}\approx{ks:.0f}$, $\kappa_{{\rm erf}}\approx{ke:.0f}$)")
    axR.legend()
    fig.suptitle("The erf kernel reshapes the spectrum of $A$ (same classical answer, different quantum coupling)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    cm.savefig(fig, "eigenspectrum_step_erf")


# ---------------------------------------------------------------------------
def _point(ev, truth, kernel, td=1e-3):
    ham = qp.build_hamiltonian(ev, epsilon=EPS, kernel=kernel, erf_sigma=td)
    sC, _ = qp.solve_classical(ham)
    q = qp.solve_quantum(ham, device="CPU", readout="statevector")
    cos = qp.quantum_metrics(q["sol"], sC, truth, 0.35)["cos_QC"]
    A = ham.A.toarray().astype(float); w = np.linalg.eigvalsh(0.5 * (A + A.T))
    kappa = w.max() / max(w[w > 1e-9].min(), 1e-9)
    return kappa, cos


def fig_kappa_cosqc(T=10):
    """The anti-HHL point: across step + erf(theta_d), MORE spectral spread
    (larger kappa) goes with HIGHER 1BQF fidelity — opposite to HHL's kappa-cost.
    Fidelity is set by how far the true-segment eigenvalues sit from the bulk at
    gamma+delta, which the erf kernel (coupling up to 2) widens."""
    ev, _ = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=1e-4, sigma_res=0.0)
    truth = qp.truth_from_event(ev)
    tds = [1e-6, 1e-4, 5e-4, 1e-3]
    pts = [("step", None, *_point(ev, truth, "step"))]
    for td in tds:
        pts.append((f"erf {td:g}", td, *_point(ev, truth, "erf", td)))
    for lab, td, k, c in pts:
        print(f"  {lab:10s} kappa={k:6.2f} cos_QC={c:.3f}", flush=True)

    # (a) scatter cos_QC vs kappa
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    for lab, td, k, c in pts:
        col = cm.C["step"] if lab == "step" else cm.C["erf"]
        axL.scatter(k, c, s=130, color=col, zorder=3,
                    edgecolor="k", linewidth=0.6)
        axL.annotate(lab, (k, c), textcoords="offset points", xytext=(8, 4), fontsize=9)
    axL.set_xlabel(r"condition number  $\kappa(A)$")
    axL.set_ylabel(r"1BQF fidelity  $\cos\theta_{QC}$")
    axL.set_title("Larger $\\kappa$ $\\Rightarrow$ HIGHER fidelity\n(opposite to HHL intuition)")
    # (b) cos_QC + kappa vs theta_d
    erf_pts = [p for p in pts if p[0] != "step"]
    step_k, step_c = pts[0][2], pts[0][3]
    tds_e = [p[1] for p in erf_pts]; ks = [p[2] for p in erf_pts]; cs = [p[3] for p in erf_pts]
    axR.set_xscale("log")
    l1, = axR.plot(tds_e, cs, "o-", color=cm.C["quantum"], label=r"erf $\cos\theta_{QC}$")
    axR.axhline(step_c, ls="--", color=cm.C["step"], label=fr"step $\cos\theta_{{QC}}={step_c:.2f}$")
    axR.set_xlabel(r"erf width  $\theta_d$ (rad)")
    axR.set_ylabel(r"$\cos\theta_{QC}$", color=cm.C["quantum"])
    axR.set_ylim(0.3, 0.95)
    ax2 = axR.twinx(); ax2.set_xscale("log")
    ax2.plot(tds_e, ks, "s--", color=cm.C["accent"], alpha=0.8, label=r"erf $\kappa$")
    ax2.set_ylabel(r"$\kappa(A)$", color=cm.C["accent"])
    axR.set_title("Sharper erf (small $\\theta_d$): stronger coupling,\nhigher fidelity AND higher $\\kappa$")
    axR.legend(loc="center right", fontsize=9)
    fig.suptitle(f"1BQF fidelity is governed by spectral separation, not $\\kappa$  ($T={T}$, clean)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cm.savefig(fig, "kappa_cosqc")


if __name__ == "__main__":
    print("== per-event / theory figures ==")
    fig_event_display_3d()
    fig_kernel_C_theta()
    fig_activation_hopfield()
    fig_angle_distribution()
    fig_eigenspectrum_step_erf()
    fig_kappa_cosqc()
    print("done.")
