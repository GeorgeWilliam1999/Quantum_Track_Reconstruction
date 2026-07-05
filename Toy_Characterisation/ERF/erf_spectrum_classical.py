#!/usr/bin/env python3
"""Classical-first spectral characterisation of the erf scattering term.

Supervisor point (1) / todo 3925d544-b9d9-8149: "the erf term gives different
weights to candidates but also significantly alters the eigenvalue spectrum —
to be seen what exactly it does."  Method mirrors the DP study
(Bifurification/dp_spectrum_classical.py): matched events (same event_key ->
same truth), common formula epsilon per noise pair, only the kernel changes.

KERNEL LADDER (the theta_d/eps ratio is the operative knob) + one discovery
made while building this study: the erf kernel value is 1 + erf((eps-angle)/
(sqrt2*theta_d)) in (0, 2], so its theta_d -> 0 limit is a HARD edge with
DOUBLED coupling (off-diag -2), NOT the true step kernel (-1).  The store's
"step reference (theta_d=1e-6)" is therefore A = sI - 2C.  The ladder splits
the erf effect into its two ingredients:

    step    kernel="step"          coupling 1, hard edge   (Eps2 convention)
    step2x  erf theta_d=1e-6       coupling 2, hard edge   (pure doubling;
                                    eigenvectors == step, spectrum dilated 2x
                                    about s — checked exactly in-code)
    erf r   erf theta_d = r*eps    coupling <=2, soft edge, extra reach
            r in {0.1, 1/3 (kink-matched sigma_kink), 1, 3}

WHAT IS COMPUTED per (pair, kernel, T):
* T in {10, 50} — EXACT dense eigh.  Per eigenpair: solution weight
  w_i ∝ (v_i^T b / lambda_i)^2, truth content (v_i^T t_hat)^2, false content
  (v_i^T f_hat)^2.  Plus an EXACT CLASSICAL EMULATION OF THE 1BQF: one-bit QPE
  with t = pi/s post-selected on ancilla=1 multiplies eigencomponent lambda by
  e^{i*lambda*t/2} cos(lambda*t/2)  (notch exactly at lambda = s = gamma+delta,
  the unconnected-false line); x_Q = |V e^{i lam t/2}cos(lam t/2) V^T b_hat|,
  P_anc = ||.||^2 (b_hat = uniform over the padded 2^n dim, as the circuit's
  H^n prepares; padded rows sit at lambda=s and are killed by the notch).
  Validated against the stored matrix-free statevector solves where the
  ham_key exists in the store (theta_d = 1e-6 and the kink width).
* T = 200 — LANCZOS spectral measures (k=300, full reorth) of b_hat, t_hat and
  f_hat; lambda_min/max by eigsh; MINRES for the classical activation metrics.

FILTER-PASS integrals (the quantitative wp99 hook):
    F_pass(v) = sum_i cos^2(pi*lambda_i/(2s)) (v_i^T v)^2   [dense]
              = sum_j omega_j cos^2(pi*theta_j/(2s))        [Lanczos measure]
for v in {t_hat, f_hat} — the fraction of truth/false direction energy the
1BQF lets through.  L_s(v) = measure mass within |lambda - s| < 0.05 tracks
how the unconnected-false line at lambda=s dissolves as theta_d grows.

Outputs results/erf_spectrum_classical.csv +
figures/erf_spectrum_{eigdensity,comb_map,filter_pass,emulation}.png
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse.linalg import eigsh, minres

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")
import qtrk_pipeline as qp  # noqa: E402
from qtrk_pipeline.metrics import rescale_to_signal  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
RES = HERE / "results"

GAMMA, DELTA = 3.0, 1.0
S = GAMMA + DELTA                    # constant diagonal; the 1BQF notch sits here
T_1BQF = np.pi / S                   # evolution time of the one-bit QPE
TAU = qp.threshold_for(GAMMA, DELTA)
PAIRS = [(1e-4, 0.0, "clean"), (3e-4, 0.01, "moderate"), (5e-4, 0.02, "heavy")]
RATIOS = [0.1, 1.0 / 3.0, 1.0, 3.0]  # theta_d = r * eps; 1/3 = kink-matched
T_DENSE = [10, 50]
T_LANCZOS = [200]
K_LANCZOS = 300
NNZ_CAP = 5e7                        # wide-theta_d high-T guard (O(T^3) corner)
EXPLODE_MAX = 50.0                   # per-solve validity gate (Eps2 §7.12)
LINE_HW = 0.05                       # |lambda - s| < LINE_HW = "the s line"

C_W, C_T, C_F = "#2a78d6", "#33322e", "#eb6834"
KCOLORS = {"step": "#33322e", "step2x": "#79776f", "r=0.1": "#9ecae1",
           "r=1/3": "#2171b5", "r=1": "#08519c", "r=3": "#041e42"}
FOOT = (r"erf-term classical-first spectral study · matched events (rep 0), common formula "
        r"$\varepsilon$ per pair: clean 0.425 · moderate 3.397 · heavy 6.646 mrad · "
        r"$\gamma$=3 $\delta$=1, s=4 · kernels: step (coupling 1) · step2x = erf $\theta_d\to0$ "
        r"(coupling 2) · erf $\theta_d=r\varepsilon$, r$\in$\{0.1, 1/3 (kink), 1, 3\}"
        "\n"
        r"T$\in$\{10,50\} exact dense eigh + exact 1BQF emulation (one-bit QPE, t=$\pi$/s, "
        r"anc=1 $\Rightarrow$ f($\lambda$)=$e^{i\lambda t/2}\cos(\lambda t/2)$, notch at "
        r"$\lambda$=s) · T=200 k=300 Lanczos measures of $\hat b,\hat t,\hat f$ · "
        r"$F_{pass}$ = $\Sigma\,\omega\cos^2(\pi\lambda/2s)$ · gate max$|x|\leq$50")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def kernel_ladder(eps: float) -> list[tuple[str, str, float | None]]:
    """(label, kernel, erf_sigma) ladder at common eps."""
    return ([("step", "step", None), ("step2x", "erf", 1e-6)]
            + [(f"r={'1/3' if abs(r - 1/3) < 1e-9 else f'{r:g}'}", "erf", r * eps)
               for r in RATIOS])


def fpass(nodes: np.ndarray, w: np.ndarray) -> float:
    """1BQF pass fraction of a (nodes, weights) spectral measure."""
    return float(np.sum(w * np.cos(np.pi * nodes / (2 * S)) ** 2) / max(w.sum(), 1e-300))


def line_mass(nodes: np.ndarray, w: np.ndarray) -> float:
    """Measure mass on the unconnected line |lambda - s| < LINE_HW."""
    m = np.abs(nodes - S) < LINE_HW
    return float(w[m].sum() / max(w.sum(), 1e-300))


def lanczos_measure(A, v0, k=K_LANCZOS):
    """Gauss nodes/weights of the spectral measure of A wrt v0 (dp_spectrum copy)."""
    A = A.tocsr()
    n = A.shape[0]
    k = min(k, n)
    Q = np.zeros((k, n))
    alpha, beta = np.zeros(k), np.zeros(max(k - 1, 1))
    q = np.asarray(v0, float).ravel()
    q = q / np.linalg.norm(q)
    Q[0] = q
    r = A @ q
    alpha[0] = q @ r
    r = r - alpha[0] * q
    kk = k
    for j in range(1, k):
        r -= Q[:j].T @ (Q[:j] @ r)
        r -= Q[:j].T @ (Q[:j] @ r)
        b = np.linalg.norm(r)
        if b < 1e-10:
            kk = j
            break
        beta[j - 1] = b
        q = r / b
        Q[j] = q
        r = A @ q - b * Q[j - 1]
        alpha[j] = q @ r
        r = r - alpha[j] * q
    theta, U = eigh_tridiagonal(alpha[:kk], beta[:kk - 1])
    return theta, U[0] ** 2


def wp99_far(x: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """(tau_wp99, eff, far) at the 1% true-quantile working point."""
    tau = float(np.quantile(x[truth], 0.01)) - 1e-12
    act = x > tau
    n_act = int(act.sum())
    eff = float((act & truth).sum() / max(truth.sum(), 1))
    far = float((act & ~truth).sum() / max(n_act, 1))
    return tau, eff, far


def emulate_1bqf(lam: np.ndarray, V: np.ndarray, n_seg: int) -> tuple[np.ndarray, float]:
    """Exact eigen-space emulation of the 1BQF statevector readout.

    b_hat = uniform over the padded 2^n dimension (the circuit's H^n); the
    padded rows are diagonal-only (lambda = s) and are nulled by the notch, so
    only the real-dim uniform component enters — with 1/sqrt(N_pad) amplitude,
    which scales P_anc but not the (renormalised) solution shape.
    """
    n_pad = 1 << int(np.ceil(np.log2(max(n_seg, 2))))
    c = V.T @ np.full(n_seg, 1.0 / np.sqrt(n_pad))
    y = V @ (np.exp(1j * lam * T_1BQF / 2) * np.cos(lam * T_1BQF / 2) * c)
    p_anc = float(np.vdot(y, y).real)
    xq = np.abs(y)
    nrm = np.linalg.norm(xq)
    return (xq / nrm if nrm > 0 else xq), p_anc


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def stored_quantum(ekey: str, eps: float, kernel: str, td, n_seg: int):
    """Stored matrix-free statevector solve for this exact ham, if on disk."""
    hkey = qp.ham_key(eps, kernel, td if td is not None else 1e-4, GAMMA, DELTA, "formula")
    skey = qp.sol_key(ekey, hkey, "quantum", "CPU", "statevector")
    try:
        back = qp.load_solution(skey)
        return np.asarray(back["sol"], np.float64)[:n_seg], float(back.get("P_anc", np.nan))
    except Exception:
        return None, np.nan


def analyse(ev, ekey, eps, label, kernel, td, truth, dense: bool,
            pair: str = "", T: int = 0):
    """One (pair, T, kernel) point -> (row, payload); cached on disk.

    The dense eigh at T=50 costs ~100 s; payload arrays are cached under
    results/spectrum_cache so figure-only iterations are free.
    """
    import json
    cache = RES / "spectrum_cache"
    cache.mkdir(exist_ok=True)
    cfile = cache / f"{pair}_T{T}_{label.replace('/', '-')}.npz"
    if cfile.exists():
        z = np.load(cfile, allow_pickle=False)
        row = json.loads(str(z["row_json"]))
        payload = {k: z[k] for k in z.files if k != "row_json"}
        payload["row"] = row
        return row, payload

    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel=kernel, gamma=GAMMA, delta=DELTA,
                               **({"erf_sigma": td} if kernel == "erf" else {}))
    A = ham.A.tocsr()
    n = int(ham.n_segments)
    if A.nnz > NNZ_CAP:
        print(f"    [skip] {label}: nnz={A.nnz:.2e} > cap", flush=True)
        return None, None
    b = np.asarray(ham.b, float).ravel()
    t_hat = truth.astype(float)
    t_hat /= np.linalg.norm(t_hat)
    f_hat = (~truth).astype(float)
    f_hat /= np.linalg.norm(f_hat)
    t0 = time.time()
    row = dict(pair=None, T=None, label=label, kernel=kernel,
               theta_d=(td if td is not None else 0.0),
               ratio=(td / eps if (td is not None and kernel == "erf") else 0.0),
               epsilon=eps, n_seg=n, n_true=int(truth.sum()), A_nnz=int(A.nnz),
               exact=bool(dense))
    payload = dict(row=row)

    if dense:
        lam, V = eigh(A.toarray())
        c_b = V.T @ b
        u = c_b / lam
        x = V @ u
        w_sol = u ** 2 / np.sum(u ** 2)
        w_tru = (V.T @ t_hat) ** 2
        w_fal = (V.T @ f_hat) ** 2
        nodes = lam
        row.update(lam_min=float(lam[0]), lam_max=float(lam[-1]))
        # exact 1BQF emulation + store validation
        xq, p_anc = emulate_1bqf(lam, V, n)
        sol_store, p_anc_store = stored_quantum(ekey, eps, kernel, td, n)
        row.update(P_anc_emu=p_anc, P_anc_store=p_anc_store,
                   cos_emu_store=(cosine(xq, sol_store) if sol_store is not None else np.nan))
        xq_r = rescale_to_signal(xq, x, TAU)
        mq = qp.metrics_at(xq_r, truth, TAU)
        tau_wp, eff_wp, far_wp = wp99_far(xq_r, truth)
        row.update(cos_QC=cosine(xq, x),
                   q_eff_abs=mq["segment_efficiency"], q_far_abs=mq["segment_false_rate"],
                   q_tau_wp99=tau_wp, q_eff_wp99=eff_wp, q_far_wp99=far_wp)
        payload.update(lam=lam, w_sol=w_sol, w_tru=w_tru, w_fal=w_fal, xq=xq_r)
    else:
        th_b, om_b = lanczos_measure(A, b)
        th_t, om_t = lanczos_measure(A, t_hat)
        th_f, om_f = lanczos_measure(A, f_hat)
        w_sol = om_b / th_b ** 2
        w_sol /= w_sol.sum()
        nodes, w_tru, w_fal = th_t, om_t, om_f
        try:
            lam_min = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                                  maxiter=4000)[0])
            lam_max = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                                  maxiter=4000)[0])
        except Exception:
            lam_min, lam_max = float(th_b[0]), float(th_b[-1])
        row.update(lam_min=lam_min, lam_max=lam_max)
        x, _ = minres(A, b, rtol=1e-8, maxiter=8000)
        x = np.asarray(x).ravel()
        payload.update(nodes_b=th_b, w_sol=w_sol, nodes_t=th_t, w_tru=om_t,
                       nodes_f=th_f, w_fal=om_f)

    mC = qp.metrics_at(x, truth, TAU)
    nodes_t = payload.get("nodes_t", nodes)
    nodes_f = payload.get("nodes_f", nodes)
    row.update(
        c_eff_abs=mC["segment_efficiency"], c_far_abs=mC["segment_false_rate"],
        exploded=bool(np.max(np.abs(x)) > EXPLODE_MAX),
        fpass_true=fpass(nodes_t, payload["w_tru"]),
        fpass_false=fpass(nodes_f, payload["w_fal"]),
        line_mass_true=line_mass(nodes_t, payload["w_tru"]),
        line_mass_false=line_mass(nodes_f, payload["w_fal"]),
        t_spectral=time.time() - t0,
    )
    row.update(pair=pair, T=T)
    import json
    np.savez_compressed(cfile, row_json=json.dumps(row, default=float),
                        **{k: np.asarray(v) for k, v in payload.items() if k != "row"})
    return row, payload


def store_gap(df):
    """Ideal-notch vs implemented-circuit wp99 on the SAME rep-0 event.

    The emulation applies the exact e^{-iAt} one-bit QPE; the stored solves
    replay the OneBQF circuit = a per-pair product formula.  For every dense
    config whose ham_key exists in the store, recompute the rep-0 classical
    solve (cheap MINRES, no eigh) and evaluate the STORED vector at wp99 —
    the far gap to q_far_wp99 is then pure implementation (Trotter) effect.
    """
    from lhcb_velo_toy.analysis import compute_epsilon
    out = []
    for ss, sr, pair in PAIRS:
        eps = float(compute_epsilon(sr, ss))
        for T in T_DENSE:
            ev, ekey = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=ss, sigma_res=sr,
                                       phi_max=0.2, hit_ineff=0.0)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            for label, kernel, td in kernel_ladder(eps):
                sol_store, _ = stored_quantum(ekey, eps, kernel, td, len(truth))
                if sol_store is None:
                    continue
                ham = qp.build_hamiltonian(ev, epsilon=eps, kernel=kernel,
                                           gamma=GAMMA, delta=DELTA,
                                           **({"erf_sigma": td} if kernel == "erf" else {}))
                x, _ = qp.solve_classical(ham)
                xs_r = rescale_to_signal(np.asarray(sol_store, float), x, TAU)
                _, eff_ws, far_ws = wp99_far(xs_r, truth)
                out.append(dict(pair=pair, T=T, label=label,
                                q_eff_wp99_store=eff_ws, q_far_wp99_store=far_ws))
    if not out:
        return df
    df = df.merge(pd.DataFrame(out), on=["pair", "T", "label"], how="left")
    have = df[np.isfinite(df.get("q_far_wp99_store", np.nan))]
    print("\n[store-gap] ideal-notch emulation vs implemented circuit, same rep-0 event:")
    for _, r in have.iterrows():
        print(f"  {r['pair']:9} T={r['T']:<3} {r['label']:7} "
              f"far_wp99 ideal={r.q_far_wp99:.3f}  implemented={r.q_far_wp99_store:.3f}  "
              f"cos(emu,store)={r.cos_emu_store:.3f}")
    return df


def main():
    from lhcb_velo_toy.analysis import compute_epsilon
    rows, dense_p, lan_p = [], {}, {}
    for ss, sr, pair in PAIRS:
        eps = float(compute_epsilon(sr, ss))
        for T in T_DENSE + T_LANCZOS:
            dense = T in T_DENSE
            ev, ekey = qp.ensure_event(n_trk=T, rep=0, sigma_scatt=ss, sigma_res=sr,
                                       phi_max=0.2, hit_ineff=0.0)
            truth = np.asarray(qp.truth_from_event(ev), bool)
            for label, kernel, td in kernel_ladder(eps):
                if not dense and td is not None and td > 1.5 * eps:
                    continue        # O(T^3) nnz corner: r=3 stays dense-only
                got = analyse(ev, ekey, eps, label, kernel, td, truth, dense,
                              pair=pair, T=T)
                if got[0] is None:
                    continue
                row, payload = got
                rows.append(row)
                (dense_p if dense else lan_p)[(pair, T, label)] = payload
                print(f"[{pair} T={T}] {label:7} λ∈[{row['lam_min']:.3f},{row['lam_max']:.3f}] "
                      f"Fpass(t)={row['fpass_true']:.3f} Fpass(f)={row['fpass_false']:.3f} "
                      f"Ls(f)={row['line_mass_false']:.3f} "
                      + (f"cosQC={row['cos_QC']:.3f} far_wp={row['q_far_wp99']:.3f} "
                         f"cos(emu,store)={row['cos_emu_store']:.4f}" if dense else "")
                      + f" [{row['t_spectral']:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df = store_gap(df)
    df.to_csv(RES / "erf_spectrum_classical.csv", index=False)
    print(f"[csv] {len(df)} rows -> results/erf_spectrum_classical.csv", flush=True)

    labels = [l for l, _, _ in kernel_ladder(1.0)]

    # ---- fig 1: exact eigenvalue densities + the 1BQF response (T=50) -------
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    bins = np.linspace(-1.0, 9.5, 300)
    for ax, (ss, sr, pair) in zip(axes, PAIRS):
        for lab in labels:
            p = dense_p.get((pair, 50, lab))
            if p is None:
                continue
            ax.hist(np.clip(p["lam"], bins[0], bins[-1]), bins=bins, histtype="step",
                    lw=1.3, color=KCOLORS[lab], label=lab)
        ax.axvline(S, color="#e34948", lw=1.0, ls="--")
        ax.axvline(0, color="#79776f", lw=0.8, ls=":")
        ax2 = ax.twinx()
        lamg = np.linspace(bins[0], bins[-1], 400)
        ax2.plot(lamg, np.cos(np.pi * lamg / (2 * S)) ** 2, color="#e34948", lw=1.0,
                 alpha=0.55)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("1BQF $\\cos^2(\\pi\\lambda/2s)$", fontsize=7.5, color="#e34948")
        ax2.grid(False)
        ax.set_yscale("log")
        ax.set_ylabel(f"{pair}\ncount")
    axes[0].legend(fontsize=7.5, ncol=3, loc="upper left")
    axes[-1].set_xlabel("eigenvalue λ")
    fig.suptitle("Exact eigenvalue densities under the erf term (T=50, matched events) — "
                 "red dashed: the 1BQF notch λ=s; curve: the filter response", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(FIGS / "erf_spectrum_eigdensity.png")
    plt.close(fig)

    # ---- fig 2: spectral weight maps at T=200 (Lanczos) ---------------------
    show = [l for l in labels if l != "r=3"]
    fig, axes = plt.subplots(3, len(show), figsize=(3.1 * len(show), 8.6),
                             sharex=True, sharey=True)
    for i, (ss, sr, pair) in enumerate(PAIRS):
        for j, lab in enumerate(show):
            ax = axes[i, j]
            p = lan_p.get((pair, 200, lab))
            if p is None:
                ax.set_axis_off()
                continue
            ax.axvline(S, color="#e34948", lw=0.8, ls="--")
            ax.axvline(0, color="#79776f", lw=0.6, ls=":")
            m, s_, _ = ax.stem(p["nodes_b"], np.maximum(p["w_sol"], 1e-12), basefmt=" ")
            plt.setp(m, markersize=2.2, color=C_W)
            plt.setp(s_, linewidth=0.6, color=C_W, alpha=0.55)
            ax.scatter(p["nodes_t"], np.maximum(p["w_tru"], 1e-12), s=7, color=C_T,
                       alpha=0.75, marker="^", zorder=3)
            ax.scatter(p["nodes_f"], np.maximum(p["w_fal"], 1e-12), s=7, color=C_F,
                       alpha=0.75, marker="v", zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(1e-9, 1)
            r = p["row"]
            ax.set_title(f"{lab} · Fp(f)={r['fpass_false']:.2f} Ls(f)={r['line_mass_false']:.2f}",
                         fontsize=7.5, loc="left")
            if j == 0:
                ax.set_ylabel(f"{pair}\nweight (norm.)")
    axes[0, 0].plot([], [], color=C_W, marker="o", ms=3, lw=0.7, label="solution weight")
    axes[0, 0].plot([], [], color=C_T, marker="^", ms=4, lw=0, label="truth measure")
    axes[0, 0].plot([], [], color=C_F, marker="v", ms=4, lw=0, label="false measure")
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("λ (Lanczos node)")
    fig.suptitle("Where solution / truth / FALSE live spectrally, step → wide erf "
                 "(T=200, k=300 Lanczos; red dashed = 1BQF notch λ=s)", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(FIGS / "erf_spectrum_comb_map.png")
    plt.close(fig)

    # ---- fig 3: THE hook — filter pass fractions vs theta_d/eps -------------
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True)
    x_of = {"step": 3e-3, "step2x": 6e-3}      # hard-edge refs pinned left of the ratio axis
    F_FLOOR = 1e-6
    for j, (ss, sr, pair) in enumerate(PAIRS):
        ax_t, ax_f = axes[0, j], axes[1, j]
        for T, mk, ls in ((10, "s", ":"), (50, "o", "-"), (200, "D", "--")):
            sub = df[(df.pair == pair) & (df["T"] == T)]
            if not len(sub):
                continue
            xs = np.asarray([x_of.get(r.label, r.ratio) for _, r in sub.iterrows()])
            o = np.argsort(xs)
            xs = xs[o]
            ax_t.plot(xs, sub.fpass_true.values[o], ls, marker=mk, ms=4, lw=1.4,
                      color=C_T, label=f"T={T}")
            ax_f.plot(xs, np.clip(sub.fpass_false.values[o], F_FLOOR, None), ls,
                      marker=mk, ms=4, lw=1.4, color=C_F)
        for ax in (ax_t, ax_f):
            ax.axvline(1 / 3, color="#2171b5", lw=0.8, ls=":")
            ax.set_xscale("log")
        ax_t.text(1 / 3, 1.02, " kink", fontsize=7, color="#2171b5")
        ax_t.set_ylim(0, 1.05)
        ax_f.set_yscale("log")
        ax_f.set_ylim(F_FLOOR / 2, 5e-2)
        ax_t.set_title(pair, fontsize=9.5)
        ax_f.set_xlabel(r"$\theta_d/\varepsilon$   (step / step2x pinned at left)")
    axes[0, 0].set_ylabel(r"$F_{pass}$(truth)")
    axes[1, 0].set_ylabel(r"$F_{pass}$(false)  [log]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("What the 1BQF lets through vs kernel width: truth passband gain (top, the "
                 "doubling) vs FALSE leakage (bottom, the wp99 reversal)", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(FIGS / "erf_spectrum_filter_pass.png")
    plt.close(fig)

    # ---- fig 4: emulation vs store + the T=10 fidelity decomposition --------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    d10 = df[(df["T"] == 10) & df.exact]
    d50 = df[(df["T"] == 50) & df.exact]
    xpos = np.arange(len(labels))
    for ax, dd, ttl in ((axes[0], d10, "T=10"), (axes[1], d50, "T=50")):
        for k, (ss, sr, pair) in enumerate(PAIRS):
            sub = dd[dd.pair == pair].set_index("label").reindex(labels)
            ax.plot(xpos + (k - 1) * 0.12, sub.cos_QC, "o-", ms=4, lw=1.2,
                    color={"clean": "#9ecae1", "moderate": "#4292c6",
                           "heavy": "#08306b"}[pair], label=pair)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, fontsize=7.5)
        ax.set_ylabel("cos(x$_{1BQF}$, x$_C$)")
        ax.set_title(f"1BQF↔classical fidelity by kernel — {ttl}", fontsize=9)
    axes[0].legend(fontsize=7)
    ax = axes[2]
    v = df[df.exact & np.isfinite(df.cos_emu_store)]
    ax.scatter(v.P_anc_store, v.P_anc_emu, s=18, c="#2a78d6")
    lims = [0, max(1e-3, float(np.nanmax([v.P_anc_store.max(), v.P_anc_emu.max()])) * 1.15)]
    ax.plot(lims, lims, color="#c3c2b7", lw=0.8, ls=":")
    ax.set_xlabel("P_anc (stored matrix-free solve)")
    ax.set_ylabel("P_anc (eigen-space emulation)")
    ax.set_title(f"emulation validation: min cos(emu, store) = "
                 f"{float(v.cos_emu_store.min()):.5f}  (n={len(v)})", fontsize=9)
    fig.suptitle("The 1BQF from the eigendecomposition alone — fidelity decomposition "
                 "(step → step2x isolates the coupling-doubling) + store validation", y=0.98)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.text(0.01, 0.004, FOOT, fontsize=6, color="#52514e", va="bottom")
    fig.savefig(FIGS / "erf_spectrum_emulation.png")
    plt.close(fig)
    print("[figs] -> figures/erf_spectrum_{eigdensity,comb_map,filter_pass,emulation}.png",
          flush=True)


if __name__ == "__main__":
    main()
