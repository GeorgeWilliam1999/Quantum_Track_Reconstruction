"""
05 -- resource vs accuracy trade-off on the BASE Hamiltonians (no fork, no
occupancy term): can a block encoding buy circuit depth or qubit count at a
measurably small cost in segment efficiency / false rate?

Knobs measured (all matrix-free through the standard store harness, wp99 +
fixed-tau metrics, reps for error bars):

  A. FABLE-family compression: rotations kept -> encoding error -> eff/far
     (isolates the accuracy axis from the alpha wall, which is reported apart).
  B. Polynomial degree per encoding domain (A-dilation / C-alpha / normalized
     walk): the metric-validated minimum degree, usually far below the
     L_inf-0.02 fit criterion of 03.
  C. Fixed-point precision of the geometry data: hit coordinates quantized to
     b bits -> rebuilt step coupling -> eff/far vs b (the qubit knob of the
     hit-level oracle: arithmetic registers scale with b).
  D. Coupling truncation: erf value cut (rescues the dictionary alpha) and
     step per-hit degree cap (window w) -> eff/far vs alpha/nnz.
  E. Qubit + depth accounting per method at the operating points B validates.

Outputs: outputs/05_{fable_pareto,degree_curves,precision_bits,truncation,
qubit_depth_table}.csv + figures fig05_*.png
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared")
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as cheb

import be_lib as bl
import qtrk_pipeline as qp
from lhcb_velo_toy.solvers.quantum.QSVT import design_line_comb_inverse

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

GAMMA, DELTA, EPS = 3.0, 1.0, 0.002
S = GAMMA + DELTA
TAU = qp.threshold_for(gamma=GAMMA, delta=DELTA)
P4_NORM = (0.5, -0.5)
HW_NORM = 0.10

MCOL = dict(fable="#2a78d6", sfable="#008300", lsfable="#e87ba4",
            camps="#eda100", dictionary="#1baf7a", szegedy="#eb6834",
            dilation="#4a3aa7", hit_oracle="#e34948", classical="#52514e")


# ----------------------------------------------------------------------
# shared plumbing
# ----------------------------------------------------------------------

_cache = {}


def get_event(T, rep):
    key = ("ev", T, rep)
    if key not in _cache:
        _cache[key] = qp.ensure_event(n_trk=T, rep=rep, sigma_scatt=1e-4,
                                      sigma_res=0.0, phi_max=0.2, hit_ineff=0.0)[0]
    return _cache[key]


def get_ham(T, rep, kernel):
    key = ("ham", T, rep, kernel)
    if key not in _cache:
        ev = get_event(T, rep)
        kw = dict(epsilon=EPS, gamma=GAMMA, delta=DELTA, kernel=kernel)
        if kernel == "erf":
            kw["erf_sigma"] = EPS / 3.0
        _cache[key] = qp.build_hamiltonian(ev, **kw)
    return _cache[key]


def base_context(T, rep, kernel):
    """A, C, truth, classical solve, spectral bounds -- computed once."""
    key = ("ctx", T, rep, kernel)
    if key not in _cache:
        ham = get_ham(T, rep, kernel)
        A = ham.A.tocsr()
        ev = get_event(T, rep)
        truth = np.asarray(qp.truth_from_event(ev), bool)
        sol_C, _ = qp.solve_classical(ham)
        C = sp.csr_matrix(abs(A - sp.diags(A.diagonal())))
        C.eliminate_zeros()
        if A.shape[0] <= 3000:
            w = np.linalg.eigvalsh(A.toarray())
            lo, hi = float(w[0]), float(w[-1])
        else:
            from scipy.sparse.linalg import eigsh
            hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False,
                             maxiter=5000)[0])
            lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False,
                             maxiter=5000)[0])
        _cache[key] = dict(ham=ham, A=A, C=C, truth=truth, sol_C=sol_C,
                           bounds=(lo, hi))
    return _cache[key]


def cheb_apply(p, M, v):
    lo, hi = p.domain
    sc = 2.0 / (hi - lo)
    sh = (lo + hi) / (hi - lo)
    X = sc * M - sh * sp.identity(M.shape[0], format="csr")
    c = p.coef
    t_prev = v
    y = c[0] * t_prev
    if len(c) > 1:
        t_cur = X @ v
        y = y + c[1] * t_cur
        for k in range(2, len(c)):
            t_next = 2.0 * (X @ t_cur) - t_prev
            t_prev, t_cur = t_cur, t_next
            y = y + c[k] * t_cur
    return y


def discriminant(C):
    deg = np.asarray(C.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    return sp.csr_matrix(sp.diags(inv) @ C @ sp.diags(inv))


def norm_comb(degree, hw=HW_NORM):
    x = np.linspace(-1, 1, 8000)
    y = np.zeros_like(x)
    for m in P4_NORM:
        y = np.maximum(y, np.exp(-(((x - m) / hw) ** 2)))
    p = cheb.Chebyshev.fit(x, y, degree, domain=[-1, 1])
    mx = float(np.max(np.abs(p(x))))
    return p / (mx / 0.95) if mx > 1.0 else p


def measure(y, sol_C, truth):
    """wp99 + fixed-tau metrics of a filtered vector."""
    sol = np.abs(np.asarray(y, float))
    nrm = np.linalg.norm(sol)
    sol = sol / nrm if nrm > 0 else sol
    mW = qp.quantum_metrics_wp(sol, sol_C, truth)
    mF = qp.quantum_metrics(sol, sol_C, truth, TAU)
    return dict(eff_wp=mW.get("segment_efficiency"),
                far_wp=mW.get("segment_false_rate"),
                tau_wp=mW.get("tau_wp"),
                eff_fix=mF.get("segment_efficiency"),
                far_fix=mF.get("segment_false_rate"))


def solve_domain(domain, degree, ctx):
    """Apply the comb of `degree` on the given encoding domain; return y."""
    A, C = ctx["A"], ctx["C"]
    d = A.shape[0]
    b = np.ones(d) / np.sqrt(d)
    lo, hi = ctx["bounds"]
    if domain == "A_dilation":
        span = hi - lo
        dom = (min(lo - 0.02 * span, 0.2), max(hi + 0.02 * span, 7.8))
        p = design_line_comb_inverse(degree=degree, s=S, domain=dom)
        return cheb_apply(p, A, b)
    if domain == "C_alpha":
        # alpha = 2^ceil(log2 Delta); the SAME 1/lambda comb response mapped
        # onto x = (s - lambda)/alpha, fitted at `degree` on [-1, 1]
        deg_arr = np.asarray((C != 0).sum(1)).ravel()
        Delta = max(1, int(deg_arr.max(initial=1)))
        alpha = 1 << int(np.ceil(np.log2(max(2, Delta))))
        span = hi - lo
        dom = (min(lo - 0.02 * span, 0.2), max(hi + 0.02 * span, 7.8))
        pA = design_line_comb_inverse(degree=200, s=S, domain=dom)  # dense ref
        x = np.linspace(-1, 1, 8000)
        lam = S - alpha * x                     # x = (s - lambda)/alpha
        y_t = np.where((lam >= dom[0]) & (lam <= dom[1]), pA(np.clip(lam, *dom)), 0.0)
        p = cheb.Chebyshev.fit(x, y_t, degree, domain=[-1, 1])
        mx = float(np.max(np.abs(p(x))))
        if mx > 1.0:
            p = p / (mx / 0.95)
        X = (S * sp.identity(C.shape[0], format="csr") - ctx["A"])  # = C signed
        return cheb_apply(p, sp.csr_matrix(X) / alpha, b), alpha
    if domain == "normalized":
        p = norm_comb(degree)
        return cheb_apply(p, discriminant(C), b)
    raise ValueError(domain)


# ----------------------------------------------------------------------
# A. FABLE-family compression -> eff/far  (T=20, reps, both kernels)
# ----------------------------------------------------------------------

def part_A(T=20, reps=(0, 1, 2)):
    rows = []
    thrs = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    for kernel in ("step", "erf"):
        for rep in reps:
            ctx = base_context(T, rep, kernel)
            A = ctx["A"]
            d = A.shape[0]
            Ad = A.toarray()
            nrm = np.linalg.norm(Ad)
            b = np.ones(d) / np.sqrt(d)
            lo, hi = ctx["bounds"]
            span = hi - lo
            dom = (min(lo - 0.02 * span, 0.2), max(hi + 0.02 * span, 7.8))
            p40 = design_line_comb_inverse(degree=40, s=S, domain=dom)

            for enc, fn in (("sfable", bl.sfable_classical),
                            ("fable", bl.fable_classical)):
                for thr in thrs:
                    Ahat, kept = fn(Ad, thr)
                    err = float(np.linalg.norm(Ahat - Ad) / nrm)
                    # accuracy axis in isolation: production comb applied to
                    # Ahat as if ideally rescaled (the alpha wall is separate)
                    y = cheb_apply(p40, sp.csr_matrix(np.clip(Ahat, None, None)), b)
                    m = measure(y, ctx["sol_C"], ctx["truth"])
                    rows.append(dict(T=T, rep=rep, kernel=kernel, enc=enc,
                                     thr=thr, rot=kept, enc_err=err, **m))
            # LS-FABLE fixed point
            Ahat, kept = bl.lsfable_classical(Ad)
            err = float(np.linalg.norm(Ahat - Ad) / nrm)
            y = cheb_apply(p40, sp.csr_matrix(Ahat), b)
            m = measure(y, ctx["sol_C"], ctx["truth"])
            rows.append(dict(T=T, rep=rep, kernel=kernel, enc="lsfable",
                             thr=np.nan, rot=kept, enc_err=err, **m))
            print(f"A: {kernel} rep{rep} done (lsfable err {err:.3f} "
                  f"far_wp {m['far_wp']:.3f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "05_fable_pareto.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# B. degree curves per encoding domain (the depth knob)
# ----------------------------------------------------------------------

def part_B():
    degrees = [8, 12, 16, 20, 24, 28, 32, 40, 54, 64, 78, 100]
    rows = []
    points = [(200, (0, 1, 2), "step"), (400, (0,), "step"), (200, (0, 1, 2), "erf")]
    for T, reps, kernel in points:
        for rep in reps:
            ctx = base_context(T, rep, kernel)
            for dom in ("A_dilation", "C_alpha", "normalized"):
                for dg in degrees:
                    out = solve_domain(dom, dg, ctx)
                    alpha = np.nan
                    if isinstance(out, tuple):
                        y, alpha = out
                    else:
                        y = out
                    m = measure(y, ctx["sol_C"], ctx["truth"])
                    rows.append(dict(T=T, rep=rep, kernel=kernel, domain=dom,
                                     degree=dg, alpha=alpha, **m))
                print(f"B: T={T} rep{rep} {kernel} {dom} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "05_degree_curves.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# C. fixed-point precision of the geometry data (the qubit knob)
# ----------------------------------------------------------------------

def quantized_step_C(T, rep, bits):
    """Rebuild the step coupling from b-bit quantized hit coordinates."""
    ev = get_event(T, rep)
    ham = get_ham(T, rep, "step")
    hx = np.array([h.x for h in ev.hits])
    hy = np.array([h.y for h in ev.hits])
    hz = np.array([h.z for h in ev.hits])
    if np.isfinite(bits):
        for arr in (hx, hy):
            lo, hi = arr.min(), arr.max()
            stp = (hi - lo) / (2 ** bits - 1)
            arr[:] = lo + np.round((arr - lo) / stp) * stp
    s2h = np.asarray(ham._segment_to_hit_ids)
    d = s2h.shape[0]
    a_idx, b_idx = s2h[:, 0], s2h[:, 1]
    vec = np.stack([hx[b_idx] - hx[a_idx], hy[b_idx] - hy[a_idx],
                    hz[b_idx] - hz[a_idx]], axis=1)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    # group segments by middle hit: seg i ends at hit h, seg j starts at h
    from collections import defaultdict
    ends = defaultdict(list)
    starts = defaultdict(list)
    for i in range(d):
        ends[int(b_idx[i])].append(i)
        starts[int(a_idx[i])].append(i)
    ii, jj = [], []
    for h in ends:
        if h not in starts:
            continue
        I = np.array(ends[h])
        J = np.array(starts[h])
        cosang = np.clip(vec[I] @ vec[J].T, -1, 1)
        ang = np.arccos(cosang)
        w = np.nonzero(ang < EPS)
        ii.extend(I[w[0]].tolist())
        jj.extend(J[w[1]].tolist())
    Cq = sp.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(d, d))
    Cq = sp.csr_matrix(Cq + Cq.T)
    Cq.data[:] = 1.0
    return Cq


def part_C(reps=(0, 1, 2)):
    rows = []
    bit_grid = [4, 6, 8, 10, 12, 16, np.inf]
    for T in (100, 200):
        use_reps = reps if T == 100 else (0,)
        for rep in use_reps:
            ctx = base_context(T, rep, "step")
            for bits in bit_grid:
                Cq = quantized_step_C(T, rep, bits)
                nnz_ref = ctx["C"].nnz
                sym_diff = int(abs((Cq - ctx["C"])).nnz)
                Aq = sp.csr_matrix(S * sp.identity(Cq.shape[0]) - Cq)
                # normalized-walk comb d=40 on the quantized coupling
                y = cheb_apply(norm_comb(40), discriminant(Cq),
                               np.ones(Cq.shape[0]) / np.sqrt(Cq.shape[0]))
                m = measure(y, ctx["sol_C"], ctx["truth"])
                # classical solve on the same quantized matrix (fair reference)
                from scipy.sparse.linalg import minres
                bq = np.ones(Aq.shape[0])
                try:
                    xq, _ = minres(Aq, bq, rtol=1e-8, maxiter=5000)
                except TypeError:
                    xq, _ = minres(Aq, bq, tol=1e-8, maxiter=5000)
                mc = qp.metrics_at(xq, ctx["truth"], threshold=TAU)
                rows.append(dict(T=T, rep=rep, bits=(bits if np.isfinite(bits) else 64),
                                 edges_changed=sym_diff, nnz_ref=nnz_ref, **m,
                                 eff_classical=mc.get("segment_efficiency"),
                                 far_classical=mc.get("segment_false_rate")))
                print(f"C: T={T} rep{rep} bits={bits}: edges±{sym_diff} "
                      f"far_wp={m['far_wp']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "05_precision_bits.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# D. coupling truncation (erf value cut; step per-middle-hit cap)
# ----------------------------------------------------------------------

def part_D(T=200, reps=(0, 1, 2)):
    rows = []
    cuts = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    for rep in reps:
        ctx = base_context(T, rep, "erf")
        C = ctx["C"]
        for cut in cuts:
            Ct = C.copy()
            Ct.data = np.where(Ct.data >= cut, Ct.data, 0.0)
            Ct.eliminate_zeros()
            alpha_dict = float(Ct.data.sum() / 2.0)
            y = cheb_apply(norm_comb(40), discriminant(Ct),
                           np.ones(C.shape[0]) / np.sqrt(C.shape[0]))
            m = measure(y, ctx["sol_C"], ctx["truth"])
            # classical on truncated A as reference
            At = sp.csr_matrix(S * sp.identity(C.shape[0]) - Ct)
            from scipy.sparse.linalg import minres
            try:
                xt, _ = minres(At, np.ones(C.shape[0]), rtol=1e-8, maxiter=5000)
            except TypeError:
                xt, _ = minres(At, np.ones(C.shape[0]), tol=1e-8, maxiter=5000)
            mc = qp.metrics_at(xt, ctx["truth"], threshold=TAU)
            rows.append(dict(T=T, rep=rep, knob="erf_value_cut", cut=cut,
                             nnz=int(Ct.nnz), alpha_dict=alpha_dict, **m,
                             eff_classical=mc.get("segment_efficiency"),
                             far_classical=mc.get("segment_false_rate")))
            print(f"D: rep{rep} erf cut={cut}: nnz={Ct.nnz} alpha={alpha_dict:.0f} "
                  f"far_wp={m['far_wp']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "05_truncation.csv"), index=False)
    return df


# ----------------------------------------------------------------------
# E. qubit + depth accounting at metric-validated operating points
# ----------------------------------------------------------------------

def part_E(dfb):
    """Operating point = min degree with mean wp99 eff>=0.99 and far<=0.01."""
    rows = []
    for T in (200, 400):
        kernel = "step"
        sub = dfb[(dfb["T"] == T) & (dfb.kernel == kernel)]
        if not len(sub):
            continue
        g = (sub.groupby(["domain", "degree"])[["eff_wp", "far_wp"]]
             .mean().reset_index())
        n_sys = int(np.ceil(np.log2(4 * T * T)))
        nnz = base_context(T, 0, kernel)["C"].nnz
        dstar = {}
        for dom in ("A_dilation", "C_alpha", "normalized"):
            gg = g[(g.domain == dom) & (g.eff_wp >= 0.99) & (g.far_wp <= 0.01)]
            dstar[dom] = int(gg.degree.min()) if len(gg) else None
        cfgs = [
            ("native 1BQF (reference)", 1, None, nnz // 2 * (2 * n_sys + 6 * max(0, 12 * n_sys - 36)), "exp(-iAt), 1 call"),
            ("dilation (simulation-only)", 1, dstar["A_dilation"], np.nan, "dense 4^n gate"),
            ("camps_C / dictionary_C", 4, dstar["C_alpha"],
             bl.model_transposition_oracle(nnz // 2, n_sys)["cx"], "transposition-compiled"),
            ("hit_oracle (C/alpha)", 45, dstar["C_alpha"],
             bl.model_hit_oracle(T, n_sys, w_pad=4)["cx"], "QROM over 5T hits"),
            ("szegedy walk (hit-prep)", n_sys + 45, dstar["normalized"],
             2 * bl.model_hit_oracle(T, n_sys, w_pad=4)["cx"] + n_sys, "2x O_P + swap"),
        ]
        for name, anc, dg, cx_call, note in cfgs:
            lcu = int(np.ceil(np.log2(dg + 1))) if dg else 0
            rows.append(dict(T=T, method=name, n_sys=n_sys, ancillas=anc,
                             lcu_qubits=lcu,
                             qubits_total=n_sys + anc + lcu,
                             degree_star=dg, cx_per_call=cx_call,
                             cx_per_solve=(dg * cx_call if dg and np.isfinite(cx_call) else np.nan),
                             note=note))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "05_qubit_depth_table.csv"), index=False)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def figures(dfa, dfb, dfc, dfd):
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.grid": True, "grid.color": "#e8e7e2",
                         "grid.linewidth": 0.6, "font.size": 11})

    # degree curves (the headline depth knob)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharex=True)
    sub = dfb[(dfb["T"] == 200) & (dfb.kernel == "step")]
    labels = {"A_dilation": ("dilation comb (A)", MCOL["dilation"]),
              "C_alpha": ("C/α comb (camps/dict/hit)", MCOL["hit_oracle"]),
              "normalized": ("normalized ±½ comb (α=1)", MCOL["szegedy"])}
    for dom, (lab, col) in labels.items():
        g = sub[sub.domain == dom].groupby("degree")[["eff_wp", "far_wp"]]
        mean, std = g.mean(), g.std()
        axes[0].errorbar(mean.index, mean.eff_wp, yerr=std.eff_wp, fmt="-o",
                         color=col, lw=2, ms=4, capsize=3, label=lab)
        axes[1].errorbar(mean.index, np.maximum(mean.far_wp, 2e-5),
                         yerr=std.far_wp, fmt="-o", color=col, lw=2, ms=4,
                         capsize=3, label=lab)
    axes[0].set_ylabel("wp99 segment efficiency")
    axes[0].set_ylim(0.4, 1.02)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("wp99 segment false rate")
    for ax in axes:
        ax.set_xlabel("polynomial degree d (walk calls ∝ depth)")
        ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Depth knob: metric-validated degree per encoding domain "
                 "(T=200, step, base Hamiltonian, 3 reps)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig05_degree_curves.png"), dpi=160)
    plt.close(fig)

    # FABLE pareto
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    sub = dfa[dfa.kernel == "step"]
    for enc, col in (("sfable", MCOL["sfable"]), ("fable", MCOL["fable"])):
        g = sub[sub.enc == enc].groupby("thr")[["rot", "enc_err", "far_wp", "eff_wp"]].mean()
        axes[0].plot(g.rot, g.enc_err, "-o", color=col, lw=2, ms=4, label=enc)
        axes[1].plot(g.rot, np.maximum(g.far_wp, 2e-5), "-o", color=col, lw=2,
                     ms=4, label=enc)
    ls = sub[sub.enc == "lsfable"][["rot", "enc_err", "far_wp"]].mean()
    axes[0].plot(ls.rot, ls.enc_err, "*", color=MCOL["lsfable"], ms=15, label="lsfable")
    axes[1].plot(ls.rot, max(ls.far_wp, 2e-5), "*", color=MCOL["lsfable"], ms=15,
                 label="lsfable")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("rotation gates kept"); axes[0].set_ylabel("encoding error ‖Â−A‖/‖A‖")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("rotation gates kept"); axes[1].set_ylabel("wp99 false rate (comb on Â)")
    axes[1].axhline(0.01, color="#52514e", ls="--", lw=1)
    for ax in axes:
        ax.legend(fontsize=9, frameon=False)
    fig.suptitle("FABLE-family compression: accuracy axis in isolation "
                 "(T=20, step, 3 reps; the α=2ⁿ wall is separate and fatal)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig05_fable_pareto.png"), dpi=160)
    plt.close(fig)

    # precision bits
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for T, mk in ((100, "-o"), (200, "-s")):
        g = dfc[dfc["T"] == T].groupby("bits")[["far_wp", "far_classical"]].mean()
        ax.plot(g.index, np.maximum(g.far_wp, 2e-5), mk, color=MCOL["szegedy"],
                lw=2, ms=5, label=f"±½ comb d=40, T={T}")
        ax.plot(g.index, np.maximum(g.far_classical, 2e-5), mk,
                color=MCOL["classical"], lw=1.5, ms=4, alpha=0.7,
                label=f"classical (same quantized C), T={T}")
    ax.set_yscale("log")
    ax.set_xlabel("fixed-point bits per hit coordinate (oracle register width)")
    ax.set_ylabel("false rate (wp99 / fixed-τ classical)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("Qubit knob: quantized geometry data → rebuilt step coupling")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig05_precision_bits.png"), dpi=160)
    plt.close(fig)

    # combined Pareto: every (implementation, QSVT degree) config as one point
    # x = total CX per filtered solve (= degree x per-call model), y = far_wp;
    # only configs holding wp99 eff >= 0.99 are shown (open markers otherwise)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    T = 200
    sub = dfb[(dfb["T"] == T) & (dfb.kernel == "step")]
    n_sys = int(np.ceil(np.log2(4 * T * T)))
    nnz = base_context(T, 0, "step")["C"].nnz
    percall = {
        "C_alpha/hit": (bl.model_hit_oracle(T, n_sys, w_pad=4)["cx"], MCOL["hit_oracle"], "o"),
        "C_alpha/camps-dict": (bl.model_transposition_oracle(nnz // 2, n_sys)["cx"], MCOL["camps"], "s"),
        "normalized/hit-prep": (2 * bl.model_hit_oracle(T, n_sys, w_pad=4)["cx"] + n_sys, MCOL["szegedy"], "D"),
    }
    dom_of = {"C_alpha/hit": "C_alpha", "C_alpha/camps-dict": "C_alpha",
              "normalized/hit-prep": "normalized"}
    for impl, (cxc, col, mk) in percall.items():
        g = (sub[sub.domain == dom_of[impl]]
             .groupby("degree")[["eff_wp", "far_wp"]].mean())
        ok = g.eff_wp >= 0.99
        ax.scatter(g.index[ok] * cxc, np.maximum(g.far_wp[ok], 2e-5), s=55,
                   color=col, marker=mk, label=impl, zorder=3)
        ax.scatter(g.index[~ok] * cxc, np.maximum(g.far_wp[~ok], 2e-5), s=45,
                   facecolors="none", edgecolors=col, marker=mk, zorder=2)
        for dg, r in g.iterrows():
            if dg in (12, 20, 40, 78):
                ax.annotate(f"d={dg}", (dg * cxc, max(r.far_wp, 2e-5)),
                            xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    # native 1BQF reference: one exp(-iAt) call, known store metrics band
    cx_1bqf = nnz // 2 * (2 * n_sys + 6 * max(0, 12 * n_sys - 36))
    ax.axvline(cx_1bqf, color=MCOL["classical"], lw=1.2, ls="--")
    ax.text(cx_1bqf * 1.1, 0.3, "native 1BQF:\none exp(-iAt) call", fontsize=8,
            color=MCOL["classical"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("total CX per filtered solve  (QSVT degree × per-call model)")
    ax.set_ylabel("wp99 segment false rate")
    ax.set_title(f"The full trade-off: QSVT polynomial degree × implementation "
                 f"(T={T}, step, base Hamiltonian)\nfilled = wp99 eff ≥ 0.99; open = efficiency lost")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig05_total_cost_pareto.png"), dpi=160)
    plt.close(fig)

    # erf truncation
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    g = dfd.groupby("cut")[["alpha_dict", "far_wp", "eff_wp", "nnz"]].mean()
    ax.plot(g.alpha_dict, np.maximum(g.far_wp, 2e-5), "-o", color=MCOL["dictionary"],
            lw=2, ms=5)
    for cut, r in g.iterrows():
        ax.annotate(f"cut={cut:g}", (r.alpha_dict, max(r.far_wp, 2e-5)),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dictionary subnormalization α = Σ|a|/2 after value cut")
    ax.set_ylabel("wp99 false rate (±½ comb d=40 on truncated C)")
    ax.set_title("erf value truncation: buying α (and dictionary viability) with weak couplings\n(T=200, erf, 3 reps)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig05_erf_truncation.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    dfb = part_B()
    dfa = part_A()
    dfc = part_C()
    dfd = part_D()
    dfe = part_E(dfb)
    figures(dfa, dfb, dfc, dfd)
    print("done")
