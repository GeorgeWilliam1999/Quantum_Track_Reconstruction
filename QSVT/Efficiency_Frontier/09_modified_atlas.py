#!/usr/bin/env python3
"""Stage 6 — the modified amplitude atlas: closed forms under the added terms.

Paper eq. 7 (the fragment/chain ladder) and the motif atlas of section 2,
generalised to the three modified Hamiltonians of the Efficiency Frontier
study.  Everything here is exact linear algebra on small motifs plus
verification against the real dp_terms systems on store events.

CLOSED FORMS (gamma=3, delta=1, s = gamma+delta; alpha = occupancy weight,
beta = fork weight; B_all/B_fork are SAME-ROLE co-hit graphs — verified
against bif.fork_graph: start-hit and end-hit groups are built separately,
so consecutive segments of one chain, which share their interior hit in
OPPOSITE roles, are never coupled):

1. Occupancy is a pure (s, delta) shift on every pure chain.
   A chain of m segments has no same-role co-hit pair inside it, so
   A_occ|chain = (s+4a) I - P_m with bias delta+4a.  The eq.-7 ladder
   generalises verbatim:
       xbar' = (delta+4a)/(s+4a-2),   r' + 1/r' = s + 4a,
       x_j   = xbar' [1 - (r'^j + r'^(m+1-j)) / (1 + r'^(m+1))],
   and the isolated attractor moves to (delta+4a)/(s+4a).
   Chain eigenvalue lines shift rigidly: lambda_k = s + 4a - 2 cos(k pi/(m+1)).

2. The fork leaves every pure-true chain EXACTLY invariant (no fork edge
   joins two true segments), and b is unchanged.  A competing same-role
   pair inside the window couples with +beta:
       isolated fork pair: x = delta/(s+beta) each,  lines s +- beta.
   A false prong forked onto a true chain's outer segment pulls BOTH:
   the (m+1)-motif "chain + prong" is solved exactly here.

3. Same-role false pair under occupancy: x = (delta+4a)/(s+6a) each,
   lines s+2a and s+6a.  The split from the shifted attractor,
       D(a) = (delta+4a) [1/(s+4a) - 1/(s+6a)] ~ 2 a delta / s^2 + O(a^2),
   is the ANALYTIC twin-break: a bare fragment keeps the attractor-track
   level while a same-role-coupled false structure drops below it.
   The floor theorem's invariant grows from the C-graph alone to the
   (C, B_all) pair — isomorphism must now match same-role hit degrees.

4. Combined (both): B_fork is a subset of B_all, so a forked pair inside
   the window carries 2a+beta; chains see the occupancy shift only:
       forked pair: x = (delta+4a)/(s+4a + 2a+beta),
       lines s+4a +- (2a+beta).  At a=0.05, b=0.5 the pair level is
       1.2/4.8 = 0.250 — numerically AT the base attractor (coincidence).

Outputs: outputs/09_modified_atlas.csv  (motif x operator levels + lines)
         printed verification report (exact micro-motifs to 1e-12 +
         on-event checks against dp_terms on store events)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

for _p in ("/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/_shared",
           "/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Bifurification",
           "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("QTRK_STORE", "/data/bfys/gscriven/qtrk_store")

import qtrk_pipeline as qp                                   # noqa: E402
import dp_terms                                              # noqa: E402
from lhcb_velo_toy.analysis import compute_epsilon           # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

S, DELTA = 4.0, 1.0
ALPHA, BETA = 0.05, 0.5


# ── closed forms ─────────────────────────────────────────────────────────────
def ladder_params(alpha=0.0):
    """(s', delta', xbar', r') for chains under occupancy weight alpha."""
    sp_, dp_ = S + 4 * alpha, DELTA + 4 * alpha
    xbar = dp_ / (sp_ - 2.0)
    r = (sp_ - np.sqrt(sp_ * sp_ - 4.0)) / 2.0
    return sp_, dp_, xbar, r


def chain_level(m, j, alpha=0.0):
    """Exact activation of segment j (1-based) in a pure chain of m segments."""
    _, _, xbar, r = ladder_params(alpha)
    return xbar * (1.0 - (r**j + r**(m + 1 - j)) / (1.0 + r**(m + 1)))


def outer_level(m, alpha=0.0):
    return chain_level(m, 1, alpha)


def attractor(alpha=0.0):
    return (DELTA + 4 * alpha) / (S + 4 * alpha)


def pair_level(coupling, alpha=0.0):
    """Symmetric level of a same-role coupled pair (no C edge)."""
    return (DELTA + 4 * alpha) / (S + 4 * alpha + coupling)


def chain_lines(m, alpha=0.0):
    k = np.arange(1, m + 1)
    return S + 4 * alpha - 2 * np.cos(k * np.pi / (m + 1))


# ── exact micro-motif verification ───────────────────────────────────────────
def solve_motif(n, c_edges, b_edges, alpha=0.0, beta=0.0, b_in_window=True):
    """Exact dense solve of a motif: C edges (continuations), B edges
    (same-role co-hit).  Fork applies beta on window B edges; occupancy
    applies 2*alpha on ALL B edges + the diagonal/bias shift."""
    A = (S + 4 * alpha) * np.eye(n)
    for i, j in c_edges:
        A[i, j] = A[j, i] = -1.0
    for i, j in b_edges:
        w = 2 * alpha + (beta if b_in_window else 0.0)
        A[i, j] += w
        A[j, i] += w
    b = (DELTA + 4 * alpha) * np.ones(n)
    return np.linalg.solve(A, b), np.linalg.eigvalsh(A)


def verify_micro():
    print("== micro-motif verification (exact dense vs closed form) ==")
    ok = True

    def check(name, got, want, tol=1e-12):
        nonlocal ok
        d = abs(got - want)
        flag = "OK " if d < tol else "FAIL"
        ok &= d < tol
        print(f"  [{flag}] {name}: exact {got:.12f} vs formula {want:.12f}")

    for a in (0.0, ALPHA, 0.10):
        x, _ = solve_motif(1, [], [], alpha=a)
        check(f"isolated  a={a}", x[0], attractor(a))
        for m in (2, 3, 4, 6):
            edges = [(i, i + 1) for i in range(m - 1)]
            x, ev = solve_motif(m, edges, [], alpha=a)
            check(f"chain m={m} outer  a={a}", x[0], outer_level(m, a))
            check(f"chain m={m} j=2    a={a}", x[1], chain_level(m, 2, a))
            want = np.sort(chain_lines(m, a))
            check(f"chain m={m} lines  a={a}", np.max(np.abs(np.sort(ev) - want)), 0.0)
    # fork pair, occ pair, both pair
    x, ev = solve_motif(2, [], [(0, 1)], beta=BETA)
    check("fork pair level  b=0.5", x[0], pair_level(BETA))
    check("fork pair lines", np.max(np.abs(np.sort(ev) - np.array([S - BETA, S + BETA]))), 0.0)
    x, ev = solve_motif(2, [], [(0, 1)], alpha=ALPHA, b_in_window=False)
    check("occ same-role pair a=0.05", x[0], pair_level(2 * ALPHA, ALPHA))
    x, ev = solve_motif(2, [], [(0, 1)], alpha=ALPHA, beta=BETA)
    check("both forked pair", x[0], pair_level(2 * ALPHA + BETA, ALPHA))
    check("both pair lines",
          np.max(np.abs(np.sort(ev) - np.array([S + 4 * ALPHA - 2 * ALPHA - BETA,
                                                S + 4 * ALPHA + 2 * ALPHA + BETA]))), 0.0)
    # true P4 chain is EXACTLY beta-invariant
    edges = [(i, i + 1) for i in range(3)]
    x0, _ = solve_motif(4, edges, [])
    x1, _ = solve_motif(4, edges, [], beta=BETA)
    check("P4 fork-invariance", np.max(np.abs(x1 - x0)), 0.0)
    # chain + forked false prong (the competing-candidate motif), m=4:
    # prong node 4 forks with outer segment 0
    xp, evp = solve_motif(5, edges, [(0, 4)], beta=BETA)
    print(f"  [num] P4+prong b=0.5: outer {xp[0]:.6f} (was {x0[0]:.6f}), "
          f"prong {xp[4]:.6f} (attractor {attractor():.6f}), "
          f"inner {xp[1]:.6f} (was {x0[1]:.6f})")
    xb, _ = solve_motif(5, edges, [(0, 4)], alpha=ALPHA, beta=BETA)
    print(f"  [num] P4+prong BOTH a=0.05 b=0.5: outer {xb[0]:.6f}, prong {xb[4]:.6f}")
    # twin-break minimal environment: false 2-chain whose end segment has a
    # same-role spectator (node 2) vs the bare true 2-chain fragment
    xf, _ = solve_motif(2, [(0, 1)], [], alpha=ALPHA)          # bare fragment
    xt, _ = solve_motif(3, [(0, 1)], [(1, 2)], alpha=ALPHA, b_in_window=False)
    print(f"  [num] twin break a=0.05: bare 2-chain outer {xf[0]:.6f} vs "
          f"same-role-attached 2-chain outer {xt[0]:.6f} "
          f"(split {xf[0]-xt[0]:+.6f}); leading-order attractor split "
          f"{2*ALPHA*DELTA/S**2:.6f}")
    print("  micro-motifs:", "ALL OK" if ok else "FAILURES ABOVE")
    return ok


# ── on-event verification ────────────────────────────────────────────────────
def verify_on_event(regime, noise, alpha, beta, rep=0, T=200):
    """Find pure-chain components of the REAL modified system and compare
    their activations to the closed forms."""
    eps = float(compute_epsilon(noise["sigma_res"], noise["sigma_scatt"]))
    ev, _ = qp.ensure_event(n_trk=T, rep=rep, **noise)
    ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                               gamma=3.0, delta=DELTA)
    A, b, _, _ = dp_terms.dp_system(ham, beta=beta,
                                    eps_B=eps if beta else None,
                                    alpha=alpha, gamma=3.0, delta=DELTA)
    A = A.tocsr()
    n = A.shape[0]
    x = np.asarray(sp.linalg.spsolve(A.tocsc(), b)).ravel()

    # off-diagonal support graph of the FULL modified operator
    Off = A.copy().tolil(); Off.setdiag(0); Off = Off.tocsr()
    ncomp, lab = connected_components(Off != 0, directed=False)
    counts = np.bincount(lab)
    report = {}
    # a pure chain in the MODIFIED operator: component whose off-diag entries
    # are all -1 (couplings only) and whose degree pattern is a path
    from collections import Counter
    stats = Counter()
    err_max = 0.0
    n_chains = 0
    for ci in np.where((counts >= 2) & (counts <= 8))[0]:
        idx = np.where(lab == ci)[0]
        sub = Off[idx][:, idx]
        vals = sub.data
        if len(vals) == 0 or np.any(np.abs(vals + 1.0) > 1e-9):
            stats["non-chain (modified couplings)"] += 1
            continue
        deg = np.asarray((sub != 0).sum(1)).ravel()
        if sorted(deg)[:2] != [1, 1] or np.any(deg > 2):
            stats["non-path"] += 1
            continue
        m = len(idx)
        n_chains += 1
        # walk the path from one end
        order = [int(np.where(deg == 1)[0][0])]
        Gd = {i: set(np.nonzero(np.asarray(sub[i].todense()).ravel())[0])
              for i in range(m)}
        while len(order) < m:
            nxt = Gd[order[-1]] - set(order)
            order.append(int(nxt.pop()))
        for j_pos, node in enumerate(order, start=1):
            want = chain_level(m, j_pos, alpha)
            err_max = max(err_max, abs(x[idx[node]] - want))
    report["chains_checked"] = n_chains
    report["max_abs_err_vs_formula"] = err_max
    report["skipped"] = dict(stats)
    print(f"  [{regime} a={alpha} b={beta}] n={n:,} comps={ncomp:,} | "
          f"pure chains checked {n_chains}, max |x - formula| = {err_max:.2e} "
          f"| skipped {dict(stats)}")
    return err_max


def main():
    ok = verify_micro()

    print("\n== on-event verification (store events, dp_terms systems) ==")
    REG = {
        "clean":    dict(sigma_scatt=1e-4, sigma_res=0.0,  phi_max=0.2, hit_ineff=0.0),
        "moderate": dict(sigma_scatt=1e-4, sigma_res=0.01, phi_max=0.2, hit_ineff=0.01),
    }
    worst = 0.0
    for regime, noise in REG.items():
        for a, bta in ((0.0, 0.0), (ALPHA, 0.0), (0.0, BETA), (ALPHA, BETA)):
            worst = max(worst, verify_on_event(regime, noise, a, bta))
    print(f"  on-event worst error: {worst:.2e}",
          "(OK)" if worst < 1e-8 else "(INVESTIGATE)")

    # ── the atlas table ──────────────────────────────────────────────────────
    rows = []
    OPS = [("base", 0.0, 0.0), ("occ_a0.05", ALPHA, 0.0),
           ("fork_b0.5", 0.0, BETA), ("occ0.05_fork0.5", ALPHA, BETA)]
    for opname, a, bta in OPS:
        rows.append(dict(operator=opname, motif="isolated (false attractor)",
                         level=attractor(a), lines=f"{S+4*a:.3f}"))
        for m, nm in ((1, "fragment m=1"), (2, "fragment m=2"),
                      (3, "fragment m=3"), (4, "true track P4 outer")):
            rows.append(dict(operator=opname, motif=nm,
                             level=outer_level(m, a),
                             lines=", ".join(f"{v:.3f}" for v in chain_lines(m, a))))
        rows.append(dict(operator=opname, motif="true track P4 inner",
                         level=chain_level(4, 2, a), lines=""))
        cpl = (2 * a + bta)
        if cpl > 0:
            rows.append(dict(operator=opname,
                             motif="same-role competing pair (in window)",
                             level=pair_level(cpl, a),
                             lines=f"{S+4*a-cpl:.3f}, {S+4*a+cpl:.3f}"))
        if a > 0:
            rows.append(dict(operator=opname,
                             motif="same-role pair (outside window)",
                             level=pair_level(2 * a, a),
                             lines=f"{S+4*a-2*a:.3f}, {S+4*a+2*a:.3f}"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "09_modified_atlas.csv", index=False)
    print("\n== the modified atlas (levels) ==")
    print(df.pivot_table(index="motif", columns="operator", values="level",
                         aggfunc="first").round(4).to_string())
    print("\nsaved outputs/09_modified_atlas.csv")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
