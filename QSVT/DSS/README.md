# DSS — Direct Structural Synthesis as the QSVT primitive

Can Xenofon's **Direct Structural Synthesis** (DSS, arXiv:2601.07766 — the 1BQF's
circuit construction) replace the qubitization/LCU construction under our QSVT comb,
and what does it save?

## The idea (2026-07-06)

DSS synthesizes controlled-$e^{-iAt}$ directly from the Hamiltonian's structure:

1. **Constant diagonal** $A = cI - B$ → exact factorization $e^{-iAt}=e^{-ict}\,e^{iBt}$;
   the diagonal is ONE phase gate $P(-ct)$ on the control (phase kickback).
2. **Two-level couplings**: $B=\sum_k B_k$, each $B_k$ a $\sigma_x$-like coupling of two
   *basis states* (segments sharing a hit within ε) → each $e^{iB_k t}$ is an exact
   **Givens rotation** (CNOT ladder to a pivot + multi-controlled $R_X(\theta)$ + uncompute).
3. **Sparsity**: total cost $\mathcal O(k\,n_s)=\mathcal O(\sqrt N\log N)$ two-qubit gates
   in the paper's regime ($k$ = interaction pairs, $n_s=\lceil\log_2 N\rceil$).

**Exactness fine print** (paper Eq. 15): the pair product is a *first-order product
formula*, error $\mathcal O(t^2)$ — but **exact on the isolated-false (noise) subspace**
($B|\psi_c\rangle=0$), so noise rejection is Trotter-immune; only signal phases blur.
Our ERF study measured this error amplifying notch physics ×10 at $d=1$.

**The obstruction for QSVT-as-built:** the qubitization dilation needs
$\sqrt{I-X^2}$ — a dense matrix function with no two-level structure. DSS cannot
synthesize it (our `QSVT.py` builds dense `UnitaryGate`s — simulation-only).

**The bridge (= resource agenda D2):** change primitive. QETU / generalized QSP builds
any **trigonometric polynomial** $\sum_k c_k\cos(k\lambda t)$ from repeated
controlled-$e^{-iAt}$ calls — i.e. from the DSS circuit itself. The comb becomes a
**Fourier comb in phase space** $\theta=\lambda t$; the 1BQF is its degree-1 member.
Width collapses to the 1BQF's ($n_s+2$: no LCU register, no dilation ancilla); per-call
gate cost = the 1BQF's; depth = ×d. The fork term ($+\beta B_\varepsilon$, constant
diagonal preserved) and the erf kernel (graded per-pair angles) run on the same
primitive unchanged.

**The open physics question:** does the $\mathcal O(t^2)$ per-pair Trotter error break
the comb's *narrow passes* ($h_w=0.18$ — far less forgiving than the 1BQF's single
cosine), and do structural mitigations fix it? Mitigations tested here:
- **edge-colouring** the coupling graph (pairs within a colour class are disjoint →
  commute exactly; error only between the ~max-degree colour classes);
- **Strang (2nd-order) splitting** (half-sweep forward + reverse, error → $\mathcal O(t^3)$);
- **t-scaling** (relative error ∝ t at depth ∝ 1/t).

## Questions this study answers

1. **Qubits saved** vs the 1BQF and vs the original (LCU/qubitization) QSVT — global
   and per-cluster (D3), measured on real store Hamiltonians.
2. **Two-qubit gate model** — measured pair counts + Hamming distances + transpiled
   per-Givens cost; per shot and per delivered solution (with AA).
3. **Trotter–comb feasibility** — exact vs product-formula comb on real events
   (T=100/400, γ=3 clean): eigenphase errors, activation ladders, segment eff/far at
   fixed τ and wp99, for natural / edge-coloured / Strang orderings; 1BQF reference.
4. **Verdict** — is the DSS-QETU comb worth building?

## Files

- `01_dss_qetu_feasibility.ipynb` — the study (§1 structure, §2 width, §3 gates,
  §4 Trotter–comb experiment, §5 verdict). Outputs → `outputs/`.

## Results (2026-07-06, γ=3 clean rep 0)

**VERDICT: worth doing — DSS-QETU is the comb's deployment path.**

| | 1BQF (DSS) | original QSVT (LCU, d=40) | **DSS-QETU comb** |
|---|---|---|---|
| width @ T=1000 | 24 | 29 | **24** (23 minimal) |
| width per-cluster (D3) | 6 | 10 (d=16) | **6** |
| 2q gates/shot @ T=400 | 1.24M | dense: 84k CX already at T=3 | **24.9M global / ≤14k per-cluster instance** |
| Trotter error | exact on noise null-space | n/a (exact, dense) | **colour-ordered: = exact comb** (97.75/0.00 fixed-τ; wp99 far 1.18 vs 1.12 exact) |

- Measured $k = 1.5$–$2.3\times\sqrt N$ (paper's regime holds); $\bar d_H$ 4–7.6; max cluster 16.
- **Coupling-graph chromatic index: median 1–2, max 4** → the product formula is nearly
  exact by geometry; **edge-colour ordering is required in the spec** (natural ordering
  broke the fixed-τ comb at T=100: eff 100→75 % via phase error on the outer-true band;
  colouring or Strang restores exactness).
- Trig comb K=40 realizable (max|p|=0.958), matches the production Chebyshev metrics;
  t/2-K=80 variant recovers contamination (99.81 %/1.36 %) — new design knob.
- Follow-ups: noise pairs + reps; fork term on the DSS primitive; qiskit QETU+DSS
  circuit cross-check vs `solve_statevector`; hardware-instance shot budget (paper §8).

## Provenance

Events/Hamiltonians: qtrk_pipeline store (`ensure_event`/`build_hamiltonian`,
γ=3, δ=1, ε=2 mrad, clean σ_scatt=1e-4). DSS construction: `OneBQF_repo/quantum_algorithms/OneBQF.py`
(`_apply_direct_controlled_u`) + arXiv:2601.07766 pp. 5–8 (Fig. 3, Eqs. 13–17).
Production comb: `lhcb_velo_toy.solvers.quantum.design_line_comb_inverse` (d=40, hw=0.18).
Original-QSVT resource anchors: QSVT/Segment_level_studies/05 (width $n_s{+}7$ at d=40,
dense-dilation transpile anchor T=3 ≈ 84k CX).
