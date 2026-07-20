# Block encodings for the segment Hamiltonian — benchmark of four literature methods on the 1BQF/QSVT systems

**Status: DRAFT (2026-07-19) — numbers being filled from outputs/*.csv**
Provenance: repo `Quantum_Track_Reconstruction/Block_Encoding` (this dir), data =
qtrk store events (deterministic keys, rep 0), library = `lhcb_velo_toy` +
`qtrk_pipeline`. Scripts: `01_…` – `04_…`; every claimed number traces to a CSV
in `outputs/`.

## 1. Intro

Our QSVT comb filter and the 1BQF notch are *spectral* algorithms: they apply a
function of the segment Hamiltonian `A = (γ+δ)I − C` to `b = 1`. On hardware the
operator enters through a **block encoding** — a unitary whose top-left block is
`A/α`. Today the production QSVT solver uses an exact dense dilation
(subnormalization-optimal, but the unitary is a dense `4^n`-element gate:
simulation-only), and the 1BQF uses per-pair Givens rotations for `e^{-iAt}`
(gates ∝ nnz). The open resource question (D5) is whether a *circuit-efficient*
block encoding exists for our matrices. Four literature candidates:

1. **S-FABLE / LS-FABLE** (Kuklinski & Rempfer, 2401.04234) — FABLE in the
   Walsh–Hadamard domain; targets *unstructured sparse* matrices.
2. **Basis-state transpositions** (Herbert, Sorci & Tang, 2309.12820) — Θ(n)
   circuits per transposition; the compiler primitive for *unstructured
   permutation oracles* (used here to price index oracles honestly).
3. **Dictionary-based block encoding** (Yang et al., 2405.18007) — LCU over
   same-value classes; α = Σ_l |A_l|; depth O(log ns) at O(n²s) ancillas.
4. **Explicit sparse-matrix circuits** (Camps, Lin, Van Beeumen & Yang,
   2203.10236) — the D_s/O_c/O_A framework, α = s; plus the Szegedy-form α = 1
   encoding of a *stochastic* matrix (§5).

## 2. Aims

- Verify each construction end-to-end on real store matrices (block extraction,
  machine-precision check where exact).
- Measure the axes that decide viability *for our filters*: subnormalization α
  (⇒ comb degree), per-call gate cost, ancillas, classical prep.
- Decide: adopt / adapt / reject, with the *why* quantified.
- If rejected: design the route that fits our matrices (hit-level geometry
  oracle, D5) with a validated cost model.

## 3. Method

Matrices from the qtrk store (γ=3, δ=1, ε=2 mrad, step & erf kernels, clean &
noisy, T = 2 … 1000). All six encodings implemented in `be_lib.py` in
verification grade (explicit unitaries allowed for oracles) + analytic cost
models; FABLE-family also in gate-exact circuit form (own UCRy/Gray-walk
implementation, machine-precision vs reference). Exact block extraction by
batched basis-column evolution up to padded dim 64. FABLE compression curves via
a classically-emulated encoder (cross-checked vs circuits at small n to ~1e-13).
Comb-degree requirements *measured* by minimum Chebyshev degree reaching L∞ 0.02
on each encoding's rescaled spectral domain (no hand-waved α-ratios).
End-to-end physics: matrix-free `p(M)b` on store events with the standard
γ-aware metrics (`qp.quantum_metrics`, τ = 0.35).

## 4. Results

### 4.1 Verification (01)

All exact constructions verify by explicit block extraction to ≤ 1.2e-13
(FABLE, S-FABLE, Camps, dictionary, Szegedy, dilation) on P4, K(1,4) hub, and
real T=2/3/5 events, both kernels (`outputs/01_verify_table.csv`). The
classical FABLE-compression emulator matches the circuits to ≤ 1e-13, licensing
the large-T compression curves. LS-FABLE — the only *approximate* method —
lands at max-entry error 2.3–4.6 (O(1), unusable) on every one of our matrices,
exactly the binary/nonnegative breakdown its own paper warns of in §5.3.

Headline verified numbers at T=3 (36 segments, padded dim 64, 18 off-diag nnz):

| method | α (step) | α (erf) | rotations | CX | note |
|---|---|---|---|---|---|
| FABLE | 256 | 256 | 3367 | 3978 | α = N·max\|a\| |
| S-FABLE | 162 | 180 | 1024 | 1350 | compression mild (29% kept at 1e-3) |
| LS-FABLE | 64 | 64 | 54 | 156 | **err 2.26 = junk** |
| Camps (C, shift-absorbed) | **2** | 3.99 | — | oracle-priced | s_pad = 2^⌈log₂Δ⌉ |
| Dictionary (C) | **2** | **17.95** | — | oracle-priced | α = χ′ (step) → Σ\|a\|/2 (erf) |
| Szegedy | **1** | 1 | — | oracle-priced | encodes D^{-1/2}CD^{-1/2} |
| dilation (production) | 1.68 | 3.36 | — | dense 4ⁿ | simulation-only |

### 4.2 Structure of the real matrices is the whole story (02)

Measured on store events, rep 0 (`outputs/02_structure.csv`):

| kernel/noise | T | n_seg | nnz_off | Δ | χ′ | distinct vals | distinct i⊕j | spec(A) |
|---|---|---|---|---|---|---|---|---|
| step clean | 100 | 40 000 | 624 | 2 | 2 | 1 | 104 | [2.38, 5.62] |
| step clean | 400 | 640 000 | 3 722 | 4 | 4 | 1 | 771 | [1.20, 6.80] |
| step noisy | 400 | 640 000 | 3 612 | 3 | 3 | 1 | 767 | [2.15, 5.85] |
| erf clean | 400 | 640 000 | 14 064 | 6 | 6 | **6 508** | 5 904 | **[−1.30, 9.30]** |
| erf noisy | 400 | 640 000 | 14 728 | 5 | 5 | 6 767 | 6 245 | [0.50, 7.50] |

Facts that decide everything downstream:
1. **C is ultra-sparse and near-1-regular**: Δ = 2–6 and greedy edge coloring
   achieves χ′ = Δ at every point — the coupling graph is essentially disjoint
   paths + rare hubs. Any α ≈ Δ encoding is already near the ‖C‖₂ ≈ 1.6–2
   lower bound.
2. **The step kernel has ONE off-diagonal value**; the erf kernel has ~nnz/2
   distinct values (every coupling unique) and drives A indefinite at T=400
   clean (λmin = −1.30).
3. **No displacement structure**: distinct i⊕j ≈ 0.2 × nnz — the pattern is
   event-random, so no circulant/Laplacian-style structured oracle exists in
   the *segment* index space (this is what breaks the papers' cheap-circuit
   examples on our matrices).
4. **The Hadamard domain is dense**: ‖HAH‖max ≈ 3.2–3.6 with **87–94 % of the
   transformed entries above 1e-3** (T ≤ 20 exact) — the measured mechanism of
   the S-FABLE compression failure: our A has a huge DC component (constant
   diagonal + nonnegative couplings), so H A H concentrates rather than
   sparsifies.

### 4.3 The four methods on our matrices (01+02+03)

- **FABLE**: exact but α = 2ⁿ·max|a| = 4.2·10⁶ at T=400 — comb degree × O(N),
  and 10¹² CX. Dead on arrival at any scale.
- **S-FABLE**: exact, but compresses *nothing useful* on our matrices: at T=20
  (N=2048) reaching tol 1e-3 needs **38 912 rotations (step) / 106 498 (erf)
  vs 1 787 on the equal-sparsity random control** — 22×/60× worse, the paper's
  §5.3 binary/nonnegative breakdown demonstrated on our data. Mechanism
  measured: ‖HAH‖max ≈ 3.2 with **74–94 % of Hadamard-domain entries above
  1e-3** (the constant diagonal + nonnegative couplings give A a huge DC
  component — H A H concentrates instead of sparsifying). And even at full
  rotation count, α = 2ⁿ‖HAH‖max ≈ 3.2 N. Dead.
- **LS-FABLE**: O(nnz) rotations but a *fixed* O(1) encoding error on binary
  nonnegative matrices: relative error **0.80 on step and 0.79 on erf vs
  4·10⁻⁵ on the random control** (T=20) — four orders of magnitude, unusable
  at any gate budget; α = 2ⁿ on top. Dead.
- **Transpositions**: not an encoder but the honest price of unstructured index
  oracles: O(nnz·n) CX per walk call when compiled from the event geometry
  (2.4·10⁶ CX at T=400 step — the same order as the native 1BQF Givens pass).
- **Camps D_s/O_c/O_A**: on C alone (diagonal absorbed into the polynomial)
  α = 2^⌈log₂ Δ⌉ = **2–8 measured** across the full grid: *excellent*, only
  1.4–1.5× the dilation's span/2 in comb degree. But the index oracle for our
  event-random pattern costs Ω(nnz·n) via transpositions — per-call cost
  equals the classical matvec, defeating the purpose at scale.
- **Dictionary**: the best literature α on the *step* kernel (α = χ′ = Δ = 2–4;
  our verified circuits confirm α = 2, exact to 2e-16) — and its LCU-over-
  matchings form connects directly to our edge-colour Trotter machinery.
  **Degenerates on the erf kernel**: every coupling value is distinct
  (6 508 distinct values at T=400), classes collapse to single entries and
  α = Σ|a_ij|/2 = **3 829** (comb degree ×723). The O(log ns) depth costs
  O(n²s) ancillas ≈ 5.6·10⁶ at T=400; the low-ancilla route pays
  transposition prices again (1.4·10⁷ CX/call measured model).
- **Szegedy α=1** (Camps §5): encodes the *degree-normalized* operator
  D^{-1/2}CD^{-1/2}. This changes the physics — see 4.5.

### 4.4 Degree & total cost (03A)

Minimum Chebyshev degree for the line comb (L∞ tol 0.02) *measured by fitting on
each encoding's rescaled domain* (`outputs/03_degree_requirements.csv`), step
kernel, clean:

| method | α (T=200) | d_req (T=200) | α (T=400) | d_req (T=400) |
|---|---|---|---|---|
| dilation (reference) | 1.62 | 26 | 2.80 | 54 |
| **szegedy (α=1)** | 1 | **32** | 1 | **32 (T-independent)** |
| camps_C / dictionary_C / hit_oracle | 2 | 40 | 4 | 78 |
| dictionary (A) | 6 | 100 | 8 | 141 |
| camps (A) | 16 | >600 | 32 | >600 |
| FABLE / LS-FABLE | 5×10⁵–2×10⁶ | unresolvable (∝α: ~10⁷) | 10⁶–4×10⁶ | unresolvable |

Two structural observations: (i) the degree of every A/C-domain encoding grows
with the spectral span (hub √m growth widens spec(A) with T and noise), while
the **normalized walk's degree is T-independent** — its domain is [−1,1] by
construction, which also eliminates the Lanczos spectral-bounds step (the
slowest classical pre-step of the current pipeline); (ii) shift-absorbing the
diagonal (encoding C, not A) is worth 2–8× in α for free — the polynomial
just gets composed with s−λ.

Total CX per comb solve = d_req × CX/call at T=400 step:
hit_oracle **1.3·10⁶** vs camps_C 1.8·10⁸ vs dictionary_C 1.1·10⁹ — and the
production 1BQF pays 2.4·10⁶ per *single* e^{-iAt} call at the same point.

### 4.5 The normalized-walk experiment (03B) — the surprise finding

The α = 1 encoding replaces A's spectrum with the normalized one, and on the
toy geometry this *collapses whole failure families onto exact fixed lines*:
the normalized P4 spectrum is {±1, ±½}, and ±1 is shared by **every** connected
bipartite component (Perron vector + bipartite symmetry) — so the only
distinguishing true-track lines are **±½**. Hubs K(1,m) sit at {±1, 0^(m−1)}
for every m (vs s±√m spreading into the band before), P3 bridges at {±1, 0},
isolated false at 0. A comb with passes only at ±½ therefore rejects hubs,
bridges and isolated falses **exactly**, with an enormous gap (nearest false
line 0.5 away) — the only survivor is, as ever, the spectrum-identical
P4-bridge (the floor theorem, representation-independent).
Measured per-component spectra at T=200 step-clean confirm it exactly
(`outputs/fig03_normalized_spectrum.png`): every pure-true component on
{−1, −½, +½, +1}, every pure-false pair on {±1}, isolated on 0.

End-to-end segment metrics on store events, ±½ comb applied matrix-free
(`outputs/03_normalized_walk_metrics.csv`; wp99 = the efficiency-first
working point, the headline convention since 2026-06-14 — the fixed-τ=0.35
columns understate eff because the −½ mode gives P4 inner segments 0.707
relative amplitude, a threshold-placement artifact, not a separation failure;
the amplitude gap true-vs-false is ~150× at T=200):

| point | solver | wp99 eff | wp99 far | classical eff / far |
|---|---|---|---|---|
| T=100 step | szegedy comb **d=20** | 1.000 | 0.0000 | 1.000 / 0.000 |
| T=200 step | szegedy comb **d=20** | 1.000 | 0.0000 | 1.000 / 0.000 |
| T=400 step | szegedy comb d=40 | 0.993 | **0.0013** | 1.000 / 0.020 |
| T=200 erf | szegedy comb d=40 | 0.990 | **0.0025** | 1.000 / 0.144 |

At T=400 step the normalized comb's false rate is **15× below the classical
solve**; on the erf kernel **58× below** — at α=1, with degree 20–40 (vs the
production deg-40 comb on the dilation), no Lanczos bounds step, and hub
rejection *exact* instead of notch-approximate. The d=12 comb is below the
resolution floor (ripple leaks the ±1 families) — d≈20 is the knee.

### 4.6 The hit-level geometry oracle (04 + design)

The wall every literature method hits: circuits priced per *matrix entry*
(Ω(nnz) = Ω(T²)), while the event only contains O(T log T) bits. Pricing the
oracle by *hits* (QROM over 5T sorted coordinates + extrapolation arithmetic +
comparator, in the state-preparation-pair form — see the injectivity subtlety
in `HIT_ORACLE_DESIGN.md`) gives O(T) per call with α = window occupancy.
Measured on real events (`outputs/04_hit_oracle_window.csv`): **coverage = 1.0
at every point** (T = 5…1000 clean, plus noisy 100/400); the 1D sorted-x band
window grows w = 1→16 (α_hit = 2→16) while the 2D disc window stays **w = 1–3
(α = 2–4) all the way to T = 1000** — the same subnormalization class as the
best literature methods. Per-call cost at T = 400: **1.6·10⁴ CX (model) vs
2.4·10⁶ (transposition-compiled Camps/1BQF Givens) and 1.4·10⁷ (dictionary
low-ancilla)** — two to three orders of magnitude, growing with T. The erf
kernel costs *nothing extra*: the value is an arithmetic function of the
computed kink angle (vs the dictionary's α exploding to 3 829). Classical prep:
one O(T log T) sort per event vs FABLE's O(N² log N) = O(T⁴ log T).

### 4.7 Resource vs accuracy on the base Hamiltonians (05)

George's question (2026-07-20): can a block encoding *meaningfully reduce
circuit depth or qubit count* with a *measurably small* loss in accuracy /
segment efficiency / false rate? Base Hamiltonians only (no fork, no occupancy
term — as everywhere in this study). Four knobs measured
(`05_resource_accuracy.py`, wp99 headline, 3 reps at T=200):

**Depth — yes, and by a lot** (`05_degree_curves.csv`, figs
`fig05_degree_curves`, `fig05_total_cost_pareto`):
- The **metric-validated QSVT degree is 12–20**, not the production 40 and not
  the L∞-0.02 fit requirement (26–78): at T=200 step, the dilation and C/α
  combs reach wp99 far = 0.000 at eff ≥ 0.997 from **d=12–16**; the ±½
  normalized comb from **d=20** (far 0.0033). At T=400: d=20 across domains.
  Degree curves are **non-monotone** (ripple positions interact with tangle
  eigenvalues — normalized d=24 spikes to 0.45 while d=20/28 are fine), so an
  operating degree needs a stability margin, not a fit criterion.
- Total depth at the validated points (T=400, `05_qubit_depth_table.csv`):
  **hit-oracle comb solve = 3.3·10⁵ CX — 7× below a SINGLE native 1BQF
  e^{-iAt} call** (2.35·10⁶); szegedy walk (hit-prep) 6.6·10⁵ (3.6× below)
  with the best far (0.0013). The transposition-compiled camps/dict route is
  20× *above* the 1BQF call. On the erf kernel the standard 1/λ comb
  **collapses at every degree** (far 0.5–1.0 at wp99) — only the normalized
  ±½ comb works (far 0.004–0.006 from d=28).

**Qubits — no; the trade runs the other way** (`05_qubit_depth_table.csv`):
implementable encodings *add* qubits over the 21-qubit native-1BQF floor:
+8 (camps/dict, 29 total), +47–49 (hit oracle, ~68–70, QROM workspace),
+65–69 (szegedy walk, ~86–90). The offer on the table is **qubits for depth
and accuracy**: ~3–4× the qubits buys 4–7× less depth per solve *and* a
false rate at or below the classical solve.

**The geometry data needs 12 bits/coordinate — a hard cliff, not a slope**
(`05_precision_bits.csv`, fig `fig05_precision_bits`; harness self-check:
64-bit rebuild reproduces the library coupling with 0 edge diff): at b=12
the rebuilt coupling differs by 6–42 edges of ~1.3–1.6k and the ±½ comb holds
far 0.000 at eff ≥ 0.996; at b=10, 190–500 boundary edges flip and *both* the
comb (far → 0.99) and the classical solve on the same quantized matrix
(eff → 0.62) break — the cliff is a property of the ε-boundary geometry, not
of the encoding. Fixed-point registers can be 12–16 bits; below that nothing
survives.

**Encoding error: magnitude is not the predictor — structure is**
(`05_fable_pareto.csv`, fig `fig05_fable_pareto`): on step, even LS-FABLE's
0.80 relative error leaves wp99 far = 0.000 (T=20; easy separation), and on
erf S-FABLE compressed to 0.05 % of its rotations (err 0.006) is still clean —
but at err 0.14 (Hadamard-domain truncation) erf collapses to far 0.95, while
LS-FABLE's smooth sin-distortion at err 0.79 stays at far 0.0. The comb
tolerates *smooth monotone* encoding error remarkably well and *structured
delocalized* error badly. (The FABLE family stays dead regardless — the
α = 2ⁿ wall multiplies the required degree by ~10⁵ before accuracy enters.)

**erf value truncation does not rescue the dictionary** (`05_truncation.csv`):
cutting couplings below 0.3 removes ~44 % of nnz but only 1–2 % of
α = Σ|a|/2 (1390 → 1376) — the α mass sits in the mid/large values. Metrics
are truncation-safe (far flat at 0.0025–0.01), but the dictionary route on erf
stays dead; arithmetic values (hit oracle) remain the only erf answer.

## 5. Conclusion

**Can any of the four methods improve our encoding? Directly, no — but two of
them supply the pieces of the thing that does.**

*Rejected outright:* FABLE / S-FABLE / LS-FABLE. Not a gate-count issue but a
structural one, confirmed on our data at every scale we measured: the segment
Hamiltonian is exactly the matrix class (binary, nonnegative, structured,
DC-dominated) their own §5.3 flags, so S-FABLE's Hadamard-domain compression
does not materialise (74–94 % of transformed entries survive; 22–60× the
control's rotations at tol 1e-3), LS-FABLE's fixed error is O(1) (0.8 vs 4e-5
on the control), and the family's α ∝ 2ⁿ multiplies the comb degree by ~10⁵.

*Rejected as circuits, adopted as analysis:* Camps' oracle framework and the
dictionary decomposition. Their α on the shift-absorbed coupling matrix C is
excellent (α = 2^⌈log₂Δ⌉ = 2–8, degree multiplier only 1.24–1.43) and the
dictionary's LCU-over-matchings *is* our edge-colour machinery in encoding
form — but for an event-random nonzero pattern both pay Ω(nnz·n) gates per
call (transposition-compiled index oracles) or O(n²s) ancillas (SBM), i.e.
the quantum circuit is as large as the classical matrix. The dictionary also
degenerates on the erf kernel (α = 3 829 at T=400). The transposition paper is
the *pricing tool* that makes these statements honest.

*The way forward (1) — the hit-level geometry oracle* (`HIT_ORACLE_DESIGN.md`):
break the per-nonzero wall by querying the O(T) hits instead of the O(T²)
entries. Same α class as the best literature methods (α = 2–4 with the 2D
window, measured, coverage 1.0 to T=1000), per-call gates ~10²–10³× below the
transposition routes, erf kernel for free, and the 1BQF inherits it as the
degree-1 member. This is D5 concretised and validated at the oracle-semantics
level; next step is the gate-level qiskit prototype (validation ladder step 2).

*The way forward (2) — the α=1 normalized walk* (Camps §5 + our spectra):
encode D^{-1/2}CD^{-1/2} directly. The normalized toy spectrum is *cleaner*
than A's: true P4 lines at ±½ are the only distinguishing lines, hubs/bridges/
isolated collapse onto {±1, 0} exactly, the domain is [−1,1] with **no
spectral-bounds estimation and a T-independent comb degree (32 measured)**,
and the ±½ comb reaches eff 1.000 at far ≤ 0.01 on step-clean events at the
efficiency-first working point (τ policy per solver, as always). On the erf
kernel it *beats the classical solve* on false rate at equal efficiency.
The floor theorem survives (P4-motif twins are spectrum-identical in any
representation) — no encoding was ever going to change that.

*Priority:* the two directions compose, with one honest gap. The hit-level
oracle in state-preparation-pair form with *uniform* window amplitudes encodes
C/w (α = w = 2–4) — that is way-forward (1) as validated. Upgrading it to the
α = 1 discriminant needs the per-segment degree normalizer √(1/deg_j) folded
into the prep; the same inverse-CDF QROM yields the *band* count at O(T) cost,
which normalizes by the window population rather than the accepted count —
i.e. an approximate discriminant whose line positions shift where band ≠ disc
occupancy. Quantifying that shift (and whether the ±½ comb tolerates it) is
the concrete next experiment; if it holds, one O(T)-per-call construction
delivers α = 1, T-independent degree 32, and no spectral-bounds step.
