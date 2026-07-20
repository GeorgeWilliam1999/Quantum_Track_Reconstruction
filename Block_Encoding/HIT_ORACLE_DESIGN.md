# The hit-level geometry oracle — a block encoding priced by hits, not by matrix entries

**Status: design + cost model (2026-07-19).** This is the "if the literature doesn't
fit, build the way forward" deliverable. It concretises resource-direction **D5**
(`QSVT/Paper_planning/RESOURCE_REDUCTION.md`) using the machinery of the four
benchmarked papers where it helps (Camps oracle framework, dictionary/LCU
subnormalization analysis, transposition compilation as the fallback cost anchor).

## 1. The information-theoretic wall the literature hits

Every circuit that encodes an *event-specific* matrix must contain the event's
information. All four papers price the circuit **per matrix nonzero**:

| method | data the circuit carries | gates on our A |
|---|---|---|
| FABLE | all N² entries | O(N²) = O(T⁴) rotations |
| S-FABLE / LS-FABLE | Hadamard-domain entries / nonzeros | O(nnz)–O(N²) rotations, **plus O(1) irreducible error on binary data** |
| Camps Thm 4.1 | index oracle c(j,ℓ) + values | O(nnz·n) via transposition compilation (no arithmetic structure in segment indices) |
| Dictionary | class membership (SBM) | depth O(log ns) but **O(n²s) ancillas**; low-ancilla route = transpositions again, O(nnz·n) |

For the segment matrix, nnz ≈ O(T²) (step, clean; worse noisy). So every
literature route pays **Ω(T²) gates per walk call** — the quantum circuit is as
big as the classical matrix.

**But A is a derived object.** The event is 5·T hits; the matrix is a *function*
of them: `C[(a,b),(b',c)] = δ_{b,b'} · [angle(a,b,c) < ε]` (step) or
`erf`-weighted (erf kernel). The information content is **O(T log T) bits, not
O(T²)**. A circuit that queries *hits* instead of *entries* can therefore be
exponentially smaller in the per-call sense: **O(T) vs O(T²)**, with the same
subnormalization class as the best literature method (α ≈ occupancy, like the
dictionary's Σ|A_l|).

## 2. Construction

Registers: segment index factored as `(layer g : 2 bits) ⊗ (hit i in layer g :
⌈log T'⌉ bits) ⊗ (hit j in layer g+1 : ⌈log T'⌉ bits)` — same total width as
the current flat index (n_seg ≈ 4T²), just *structured*.

Within each layer, hits are **classically pre-sorted by slope/position**
(O(T log T), once per event — compare FABLE's O(N² log N) = O(T⁴ log T)).

**Index oracle O_c (slot ℓ = "ℓ-th continuation candidate"):**
`c((g,a,b), ℓ) = (g+1, b, base(g,a,b) + ℓ)` — register arithmetic:
1. increment the 2-bit layer field (Camps' L-shift, O(1)),
2. copy field b into the first-hit field (n/2 CNOTs),
3. `base` = index of the first layer-(g+2) hit inside the ε-window of the
   extrapolated line through (a,b): one fixed-point extrapolation
   (2 subtractions + 1 multiply-add on ~16-bit registers) plus **one QROM
   lookup of the inverse-CDF of sorted hit positions** — a table of 5T (not 4T²)
   entries — followed by an adder for +ℓ.

**Form of the encoding (important subtlety).** Camps' in-place `O_c` requires
`c(·,ℓ)` injective per slot — false here: segments `(a,b)` and `(a',b)` sharing
an endpoint have overlapping continuation windows. The construction therefore
uses the **state-preparation-pair / Szegedy form** (Camps Thm 5.1, Gilyén
Lemma 48): `U_R |0^n⟩|j⟩ = (1/√w) Σ_{ℓ<w} |window_ℓ(j)⟩|accept-flag⟩|j⟩`, and
`U = U_L† · SWAP · U_R` block-encodes `C/w` — no injectivity needed, at the
price of a second n-qubit register (which the qubitization walk carries anyway).
The window enumeration + accept flag below implement U_R.

**Value oracle (the accept flag):** re-compute the kink angle from hit
coordinates (two QROM reads + fixed-point arithmetic), then
- step kernel: **comparator** against ε → flag qubit (value −1 fixed rotation),
- erf kernel: rotation by `arccos(erf-weight(angle))` evaluated arithmetically —
  **the erf kernel costs no extra data**, unlike the dictionary method where
  continuous values explode the class count to Σ|a_ij|.

Diagonal (γ+δ)I is absorbed into the polynomial (shift), not encoded.

**Subnormalization:** α = w_pad = 2^⌈log₂ w⌉ where w = max hits inside an
ε-window ≈ local occupancy (measured Δ on clean events: 2–3; noisy: O(10)).
Same class as dictionary/Camps-on-C — near the ‖C‖ lower bound — and ~half the
span-based α of the exact dilation is unreachable anyway (α ≥ ‖C‖₂ theorem,
dictionary paper Remark 1).

## 3. Cost model (per walk call)

| piece | gates (CX-equivalents) | notes |
|---|---|---|
| layer shift + field copy | O(n) | Camps shift circuits |
| extrapolation + compare | ~30 × 16-bit ops ≈ 500 | fixed-point |
| QROM (2 reads, 5T entries) | ~8·5T linear, or O(√T) T-depth select-swap | the dominant term |
| uncompute | ×2 | |

**Per call: O(T) vs O(T²·n) for the best literature route** (transposition-
compiled dictionary/Camps). At T=1000: ~4·10⁴ vs ~5·10⁷ CX per call —
three orders of magnitude — while α stays at the occupancy floor, so the comb
degree penalty vs the (simulation-only) dilation is the same ×(α/α_dil) ≈ 2–3
factor both routes pay.

The same oracle applied at degree 1 (T₁ = one walk call) is a **hardware-route
1BQF**: cos-notch semantics with O(T) gates instead of the native Givens
product's O(nnz·n) = O(T²·n).

## 4. What it does NOT fix (honesty)

- The **floor theorem** (spectrum-identical false structures) is representation-
  independent — no encoding touches it.
- α is still ≈ occupancy: the degree multiplier vs the exact dilation (×2–3)
  is inherent to *any* gate-efficient LCU-class encoding (α ≥ ‖C‖₂ ≈ ρ(C)).
- Noise raises occupancy w and hence α — same trend as every sparse method.
- The classical sort must be redone per event (O(T log T)) — negligible.

## 5. Validation ladder (next steps)

1. numpy oracle emulator: verify `base()` window covers all accepted couplings
   on store events (clean + noisy, step + erf), measure w distribution vs T.
2. qiskit prototype of O_c on a T=3 event (12–14 qubits), block-extraction test
   against C exactly as `01_verify_and_measure.py` does for the others.
3. Full resource table vs the measured Δ/χ′/nnz curves from `02_structure_scaling`.
4. Wire into the walk/LCU layer of `lhcb_velo_toy.solvers.quantum.QSVT` as a
   third backend (after `solve_statevector` and the dense-dilation circuit).
