# Reducing QSVT qubits & two-qubit gates — research directions to compete with the 1BQF

**Goal.** The current QSVT build (LCU-of-Chebyshev over the qubitization walk) delivers
a strictly better segment-metric frontier than the 1BQF (false rate at 99 % efficiency
~1 % vs 44 % at T=400), but at a resource premium: ~5 extra ancillas, a degree-$d$ depth
multiplier, and *controlled* walk operators. This note is the research agenda for closing
that gap — making the overhead a **small constant**, not a scaling, and removing every
**superlinear** overhead.

## 0. The resource gap to close (the honest scoreboard)

| axis | 1BQF (TrackHHL) | QSVT — current LCU build | source of the gap |
|---|---|---|---|
| system qubits | $\lceil\log_2 4T^2\rceil$ | $\lceil\log_2 4T^2\rceil$ | same (global solve) |
| ancillas | 2 (clock + flag) | $1 + \lceil\log_2(d{+}1)\rceil \approx 7$ at $d{=}40$ | **LCU register** |
| walk calls (depth) | 1 ($\cos = T_1$) | $d \times \pi/(4\sqrt P)$ | richer filter (irreducible $d$) + amplification |
| per-call 2-qubit cost | Givens product (sparse, cheap) | controlled-$W^{2^j}$ (controlled dilation block-encoding) | **control + dense dilation** |
| amplification | $\pi/(4\sqrt P)$, $P\propto1/T$ | $\pi/(4\sqrt P)$, $P\propto1/T$ | same (uniform $b$ over $4T^2$) |

The 1BQF **is** the $d=1$ member of the same family, so QSVT can never be strictly cheaper
at a fixed primitive. The realistic target: **(i)** match the 1BQF's *width* and *per-call
2-qubit primitive*, **(ii)** reduce the depth multiplier $d$ toward its floor (~10–16), and
**(iii)** kill the superlinear $\sqrt T$ amplification. Six directions, ordered by impact.

---

## D1 — Eliminate the LCU register: genuine QSP / generalized QSP
*(the single biggest win — hits both axes at once)*

The LCU-of-Chebyshev needs $\lceil\log_2(d{+}1)\rceil$ ancillas **and** a SELECT of
*controlled* $W^{2^j}$. True QSVT/QSP (Gilyén–Su–Low–Wiebe 2019) realises the same $p(A)$
with **one** signal-processing ancilla and $d$ **uncontrolled** walk applications
interleaved with single-qubit $Z$-rotations.

- **Qubits:** drops $\sim6$ (back to 1BQF-level, $\text{sys}+2$).
- **2-qubit gates:** the walks become *uncontrolled* — controlling a multi-qubit unitary is
  the dominant 2-qubit overhead, so this is a large per-call saving.
- **Cost:** need the phase-angle sequence (pyqsp / QSPPACK) and definite-parity handling
  (even/odd split, or the generalised-QSVT linear combination).

This is the "as-built → as-designed" upgrade already flagged as the next build step; it is
the prerequisite for a fair resource comparison.

## D2 — Reuse the 1BQF's own primitive: generalized QSP over $e^{-iAt}$
*(the biggest 2-qubit-gate win — "compete with the 1BQF" most literally)*

Do **not** block-encode $A$ at all. The 1BQF already has a cheap, sparse, hardware-validated
circuit for $e^{-iAt}$ (the two-level / Givens product: one controlled-$R_X$ per
$\varepsilon$-window interaction pair). **Generalized QSP** (Motlagh–Wiebe, PRX Quantum
2024, arXiv:2308.01501) builds a trigonometric polynomial $p(e^{iAt})$ with a **single
ancilla** and $d$ **uncontrolled** $e^{iAt}$ calls.

- The comb response is a function of $\lambda$; over $U=e^{-iAt}$ it is a Fourier
  (trig-polynomial) series in $\lambda t$ — exactly what generalized QSP implements, and
  $\cos(\lambda t)$ is precisely the quantity the 1BQF's Hadamard test already produces.
- **Result:** the QSVT circuit becomes *"the 1BQF circuit, repeated $d$ times, with
  single-qubit rotations between"* — **same width ($\text{sys}+1$), same 2-qubit primitive,
  depth $=d\times$(1BQF depth)**. The only overhead vs the 1BQF is the constant $d\sim12$–16.
- Reuses the TrackHHL primitive that already ran on Quantinuum H2 / IBM Heron — so the
  hardware path is inherited, not rebuilt.

**D1+D2 together** = generalized QSP over the 1BQF evolution: 1BQF width, 1BQF 2-qubit
primitive, $d$ uncontrolled repetitions. This is the headline competitive construction.

## D3 — Block-diagonal (cluster) decomposition: break $\log_2 4T^2$ and kill amplification
*(the biggest system-qubit win — and it removes the $\sqrt T$)*

$A$ is **block-diagonal over connected components** of the continuation graph. Pure-P4
true clusters are 4 segments; contaminated tangles are 5–16 (measured). Solve
**cluster-by-cluster**: the quantum register holds one cluster, $\lceil\log_2(\text{max
cluster})\rceil \approx 4$–5 qubits, **independent of $T$**.

- **System qubits:** $22 \to \sim5$ at $T=1000$ — a 4× cut *and constant in $T$*.
- **Amplification:** the $P\propto1/T$ came entirely from the uniform $|b\rangle$ diluting
  over $4T^2$ segments. Per cluster, $|b\rangle$ is concentrated $\Rightarrow P=\mathcal O(1)$,
  so the $\sqrt T$ amplitude-amplification overhead **vanishes**.
- **Hardware:** each cluster circuit is tiny — the memory already identifies an 8–9-qubit
  contaminated component as a near-term hardware candidate.
- **Honesty (WP6):** connected-components is $O(n)$ classical preprocessing (cheap,
  geometry-driven) — but it benefits the 1BQF identically, so the *comparison at fixed
  cluster* is the apples-to-apples one, and the quantum content (the per-cluster spectral
  filter) is unchanged. This reframes the story from "one global log-qubit solve" to "many
  tiny independent solves," which is both cheaper and more NISQ-realistic.

## D4 — Lower the degree: minimax / Lin–Tong optimal filtering
*(fewer walk calls → fewer 2-qubit gates and fewer LCU qubits)*

nb05 shows $d\approx16$ suffices to $T=700$, but the Chebyshev least-squares fit is
degree-inefficient and non-monotonic (intermediate degrees cliff). A **minimax (Remez)**
design, or the **Lin–Tong optimal eigenstate filter** (arXiv:1910.14596, provably minimal
degree for a target gap-resolution), reaches the comb resolution at lower $d$.

- The comb must resolve a $\sim0.2$-wide gap at 4 lines; Lin–Tong gives $d\sim
  (1/\text{gap})\log(1/\varepsilon)$, and a **rational / 4-projector** design may beat the
  single polynomial.
- Lower $d$ shrinks depth ($\propto d$) **and** the LCU register ($\lceil\log_2(d{+}1)\rceil$,
  if D1 not yet applied). The irreducible floor is $d\gtrsim7$ (3 interior zeros between 4
  passes + band edges).

## D5 — Cheaper block-encoding via the detector-geometry oracle
*(fewer 2-qubit gates per call, if a block-encoding is kept instead of D2)*

$A = sI - C$ has a nonzero pattern fixed by geometry: a segment couples only to neighbours
sharing a hit in an adjacent layer, with values in $\{s, -1\}$. So the sparse-access oracles
are cheap:
- **Column-index oracle** = an *arithmetic* function of segment indices (hit/layer
  arithmetic), not a generic QRAM lookup.
- **Value oracle** = a fixed phase (single nonzero off-diagonal value).

This makes the qubitization block-encoding $O(\text{polylog}\,n)$ with small constants —
comparable to the 1BQF's Givens product, and far below the dense-dilation build
($O(4^{n_s})$, simulation-only). The research content is the explicit geometry-oracle
construction and its gate count.

## D6 — Parity halving and amplification-free readout
*(constant-factor cleanups)*

- The comb is (after centring on $s$) approximately **even** in $(\lambda-s)$, so a
  definite-parity QSP halves the angle/rotation count.
- **Non-uniform $|b\rangle$:** a classical prior that pre-weights $b$ toward likely-true
  segments raises $P$ globally (an alternative to D3 when a global solve is required).

---

## Synthesis — the competitive construction

**D2 + D3 + D4**: a *per-cluster generalized-QSP solver using the 1BQF's $e^{-iAt}$ primitive
at minimax-minimal degree*:

| axis | 1BQF (global) | this construction |
|---|---|---|
| qubits | $\lceil\log_2 4T^2\rceil + 2 \approx 24$ @ T=1000 | $\lceil\log_2(\text{max cluster})\rceil + 1 \approx 5$, **const in T** |
| 2-qubit primitive | Givens product | **identical** (the 1BQF's $e^{-iAt}$) |
| walk calls | 1 | $d \sim 12$–16 (minimax), **no $\sqrt T$ amplification** |
| metric frontier | 44 % false @ 99 % eff (T=400) | **~1 %** |

The residual overhead vs the 1BQF collapses to a **single constant depth multiplier
$d\sim15$ per cluster** — the irreducible price of a 4-line filter — while the *width drops
below the global 1BQF* and the amplification overhead disappears. That is the regime where
QSVT competes: same hardware primitive, constant depth premium, strictly better physics.

## Validation ladder (cheap → expensive)

1. **Minimax/Lin–Tong degree** (classical, numpy): re-run the nb05 sweep with a Remez comb;
   target $d$ at which far@99 %-eff floors. *(D4)*
2. **Per-cluster resource model** (classical): histogram cluster sizes vs T from the store;
   derive the width law and per-cluster $P$. *(D3)*
3. **Generalized-QSP-over-$e^{-iAt}$ prototype** (qiskit, small cluster): build the comb as a
   trig polynomial of the 1BQF evolution; validate against `solve_statevector`; count
   2-qubit gates vs the dilation build. *(D1+D2)*
4. **Geometry-oracle block-encoding** gate count (analytic + small qiskit). *(D5)*
5. **Hardware candidate**: one contaminated 8–9-qubit component, generalized-QSP, on the
   inherited TrackHHL backend. *(WP3)*

## References

- Gilyén, Su, Low, Wiebe, *QSVT and beyond*, STOC 2019 (arXiv:1806.01838) — QSP, one ancilla.
- Motlagh, Wiebe, *Generalized Quantum Signal Processing*, PRX Quantum 2024 (arXiv:2308.01501)
  — single-ancilla trig-polynomials of a unitary; the $e^{-iAt}$ reuse path.
- Lin, Tong, *Optimal polynomial based quantum eigenstate filtering* (arXiv:1910.14596) —
  minimal-degree band filters.
- Low, Chuang, *Hamiltonian simulation by qubitization* (arXiv:1610.06546) — the walk operator.
- Internal: `QSVT/Segment_level_studies/05_circuit_depth_and_qubit_scaling.ipynb` (the measured
  baseline this note proposes to beat); the first-principles math Notion page (resolution law,
  $1/T$ derivation, floor theorem).
