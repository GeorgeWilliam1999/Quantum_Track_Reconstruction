# Paper plan — QSVT spectral-filter engineering for particle track reconstruction

**Follow-up to:**
1. *A quantum algorithm for track reconstruction in the LHCb vertex detector*,
   Nicotra et al., JINST **18** P11028 (2023) — the Ising-like segment Hamiltonian;
   classical matrix inversion + HHL; exponential-speedup potential, limited by
   Hamiltonian-simulation depth and readout.
2. *TrackHHL: The 1-Bit Quantum Filter for particle trajectory reconstruction*,
   Chiotopoulos, Nicotra, Scriven, et al., arXiv:2601.07766 — reformulates tracking
   from matrix inversion to **binary ground-state filtering**; single-ancilla
   spectral threshold (the 1BQF); O(√N log N) gates; benchmarked on Quantinuum H2
   and IBM Heron.

**This paper (working thesis).** Track finding on the segment Hamiltonian is not a
linear-solve problem and not a one-notch problem — it is a **spectral
discrimination** problem with **two design knobs**: the spectral response and the
operator itself. The 1BQF is the degree-1 member of the Chebyshev/QSVT filter
family; generalising the polynomial turns the filter into a design space. We
(i) characterise the failure modes of inversion and of the 1-bit filter as
*spectral populations* of the Hamiltonian, (ii) prove what any spectral filter can
and cannot remove (the same-length-degeneracy floor), (iii) engineer the optimal
response — a **line-comb inverse** that passes only the geometry-pinned true-track
eigenvalue lines — and the optimal operator — the **ε-windowed bifurcation term**,
a sparse, constant-diagonal modification that separates the hit-topology errors
the spectrum cannot see, lifting the floor — and (iv) show on the standard
campaign that the co-designed system **dominates both classical inversion and the
1BQF simultaneously** at efficiency-first working points, at a quantified,
constant (degree × walk-call) cost with ~30-qubit width at HL-LHC multiplicity.

---

## 1. The narrative arc (Introduction skeleton)

| Paper | Verb | Circuit | Response f(λ) | Limitation exposed |
|---|---|---|---|---|
| JINST 2023 | **invert** | full HHL: multi-qubit QPE + rotation | ≈ 1/λ everywhere | depth (QPE precision), readout; froth passed at 1/λ |
| TrackHHL 2026 | **filter** | 1 clock qubit, Hadamard test | cos(λt/2): one notch at λ=γ+δ | erases only the *isolated* false; halves outer true segments; coupled false untouched |
| **this paper** | **engineer** | d walk calls + LCU/QSVT phases | designed polynomial p(λ) | the provable floor: spectrum-identical false (same-length bridges / contaminated clusters) |

One-line pitch: *"From inverting the matrix, to filtering one bit, to engineering
the spectral response AND the Hamiltonian — with the theorem that says which knob
each error type needs."*

## 2. Core contributions (the claims list)

C1. **Spectral taxonomy of tracking failure modes.** Closed-form eigen-atlas of the
    segment Hamiltonian's cluster types: true P4 chains (λ = s − 2cos(kπ/5),
    s = γ+δ), isolated false (λ = s exactly), false P_m bridges (interleaving
    lines), hubs K(1,m) (λ = s ± √m, plus m−1 degenerate modes on s). Why the
    1BQF works (the isolated bulk is a degenerate line on its notch) and why it
    stops there (the coupled false straddle the true band; no evolution time
    reaches them — the two-notch no-go scan).

C2. **The filter-design theorem (the floor).** Any spectral method — HHL, 1BQF,
    multi-bit QPE, QSVT, classical eigenfilter — acts as a function of the
    spectrum, so it cannot separate spectrum-identical populations: a false
    bridge of the true track length (P4 ≅ P4, identical block + identical
    uniform-b projection) and, dually, a true track inside a contaminated
    cluster. Quantified empirically vs multiplicity and acceptance window
    (0 % at T ≤ 200, 0.6 % at T = 400/ε = 4 mrad; ~2 % efficiency analogue at
    T = 400). Anything *not* on the floor is removable by polynomial design.

C3. **The line-comb inverse.** The exploitable structure is the *discreteness* of
    the true spectrum (detector geometry pins true tracks to four lines).
    Production filter: narrow ~1/λ passes at the P4 lines only, γ-aware
    (rigid shift with s), degree ~40. Includes the negative result that makes it
    credible: the contiguous band-limited inverse — the "obvious" design, and our
    own first attempt — collapses at density (false rate 5 % → 44 % from T = 100
    → 400) because tangles fill any band; the comb fixes it because lines admit
    no between-line modes.

C4. **A fair comparison methodology.** Solver-specific, efficiency-first
    thresholds (τ placed at the (1−e) true-amplitude quantile) + ROC, instead of
    judging all solvers at the classical cut. This reverses naive conclusions:
    at ≥99 % efficiency the 1BQF pays 44 % false rate at T = 400 (its halved
    outer-true band sits inside its coupled-false population) while the comb
    pays 1.3 % — *below classical inversion's 1.6 %*.

C5. **Circuit realisation & resource accounting.** LCU-of-Chebyshev over the
    qubitization walk operator (no phase-angle synthesis needed; the 1BQF is
    literally the d = 1 member, cos = T₁(walk)): exact dilation block-encoding,
    ⟨0|W^k|0⟩ = T_k(X), m = ⌈log₂(d+1)⌉ LCU qubits. Width
    ⌈log₂ 4T²⌉ + 1 + m: **25 qubits at T = 200, 29 at T = 1000** — width is not
    the obstacle at HL-LHC scale; cost is depth (d walk calls vs 1) and
    post-selection (success ~10⁻³, ~1/√p amplitude-amplification rounds).
    Exact-simulation validation chain: explicit circuit ≡ streaming gate
    sequence ≡ matrix-free semantics ≡ eigendecomposition (10⁻⁹), to
    T = 40 / 20 qubits locally.

C6. **Production-scale benchmark on the standard campaign.** Three solvers
    (classical MINRES, 1BQF, QSVT comb) on identical events/Hamiltonians in the
    same store: γ ∈ {1,2,3} × hit-drop {0,1 %} × T = 10…400 × reps; segment
    efficiency/false-rate 2×2, amplitude and angular distributions, working
    points. (Stretch: real Run-3 VeLo events via the existing Run3_Verification
    loader.)

C7. **Operator–response co-design: the ε-windowed bifurcation term lifts the
    floor.** The floor populations (C2) are spectrum-identical *under the base
    Hamiltonian* — but they are not hit-topology-identical: every one of them
    involves near-collinear competitors sharing a hit. The **ε-windowed
    Denby–Peterson fork term** (from `Toy_Characterisation/Bifurification`)
    adds a repulsive coupling +β on exactly those pairs,
    A′ = (γ+δ)I − C + βB_ε, while **staying sparse** (nnz(B_ε) = O(n_seg) —
    the block-encoding/walk cost is unchanged) and **constant-diagonal** (every
    segment touches two hits ⇒ 1BQF/QSVT compatibility preserved). It moves the
    error configurations the filter cannot see: cleanly separates the
    competing-continuation errors in the operator, leaving the comb to do the
    spectral discrimination. Includes its own negative result: the *dense* fork
    (all co-hit pairs) breaks both the sparsity invariant (O(T³)) and the 1BQF
    (AUC → 0.55, ~20× slower) — the acceptance window is what makes the term
    admissible, mirroring how the line-comb (not the band) makes the filter
    admissible. **Measured** (deep dive, `01_codesign_deep_dive.ipynb`):
    - *Surgical and provably safe:* the fork touches 1.8 % of segments at
      T = 400; **0/1564** pure-P4 true segments carry a fork edge ⇒ the clean
      true spectrum is exactly β-invariant (stays on the comb lines, any β).
    - *Contamination recovery:* trajectory analysis shows the mechanism — the
      true-weighted modes of contaminated clusters migrate **into** the comb
      bands as β grows. ε = 2 mrad, T = 400: misses 36 → 20 (eff 97.8 → 98.9 %)
      at comb far = 0; ε = 4 mrad, T = 400: comb eff 88.5 → 94.3 % and
      classical far 19.4 → 11.1 % (β 0 → 1).
    - *The honest limit:* an **isolated** floor bridge whose competitors lie
      outside the fork window is β-invariant (harvested example: exact P4
      spectrum, zero fork edges, flat trajectories) — the ε-window that makes
      the term sparse also bounds its reach. **The fork window is a design
      parameter separate from the acceptance window** (a wider B-window stays
      sparse while reaching the bridges' competitors): quantifying that
      decoupling is the core of WP7.
    - *The β cost:* the spectral half-span grows only where the fork acts
      (T ≤ 200: envelope pinned at the P4 extremes, no widening at all;
      T = 400: 2.8 → 4.7 at β = 1 ⇒ comb degree 40 → ~67 for fixed line
      resolution, ≤ 1 extra LCU qubit). Width, oracle sparsity
      (nnz(B_ε) ≤ 5 % of nnz(A)) and success probability (∝ 1/T) are unchanged.

## 3. Theory section content

- The segment Hamiltonian and its cluster decomposition (block-diagonal over
  connected components of the continuation graph); solution in the eigenbasis
  x_i = Σ_k (β_k/λ_k) u_k(i).
- The filter family: x^f = Σ_k β_k f(λ_k) u_k; classical inversion f = 1/λ;
  1BQF f = cos(λt/2) derived from the Hadamard test; QSVT/LCU f = p(λ),
  any bounded polynomial of definite parity (or LCU sum without parity
  constraint, at an ‖c‖₁ success-probability cost).
- Closed-form spectra of the canonical clusters (P_m chains, K(1,m) stars,
  isolated 1×1) — from the existing amplitude atlas; the QPE-phase picture
  φ = λ/2s linking back to TrackHHL's threshold language.
- The no-go results: (a) two-notch scan — no second evolution time helps the
  1BQF; (b) the floor theorem (C2) with the graph-isomorphism argument;
  (c) the band-design failure as a corollary (any *interval* admits tangle
  modes whose density grows with multiplicity).
- The comb design: target response, Chebyshev fit, degree-vs-gap scaling
  (d ≈ 2/Δ̃ with Δ̃ the rescaled line spacing), realizability (max|p| ≤ 1),
  γ-awareness, the hw trade-off (hw → band reproduces the failure; hw → 0
  needs degree → ∞ and collides with noise broadening).
- Qubitization/LCU construction and resource theorem (C5); relation to the
  1BQF's O(√N log N) gate count — the comb costs a *constant factor d* more
  walk calls at the same width scaling, preserving the sparse-A invariant.
- The modified Hamiltonian (C7): the Denby–Peterson occupancy penalty, its two
  diagonal treatments (both constant-diagonal because every segment touches
  exactly two hits), the ε-window restriction B_ε and its sparsity bound;
  the analytic single-fork example (x_f = (δ − βx_s)/(γ+δ) → 0); why the dense
  fork fails (uniform down-scaling classically; lifts the false bulk off the
  notch quantumly); spectrum-widening vs β and the implied comb-degree cost.
  (All derivations exist in `Bifurification/bifurcation_hamiltonian.md` §1–9.)

## 4. Evidence inventory — what exists vs what is missing

### Already produced (repo paths; regenerate at paper quality)
| Paper figure (working title) | Source | Status |
|---|---|---|
| Fig 1: failure-mode spectral atlas (true lines vs false populations) | `QSVT/Initial/00_spectral_census` (`spectral_overlap`, `gap_degree`) + `Toy_Characterisation/Bifurification/04` (`fp_atlas_T200`, `phase_filter_map`) | done (toy, T=200) |
| Fig 2: 1BQF no-go (two-notch scan) | `Bifurification/04` (`two_notch_scan`) | done |
| Fig 3: filter responses f(λ): 1/λ, cos, band, comb over the spectrum | `QSVT/Initial/01` + comb (`qsvt_efficiency_dip_diagnosis` style) | needs a clean combined figure |
| Fig 4: the 2×2 (3 solvers, T ≤ 400, γ=3 clean + 1 % drop) | `QSVT/Segment_level_studies/01` (`qsvt_2x2_*`) | done |
| Fig 5: working points — ROC @ T=400 + far@99 % eff vs T | `QSVT/Segment_level_studies/01` (`qsvt_working_points`) | done |
| Fig 6: amplitude distributions true/false × 3 solvers @ T=400 | `QSVT/Segment_level_studies/02` (`qsvt_amplitudes_T400`) | done |
| Fig 7: band-design failure vs comb (the negative result) | numbers exist (5.2/15.4/43.6 %); needs a dedicated comparison figure | **to make** |
| Fig 8: qubit-width scaling + local exact-simulation ceiling | `QSVT/Initial/05` (`first_build_qubits`) | done |
| Fig 9: efficiency-dip / floor quantification (contaminated clusters; irreducible fraction vs T, ε) | `QSVT/Segment_level_studies/01` §3 + `QSVT/Initial/00` (`irreducible_floor`) | done |
| Fig 10: the ε-windowed fork term — sparsity, targeted suppression, dense-fork failure | `Bifurification/03` (`eps_fork_sparsity`, `eps_vs_dense_classical`, `eps_fork_quantum`) + `Bifurification/02` (`classical_vs_quantum_beta`) | done |
| Fig 11: co-design — eigenvalue trajectories λ(β) (recovery vs the invariant bridge), ε=4 mrad floor test, spectrum/degree trade-off | `Paper_planning/01_codesign_deep_dive` (`codesign_spectral_trajectories`, `codesign_event_level`) | done (toy; ε_B scan = WP7) |
| Fig 12: scaling — sparsity O(n_seg), width to 29 q at T=1000, success ∝ 1/T (measured) | `Paper_planning/01_codesign_deep_dive` (`codesign_scaling`) | done |
| Tab 1: solver comparison at fixed τ and at working points | nb01 tables | done |
| Tab 2: resource accounting (qubits, walk calls, success prob, AA rounds) vs 1BQF and HHL | `QSVT/Initial/03` | done (extend) |

### Gaps — work packages before drafting
- **WP1 (robustness):** comb noise scan — σ_scatt, σ_res, hit-drop sweeps
  (re-run `Initial/04` with the comb; hit-drop is the known weak spot: P4 → P3
  off-comb; quantify and present the τ-retune / P3-satellite-line mitigation and
  its degeneracy cost). Also ε-window sweep at fixed comb.
- **WP2 (scale):** extend the qsvt campaign to T = 700/1000 (classical partners
  exist; quantum/1BQF does not — state that QSVT matrix-free semantics scale
  where 1BQF statevector did not). Optional: phi_max/density variations.
- **WP3 (shots & noise model):** the paper-2 reviewers will ask: sampling
  readout (finite shots vs exact statevector), and a hardware-noise study.
  Minimum: shot-noise scaling of the working points (success prob 10⁻³ →
  shots budget; amplitude amplification accounting). Stretch: small comb
  instance (one P4 + parasite cluster, ≤ 8–10 qubits after the dilation) on
  IBM Heron / Quantinuum H2 — the direct continuation of paper 2's hardware
  section. Requires pyqsp (or QSPPACK) phase angles on a networked machine, or
  the LCU circuit as-built.
- **WP4 (theory hardening):** write the floor theorem properly (graph
  isomorphism + uniform b ⇒ identical filtered amplitudes for ANY f);
  degree-vs-gap and degree-vs-noise-broadening scaling; formal statement that
  the comb is optimal among spectrum-only filters up to the floor.
- **WP5 (real data, stretch):** run the comb on the Run-3 VeLo events
  (`Run3_Verification` loader exists; toy taxonomy already validated there) —
  even a single figure would lift the paper from toy-only.
- **WP6 (fairness/baselines):** classical baseline beyond MINRES-at-τ: the
  classical eigenfilter (apply the comb classically — same response, no quantum;
  makes the quantum claim honest: the *quantum* content is the resource scaling,
  not the response itself). This must be stated clearly to pre-empt the
  "dequantisation" referee.
- **WP7 (co-design, C7):** the comb × ε-fork study at scale. Core new question
  from the deep dive: **decouple the fork window ε_B from the acceptance ε** —
  scan ε_B > ε so the term reaches the isolated bridges' competitors while
  staying sparse (measure nnz(B_{ε_B}) growth and the bridge-floor reduction vs
  ε_B, alongside the β scan); the β trade-off curve (spectrum widening →
  comb-degree cost, optimal (β, ε_B)); whether the fork restores a *fixed*
  per-solver τ at T = 400 (the working-point τ drop to ~0.08 was driven by the
  contaminated clusters); store-campaign variant with A′ = A₀ + βB_{ε_B} (new
  ham provenance keys) so the 2×2 and working points can be reported for the
  co-designed system. Deep dive done (`01_codesign_deep_dive.ipynb`: spectral
  mechanics, ε=4 mrad floor test, scaling to T = 1000, Condor feasibility);
  script basis: `Bifurification/bif.py` (`fork_graph_eps`) +
  `QSVT/qsvt_store_campaign.py`.

### Simulation reach (measured — scopes WP1/2/3/7; details in `01_codesign_deep_dive.ipynb` §C–D)

| problem | method | reach | per-job cost |
|---|---|---|---|
| physics campaigns (2×2, working points, β/ε_B/γ/noise scans) | matrix-free exact semantics | **T = 1000+** (n = 4 M segments, 29 qubits-equivalent) | ~1.6 s solve + ~2 min build, ≲ 8 GB → a 1 000-solve campaign ≈ 50–80 CPU-h on Condor |
| exact explicit-circuit validation | dense-walk statevector (streaming) | T ≈ 64 (21 q) comfortably; T ≈ 90 (22 q) hero job | n_s=14: ~1 h + ~60 GB (bigmem); n_s=15: days + ~200 GB |
| shot-based + hardware-noise studies | Aer, cluster-level circuits | 5–60-segment components → 8–13 qubits | seconds–minutes per 10⁴ shots; full noise sweeps farmable |
| hardware demo (WP3) | Quantinuum H2 / IBM Heron | one contaminated component, comb degree ~12–16 | **8–9 qubits** — realistic continuation of paper 2's hardware section |
| full-event circuit at HL-LHC scale | — | 29–30 q *width* but 2ⁿ statevector | not classically simulable — the point of the paper |

## 5. Proposed outline

1. **Introduction** — HL-LHC tracking; the arc invert → filter → engineer (§1
   table); contributions.
2. **The segment Hamiltonian and its spectrum** — model recap (from papers 1–2),
   cluster decomposition, the failure-mode atlas (C1).
3. **Spectral filters and their limits** — filter family; why the 1-bit notch
   works and stops; the two-notch no-go; the floor theorem (C2).
4. **Engineering the response** — band design and its density failure (C3
   negative result); the line-comb inverse; degree/realizability; γ-awareness.
5. **Engineering the operator** — the ε-windowed bifurcation term (C7): the
   occupancy penalty, the acceptance-window restriction that keeps it sparse
   and constant-diagonal, the dense-fork negative result, and the co-design
   result (floor recovery at pinned-zero false rate; the β/spectrum-widening
   trade-off).
6. **Quantum realisation** — qubitization/LCU construction; 1BQF as d = 1;
   resources (C5) — unchanged under A′ (sparsity + diagonal preserved);
   validation chain; (WP3 hardware/shots).
7. **Results** — campaign setup; the 2×2; amplitudes/angles; working-point
   methodology (C4) and the headline table; co-design results (WP7);
   robustness (WP1); scale (WP2); (WP5 Run-3).
8. **Discussion** — what the floor means for spectral methods at HL-LHC
   density; the three-knob division of labour (operator: hit-topology errors;
   response: spectral errors; occupancy/track-level: the rest); dequantisation
   honesty (WP6); outlook (fault-tolerant costing, parity/phase-angle QSVT,
   amplitude amplification).
9. **Conclusion.**

## 6. Title candidates

- *Engineering the spectral response: QSVT filters for particle track reconstruction*
- *From one bit to a comb: polynomial spectral filters for tracking at the HL-LHC*
- *TrackQSVT: spectral-filter engineering for particle trajectory reconstruction*
- *The limits and design of quantum spectral filters for track finding*

## 7. Venue & logistics

- **Venue:** JINST (continuity with paper 1; instrumentation audience) or
  *Quantum* / PRA-applied if the theory content (floor theorem + QSVT
  construction) is foregrounded. Suggest: target JINST, arXiv first.
- **Data/code:** both repos are public; the qtrk store campaign is reproducible
  (`QSVT/qsvt_store_campaign.py`); pin a tag at submission.
- **Figures:** regenerate at paper style (consistent palette: classical green /
  1BQF red / QSVT purple as in the notebooks; PDF 600 dpi already standard).
- **Author tasks:** to be assigned — WP1–2 are mechanical (this repo), WP3
  hardware needs the paper-2 hardware authors, WP4 is a theory write-up, WP5
  uses the existing Run-3 loader.

## 8. Anticipated referee pressure (prepare answers in-text)

1. *"The comb is classically simulable — where is the quantum advantage?"* →
   WP6: the response is solver-agnostic; the quantum content is applying it in
   O(poly log N) width with d sparse walk calls vs classical O(N) eigen/solve
   work — the same advantage logic as papers 1–2, now with a response that is
   actually *better* than inversion, not an approximation of it.
2. *"Success probability 10⁻³ kills you."* → amplitude amplification (~30
   rounds), and the success probability is dominated by the signal fraction
   4T/4T² = 1/T — the *same* readout bottleneck as papers 1–2; the comb does not
   worsen it (cf. 1BQF 0.003 vs comb 0.002 at T = 100).
3. *"Degree-40 polynomials are fragile under hardware noise."* → WP1/WP3; line
   width hw = 0.18 is the noise budget; degree-vs-broadening scaling in §4.
4. *"Toy model only."* → WP5 Run-3 figure; geometry-pinned line structure is the
   only model-specific assumption, and it holds for fixed-layer-count detectors.
5. *"The threshold is tuned per solver — unfair."* → C4 is the fair version
   (efficiency-first working points + full ROC for every solver, including the
   baselines).

## 9. Suggested first steps

1. Approve/adjust this plan (especially venue + WP3 hardware scope).
2. WP1 + WP2 runs (1–2 days of local compute, scripts exist).
3. Fig 7 (band vs comb) and Fig 3 (response overview) — the two missing figures.
4. WP4 floor-theorem write-up → becomes §3.
5. Skeleton LaTeX (JINST class) with the figure inventory wired in.
