# Section drafts — slide content

<!-- STATUS: draft -->
<!-- All numbers traceable to either main.tex (paper_comments) or segment_level_analysis.ipynb -->

## Slide: Paper recap (1 slide)
- 1-Bit Quantum Filter: reformulation of HHL-style track reconstruction from matrix inversion to **spectral filtering** (signal vs. combinatorial-noise eigenspaces).
- Direct Structural Synthesis (DSS) instead of Trotterisation → shallower circuits, $\mathcal{O}(\sqrt N \log N)$ gate complexity.
- Submitted to *Communications Physics*; **3 reviewers**, 2 weeks left to resubmit.

## Slide: Reviewer overview
- **Reviewer 1**: co-review notice only, no scientific content.
- **Reviewer 2**: 3 major comments (classical alternatives + load complexity, realism/limitations, generality) + 5 minor.
- **Reviewer 3**: positive overall; asks for HL-LHC context, fault-tolerant scaling, line-by-line edits.

## Slide: Reviewer 1 — for completeness
- *"I co-reviewed this manuscript with one of the reviewers ..."*
- No scientific comments; included here for completeness.
- Implication (Jacco): standards may be slightly more flexible than full *Nature*.

## Slide: R2 Major-1 — Classical alternatives & O(N) load
- **Concern**: classical limits not quantified; load complexity $\mathcal{O}(N)$ may erase quantum speedup (cites Phys. Rev. D 101, 094015).
- **Our response**:
  - $\mathcal{O}(N)$ data loading **does not apply**: $\ket b$ loaded by Hadamards (cost $\mathcal{O}(\log N)$), $\mathbf A$ embedded via DSS gate-by-gate. Both already inside the $\mathcal{O}(\sqrt N \log N)$ count.
  - We will add a paragraph in the Gate Complexity section making this explicit.
- **Open**: classical-limit prose (HL-LHC throughput / GPU farm scaling) is still placeholder; "outperform classical" sentence not yet written.

## Slide: R2 Major-2 — Realism & limitations
- **Concern**: only straight tracks, ≤3 layers, ≤4 particles; no realistic detector effects.
- **Clarifications**:
  - The *toy* did include multiple scattering and detector resolution (text was unclear).
  - The 5-layer / 4-particle limit refers to **current quantum hardware noise**, not the algorithm. Statevector simulations already scale to 1000 particles.
- **New work to address**: re-run **Davide Nicotra's MC events** (`arXiv:2308.00619`, $B_s\to\phi\phi$, full Pythia/EvtGen/Geant4) through the 1-bit filter and benchmark vs. **Search-by-Triplet** at the segment level.
  - Draft numbers in our response: 94.2 % segment efficiency (1BQF) vs. 94.8 % (SBT); raw fake rate 5.1–13.7 % vs. 4.3 % post track-building [Nicotra et al., fig. 6].
  - **Status**: simulation in progress (Xeno) — final numbers not yet ready.

## Slide: R2 Major-3 — Generality of the conceptual advance
- **Concern**: spectral-filter trick may be too tailored to this Hamiltonian.
- **Planned response (consensus)**:
  - Method extends naturally to **curved tracks** (original Hamiltonian, Debney/Pearson) — Panos working in this direction.
  - More generally: any problem where noise eigenstates collapse onto a known eigenvalue can be filtered the same way.
- **Status**: prose not yet drafted. Looking for non-tracking example problems.

## Slide: R2 Minor comments
- **Hardware run** (real device): planned via Quantinuum + IBM (QTI free time). 6 weeks notional; 2 weeks left. Status uncertain.
- **Fig. 2(c)**: log axes + caption sub-panel labels — TODO George.
- **Eq. 6** ($\ket b$ as $\ket+^{\otimes n_s}$): clarified, defined $\ket b = H^{\otimes n_s}\ket 0 = \ket+^{\otimes n_s}$ before Eq. 6, and explained eigenbasis rewrite. **DONE**.
- **Eq. 10** intermediate step (uncomputation): drafted in the "test area" of the response doc; not yet merged into manuscript.
- **Error mitigation**: none used; will state explicitly and flag as future work.

## Slide: R3 — General comments
- **At what N does the asymptotic advantage matter for HL-LHC?** Will add a "HL-LHC event size" reference line on the complexity figure (Xeno's standing slide).
- **HL-LHC timeline** in introduction: yes, framed as scaling/dependency on fault-tolerant hardware, not as a deliverable for the paper.
- **Fault-tolerant discussion**: short paragraph in Results — connectivity, noise floor, expected scaling for both QPU technologies. **TBD** (no current expert in author list).
- **Sparsity at HL-LHC**: refer back to Davide's paper which already shows the matrix becomes *more* sparse at higher event size.
- **PV / dense-observable motivation**: now expanded in Discussion — explicit example (primary vertex compresses N tracks → 1 geometric feature, avoids O(N) tomography). **DONE**.

## Slide: R3 — Line-by-line (summary)
- 13 line-by-line edits to the Methods section + bibliography cleanup.
- **DONE**: theta definition, ordering of "positive semi-definite" remark, beta → c_j renaming, repetition removals at L.179/183, Fig. 4 caption clarifying #p labels, Aer wording, BW-friendly linestyles for Fig. 5, full bibliography rebuild (DOI uniform, missing DOIs added, capitalisation fix).
- **TODO**: $n_s$ definition (L.221), reference for "exhaustive search methods" (L.240), L.316-317 fake-rate expansion, r1/r2/r3 notation consistency.

## Slide: Toy segment-level analysis — what & why
- **Question (mine)**: at what multiplicity does the Hamiltonian + threshold strategy break, and is it the *Hamiltonian* or the *tracker* that fails?
- Setup: $\gamma=3, \delta=1, \varepsilon=2$ mrad, threshold $\tau=0.35$, 5 modules, $\Delta z=33$ mm. Multiplicities $n=10,30,100,300,500,750,1000$, 3 reps each.
- Notebook: `Toy_Characterisation/Verify_new_results/segment_level_analysis.ipynb`.
- Headline figures:
  - `fig13d_fixed_epsilon_drop1pct_logx.pdf` — segment-level efficiency / false-rate vs. $n_\mathrm{trk}$, with vs. without 1 % hit drop.
  - `fig14_solver_segment_efficiency_overlay_drop1pct.pdf` — solver-level segment efficiency overlay.
  - `fig17_tracker_ab.pdf` — CC vs. layered tracker A/B (extra slide if time).

## Slide: Headline result — segment efficiency stays at 100 %
- **Segment-level efficiency = 100 %** at every multiplicity up to $n=1000$ (clean events).
- **False-positive segment rate** (above $\tau=0.35$) climbs with $n$: 0 % at small $n$, 1.2 % at $n=300$, 3.4 % at $n=500$, 11.3 % at $n=750$, **19.8 % at $n=1000$**.
- The Hamiltonian's true/false separation gap **stays clean** at every $n$ — the false-positive tail is a small set of topologically special segments (≥3 compatible neighbours sharing a middle hit), not solver instability.
- **1 % hit-drop**: segment efficiency degrades only marginally; full numbers on slide from `fig13d` overlay. <!-- TODO confirm numerical degradation -->

## Slide: Why this matters for the reviewers
- Directly addresses **R2 Major-2**: the algorithm scales to 1000 particles in the toy at 100 % segment efficiency, 80 % even with a 19.8 % FP-segment tail, and survives a 1 % hit-drop. Not "limited to 4 particles".
- Directly addresses **R3** (sparsity & HL-LHC): the matrix stays well-conditioned ($\kappa$ grows by only ~3.4× over two decades of $n$).

## Slide: Optional — tracker A/B (§17)
- The 100 % segment efficiency hides a subtlety: when paired with a naive `get_tracks` (connected components), track-level efficiency collapses 99 % → 44 % between $n=300$ and $n=1000$.
- Replace with `get_tracks_layered` (angle check + module exclusivity, **same** Hamiltonian, same solver, same threshold) and track efficiency **recovers to ≥99.96 %** at every $n$.
- Conclusion: the Hamiltonian configuration is validated; previous lower numbers were a tracker artefact, not a method limit.

## Slide: Hardware results (Xeno + Alain)
- Hellinger fidelity and Signal-Separation Index (SSI) for 5 problem configurations (`2T·3L … 8T·3L`), three QPU architectures (H2 Quantinuum, IBM-Fez, IBM-Pittsburgh).
- Solid lines = noise emulator, dotted = real hardware <!-- CLARIFY: confirm which configurations actually ran on hardware -->.
- Headline: H2 emulator preserves Hellinger fidelity > 0.4 up to 4T·5L; SSI > 1 (signal above noise floor) for all H2 configurations except 8T·3L.
- IBM hardware/emulator drop below SSI = 1 from 4T·3L onward — connectivity / two-qubit-gate noise is the limiter.

## Slide: Realistic MC vs. Search-by-Triplet (Xeno, pending)
- 1-bit filter applied to Davide's MC dataset (~1000 hits), benchmarked against Python implementation of SBT (`github.com/dcampora/velopix_tracking`).
- Draft numbers in response: **94.2 % segment efficiency (1BQF) vs. 94.8 % (SBT)**; ghost rate 5.1 % at low n, 13.7 % at high n; rises with hits.
- **Final numbers not yet in hand** — slide is a placeholder.

## Slide: Open items (2 weeks)
| Comment | Status | Owner |
|---|---|---|
| O(N) load complexity paragraph | drafted | Xeno |
| Classical limits / HL-LHC throughput prose | TODO | TBD |
| "Will it outperform classical" sentence | TODO | TBD |
| Realistic MC + SBT benchmark results | running | Xeno |
| Generality of method (Major 3) | TODO | TBD |
| Hardware on real device (Quantinuum) | uncertain | Xeno + Alain? |
| Fig 2 axes + sub-panel labels | TODO | George |
| Eq. 10 intermediate step | drafted, merge pending | Xeno |
| Error mitigation paragraph | TODO | Xeno |
| L.316-317 fake-rate expansion | TODO | TBD |
| L.221 $n_s$ def, L.240 ref, r1/r2/r3 consistency | TODO | Xeno |
| Fault-tolerant scaling paragraph (R3) | TODO, no expert | TBD |

## Slide: Schedule
- Today: 2026-04-29. Resubmission deadline ≈ 2 weeks.
- Critical-path items: Xeno's MC benchmark (R2-M2), classical-limits prose (R2-M1), generality prose (R2-M3), hardware run.
- Stretch items: fault-tolerant paragraph, error-mitigation prose.
