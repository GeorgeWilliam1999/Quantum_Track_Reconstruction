# Paper 1 — plan for the sections still to write (13 Oct 2026)

Rules that bind every section below (decided 16 Sep 2026):

- One set of events for the whole paper: 5 µm, 0.15 mrad, 1 % hit loss, cone 0.2 rad, window 1.70 mrad,
  T = 10 … 1000, 10 events each (`../1BQF_to_spectral_filter_corpus/outputs_5um/`). The chapter figures
  made on the old 2 mrad clean store (`clean_headline.pdf`, `comb_degrees.pdf`, `readout_probability.pdf`,
  `fragmentation.pdf`) are **not** used; their successors are rebuilt from the corpus.
- Reading rule: classical and one-bit filter at the threshold that keeps 99 % of true segments; the
  polynomial at the lowest threshold with a false rate of at most 1 %. Both working points are reported
  for all three solvers where a table allows it.
- Every number in the text is a macro from `numbers.tex` (`13_paper_numbers.py`); every figure has a
  script in the corpus folder and a line in `figures/SOURCES.md`.
- Communications Physics: about 5,000 words of main text, five to six main figures, Methods, a supplement.
  The main text already holds five figures and one table, so 2.3 and 2.4 may add at most two figures
  between them; everything else goes to the supplement.

---

## Results 2.3 — Results against track multiplicity

**Message.** The polynomial keeps 92 % of true segments at 1000 tracks at a false rate below 1 %, where
the classical solver and the one-bit filter keep 24 % and 49 %; read the other way, their false rate at
99 % efficiency rises from 0.12 at 200 tracks to 0.77 at 1000. The crossover for the classical solver is
near 400 tracks. The one-bit filter collapses to 0.49 from 400 tracks on, because its threshold must
reach the level that the false pairs share with the fragments.

**Paragraphs.**
1. Efficiency at a false rate of at most 1 % against T for the three solvers (figure 5a): the
   polynomial's curve, its worst event, the classical crossover, the one-bit plateau at 0.49 and why.
2. False rate at 99 % efficiency against T (figure 5b): classical and one-bit identical on the median
   event (say so plainly; it is the honest caveat), the polynomial's value and why it is high.
3. The full efficiency-against-false-rate curves at 200, 400, 700 and 1000 tracks (figure 6): the
   polynomial separates, the other two overlap; where each reading rule sits on the curve.
4. The degree-60 variant with half-width 0.12: 2 to 3 points at 700 and 1000 tracks for 1.5× the calls
   (numbers exist from the teeth scan); one paragraph, no figure.
5. Event-to-event spread: medians with the lowest event, pooled against per-event (one sentence each).

**Figures.**
- Figure 5 (new, main): two panels, efficiency at ≤ 1 % false rate and false rate at 99 % efficiency
  against T, three solvers, median line with the 10-event range as a band, degree-60 variant as a thin
  dashed line in (a). Data: `outputs_5um/summary_by_event.csv` (eff_at_far1, far99 per event) and
  `outputs_teeth/teeth_scan.csv` (degree 60). Script: `16_fig_multiplicity.py`.
- Figure 6 (new, main or supplement): efficiency against false rate for the three solvers at T = 200,
  400, 700, 1000 (four panels), with the two reading rules marked. Data: the stored amplitudes
  `outputs_5um/amplitudes/T*_rep*.npz` (80 files; thresholds swept per event, then pooled or median).
  Script: `17_fig_curves.py`. If the figure budget is tight this goes to the supplement and one sentence
  carries it in the text.
- Supplement: the error atlas at 400 tracks (exists, `fig3_error_atlas_T400_with_qsvt.pdf`); a table of
  the 25 teeth-scan designs (from `outputs_teeth/teeth_scan.csv`).

**Numbers to add to `13_paper_numbers.py`.** Efficiency at ≤ 1 % and false rate at 99 % for every T
(not only 200/400/700/1000) if quoted; the T at which the classical curve drops below the polynomial's;
the one-bit plateau value; degree-60 numbers already exist (TeethBest*, TeethFourSixty*).

**Depends on.** Nothing. Can be written now.

---

## Results 2.4 — Computational cost

**Message.** All three algorithms call the same primitive; what differs is how many times and how the
matrix enters the circuit. Width grows as the logarithm of the event, calls per shot are constant in T
(2 for the one-bit filter, 40 for the phase sequence, 63 for the Chebyshev walk, 510 for eight-bit HHL),
and the price per call is set by the block encoding: the controlled evolution costs about 150·T^1.37
two-qubit gates (measured), a block encoding priced by hits about 100·T^0.85 (modelled, section 2.5).
Post-selection succeeds with probability about 0.44/T, so amplitude amplification multiplies the calls
by about 1.8·√T. The one currency is two-qubit gates per solved event.

**Paragraphs.**
1. The five factors: width, calls per shot, gates per call, amplification, readout samples; one
   sentence each, then the product.
2. Width (table 2 / figure 7a): ⌈log2 4T²⌉ system qubits plus 2 (one-bit filter), 4 (phase sequence),
   7 (walk and linear combination), n_c + 1 (HHL); 24 to 31 logical qubits at LHCb multiplicity.
3. Calls per shot and gates per call (figure 7b): measured transpilations against extrapolations,
   marked as such; HHL as a structure-level estimate, labelled.
4. Amplification and readout: post-selection probability measured on the corpus; repetitions 1/P
   against π/(4√P); readout M ln M samples with M ≈ 4T; the total d·T^1.5·log T walk calls against
   d·T² classical operations, said without overclaiming.
5. Which realisation is cheaper (the question left open in 2.2): phase sequence (fewer qubits, 40
   calls) against the walk (7 ancillas, 63 calls, success reduced by the one-norm). State the answer
   and keep the other as a column.
6. Classical loading: the hit oracle needs 0.4 ms and 8 KB at T = 400 (the paragraph the hub page
   lists as missing).

**Figures and table.**
- Table 2 (new, main): the five factors and their product at T = 400 and 1000 for HHL (eight bits),
  the one-bit filter, the phase sequence, the walk, each with the dilation price and the hit-priced
  price. Data: `../section4_qsvt/outputs/02_resource_model.csv` (widths, calls, P, totals), the measured
  transpilation fits (chapter 11.5; locate the CSV and record it in SOURCES.md), the corpus amplitudes
  for P. Script: extend `13_paper_numbers.py` with Cost* macros.
- Figure 7 (new, main): logical qubits (a) and two-qubit gates per solved event (b) against T on the
  corpus multiplicities, transpiled points filled and extrapolated points open, HHL as a shaded band
  labelled "estimate", the hit-priced encoding as a second line in (b). Successor of
  `qubit_scaling.pdf` + `resource_model.pdf`. Script: `18_fig_cost.py`.
- Supplement: post-selection probability against T measured on the corpus with the 0.44/T fit
  (successor of `readout_probability.pdf`); the surface-code estimate (per-cluster comb about 10³
  physical qubits, global comb about 10⁴) as a short table with its assumptions.

**Numbers to add.** Width at T = 400 and 1000 per route; calls per shot per route; CX per call
(measured and modelled) at T = 400 and 1000; P at T = 400 and 1000 on the corpus; the amplified call
budget; the loading time and size.

**Depends on.** The 2.5 scope decision (whether the hit-priced column appears here as a model); the
location and provenance of the transpilation measurements (chapter 11.5) before any number is quoted.

---

## Results 2.5 — A block encoding priced by detector hits

**Decision needed from George (Blocked to-do since 14 Sep).** Two options:
- *Cost model only (recommended).* One page: what the hit oracle is (each call priced by the hits, not
  the matrix entries), its qubit cost (about 45 workspace qubits over the floor) and gate cost
  (1.6 × 10⁴ CX per call at T = 400 against 2.4 × 10⁶ entry-priced), stated as a model with the
  circuit not synthesised, and one column in table 2. No new figure; the line in figure 7b carries it.
  Cite the Block Encoding IV write-up / paper 4 for the construction.
- *Full construction.* Moves the paper to PRX Quantum and onto the critical path of the Xenofon
  conversation. Not planned here.

**Depends on.** George's decision. Until then the subsection keeps its stub.

---

## Discussion

**Message.** A single zero removes exactly the isolated false segments and nothing else because its
shape is fixed; the spectrum of the segment Hamiltonian is a design surface on which any response can
be placed; a degree-40 polynomial on the same primitive solves 1000-track events at a false rate below
1 %; what it cannot separate is set by the graph spectra (hubs 0.11 from the track line, contaminated
tracks off every line); the price is depth per call, which a hit-priced block encoding converts into
qubits.

**Paragraphs (five, no figure).**
1. What the single zero leaves and why: one zero, fixed shape; the pairs, triples and hubs survive at
   the level of the fragments (the honest line: classical and one-bit read the same at 99 %).
2. The spectrum as the design surface: the four lines of a complete track are fixed by geometry; bright
   and dark modes; the polynomial is designed from the graph spectra, not fitted to events (the teeth
   scan confirms the design on held-out events).
3. The limits of every spectral filter: the three-prong hub line at 2.268 and the contaminated track;
   what would move them (degree 60 with narrower passes; modified operators and refit, held for the
   next paper; hit-uniqueness post-selection as a classical step).
4. Cost: calls constant in T, width logarithmic, price per call set by the encoding; the hit-priced
   route; fault-tolerant framing (no QSVT has run on NISQ hardware); the per-cluster route as the
   realistic first hardware target.
5. Outlook: detector noise moves false weight onto nameable lines and demands a refit; real Run-3
   geometry turns the lines into a band; both are the sequels.

**Depends on.** 2.3 and 2.4 written.

---

## Methods (eight subsections, no main figures)

| Subsection | Content | Source |
|---|---|---|
| Detector simulation and the set of events | geometry, cone, the three effects, event counts, measured residual and hit loss (already there), the six checks every event passed (C1–C6) | `01_build_corpus.py`, `outputs_5um/checks.csv`; supplement table of the checks |
| The acceptance window | kink as second difference; variance σ²_scatt + 6(σ_res/Δz)²; Rayleigh; the 3√2 rule and e⁻⁹; measured against predicted width | `lhcb_velo_toy.analysis.compute_epsilon`; checks.csv C6 |
| The Hamiltonian and the classical solver | index set and N ≈ 4T²; the three terms with the ½ factors; A x = b; block structure; exact per-block solution from eigenpairs; the shared pipeline (same events and segments for every solver) | `qtrk_pipeline`, `dp_terms`; corpus builder |
| The one-bit filter: circuit and state | the state after each stage Ψ1…Ψ4 (promised to Jochen); flag and uncompute; p_i; n = ⌈log2 N⌉ and the unused basis states; t = π/λ_c | 1BQF paper; `one_bqf.py` for the padding convention (to be read) |
| Polynomial design and phase factors | target g(λ); fit interval from Lanczos extremes with the margin Δ; Chebyshev least squares; rescale to 0.95; parity split; phases (Dong et al.); the teeth-scan protocol (chosen on events 0–4, reported on 5–9) | `design_line_comb_inverse`; `01_phase_factors.json`; `05_teeth_scan.py` |
| Exact evaluation of the filters | eigenvalue route against the three-term recursion on the sparse matrix (agreement 10⁻⁷, macro CheckMaxDiff); every curve is the exact readout, not the package circuit (hub-page requirement) | `03_terms_check.py`, `01_build_corpus.py` |
| Segment efficiency, false rate and working points | definitions; the 99 % threshold; the lowest threshold with ≤ 1 %; medians and lowest event | `01_build_corpus.py` curve functions |
| Cost model | the five factors; measured transpilations and the fit; extrapolation rule; HHL estimate assumptions; surface-code estimate assumptions | `02_resource_model.py`, chapter 11.5 |

**Depends on.** Nothing for the first seven; the cost subsection follows 2.4.

---

## Back matter

- **Data availability.** The corpus (80 events, amplitudes, metrics, checks) needs a home with a DOI
  (Zenodo or the Nikhef data store) — decision for George. State the version.
- **Code availability.** `lhcb_velo_toy` v2.2.0 is on `origin/centralise-2.1`, merge to main is a
  Blocked to-do; either merge or tag the release on the branch before submission. The corpus scripts
  and the figure scripts are in `1BQF_to_spectral_filter_corpus/`; state the commit.
- **Acknowledgements, Author contributions.** With the co-authors; the author list and affiliations are
  still "to be agreed".
- **Abstract.** Final sentence on cost once 2.4 exists; remove the "draft" marker.

---

## Open inside the drafted sections (not new sections)

- Job 3: move the block-spectrum paragraph into 2.1 and give 1/4, 1/3, 4/11, 5/11 through eigenpairs;
  expand bright and dark modes with the permutation argument and the chain eigenvectors; add the
  degree-one derivation from the circuit. Agreed with George before it is done.
- Marcel's comments: replies still on hold.
- Figure budget: with figures 5 and 7 the main text has seven figures and two tables; one of figure 3
  (error atlas) or figure 6 (curves) should go to the supplement to stay near six.

## Suggested order of work

1. 2.3 (data ready; two scripts; one day).
2. Methods subsections 1–7 (text only; the Ψ-state derivation is the only derivation to write).
3. George decides 2.5; locate and ledger the transpilation data; then 2.4 with table 2 and figure 7.
4. Discussion, abstract, back matter.
5. Job 3 and the figure-budget cut, then strip the review mark-up for the group's second round.
