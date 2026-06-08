# Presentation Planning — Inventory of `paper_comments/main.tex`

<!-- STATUS: draft -->
<!-- SOURCES: vscode-local:.../paper_comments/main.tex -->

## Document structure as it stands

1. **Plan on Major Comments** (pre-reviewer brainstorm)
   - Xenofon: hardware runs (IBM easy via QTI; Quantinuum needs token negotiation).
   - Xenofon: re-run Davide's MC events in the simulator and compare to Search-by-Triplet at segment level (the "strongest response").
   - Marcel: clarifies that the *toy* did include MS + resolution; the *quantum* sims did not.

2. **Reviewer 1** — co-review notice only, no scientific comments. Jacco notes this means standards may not be strict "Nature" level.

3. **Reviewer 2** (3 major comments + 4 minor)
   - **Major 1** Classical alternatives + O(N) load complexity + classical benchmarks.
   - **Major 2** Limitations + realism (toy: straight, ≤3 layers, ≤4 particles — note this is a misreading per Jacco/Marcel).
   - **Major 3** Generality of the conceptual advance beyond this Hamiltonian.
   - **Minor a** Run on real hardware.
   - **Minor b** Fig. 2(c) log scales + caption mismatch.
   - **Minor c** Eq. 6 |b⟩ should be |+⟩^n.
   - **Minor d** Eq. 10 add intermediate uncomputation step.
   - **Minor e** Error mitigation comment.

4. **Reviewer 3** (general comments + line-by-line)
   - General: at what N does asymptotic advantage matter vs HL-LHC? HL-LHC timeline. Fault-tolerant scenario discussion.
   - Methods line-by-line: ~13 items, mostly already addressed (olive = done).
   - Discussion: motivate PVs / z-vertices for non-LHCb readers; sparsity at HL-LHC.
   - References: DOI/arXiv/capitalisation cleanup (done).

5. **Response to Reviewers section** — drafts of varying completeness.
   - Major 1: O(N) load — fully drafted. Classical limits/timescales — placeholder ("blabla"). Classical benchmarks — drafted (refers to new SBT comparison). Outperform classical — placeholder.
   - Major 2: drafted at length, references Nicotra et al. 2308.00619, quotes 94.2% segment efficiency vs 94.8% SBT, fake rates 5.1%–13.7% raw vs 4.3% post track-building.
   - Major 3: bullet outline only, no prose.
   - Minor 1 (hardware): "Quantinuum loading" — not yet drafted.
   - Minor 2 (Fig 2): TODO George.
   - Minor 3 (Eq 6): drafted.
   - Minor 4 (Eq 10 intermediate): empty — but the "test area" at the end has a derivation candidate.
   - Minor 5 (error mitigation): empty.
   - PVs / dense observable: drafted.
   - References: drafted.
   - Line-by-line responses: most done; figure 4 uncertainty needs polish; L.316-317 TODO; r1/r2/r3 consistency TODO.

6. **New plots** section — orphaned figures, no captions in most cases:
   - `hardware_results.png` — Hellinger fidelity + SSI (this is one of the attached images).
   - `algorithm_accuracy_binned.png` — ghost rate + segment matching efficiency vs hits (the other attached image).
   - Several `george_new/fig13d_fixed_epsilon_*.pdf` variants — captions empty.
   - `george_new/fig14_solver_segment_efficiency_drop1pct.pdf` — caption empty, marked "FINAL".

7. **Test area** at end — alternative derivations of Steps 4 and 5 (Eq. 10 / uncomputation). Two versions, second is the cleaner "idealised" grouping.

## Internal disagreements / unresolved
- Whether O(N) load comment is a misunderstanding by reviewer (Xenofon) vs needs clearer text (Kurt).
- Whether the "generality" point can be answered with curved tracks (Marcel) or needs cross-domain examples (Xenofon, Kurt sceptical).
- Whether error mitigation needs to actually be run (Jacco "easy?") or just clarified as not used (Xenofon).
- Whether hardware authorship needs to include Alain et al. (Marcel raised; unresolved).

## Likely status of "what's been done"
Based on inline `\textcolor{olive}` markers (done) vs `\textcolor{blue}` (proposed/planned):
- DONE: most line-by-line minor fixes in the manuscript text, reference cleanup, eq.6 clarification, Fig 4 caption, PV/dense-observable expansion, MC + SBT comparison results (figures exist), Hellinger/SSI hardware-emulator figure exists.
- IN PROGRESS / TODO: Fig 2 axes, L.316-317 expansion, classical timeline prose, generality prose, hardware-on-real-device run, error mitigation statement, eq.10 intermediate step (drafted in test area but not merged), L.221 n_s definition, L.240 reference, r1/r2/r3 consistency.
