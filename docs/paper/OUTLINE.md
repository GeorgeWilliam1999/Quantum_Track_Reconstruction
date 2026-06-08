# Paper Layout — Hopfield-Hamiltonian Track Reconstruction in a Toy VELO

**Target:** Nature Reviews–style review/characterisation article (broad readership, methods-heavy but accessible; review papers in this family are typically 8–15k words with ~6–8 display items and a long reference list).
**Affiliations:** Maastricht University · Universiteit Hasselt · Nikhef.
**Not** an LHCb collaboration paper — written as an external methods/review contribution that *uses* an LHCb-like detector geometry as a substrate.

---

## 0. Framing decisions (set these before drafting)

- **Genre:** review-with-original-results. Nature Reviews journals accept "Technical Reviews" and "Perspectives" that combine literature synthesis with new characterisation work. We position the paper as: *a unified characterisation of a class of Hopfield/Hamiltonian segment finders, with a controlled toy as the experimental vehicle.*
- **One-sentence pitch:** "We characterise — analytically and empirically in a controlled toy — when the Hopfield-Hamiltonian formulation of track segment finding works, when it fails, and how it fits into a complete reconstruction pipeline; this clarifies the regime in which quantum linear-system primitives such as 1BQF can be expected to deliver real benefit."
- **Scope guardrails:** keep the toy front-and-centre. No real LHCb data, no MC samples, no physics performance claims beyond the toy. This is what allows non-collaboration authorship.
- **Quantum content:** present, but framed as motivation and outlook, not as the main result. The Hamiltonian/spectral characterisation is venue-neutral; quantum is the "why this matters now" angle.

---

## 1. Title / abstract / display-item budget

- **Working title:** *Characterising the Hopfield-Hamiltonian Approach to Charged-Particle Track Finding: Operating Regimes, Failure Modes, and Implications for Quantum Linear-System Solvers.*
- **Alt title (shorter):** *When does the Hopfield Hamiltonian reconstruct tracks? A toy-model characterisation.*
- **Abstract structure (~200 words):**
  1. Background — Hopfield/Denby/Stimpfl-Abele lineage; renewed interest via QUBO/quantum solvers.
  2. Gap — no systematic characterisation of the *operating envelope* of the Hamiltonian itself, separated from solver and detector noise.
  3. Approach — minimal LHCb-VELO-like toy; closed-form acceptance threshold $\varepsilon$; spectral analysis; segment- and track-level sweeps.
  4. Findings — three quantitative regimes (efficient / degenerate / saturated); identification of the narrow-angle separation dip; analytic geometric clipping artefact.
  5. Implications — concrete conditions under which one-bit quantum filtering (1BQF / HHL-class) is expected to be advantageous.
- **Display-item budget:** 6 main figures, 1 main table, 4 boxes, ~3 supplementary figures.
  - Fig. 1 — pipeline schematic + toy geometry + clean event display.
  - Fig. 2 — Hamiltonian construction & spectral structure (bimodal spectrum, Gershgorin discs).
  - Fig. 3 — segment-level performance (4-panel: scattering / resolution / density / fixed-$\varepsilon$ extrapolation).
  - Fig. 4 — recovery & separation dip at narrow generation angles (mechanism + sweep).
  - Fig. 5 — track-level closure (efficiency, ghost rate, clone rate vs. density; tracker A/B).
  - Fig. 6 — operating-regime phase diagram in $(n_\text{tracks}, \sigma_\text{scatt})$ with shaded efficient/degenerate/saturated regions.
  - Table 1 — symbols, fixed parameters, derived invariants ($\beta$, $s^\star_\text{outer}$, $\tau$).
  - Box 1 — *What is a Hopfield Hamiltonian for tracking?* (pedagogical).
  - Box 2 — *The acceptance threshold $\varepsilon$* (derivation in 8 lines).
  - Box 3 — *1BQF in one paragraph* (pointer to companion work).
  - Box 4 — *Reproducibility* (notebooks, cached sweeps, Zenodo DOI).

---

## 2. Section-by-section layout

### 1. Introduction (≈1.5 pp)
- Charged-particle tracking as a combinatorial problem at HL-LHC scale.
- Three historical families: track following (Kalman), cellular automata, and **energy-minimisation / Hopfield**.
- Renewed interest: QUBO formulations, quantum annealing, gate-model linear-system solvers (HHL, QSVT, 1BQF).
- *What is missing in the literature:* a clean separation between (a) Hamiltonian model, (b) solver, (c) detector realism. Reviews so far conflate them.
- *Contribution of this paper:* (i) review the Hamiltonian family; (ii) characterise the model in isolation using a controlled toy; (iii) map operating regimes; (iv) state precise structural conditions for downstream quantum solvers.
- Roadmap paragraph.

**Sources to lift:** intro of `recovery_separation_analysis.tex`, `characterisation_report.tex`, generalisability mini-report §1.

### 2. Background and related work (≈2 pp, review content)
- 2.1 Hopfield networks for combinatorial optimisation (Hopfield 1982; Hopfield–Tank 1985).
- 2.2 Hopfield for tracking: Denby 1988; Stimpfl-Abele & Yepes 1991; Peterson 1989; Passaleva 2008; Funke et al. (TrackML).
- 2.3 Modern QUBO/Ising tracking: Bapst et al. 2020 (D-Wave); Zlokapa et al.; Schwemmer et al.; Crippa et al.; Wei et al. (graph-based QUBO).
- 2.4 Quantum linear-system solvers relevant to the segment problem: HHL (Harrow–Hassidim–Lloyd 2009); QSVT (Gilyén et al. 2019); 1-bit phase estimation (Kitaev; Higgins et al.); your own 1BQF work (cite as companion).
- 2.5 Where the gap is — none of the above isolate the *Hamiltonian's* operating envelope from solver noise or detector realism.

### 3. The toy model and Hamiltonian (≈2 pp, methods)
- 3.1 Detector geometry & event generator (5 modules, $\Delta z=33$ mm, generator `StateEventGenerator`). One concise figure (Fig. 1).
- 3.2 Segments, triplets, and the truth definition.
- 3.3 The acceptance threshold $\varepsilon$ — Box 2 with the derivation; full form in equation:
  $$\varepsilon=\sqrt{2\theta_s^2+12\theta_r^2+2\theta_{\min}^2}.$$
- 3.4 The Hamiltonian: diagonal penalty $\gamma$, bias $\delta$, off-diagonal angular coupling. Hopfield fixed points $\beta=\delta/(\delta+\gamma)$ and the outermost-segment fixed point $s^\star_\text{outer}$ used as invariants.
- 3.5 Solvers: linear $\mathbf{A}\mathbf{x}=\mathbf{b}$ vs. relaxation; classical reference solver used in this paper.
- 3.6 Validation harness (`EventValidator`, LHCb-style match cuts). Reproducibility note.

**Sources to lift:** `01_methodology.md`, `characterisation_report.tex` §2–§3, `segment_level_report.tex` early sections, `recovery_separation_analysis.tex` notation tables.

### 4. Spectral characterisation of the Hamiltonian (≈2 pp, partly review, partly new)
- 4.1 Bimodal spectrum: signal cluster vs. noise cluster at $\lambda_n\!\approx\!\gamma$. Figure 2.
- 4.2 Gershgorin bounds — analytic estimates of the noise cluster and the spectral gap $\Delta$.
- 4.3 Conditioning vs. event density: how the gap closes with occupancy and how this dictates solver requirements.
- 4.4 Invariants we observe across all sweeps (fixed points, eigenvalue plateaux). Connect to the structural conditions S1–S3 from the generalisability note (bimodal spectrum, gate-cheap RHS, low-weight observable).

**Sources:** §16 of master notebook; `recovery_separation_analysis.tex` §4–§6; generalisability mini-report §1.

### 5. Where the Hamiltonian works — operating regimes (≈2.5 pp, headline result)
- 5.1 Segment-level efficiency and false-pair rate vs. scattering $\sigma_s$, resolution $\sigma_r$, density $n$, and angular cone $\phi_\text{max}$. Figure 3 (the existing 4-panel paper plot from §13).
- 5.2 Fixed-$\varepsilon$ extrapolation to 1000 tracks; power-law fits; comparison to the analytic combinatorial scaling.
- 5.3 **Phase diagram** (Fig. 6) — three regions:
  - *Efficient:* $\Delta\gg$ noise scale; spectrum clean; segment efficiency $>95\%$.
  - *Degenerate:* spectral gap closes; recovery dip appears; ghost segments dominate.
  - *Saturated:* combinatorial explosion; ε no longer separates true from false.
- 5.4 Quantitative thresholds for each transition — these are the numbers we want reviewers to remember.

**Sources:** §3–§13 of master notebook; `02_results_segment_sweeps.md`; `characterisation_report.tex` §4–§5.

### 6. Where the Hamiltonian fails (≈2 pp, original results)
- 6.1 The narrow-angle separation dip — full mechanism: cross-track angle pile-up, ε saturation, near-degenerate eigenvectors. Figure 4.
- 6.2 Hit competition and occupancy structure (1-paragraph summary; full study deferred to companion / appendix).
- 6.3 Geometric clipping artefact — segment-grass — covered briefly with pointer to Appendix A.

**Sources:** `recovery_separation_analysis.tex` (the bulk of §3–§7); `hit_competition_study_report.tex`; `segment_grass_report.tex`.

### 7. Closing the loop — track-level reconstruction (≈1.5 pp)
- 7.1 From activated segments to tracks: connected-components vs. layered tracker (`get_tracks` vs. `get_tracks_layered`). One paragraph + Fig. 5.
- 7.2 Track efficiency, ghost rate, clone rate, hit purity vs. density.
- 7.3 Robustness to a 1% hit-drop noise model (the §14 study), demonstrating the closure between segment- and track-level metrics.
- 7.4 What the *rest* of a realistic pipeline would add (Kalman fit, momentum, vertexing) — and which of those steps the Hamiltonian formulation does *not* affect.

**Sources:** §14–§17 of master notebook; `03_results_solver_tracker.md`.

### 8. Implications for quantum linear-system solvers (≈1.5 pp, discussion)
- 8.1 Map characterisation findings → the three structural conditions for 1BQF / HHL-class advantage (S1: bimodal spectrum; S2: gate-cheap RHS; S3: low-weight observable).
- 8.2 Translation: the *Efficient* regime of §5 is exactly the regime in which the spectral gap is large enough for a single phase-estimation qubit to resolve signal from noise. The *Degenerate* regime is precisely where 1BQF breaks.
- 8.3 Beyond tracking — concise list of problem classes inheriting the same structure (associative memory, kernel methods, graph-Laplacian QUBO). Compress the generalisability mini-report into one paragraph + a list.

**Sources:** `generalisability_mini_report.tex` (compressed heavily).

### 9. Limitations (≈0.5 pp)
- Toy detector — no realistic $B$-field, no material budget, no time information.
- Single solver class.
- Performance numbers are not transferable to full LHCb; they are *envelope* numbers for the model.

### 10. Conclusions and outlook (≈0.5 pp)
- Three headline numbers (regime boundaries) reviewers can quote.
- Two open questions: behaviour under realistic field maps; behaviour with truly quantum solvers in the *Degenerate* regime.

### Appendix A — Geometric clipping ("segment grass") artefact
Full content of `segment_grass_report.tex` lifted with light edits.

### Appendix B — Numerical methods, Numba kernels, vectorised Hamiltonian builder
One page; lift from `01_methodology.md` + §2b/§14 sanity checks.

### Appendix C — Reproducibility & data availability
- Notebook list with one-line description.
- Cached sweep artefacts (`outputs/segment_analysis/cache/*.pkl`).
- Zenodo DOI placeholder.

---

## 3. Mapping: existing files → paper sections

| Paper section | Primary source(s) | Status |
|---|---|---|
| §1 Intro | `recovery_separation_analysis.tex` §1; `characterisation_report.tex` abstract | needs new prose synthesising the three threads |
| §2 Background | none yet | **new writing required** (literature review) |
| §3 Toy & Hamiltonian | `01_methodology.md`; `characterisation_report.tex` §2–§3 | lift, condense |
| §4 Spectrum | §16 master notebook; `recovery_separation_analysis.tex` §4–§6 | lift |
| §5 Operating regimes | `02_results_segment_sweeps.md`; §3–§13 master notebook; `characterisation_report.tex` §4–§5 | promote MD→prose; redraw Fig. 3, 6 |
| §6 Failure modes | `recovery_separation_analysis.tex`; `hit_competition_study_report.tex` | lift core mechanism, defer detail |
| §7 Track-level | `03_results_solver_tracker.md`; §14–§17 master notebook | promote MD→prose |
| §8 Quantum implications | `generalisability_mini_report.tex` | compress aggressively |
| App. A | `segment_grass_report.tex` | lift verbatim |
| App. B | `01_methodology.md` | lift |

## 4. Open questions for the authors

1. **Journal target within the Nature Reviews family.** *Nature Reviews Physics* fits best (publishes Technical Reviews and Perspectives that combine review with original characterisation). Confirm or pick alternative. Note: *Nature Reviews Physics* discourages first-time presentation of large blocks of unpublished results; we may need to first post a companion preprint (e.g. on arXiv `physics.ins-det` / `hep-ex`) covering the segment-level sweeps, then cite it from this review. Worth deciding now.
2. **Author order and corresponding author** across Maastricht / UHasselt / Nikhef.
3. **Quantum framing intensity.** Two viable stances:
   (a) "Characterisation paper that happens to enable quantum solvers" — quantum confined to §8 + Box 3.
   (b) "Why the Hopfield Hamiltonian is a natural quantum substrate" — quantum elevated to a co-headline.
   Pick one before drafting §1.
4. **Companion preprint?** If yes, the segment-level + recovery-dip results live in the preprint, and this paper *cites* them while keeping methods + regime map + quantum implications. If no, all numbers go here and the paper is longer.
5. **Reproducibility/data plan.** Confirm Zenodo deposit is acceptable to all three institutions; otherwise we use an institutional repository.
