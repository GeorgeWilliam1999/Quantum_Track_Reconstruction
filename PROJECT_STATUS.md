# Project Status & TODO — Quantum Track Reconstruction

_Last updated: 2026-07-17_

### 🆕 2026-07-17 — occupancy as post-selection · erf Trotter arbiter · fork quantum NO-GO
Three studies closed in one pass (all store-backed, committed `7e755992`/`747fed7a`, write-ups
Provisional/verify-ready in Notion):
- **Hit-uniqueness post-selection** (`qtrk_pipeline.postselect`, study script
  `Bifurification/dp_postselect_uniqueness.py`): Denby occupancy applied AFTER the solve as a
  per-(hit, role) slot filter. Heavy T=200: far 0.65→**0.04** at eff 0.94 (far-optimal; the
  in-matrix term never reaches it); moderate T=200: **0.9975/0.0000** — beats in-matrix outright
  at the eff≥0.99 gate with zero solver change. Composes with the 1BQF notch (0.92/0.05).
  _2026-07-17 (late): figures rebuilt_ — the original 2×2 PNG was corrupted by a pandas `.T`
  footer bug; now split into `_frontiers/_census/_robustness/_wpmap` + fixed overview, and the
  write-up was rebuilt (v2, with scope/theory/relevance) superseding the v1 page.
- **ERF exact-evolution arbiter** (`ERF/erf_exact_evolution_check.py`): the kink-width wp99
  reversal is **Trotter-dominated** (far 0.834→0.218 moderate / 0.943→0.576 heavy without the
  product formula); step2x deficit is real doubled-coupling physics; a genuine residual remains
  at heavy kink. Circuit fix path = QSVT X edge-colour-ordered DSS.
- **Fork penalty on real solvers = NO-GO** (`QSVT/Codesign/03_fork_quantum_noisy.py`, store study
  `QSVT_fork_noisy`): with γ*(β) validity, auto-tracked notch and a measured-spectrum comb, the
  1BQF worsens strictly with β (far 0.65→0.73→0.96); the atlas's emulated 99.1/0.9 three-knob
  target does not survive; the uniqueness gate, not the fork, controls the false rate quantum-side.
  Heterogeneous β=0.5 breaks the product formula (cos 0.944) — same split as the erf arbiter.
Also: audit content-check closed (2 prose fixes; both Verify store notebooks re-executed against
the current 2772-row store — wp99 curves now carry the WP_TAU_FLOOR fix).

### 🧮 QSVT (sub-project, 2026-06-10/11) — `QSVT/`
Polynomial spectral filters generalising the 1BQF's single cosine notch — now a
full paper programme (`QSVT/Paper_planning/PAPER_PLAN.md`, C1–C7 / WP1–WP7).
**Done:** feasibility Steps A–E (`Initial/`); the circuit-level solver in the
package (LCU-of-Chebyshev qubitization; circuit == streaming == matrix-free to
1e-9); the store campaign (`solver='qsvt'`, now to **T=1000**: comb far 1.0 %
vs classical 20.5 %); the **line-comb inverse** (band fails at density) and the
**minimax comb** (`design_minimax_comb`, degree 6–10); hit-drop fragments +
**occupancy gate**; **per-solver efficiency-first thresholds** (T=400 clean:
classical 100 %/1.6 %, 1BQF 99.5 %/43.9 %, **QSVT 98.3 %/0.96 %**);
**WP1** noise robustness (comb == classical to ≤0.7 % through the realistic
range; acceptance wall + fragment degeneracy are encoding-level); **depth &
qubit scaling** (29 q measured at T=1000; total walk calls flat ~50 at d≈16 to
T=700); **resource directions D1–D6** with **D3 cluster decomposition**
(3–6 qubits flat in T, exact, kills the √T amplification) and **D4 minimax**
executed; **co-design** (ε-fork deep dive + **WP7** (β, ε_B) scan: sweet spot
(0.5–1, ε_acc) kills half the filter-immune floor, strict Pareto); **WP5
Run-3 real data** (variable length ⇒ band+grass-notch design; the fragment
floor caps all solvers; m≥3 target: design concedes 5e-4 AUC to classical);
**WP3/WP6 readout & dequantisation** (d·T^1.5·lnT walk calls vs d·T² float-ops,
honest claim fixed); the **activation-spectrum unification** (closed-form
ladder; toy AUC ~1 for all solvers — differences are tail/ladder phenomena);
**the toy design closure** (2026-06-12, `Segment_level_studies/07`): the
**shared-line theorem** (contaminated-P5 odd lines ≡ hub K(1,3) lines at s±√3;
any recovering filter admits a hub population ≥ 3/2 × the recovered level;
leg-null = count optimum, built & ladder-verified), the census (contamination
*is* the efficiency story; hubs/bridges rare), the measured veto of line
surgery (heterogeneity + mirror tax; wide tails = graded broadband recovery),
and the (d, hw) scan closing the family at the production point d=40
hw 0.12–0.18 — production comb stays, **sharp d120 = high-purity variant**
(far 0.36 % vs 1.0 % fixed-τ at T=1000), Notion Write-Up *QSVT VIII*.
**Notion:** project page *QSVT* + Write-Ups *QSVT I Theory*, *QSVT II
Application* (**§1–§18**, every figure with full mathematical description),
the **first-principles mathematics** page (**§1–§12**, all derivations incl.
floor-theorem proof, activation ladder, readout maths), and the chronological
log — dual-linked to *Quantum LHCb Toy* and *QSVT*.
**Open:** D1/D2 generalized-QSP over e^{-iAt} (the hardware path, 8–9 qubits);
real-data fork campaign; length-aware response design; WP4 LaTeX; paper
figures 3 & 7; paper assembly.

Root-level tracker for the whole workspace: outstanding work, a summary of
what's done, and links to every written (compiled) report. Studies that are
**finished but have no written report** are flagged with 🚩.

Operational data/pipeline guides live in
[Toy_Characterisation/DATA_INDEX.md](Toy_Characterisation/DATA_INDEX.md) and
[Toy_Characterisation/DATA_GENERATION_GUIDE.md](Toy_Characterisation/DATA_GENERATION_GUIDE.md)
(the old per-study `AUDIT.md` tracker was removed 2026-06-14 — superseded by this file + Notion).

> **📌 Notion is the canonical tracker** (since 2026-06-08 reconcile). The Notion
> project [**Quantum LHCb Toy**](https://app.notion.com/p/3265d544b9d980b0befcef00eb67ab9c)
> holds three databases that mirror this file:
> - **Write Up for PhDs** — one row per study/report (status + date + repo path).
> - **To do** — the live shared task list.
> - **Literature and Resources** — external references + technical specs only.
>
> Every study/report below has a matching Write-Up row. Keep them in sync: when a
> study changes state here, update its Notion row (and vice-versa).

### 🔄 2026-06-08 reconcile (repo ↔ Notion)
- Added **17 Write-Up rows** for every report/study that had none (the 5 Initial/
  Grass/Recovery reports, segment-level, quantum_algorithm, comparison,
  generalisability, reviewer-response, OneBQF audit, track-density legacy, ERF,
  T3, T4, Verify_new_results, Quantum_Toy_Study); set Epsilon_study_2 → **Done**.
- Closed stale todos (Eps2 GPU pipeline, ERF Condor submit = done); added open
  todos for T3/T4 aggregation, the Verify notebook, scratch-notebook cleanup, and
  the IBM-token security fix.
- Relabelled the 5 report rows in **Literature** as `[ARCHIVED PDF]` pointers —
  canonical tracking is now in Write-Up (the Notion MCP can't move file uploads).
- ⚠️ **IBM Quantum API token is in plaintext on the Notion project page** — revoke,
  regenerate, store outside Notion. Tracked as a todo dated 2026-06-09.

---

## ⏳ Outstanding TODO

_The 2026-06-06 Condor-submission TODO/Blockers (held jobs, "T3/T4 never submitted",
the 24 h GPU walltime cap, the absolute-0.35 worker fix) are **history** — all five
studies (T1–T5) are complete and store-backed (see the summary table below and Notion).
The `qtrk_store` pipeline replaced the per-study Condor flow; the matrix-free 1BQF +
exact cKDTree A-build (`SCALING_DEEP_DIVE.md`, 2026-06-14) removed the OOM/walltime
problems entirely. Current open work only:_

### Toy_Characterisation
- [ ] **ERF (T5)** — Youden-J / EER threshold optimisation on pooled per-segment store
  scores (`qp.load_solution` + truth), then finalise the report.
- [x] **Verify_new_results** — store-backed notebooks re-executed & frozen 2026-07-17 against
  the 2772-row store (legacy local-pkl notebook deleted 2026-07-03).
- [ ] Run the segment analysis with a 0.5 % drop rate.
- [ ] **Quantum_Toy_Study** — decide whether it needs its own write-up or is fully
  superseded by the `quantum_segment` reports, then close out.
- [ ] Housekeeping — delete scratch notebooks (`Quantum_Toy_Study/Untitled-2.ipynb`,
  `Verify_new_results/Untitled-4.ipynb`, `Recovery_Seperation_analysis/Untitled-{2,3,5}.ipynb`);
  triage backups.

### QSVT
- [ ] D1/D2 generalized-QSP over the 1BQF's own e^{-iAt} (the hardware path, 8–9 qubits).
- [ ] Real-data fork campaign; length-aware response design (the Run-3 frontier).
- [ ] WP4 LaTeX transcription; paper figures 3 & 7; paper assembly.

---

## ✅ Summary of what's done

| Study | Outcome | Report |
|-------|---------|--------|
| **Initial — baseline** | Default config + low-density 100% efficiency baseline | ✅ written |
| **Initial — characterisation** | True vs false segment-pair angle distributions | ✅ written |
| **Initial — hit competition** | High-density degradation, ROC, scattering scans | ✅ written |
| **Segment Grass** | "Grass" sub-peaks traced to geometric clipping at cone edge | ✅ written |
| **Recovery & Separation** | Near-coincident track degeneracy, condition-number study | ✅ written |
| **Segment-level (classical)** | Core segment discriminator, calculated ε derivation | ✅ written |
| **Quantum segment (OneBQF/HHL)** | 1-bit HHL segment solver, τ-threshold optimisation | ✅ written |
| **Quantum vs classical segment** | Solver comparison, fidelity, timing | ✅ written |
| **OneBQF parameter audit** | Parameter audit of the quantum solver | ✅ written |
| **Generalisability** | Mini-report on generalisability | ✅ written |
| **Track-density (OLD_TOY_CODE)** | Legacy density study | ✅ written |
| **Epsilon_study_2 (T2)** | Calculated ε + sensitivity grid; 1121 pkls, aggregated, 82 figures | ✅ **report.pdf** (22 pp, +stability §); **Done 2026-06-07**, headline **re-derived from qtrk_store 2026-06-14 — unchanged** |
| **Quantum_Toy_Study** | OneBQF benchmarking runs complete (3367 pkls) | 🚩 **no dedicated report** (largely folded into quantum_segment) |
| **Larger_Scatter (T3)** | Scatter × hit-inefficiency grid | ✅ **store-backed (qtrk_store, 3920 rows); `store_analysis.py` → report + 4 figs; Done 2026-06-14** |
| **Larger_Scatter_Density (T4)** | Tight cone × scatter grid | ✅ **store-backed (1418 rows); `store_analysis.py` → report + 3 figs; Done 2026-06-14** |
| **ERF (T5)** | Single-event comparison + full multiplicity sweep from store | 🟡 **single-event report + full store sweep (3190 rows, `store_landscape.py`) added**; Youden-J/EER on pooled scores pending |
| **Run3_Verification** | Toy segment framework on real LHCb Run-3 Bs→φφ events | ✅ **classical 0.88/0.94 + 1BQF + false-type census; Write-Up created 2026-06-14** |
| **EpsilonStudies (legacy)** | Superseded by Epsilon_study_2 | n/a (no report needed) |

---

## 📄 Written reports (compiled PDFs + LaTeX sources)

### Curated reports — `docs/reports/`
- [segment_level_report.pdf](docs/reports/segment_analysis/segment_level_report.pdf) · [.tex](docs/reports/segment_analysis/segment_level_report.tex)
- [quantum_algorithm_report.pdf](docs/reports/quantum_segment/quantum_algorithm_report.pdf) · [.tex](docs/reports/quantum_segment/quantum_algorithm_report.tex)
- [comparison_report.pdf](docs/reports/quantum_vs_classical_segment/comparison_report.pdf) · [.tex](docs/reports/quantum_vs_classical_segment/comparison_report.tex)
- [generalisability_mini_report.pdf](docs/reports/generalisability/generalisability_mini_report.pdf) · [.tex](docs/reports/generalisability/generalisability_mini_report.tex)
- [reviewer_response_status.pdf](docs/reports/presentation_planning/reviewer_response_status.pdf) · [.tex](docs/reports/presentation_planning/reviewer_response_status.tex)

### Toy_Characterisation study reports
- [baseline_report.pdf](Toy_Characterisation/Initial/baseline_report.pdf) · [.tex](Toy_Characterisation/Initial/baseline_report.tex)
- [characterisation_report.pdf](Toy_Characterisation/Initial/characterisation_report.pdf) · [.tex](Toy_Characterisation/Initial/characterisation_report.tex)
- [hit_competition_study_report.pdf](Toy_Characterisation/Initial/hit_competition_study_report.pdf) · [.tex](Toy_Characterisation/Initial/hit_competition_study_report.tex)
- [recovery_separation_analysis.pdf](Toy_Characterisation/Recovery_Seperation_analysis/recovery_separation_analysis.pdf) · [.tex](Toy_Characterisation/Recovery_Seperation_analysis/recovery_separation_analysis.tex)
- [segment_grass_report.pdf](Toy_Characterisation/Segment_Grass/segment_grass_report.pdf) · [.tex](Toy_Characterisation/Segment_Grass/segment_grass_report.tex)
- [OneBQF_parameter_audit.pdf](Toy_Characterisation/Verify_new_results/OneBQF_parameter_audit.pdf) · [.tex](Toy_Characterisation/Verify_new_results/OneBQF_parameter_audit.tex)

### Legacy
- [track_density_study_report.pdf](OLD_TOY_CODE/track_density_study_report.pdf) · [.tex](OLD_TOY_CODE/track_density_study_report.tex)

> Markdown report drafts (not yet PDF) also exist under
> [docs/reports/quantum_segment/](docs/reports/quantum_segment/) (00–05 `.md`)
> and [docs/reports/presentation_planning/](docs/reports/presentation_planning/).

---

## 🚩 Complete but no report (write-ups owed)
1. **Larger_Scatter (T3)** & **Larger_Scatter_Density (T4)** — data is in
   (1224 / 472 pickles) but **not yet aggregated**; no analysis or report. Highest-
   value owed write-ups now that Eps2 is done. Tracked in Notion Write-Up (In progress).
2. **Quantum_Toy_Study** — all benchmarking runs are complete with figures, but
   it has **no standalone report**. Confirm whether it should be written up or
   is fully covered by the `quantum_segment` reports, then close it out.
3. ~~**Epsilon_study_2 (T2)**~~ — **DONE 2026-06-07**: `report.pdf` (22 pp,
   +stability §), 1121 pkls aggregated, 82 figures. Notion Write-Up → Done.
