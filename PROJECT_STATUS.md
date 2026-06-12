# Project Status & TODO — Quantum Track Reconstruction

_Last updated: 2026-06-12_

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

Detailed per-study audit for the active campaign lives in
[Toy_Characterisation/AUDIT.md](Toy_Characterisation/AUDIT.md).

> **📌 Notion is the canonical tracker** (since 2026-06-08 reconcile). The Notion
> project [**Quantum LHCb Toy**](https://app.notion.com/p/3265d544b9d980b0befcef00eb67ab9c)
> holds three databases that mirror this file:
> - **Write Up for PhDs** — one row per study/report (status + date + repo path). 21 rows.
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

### ✅ Done 2026-06-06 (this session)
1. **T3/T4/T5 submitted to Condor** — 2597 jobs. T3 `Larger_Scatter` (1225,
   clusters 4799949–52), T4 `Larger_Scatter_Density` (490, 4799953–56), T5 `ERF`
   (882, 4799957–60). Worker confirmed high-performance (`SimpleHamiltonianFast` +
   Numba `fast_segment_metrics`). Per-T quantum cutoff baked into every `gen_params.py`:
   **`run_quantum=1` for T≤200, `0` for T≥400** (OneBQF statevector is ~6 h at T=200,
   intractable above; classical is ~10 s at any T). `mem_tier_gb` bug fixed in T3/T4
   (T=1000 now 128 GB, was 64).
2. **Held jobs resolved → queue 0 held.** 31 held eps2 jobs (T=200 walltime, T=700
   "GPU unused") re-run **classical-only** (19 deduped rows, clusters 4799974/75). The
   two dead Verify_new_results T2000 jobs removed (incl. the 256 GB OOM). Root cause of
   the walltime/GPU holds: `submit_gpu.sub` still had `+MaxRuntime=86400`, and T≥500
   quantum is simply intractable (T=200≈6 h, T=1000≈weeks; sampling has the same cost).
3. **T2 Epsilon_study_2 report enhanced + recompiled** (22 pp). Added §4.10 *Algorithm
   Stability: The Operating Envelope* (stable ≤0.01 mm; segment breakdown ≥0.02 mm &
   T≳50; classical cliff-edge vs quantum graceful purity loss) + abstract envelope
   sentence. `Epsilon_study_2/report.pdf`. (The "no report" 🚩 was stale — a full
   report was written 2026-06-02.)

### 🔭 What's next (priority order)
1. **Monitor T3/T4/T5 on Condor** (`condor-monitor`). High-T classical jobs run
   `collect_segment_pair_angles` (O(T³)) so T=1000 takes hours and ~64–128 GB — a few
   of the densest (clean, no-drop) T=1000 cells may OOM at 128 GB; bump to 192 or drop.
2. **Aggregate + write T3/T4/T5 reports** as pickles complete (`aggregate-ready` →
   `notebook-runner` → `report-drafter`). ERF analysis already has the local head start.
3. **Verify_new_results** — finish `Quantum_segment_level_analysis.ipynb`.

### Blockers
- [x] ~~**Clear the 286 HELD Condor jobs**~~ — **RESOLVED 2026-05-30** (history). 166
  walltime + 120 memory holds, fixed in-place via `condor_qedit` + `condor_release`.
  See [Toy_Characterisation/AUDIT.md](Toy_Characterisation/AUDIT.md) §6.
- [ ] **33 NEW held jobs (2026-06-06), all Epsilon_study_2 T=1000** — 24 walltime
  (still hitting the 24 h `MaxWallTime=86400` site cap on GPU jobs), 8 *"GPU claimed
  but no GPU usage"* (GPU statevector jobs not exercising the GPU), 1 OOM at **256 GB**
  (used 261624 MB). Needs a decision, not a blind `condor_release`. See AUDIT.md §6b.
- [x] ~~**Bake the memory/walltime fix into the T3–T5 submit config**~~ — `submit_base.sub`
  now has `+JobCategory="long"`, `+MaxRuntime=259200`, and per-study `gen_params.py`
  emit 16/32/64/128 GB tiers. **Note the 24 h GPU walltime cap above is a separate,
  site-imposed limit** the `MaxRuntime` override does not clear.
- [ ] `_shared/run_worker.py` and `ERF/hamiltonian_comparison.ipynb` are **untracked
  in git** — the absolute-0.35 threshold fix (below) is live on disk but not committed.

### Worker correctness fix (2026-06-06)
- [x] **Segment threshold made absolute in `_shared/run_worker.py`.** Default-metric
  reporting now uses `threshold = tau_default` (absolute 0.35) on both `sol_C` and the
  rescaled `sol_Q`, matching `segment_level_analysis.ipynb` `SOLVER_THRESHOLD`. The old
  relative `tau_default * max(sol)` collapsed below the 0.25 attractor / 0.375 plateau
  and produced a spurious FAR≈0.98. **All four studies share this worker**, so newly
  submitted jobs get correct stored defaults; already-completed pkls keep the full
  `sol_C/sol_Q/truth`, so analysis must recompute from the vectors, not trust
  `*_metrics_default`.

### Supervisor 5-task series
- [ ] **T2 Epsilon_study_2** — 1102 pkls; aggregated; figures done. 33 T=1000 GPU jobs
  held + 51 idle (incremental reps). **Write the report** 🚩 (does not need to wait).
- [ ] **T3 Larger_Scatter** — params generated (1222 rows) but **0 pkls, never
  submitted**. Run `submit.sh`, aggregate, analyse, report.
- [ ] **T4 Larger_Scatter_Density** — params generated (490 rows) but **0 pkls, never
  submitted**. Same pipeline.
- [ ] **T5 ERF** — **partially run locally** (162 CPU pkls, T∈{10,20,50}, all 18
  (θ_d × noise) cells); deep step-vs-ERF comparison done in `hamiltonian_comparison.ipynb`
  (eigenspectra, solution vectors, segment metrics). **Condor study not yet submitted**
  (882 rows). After submission: Youden-J / EER histogram threshold analysis, report.

### Quantum verification
- [ ] **Verify_new_results** — finish `Quantum_segment_level_analysis.ipynb`; resubmit
  any held seg14e high-T sampling jobs. Outputs live under `outputs/quantum_segment_analysis/`.

### Housekeeping
- [ ] Triage scratch notebooks: `Quantum_Toy_Study/Untitled-2.ipynb`,
  `Verify_new_results/Untitled-4.ipynb`,
  `Recovery_Seperation_analysis/Untitled-2/3/5.ipynb`.
- [ ] Decide whether `Quantum_Toy_Study` needs its own write-up or is fully
  superseded by the `quantum_segment` reports (see 🚩 below).

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
| **Epsilon_study_2 (T2)** | Calculated ε + sensitivity grid; 1121 pkls, aggregated, 82 figures | ✅ **report.pdf** (22 pp, +stability §); **Done 2026-06-07** |
| **Quantum_Toy_Study** | OneBQF benchmarking runs complete (3367 pkls) | 🚩 **no dedicated report** (largely folded into quantum_segment) |
| **Larger_Scatter (T3)** | Scatter × hit-inefficiency grid | 🟡 **1224 pkls in, drained 2026-06-07** — not yet aggregated → analyse + report |
| **Larger_Scatter_Density (T4)** | Tight cone × scatter grid | 🟡 **472 pkls (draining 2026-06-08)** — not yet aggregated → analyse + report |
| **ERF (T5)** | Local T≤50 + comparison notebook; Condor grid drained | 🟡 **712 pkls + 44 figs; report.pdf drafted** — Youden-J/EER + finalise (due 6/13) |
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
