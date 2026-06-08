# Plan: Complete T2–T5 to LHCb Internal Review Standard

> **Status update — 2026-06-06** (plan authored ~05-30; progress against it below)
>
> | Phase | Plan item | Status |
> |-------|-----------|--------|
> | 1.1 | Fix submit config (long + 3-day, 128 GB tier) | ✅ `submit_base.sub` + `gen_params.py` tiers done |
> | 1.2 | Submit T3 / T4 / T5 | ❌ **T3 & T4 still 0 pkls (never submitted)**; T5 ERF only run **locally** (162 CPU pkls, T≤50), Condor not submitted |
> | 1.3 | Verify T2 queue clear | ⚠️ **33 new holds** on T2 T=1000 GPU (24 h site walltime / "GPU claimed unused" / 256 GB OOM) — see AUDIT §6b |
> | 2 | T2 aggregation + report | 🟡 1102 pkls aggregated, figures done; **report not written** 🚩 |
> | 3 | Finish `Quantum_segment_level_analysis.ipynb` | 🟡 in progress |
> | 4 | T3/T4/T5 reports | ⛔ blocked on 1.2 |
> | — | (new) ERF threshold bug fixed in `_shared/run_worker.py`; new `ERF/hamiltonian_comparison.ipynb` | ✅ this week |
>
> **Critical path now:** submit T3 + T4 (longest pole, idle >1 week) → submit T5 ERF →
> draft T2 report in parallel → decide on the held T=1000 GPU quantum jobs. The 24 h GPU
> walltime hold is a **site cap** the `MaxRuntime` bump does not clear; treat T=1000 quantum
> as optional for the T2 report rather than a blocker.

## Context

A supervisor/group review meeting is scheduled in the coming weeks. All four
outstanding supervisor tasks — T2 (Epsilon_study_2), T3 (Larger_Scatter),
T4 (Larger_Scatter_Density), T5 (ERF) — need to be in the review package.

Current blockers:
- T2 has 1049 pickles, aggregated CSVs, and polished figures but **no written report**.
- T3/T4/T5 have params generated but **have never been submitted** to Condor.
- The 64 GB / 24 h walltime limits that caused 286 held jobs in T2 have not yet
  been baked into T3–T5 submit configs — they will repeat the same holds if submitted as-is.
- `Verify_new_results/Quantum_segment_level_analysis.ipynb` is incomplete.

The critical path is: **fix memory config → submit T3/T4/T5 (start compute clock) →
write T2 report while they run → write T3/T4/T5 reports as they complete**.

---

## LHCb Internal Review Standard

Each study report must match the format of
`docs/reports/segment_analysis/segment_level_report.tex`:

- `\documentclass[11pt,a4paper]{article}`, 2.5cm margins, `natbib` (numbers)
- **Abstract** (~150 words) with concrete numbers
- **Introduction**: physics motivation, place in the task series, what question this task answers
- **Methodology**: detector model (refer to T1 baseline for full details), Hamiltonian
  formulation, calculated ε formula, solver variants, metrics definitions.
  Cite Nicotra et al. J.Inst. 18 P11028 (2023) and arXiv:2511.11458v1.
- **Results**: all figures as PDF (vector) + PNG (300 DPI); quantitative tables
  with mean ± SEM across replicas; sensitivity heatmaps where applicable.
- **Discussion**: comparison to prior tasks, interpretation of degradation with occupancy.
- **Conclusions**: bullet-form key takeaways.
- **References**: at minimum Nicotra (2023), arXiv:2511.11458, LHCb VELO TDR.

Shared notation across all reports: θ for angle, ε for acceptance threshold, σ_s /
σ_r for scattering / resolution, T for n_trk, τ for solver threshold, γ/δ for
Hamiltonian penalties.

---

## Phase 1 — Unblock T3/T4/T5 and start compute (Day 1)

**Submit T3/T4/T5 to Condor immediately to start the compute clock. Apply the
memory fix first — without it they will repeat T2's 286-job hold.**

### Step 1.1 — Fix the submit config

Edit `Toy_Characterisation/_shared/submit_base.sub` to replace the existing
`+JobCategory = "medium"` and `MaxRuntime = 86400` with:

```
+JobCategory    = "long"
+MaxRuntime     = 259200        # 3 days
```

Update the memory tier logic in each study's `gen_params.py` so rows with
n_trk ≥ 400 are assigned to the 128 GB tier. Safe thresholds from T2 OOM
incidents (AUDIT.md §6):

| n_trk    | RequestMemory   |
|----------|-----------------|
| ≤ 150    | 16 384 MB       |
| ≤ 400    | 32 768 MB       |
| ≤ 700    | 65 536 MB       |
| ≤ 1000   | 131 072 MB      |
| > 1000   | 196 608 MB      |

Files to edit:
- `Toy_Characterisation/_shared/submit_base.sub`
- `Toy_Characterisation/Larger_Scatter/gen_params.py`
- `Toy_Characterisation/Larger_Scatter_Density/gen_params.py`
- `Toy_Characterisation/ERF/gen_params.py`

### Step 1.2 — Regenerate params CSVs and submit

Run `params-preflight` agent on each CSV first, then:

```bash
cd Toy_Characterisation/Larger_Scatter      && python gen_params.py && bash submit.sh
cd Toy_Characterisation/Larger_Scatter_Density && python gen_params.py && bash submit.sh
cd Toy_Characterisation/ERF                 && python gen_params.py && bash submit.sh
```

Total: 1224 + 489 + 881 = 2594 jobs.

### Step 1.3 — Verify T2 queue is clear

Use `condor-monitor` agent to confirm T2's held clusters (4696265, 4696266,
4719401, 4719402, 4720866) are running/idle. If any remain held, re-apply the
`condor_qedit` + `condor_release` fix from AUDIT.md §6.

---

## Phase 2 — Complete and write T2 report (Days 1–5)

Run in parallel with T3/T4/T5 compute.

### Step 2.1 — Final T2 aggregation

Use `aggregate-ready` agent to confirm pkl count, then:

```bash
python Toy_Characterisation/_shared/aggregate.py \
  --indir Toy_Characterisation/Epsilon_study_2/results \
  --outprefix Toy_Characterisation/Epsilon_study_2/results/eps2 \
  --group-keys sigma_scatt,hit_ineff,sigma_res \
  --emit-angles
```

### Step 2.2 — Run validation gate

Use `validation-gate` agent. Gate must pass (eps2val within ±2 combined SEM
of Verify_new_results on `angle_segment_efficiency`, `cls_default_segment_purity`,
`cls_default_segment_false_rate`) before report is written.

### Step 2.3 — Refresh T2 figures

Use `notebook-runner` agent on:
- `Epsilon_study_2/analysis.ipynb`
- `Epsilon_study_2/segment_metrics_calc_epsilon.ipynb`

Then use `figure-auditor` agent to confirm all expected PDFs and PNGs exist
and are newer than the source CSV.

### Step 2.4 — Write T2 LaTeX report

File: `Toy_Characterisation/Epsilon_study_2/epsilon_study_2_report.tex`
Format: copy preamble from `docs/reports/segment_analysis/segment_level_report.tex`

Report structure:
1. **Abstract** — calculated ε formula, validation result (matched Verify within N SEM),
   sensitivity range, quantum cos_QC agreement
2. **Introduction** — T2 in the 5-task series; T1 established default config;
   T2 replaces hand-tuned ε with the closed-form expression
3. **Method** — reference T1; describe ε derivation, sensitivity grid (4×3×8 combos),
   seg14e companion setup
4. **Results — Validation gate** — table of angle_segment_efficiency ± SEM vs T for
   eps2val vs Verify_new_results; state agreement explicitly
5. **Results — Sensitivity** — heatmaps (σ_res × σ_scatt) at T ∈ {100, 500, 1000};
   key figures: `heatmap_efficiency`, `heatmap_reco_ghost_rate`, `activation_spectrum`
6. **Results — Classical vs quantum** — seg14e companion; cos_QC table vs T;
   key figures: `heatmap_clone_fraction`, `occ_std_vs_ghost`
7. **Discussion and Conclusions**

Compile: `pdflatex → bibtex → pdflatex × 2`

---

## Phase 3 — Complete Verify_new_results notebook (Days 1–5, parallel)

Finish `Toy_Characterisation/Verify_new_results/Quantum_segment_level_analysis.ipynb`.
Incomplete sections (identified from notebook structure): τ-threshold sweep,
dominance analysis, timing figures. Use `notebook-runner` agent after completing cells.

This feeds the T2 seg14e companion analysis and updates the quantum verification
claim in the existing `OneBQF_parameter_audit.pdf`.

---

## Phase 4 — Write T3/T4/T5 reports as jobs complete (Days 5–14)

Monitor with `condor-monitor` agent; trigger aggregation via `aggregate-ready` agent
at ≥95% pickle completion. For each study:

1. `aggregate-ready` → `aggregate.py`
2. `notebook-runner` → refresh `analysis.ipynb`
3. `figure-auditor` → confirm PDF + PNG pairs exist and are fresh
4. `report-drafter` agent → generate LaTeX skeleton from summary CSV
5. Fill in physics interpretation manually
6. Compile PDF

### T3 report focus
Key claim: efficiency and ghost rate vs σ_scatt (5 levels, 1–10×10⁻⁴) and
hit-inefficiency p_drop (0–10%) across T=10–1000. Identify the degradation boundary.
File: `Toy_Characterisation/Larger_Scatter/larger_scatter_report.tex`

### T4 report focus
Key claim: how tightening φ_max interacts with σ_scatt. Report n_segments vs φ_max
(tighter cone = sparser Hamiltonian = better conditioned). Identify the
precision/efficiency trade-off point.
File: `Toy_Characterisation/Larger_Scatter_Density/larger_scatter_density_report.tex`

### T5 report focus
Key claim: Youden-J and EER thresholds from pooled true/false angle histograms vs
θ_d (ERF width). Confirm θ_d=10⁻⁶ reproduces the step-function result (regression).
Report the optimal (θ_d, τ) operating point.
File: `Toy_Characterisation/ERF/erf_report.tex`

---

## Phase 5 — Housekeeping (any time, ≤ 30 min)

- Delete scratch notebooks: `Quantum_Toy_Study/Untitled-2.ipynb`,
  `Verify_new_results/Untitled-4.ipynb`,
  `Recovery_Seperation_analysis/Untitled-2/3/5.ipynb`
- Quantum_Toy_Study: add `README.md` note saying findings are superseded by
  Verify_new_results and pointing to relevant outputs. No standalone report needed.
- Update `PROJECT_STATUS.md` and `Toy_Characterisation/AUDIT.md` as each task closes.

---

## Verification checklist before the review meeting

- [ ] T3/T4/T5 submitted with correct memory tiers (128 GB for n_trk ≥ 400, long category)
- [ ] T2 validation gate passes (eps2val within ±2 SEM of Verify_new_results)
- [ ] T2 `epsilon_study_2_report.pdf` compiled without LaTeX errors
- [ ] T3/T4/T5 aggregated, analysis notebooks clean, reports compiled as PDFs
- [ ] All figures exist as PDF (vector) + PNG (300 DPI)
- [ ] `Quantum_segment_level_analysis.ipynb` all cells execute without errors
- [ ] `PROJECT_STATUS.md` updated: T2–T5 all show ✅ with report links
- [ ] Scratch notebooks deleted

---

## Timeline estimate

| Action | When |
|--------|------|
| Fix submit config; submit T3/T4/T5 | Day 1 |
| T2 aggregation + validation gate + figure refresh | Day 1–2 |
| T2 LaTeX report written + compiled | Day 2–4 |
| Verify_new_results notebook complete | Day 3–5 |
| T3/T4/T5 complete on Condor (estimate, cluster-dependent) | Day 5–10 |
| T3/T4/T5 aggregation + analysis | Day 8–11 |
| T3/T4/T5 LaTeX reports written + compiled | Day 10–14 |
| Housekeeping + status doc updates | Day 14 |
