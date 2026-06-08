---
name: project-epsilon-study2
description: Epsilon_study_2 — noise sensitivity study for 1BQF segment algorithm, LaTeX report in progress
metadata:
  type: project
---

Study sweeps (sigma_res, sigma_scatt, T) to characterise how detector noise affects the 1BQF quantum segment-level reconstruction algorithm. Replaces hand-tuned epsilon with the closed-form formula from the segment-level report.

Key results already written up in `report.tex`:
- Formula achieves >=99.97% coverage across the full noise grid
- Classical efficiency collapses to 6% at T=200, sigma_res=0.05mm
- Quantum (1-bit HHL) maintains ~100% efficiency everywhere with correct absolute threshold (x_Q > 1e-6)
- Quantum purity degrades to ~9% at worst case (T=200, sigma_res=0.05mm)
- Quantum statevector simulation scales as T^4.5

**Why:** Supervisor task 2 — replace hand-tuned epsilon with computed formula and characterise sensitivity.

**How to apply:** The report.tex is at `Toy_Characterisation/Epsilon_study_2/report.tex`. All 9 analysis sections (A-I) are complete. Report may need updating if figures are regenerated or new results added.

**Status as of 2026-06-02:** report.tex appeared complete (all sections written) but we were still working on it when the session disconnected.
