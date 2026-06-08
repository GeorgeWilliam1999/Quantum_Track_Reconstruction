# Quantum Track Reconstruction in the LHCb VELO — Mini Quantum Workshop (UM, 11 June 2026)

**Speaker:** George Scriven (Nikhef / Maastricht / Hasselt)
**Basis:** TrackHHL — *The 1-Bit Quantum Filter for particle trajectory reconstruction*, Chiotopoulos, Nicotra, Scriven *et al.*, [arXiv:2601.07766](https://arxiv.org/abs/2601.07766); Nicotra *et al.*, JINST 18 (2023) P11028.

This document is the narrative backbone + figure map for the talk and the source for the Notion page.
**All figures/animations were regenerated fresh (June 2026) from the `qtrk_store` data + the `lhcb_velo_toy` package** — no reuse of pre-threshold-fix PNGs.

> Provenance: aggregate figures ← `qtrk_store/manifest/metrics.csv` (7682 rows, recomputed segment metrics at absolute τ=0.35). Quantum vs-T ← fresh CPU statevector sweep `assets/quantum_sweep.csv` (T=10/20/50, step+erf, 3 reps) + the campaign's T=200 quantum (store). Per-event/spectrum ← the package on deterministic stored events. Scripts: `scripts/`.

---

## 0. Motivation (1–2 slides)
- The LHCb VELO is the silicon vertex detector nearest the IP; HL-LHC pile-up pushes track reconstruction into a combinatorial bottleneck.
- **Idea:** recast track finding as a *linear system* `A x = b` over detector **segments**, solvable classically **or** with a quantum linear-systems algorithm.
- This talk: a controlled **toy VELO** to characterise the reformulation and the **1-Bit Quantum Filter (1BQF)** — where it works, how it treats the Hamiltonian spectrum, and why a single phase bit beats full HHL here.

## 1. The toy & its architecture (2 slides)  — *why it's useful*
- A controlled, deterministic testbed: 5 planes at z=33–165 mm (Δz=33 mm), ±40 mm; configurable scattering σ_scatt, resolution σ_res, hit-drop, density (cone φ_max), multiplicity T∈{10…1000}.
- Decoupled `qtrk_pipeline`: events stored once (deterministic seed), matrix A regenerated on demand (never stored), solutions stored, **metrics are a recomputed view** (closes the relative-threshold bug class structurally). Sparse end-to-end.
- FIG `event_display_3d.png` · ANIM `event_3d.gif` — a generated event.
- FIG `pipeline_architecture.png` — data flow geometry → event → A,b → {classical, 1BQF} → metrics.

## 2. The Hamiltonian reformulation (2–3 slides)  — *theory*
- Segment = directed hit→hit between adjacent planes. Two segments sharing a middle hit = a **triplet** with **kink angle** θ.
- Solve `A x = δ1`, with `A_ii = γ+δ`, `A_ij = −C(θ_ij)` (attractive coupling on triplets). Defaults γ=3, δ=1.
- **Acceptance angle** ε from the noise: `ε = √(2(s·σ_scatt)² + 12·arctan²(s·σ_res/Δz) + 2·θ_min²)`, s=3, θ_min=1.5e-5.
- Two compatibility kernels: **step** `C=1 (θ<ε) else 0`, and **erf** `C = 1+erf((ε−θ)/(θ_d√2)) ∈ [0,2]`.
- FIG `hamiltonian_schematic.png` · `kernel_C_theta.png` · `angle_distribution.png` (true triplets θ≈0, false spread; ε cuts cleanly).
- **Hopfield fixed points:** isolated/false segments relax to `δ/(δ+γ)=0.25`; true segments sit on a ~0.375 plateau ⇒ an **absolute** threshold **τ=0.35** separates them (γ-aware: τ=δ/(δ+γ)+0.10).
- FIG `activation_hopfield.png` · ANIM `hopfield_relaxation.gif`.

## 3. Classical inversion & outputs (1–2 slides)
- `x = A⁻¹b` via sparse LU / CG. Threshold at 0.35 → active segments → group into tracks.
- Clean baseline: **efficiency = purity = 1.000** across all T (FIG `seg_metrics_vs_T.png`).
- Sparse end-to-end: `A_nnz ~ n_seg ~ O(T²)`, never O(T³) (FIG `scaling_nseg_nnz.png`); solve time FIG `timing_vs_T.png`.

## 4. Quantum inversion — HHL → 1BQF (3–4 slides)  — *the heart*
- **Original HHL:** QPE onto n (~4–8) time qubits with dense `e^{iAt}`, IQFT, a **1/λ R_y inversion ladder**, post-select ancilla. Cost: O(N²) dense unitaries, depth grows with eigenvalue precision.
- **1-Bit Quantum Filter (OneBitHHL):** **one** time qubit; sparse `e^{iAt}` via two-level (Givens) gates — **O(A_nnz)**; replace the 1/λ ladder by a single `X–CX–X` ancilla flip = a **one-bit spectral filter**. Shallow, fixed depth.
- FIG `hhl_vs_1bqf_circuit.png` — side-by-side structure + cost.
- **Why a single bit suffices here:** the spectrum is **bimodal** — a huge bulk at λ=γ+δ (false/isolated segments) and a thin true-segment band spread away from it. With `t=π/(γ+δ)`, 1-qubit QPE retains an eigenstate with prob `cos²(πλt/2π)`, whose **zero lands exactly on the bulk** ⇒ the bulk is rejected, the true band kept.
- FIG `eigenvalue_filter.png` · ANIM `hhl_vs_1bqf_filter.gif` (notch slides onto the bulk at t=π/(γ+δ)).
- FIG `solution_hist_CQ.png` — on one event, 1BQF drives false→0 and true→high, so **segment classification is perfect (eff=purity=1) even at cos_QC≈0.34**: the fidelity metric understates practical performance (it is dominated by the many false segments where classical=0.25 vs quantum≈0).
- FIG `quantum_cost_vs_T.png` — qubits ~log₂n_seg; statevector sim cost ~O(A_nnz·2^{n_sys}).

## 5. Treating the eigen-spectrum differently (2–3 slides)  — *step vs erf*
- The erf kernel reshapes the spectrum: coupling up to **2** (vs step's 1) pushes true eigenvalues **further** from the bulk (FIG `eigenspectrum_step_erf.png`; ANIM `spectrum_step_to_erf.gif`).
- **Key result (fresh, T=10):** erf has a *larger* κ yet *higher* 1BQF fidelity than step — **opposite to HHL's "large κ = costly" intuition.** Fidelity is set by **spectral separation from the bulk**, not κ.
  - step: κ≈2.4, cos_QC≈0.47 · erf(θ_d→0): κ≈9.5, cos_QC≈0.84 · erf(θ_d=1e-3): κ≈3.1, cos_QC≈0.53.
- FIG `kappa_cosqc.png` (left: cos_QC vs κ scatter; right: cos_QC & κ vs θ_d, step baseline).

## 6. Results: classical vs quantum, and where it breaks (3–4 slides)
- **Segment fidelity vs T:** cos_QC falls with T (10→0.47, 20→0.34, 50→0.22, 200→~0.14 store); **erf > step at every T** (FIG `cos_QC_vs_T.png`). Ancilla success P_anc shrinks with T (FIG `P_anc_vs_T.png`).
- **After threshold, the 1BQF still recovers the true segments** (FIG `quantum_seg_metrics_vs_T.png`).
- **Classical failure mode (honest, fresh data):** noise-scaled ε keeps **efficiency robust**, but **purity collapses** under resolution smearing (1.00→0.17 at σ_res=0.05, T=200; →~0.01 at T=700) and under **high density** (tight cone → purity ~0.02). Hit-drop lowers **efficiency** (→0.79 at 10% drop).
  - FIG `noise_failure_mode.png` (efficiency robust | purity collapses) · `purity_vs_T_noisy.png` · `scatter_drop_heatmap.png` · `density_vs_phi.png`.
- ERF under heavy smearing: FIG `erf_vs_step_noise.png`.

## 7. Bottom line (1 slide)
One reformulation (`A x = δ1`, `A_ii=γ+δ`, `A_ij=−C(θ)`), one truth (same-track adjacent hits), one operating point (x>0.35), one fidelity (cos_QC). The **1-Bit Quantum Filter** trades HHL's expensive 1/λ eigenvalue inversion for a **single phase-bit band filter** matched to the bimodal track-finding spectrum — far fewer qubits, O(A_nnz) sparse evolution, and (via the erf kernel) higher fidelity exactly where HHL intuition says it should be worse.

---

### Figure inventory
**figures/** seg_metrics_vs_T · noise_failure_mode · purity_vs_T_noisy · scatter_drop_heatmap · density_vs_phi · scaling_nseg_nnz · timing_vs_T · erf_vs_step_noise · event_display_3d · kernel_C_theta · activation_hopfield · angle_distribution · eigenspectrum_step_erf · kappa_cosqc · pipeline_architecture · hamiltonian_schematic · hhl_vs_1bqf_circuit · eigenvalue_filter · cos_QC_vs_T · P_anc_vs_T · quantum_seg_metrics_vs_T · solution_hist_CQ · quantum_cost_vs_T
**animations/** event_3d · hopfield_relaxation · spectrum_step_to_erf · hhl_vs_1bqf_filter

### Note for the record
The fresh data refines an earlier `Epsilon_study_2/report.tex` claim ("classical efficiency collapses to ~6%"): with the **noise-scaled ε** and the **absolute-0.35** segment metric, classical **efficiency stays high**; it is **purity** that collapses under resolution smearing / density. Efficiency is degraded instead by **hit inefficiency (drop)**.
