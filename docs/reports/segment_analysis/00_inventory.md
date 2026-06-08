# Inventory — `segment_level_analysis.ipynb`

<!-- STATUS: final -->
<!-- SOURCES: Toy_Characterisation/Verify_new_results/segment_level_analysis.ipynb -->

Single source notebook: `Toy_Characterisation/Verify_new_results/segment_level_analysis.ipynb` (96 cells, ~5 660 lines).

Cached artefacts (`outputs/segment_analysis/`):
- 2×2 paper plots per experiment (PDF + PNG): scattering/resolution sweeps, density scan, fixed-ε, zero-noise, extended (to 1000 tracks), §14 solver, §15 track-level, §16 spectrum, §17 tracker A/B.
- CSVs: `segment_{scattering,resolution,density,zero_noise}_sweep[_ext].csv`.
- `cache/*.pkl` for every expensive sweep (all sweeps cache-protected — results are persistent and re-runnable).

Key global settings (from §1, §2):

| Parameter | Value |
|---|---|
| Geometry | 5 modules, $\Delta z = 33$ mm, $z \in \{33,66,99,132,165\}$ mm |
| Module half-width | 40 mm (80 mm full) for §3–§9, §11–§12; 80 mm (160 mm full) for §10, §10b |
| Hamiltonian | $\gamma=3.0$, $\delta=1.0$, scale $=3.0$, $\theta_{\min}=1.5\times10^{-5}$ rad |
| ε formula | $\varepsilon = \sqrt{2(\text{scale}\,\sigma_s)^2 + 12\,\arctan^2(\text{scale}\,\sigma_r/\Delta z) + 2\theta_{\min}^2}$ |

Cell-level summary:

| Cell | Section | Content |
|---|---|---|
| 1–8 | §1–§2b | Intro, imports, geometry, pure-Python & Numba segment-metrics kernels |
| 9–14 | §3 | Single-event baseline (20 tracks, low noise) |
| 15–17 | §4 | Scattering histograms (7 σ_s values × 10 events) |
| 18–20 | §5 | Segment eff/FR vs σ_s × n_tracks × φ_max |
| 21–23 | §6 | Segment eff/FR vs σ_res × n_tracks × φ_max |
| 24–26 | §7 | Density scan (5–100 tracks, φ_max=0.2) |
| 27–28 | §8 | ROC acceptance curves |
| 29–31 | §9 | Summary tables & CSV export |
| 32–34 | §10 | Fixed-ε (triplet, new gen) |
| 35–37 | §10b | Fixed-ε (pairwise, old gen) + §10c power-law fits |
| 38–42 | §11 | Zero-noise benchmark (dense vs non-dense) |
| 43–60 | §12a–g | Extended sweeps to 1000 tracks (Numba) + paper figures §13 |
| 61–68 | §13 | Paper-ready figures (2×2) |
| 69–77 | §14 | Solver segment efficiency (classical Hamiltonian + 1% drop) |
| 78–87 | §15 | Track-level metrics (EventValidator, knee analysis) |
| 88–91 | §16 | Spectral diagnostics + solution histograms |
| 92–96 | §17 | Tracker A/B (`get_tracks` CC vs `get_tracks_layered`) |
