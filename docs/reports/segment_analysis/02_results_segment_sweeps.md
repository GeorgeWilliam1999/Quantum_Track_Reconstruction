# Results — segment-level sweeps (§3–§13)

<!-- STATUS: final -->
<!-- SOURCES: segment_level_analysis.ipynb §3–§13 -->

## §3 Single-event baseline
[source: cell #VSC-0cc48192 + baseline_angle_distributions.png]
20 tracks, $\sigma_s = 0.1$ mrad, $\sigma_r=0$, $\phi_{\max}=0.02$.
True-pair angles peak near zero (well below ε ≈ $0.7$ mrad in this configuration);
false-pair distribution extends over tens of mrad. The cleanly separated
log-density plot (`outputs/segment_analysis/baseline_angle_distributions.png`)
justifies the use of ε as an acceptance threshold.

## §4 Scattering-angle histograms
[source: §4 cells + `scattering_sweep_histograms.pdf`; cache `scatt_angle_results.pkl`]
7 σ_s values × 10 events, 20 tracks. True-angle mean grows roughly linearly with σ_s
(theory: $\langle|\theta_\text{true}|\rangle \propto \sigma_s$). At each σ_s the formula ε
tracks the rising distribution and keeps the true-pair acceptance > 80% while the false-pair
contamination is minimal (dominant false support is $\gg$ ε).

## §5 Segment eff / FR vs σ_scatt
[source: §5 cells; `scattering_sweep_phi{0.02,0.2}.pdf`; CSV `segment_scattering_sweep.csv`]
Sweep: σ_s ∈ {0.05,0.1,0.2,0.4,0.6,0.8,1.0} mrad, $n_\text{tracks}$ ∈ {10,20,50,100}, both angle cones, 10 repeats each.
Segment efficiency is flat at 100% across all σ_s for $n_\text{tracks} \le 100$ — the ε-formula successfully tracks the true-pair distribution. False rate rises monotonically with σ_s because ε widens and catches more unrelated combinatorial pairs.

## §6 Segment eff / FR vs σ_res
[source: §6 cells; `resolution_sweep_phi{0.02,0.2}.pdf`; CSV `segment_resolution_sweep.csv`]
σ_res ∈ {0, 5, 10, 15, 20, 30, 50} µm at fixed σ_s = 0.1 mrad, same grid as §5.
Behaviour qualitatively identical to §5 but driven by the $12\theta_r^2$ term;
ε grows sub-linearly with σ_res (because $\theta_r=\arctan(\text{scale}\cdot\sigma_r/\Delta z)$
is quasi-linear for our range).

## §7 Density scan
[source: §7 cells; `density_scan.pdf`; CSV `segment_density_scan.csv`]
$n_\text{tracks}$ = 5–100 in steps of 5, σ_s = 0.1 mrad, σ_r = 0, $\phi_{\max}=0.2$.
True pairs $\propto n$, false pairs scale as $\propto n^3$ for the shared-middle-hit method — confirmed by the reference $n^3$ line in panel (d).

## §8 ROC acceptance curves
[source: §8 cells; `acceptance_curves.pdf`]
For each σ_s the cumulative acceptance fraction is plotted as a function of threshold ∈ [0, 10] mrad; the operating point ε from the formula sits on the knee between high true-acceptance (>98%) and low false-acceptance (< 1% for σ_s ≤ 0.4 mrad).

## §10 / §10b Fixed-ε comparison (triplet vs pairwise)
[source: §10, §10b; `fixed_epsilon_comparison_new_vs_old.pdf`]
Both methods generate events with *identical* physics parameters (σ_s=0.1 mrad, σ_r=5 µm, PV σ=(1,1,1), 160 mm modules, MIP particles, 20 tracks/event) and use the same fixed ε=2 mrad. The *only* difference is the segment-pair definition.

Efficiency is 100% for both methods at every multiplicity tested (20–1000 total tracks) — both methods count all truth segments in the "true" pool and ε=2 mrad is wide enough to accept all of them.

False rate diverges:
- Triplet (shared middle hit): grows because $N_\text{false} \sim T^3$ while $N_\text{true} \sim T$ so even a tiny false-acceptance probability swamps true at high $T$.
- Pairwise (truth segments only): false pool is $\sim T^2$; even with identical ε, far fewer false acceptances.

## §10c Power-law scaling fits
[source: §10c cells; `scaling_power_law_fits_new_vs_old.pdf`]
Weighted non-linear fits to $N = a T^b$ using per-point SE as σ (curve_fit with `absolute_sigma=True`).
The notebook's markdown tabulates the fitted exponents for $N_\text{false}$, $N_\text{false\,accepted}$, $N_\text{true}$, $N_\text{true\,accepted}$ for both methods and compares against the combinatorial expectation (triplet: $b\approx 3$, pairwise: $b\approx 2$ for false; $b\approx 1$ for true either way). Exact fitted values are in the notebook cell output [source: §10c cell #VSC-ae61d74f stdout — table prints "b = ... ± ..." for each quantity].

<!-- TODO: extract exact fitted b values if final report needs them quantitatively -->

## §11 Zero-noise benchmark
[source: §11 cells; `zero_noise_benchmark.pdf`, `zero_noise_benchmark_paper.pdf`]
σ_s ≈ 1×10⁻⁸ (effectively zero), σ_r = 0. ε reduces to $\sqrt{2}\theta_{\min} \approx 0.021$ µrad (≈0.0000212 mrad). Two cone regimes:
- Non-dense ($\phi_{\max}=0.2$): segment eff ≈ 100%, false-rate baseline.
- Dense ($\phi_{\max}=0.02$): small-angle tracks more likely to satisfy ε by coincidence → higher false-rate tail even at zero noise.

Used as a floor to quantify combinatorial contamination in the absence of physical noise.

## §12 Extended sweeps (to 1000 tracks)
[source: §12 cells; `scattering_sweep_ext_phi*.pdf`, `resolution_sweep_ext_phi*.pdf`, `density_scan_ext.pdf`, `zero_noise_benchmark_ext*.pdf`; `segment_*_ext.csv`]
Identical sweeps to §5–§7, §11 but with $n_\text{tracks}$ extended to {10,20,50,100,200,500,1000}. Numba kernel makes the $O(T^3)$ triplet enumeration tractable (the $n=1000$ sweep uses $\sim 4 \times 10^6$ false triplets per event; ~seconds/event with the vectorised builder).

Behaviour at high $n$: efficiency stays at 100% across σ_s and σ_r ranges; false rate grows roughly as $n$ (because for the triplet method, false pairs $\propto n^3$ and total accepted $\propto n$ under a fixed angular cut).

## §13 Paper figures
[source: §13; `outputs/segment_analysis/paper/fig13{a,b,c,d,e}.pdf`]
2×2 "publication" versions of §4, §5, §6, §7, §11. §13d uses the pairwise all-pairs metric (`fast_pairwise_metrics`) validated against the pure-Python reference and is the figure referenced from the paper write-up.
