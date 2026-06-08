# Larger_Scatter — larger material scattering and hit inefficiency

## Motivation

Task 3: study the segment-level algorithm under *more realistic* detector
conditions, individually and combined:

- $\sigma_{\rm scatt}$ ranges over $\{10^{-4},\,3\!\times\!10^{-4},\,5\!\times\!10^{-4},\,7\!\times\!10^{-4},\,10^{-3}\}$ rad
  (the legacy nominal of $10^{-4}$ is the lower edge).
- hit inefficiency $p_{\rm drop} \in \{0,\,1\%,\,2\%,\,5\%,\,10\%\}$
  (per-hit Bernoulli dropout, applied after generation but before the
  Hamiltonian build).

$\sigma_{\rm res}$ is held at 0 mm to isolate the two effects (resolution
is the focus of Epsilon_study_2). Convolution remains **off** (step
Hamiltonian) per the supervisor's instruction.

## Three sub-experiments — cartesian grid

We run the full $5 \times 5 = 25$ Cartesian grid in one CSV (which
naturally contains the "scatter only" rows at $p_{\rm drop}=0$, the
"inefficiency only" rows at $\sigma_{\rm scatt}=10^{-4}$, and all 16
combined points). Analysis filters into the three views.

| axis | values |
|------|--------|
| $\sigma_{\rm scatt}$ (rad) | $\{1, 3, 5, 7, 10\}\!\times\!10^{-4}$ |
| $p_{\rm drop}$ | $\{0,\,0.01,\,0.02,\,0.05,\,0.10\}$ |
| $T$ | $\{10, 20, 50, 100, 200, 400, 700, 1000\}$ |

Reps taper as in Phase 1 (10/5/3).

Epsilon is computed per row via `compute_epsilon(0, sigma_scatt)` so the
geometric acceptance widens consistently with the larger scattering.

## Diagnostics

- Segment efficiency / purity / false-rate vs $T$ for each grid cell.
- Pair-angle histograms at high $T$ to verify $\varepsilon$ still
  separates true from false.
- Combined-effect surface plots (eff vs $\sigma_{\rm scatt}$ and
  $p_{\rm drop}$).
