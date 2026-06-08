# Larger_Scatter_Density — tighter angular cone with larger scattering

## Motivation

Task 4: combine *larger* material scattering with *tighter* angular
acceptance ($\phi_{\max} = \theta_{\max}$). A narrower cone packs the same
number of tracks into less solid angle, raising the local track density
and stressing the segment-level discriminator more than either knob
alone. Convolution remains **off**.

## Grid

| axis | values |
|------|--------|
| $\phi_{\max}$ (rad)         | $\{0.2,\,0.1,\,0.05,\,0.02,\,0.01\}$ |
| $\sigma_{\rm scatt}$ (rad)  | $\{1\!\times\!10^{-4},\,3\!\times\!10^{-4}\}$ |
| $T$                         | $\{10, 20, 50, 100, 200, 400, 700, 1000\}$ |

$\sigma_{\rm res}=0$, hit inefficiency $=0$. Reps: 10/5/3.

$\varepsilon$ is `compute_epsilon(0, sigma_scatt)` per row.

## Diagnostics

- Segment efficiency vs $T$ for each $(\phi_{\max}, \sigma_{\rm scatt})$.
- Pair-angle histograms at the tightest cone, highest $T$ to detect
  $\varepsilon$ overlap.
- Quantum-classical cosine vs density proxy
  $T / \phi_{\max}^2$.
