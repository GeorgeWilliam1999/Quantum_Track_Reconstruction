# ERF — smooth angular cost with histogram-based thresholds

## Motivation

Task 5: replace the step-function angular cost (used by Tasks 2-4) with
the smooth ERF variant

$$
\mathcal{C}(\theta) = 1 + \mathrm{erf}\!\left(\frac{\varepsilon-\theta}{\theta_d\sqrt{2}}\right),
$$

and study how the soft-threshold width $\theta_d$ (`erf_sigma` in the
codebase) interacts with $(\sigma_{\rm res}, \sigma_{\rm scatt})$. This
is the only experiment with `convolution=1`.

Because the ERF cost is continuous, the right operating threshold for
the *solver* output is no longer a hand-picked $\tau \cdot \max(\mathrm{sol})$;
we determine it from the histograms of true/false segment scores
(stored verbatim in every pickle).

## Grid

| axis | values |
|------|--------|
| $\theta_d$ (rad)        | $\{10^{-6},\,10^{-5},\,5\!\times\!10^{-5},\,10^{-4},\,5\!\times\!10^{-4},\,10^{-3}\}$ |
| $(\sigma_{\rm scatt}, \sigma_{\rm res})$ | $\{(10^{-4},\,0),\,(3\!\times\!10^{-4},\,0.01),\,(5\!\times\!10^{-4},\,0.02)\}$ |
| $T$                     | $\{10, 20, 50, 100, 200, 400, 700, 1000\}$ |

$\theta_d = 10^{-6}$ is the **regression check** — at this width the ERF
collapses back to a step, so each $(\sigma_{\rm res},\sigma_{\rm scatt})$
row should reproduce its Phase 1/2 baseline.

Reps: 10/5/3. Convolution = 1 throughout. $\phi_{\max}=0.2$, hit
inefficiency 0.

## Threshold strategy

The pickle for every event stores `true_angles`, `false_angles`,
`sol_C`, `sol_Q`, and `truth`. The analysis notebook does:

1. Pool true and false segment scores across reps within each grid cell.
2. Find the score threshold that maximises Youden's J statistic
   (`tpr - fpr`) and the threshold at the equal-error rate.
3. Re-evaluate efficiency / purity / false-rate at the histogram-driven
   threshold.
4. Compare to the default $\tau=0.35\cdot\max$ baseline.
