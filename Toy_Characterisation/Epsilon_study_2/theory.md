# Calculated epsilon — derivation note

The 1BQF segment Hamiltonian (constructed in `SimpleHamiltonianFast`)
penalises segment triplets whose mid-hit kink angle exceeds a threshold
$\varepsilon$. Whether a true triplet survives the cut depends on two
independent sources of angular spread:

1. **Multiple scattering** in the detector material rotates the
   particle direction by $\Delta\theta \sim \mathcal{N}(0,\sigma_{\rm scatt}^2)$
   between adjacent planes. The total kink angle accumulated over two
   inter-plane gaps is a sum of two Gaussian components plus their
   correlation, with effective variance $2\sigma_{\rm scatt}^2$.

2. **Measurement resolution** $\sigma_{\rm res}$ on the (x, y) hit
   position propagates into the apparent kink angle:
   small perturbations of all three hits, by independent
   $\mathcal{N}(0,\sigma_{\rm res}^2)$ noise in each transverse
   coordinate, induce a small-angle deviation. For inter-plane spacing
   $\Delta z$, leading order gives
   $\Delta\theta_{\rm res} \approx \sigma_{\rm res}/\Delta z$ per
   plane, and the cumulative variance over the three hits feeding the
   triplet, projected onto the kink, comes out to
   $\sigma_\theta^2 \approx 6 \arctan^2(\sigma_{\rm res}/\Delta z)$.

3. **Pixel binning floor** $\theta_{\min}$: even noise-free triplets
   inherit a residual angular dispersion of order
   $\theta_{\min}\sim 1.5\!\times\!10^{-5}$ rad from the pixel pitch.

We choose a "scale" factor $s$ (number of standard deviations we are
willing to accept; the legacy code uses $s=3$). The cumulative
acceptance threshold is then

$$
\varepsilon = \sqrt{2(s\sigma_{\rm scatt})^2 + 12\arctan^2\!\Bigl(\frac{s\sigma_{\rm res}}{\Delta z}\Bigr) + 2\theta_{\min}^2}.
$$

The factor of 2 on the scattering term and 12 on the resolution term
both come from summing two contributions and applying the $s^2$ rescale.
The $\arctan$ keeps the expression sensible for large $\sigma_{\rm res}/\Delta z$
(in our regime $\arctan(x)\approx x$ and the term reduces to
$12 (s\sigma_{\rm res}/\Delta z)^2$).

Implementation: `lhcb_velo_toy.analysis.compute_epsilon` (and the
re-export from `Toy_Characterisation._shared.helpers`).

References: `docs/reports/segment_analysis/segment_level_report.pdf` §2.
