# Calculated epsilon — derivation note

The 1BQF segment Hamiltonian (constructed in `SimpleHamiltonianFast`)
penalises segment triplets whose mid-hit **kink angle** exceeds a threshold
$\varepsilon$. A triplet is three consecutive hits $(\mathbf r_1,\mathbf r_2,\mathbf r_3)$
on three planes spaced by $\Delta z$; the kink is the change of direction at
the middle hit. Because a track lives in 3D, the kink has **two independent
transverse projections** ($x$ and $z$ wrt the beam, $y$ and $z$), and the cut
acts on the 3D kink magnitude $\theta=\sqrt{\theta_x^2+\theta_y^2}$.

Two independent sources broaden each projected kink:

1. **Multiple scattering.** At the middle plane the particle direction is
   rotated by a projected Coulomb-scattering angle
   $\Delta\theta\sim\mathcal N(0,\sigma_{\rm scatt}^2)$ *in each projection
   separately*. This adds directly to the projected kink, contributing
   variance $\sigma_{\rm scatt}^2$ **per projection**. (In detector terms
   $\sigma_{\rm scatt}=\theta_0$ is the Highland angle,
   $\propto (1/p_T)\sqrt{x/X_0}$ — see `detector_physics_report.md`.)

2. **Measurement resolution.** In one projection the measured kink is the
   discrete second difference of the three hit coordinates,
   $$
   \theta_x \;=\; \frac{x_3-x_2}{\Delta z}-\frac{x_2-x_1}{\Delta z}
              \;=\; \frac{x_1-2x_2+x_3}{\Delta z}.
   $$
   With independent errors $x_i\to x_i+\mathcal N(0,\sigma_{\rm res}^2)$ and
   second-difference weights $(1,-2,1)$,
   $$
   \mathrm{Var}(\theta_x)=\frac{1^2+2^2+1^2}{\Delta z^2}\,\sigma_{\rm res}^2
                         =\frac{6\,\sigma_{\rm res}^2}{\Delta z^2}.
   $$
   The factor **6** is exactly $1+4+1$ from the $(1,-2,1)$ stencil. For large
   angles $\sigma_{\rm res}/\Delta z\to\arctan(\sigma_{\rm res}/\Delta z)$.

3. **Pixel-pitch floor** $\theta_{\min}\sim1.5\times10^{-5}$ rad: a residual
   angular dispersion even for noise-free triplets.

Adding the two sources in quadrature gives the **per-projection kink variance**
$$
\boxed{\;\sigma_p^2 \;=\; \sigma_{\rm scatt}^2
        + 6\,\arctan^2\!\bigl(\sigma_{\rm res}/\Delta z\bigr)\;}
        \;\approx\; \sigma_{\rm scatt}^2 + 6\,\sigma_{\rm res}^2/\Delta z^2 .
$$
The 3D magnitude $\theta=\sqrt{\theta_x^2+\theta_y^2}$ with $\theta_{x,y}\sim
\mathcal N(0,\sigma_p^2)$ is **Rayleigh-distributed** with scale $\sigma_p$, so
$\mathbb E[\theta^2]=2\sigma_p^2$ and the survival function is
$P(\theta>\varepsilon)=\exp(-\varepsilon^2/2\sigma_p^2)$.

We accept up to $s$ standard deviations (the legacy code uses $s=3$). Setting
$\varepsilon^2=2(s\sigma_p)^2$ — the factor **2** is the two projections /
$\mathbb E[\theta^2]=2\sigma_p^2$, **not** two inter-plane gaps — gives

$$
\varepsilon = \sqrt{2(s\sigma_{\rm scatt})^2
                    + 12\,\arctan^2\!\bigl(s\sigma_{\rm res}/\Delta z\bigr)
                    + 2\theta_{\min}^2}
            \;=\; s\sqrt2\,\sigma_p .
$$

The $2$ on scattering and $12=2\times6$ on resolution are the per-projection
coefficients ($1$ and $6$) times the projection factor $2$, with the $s^2$
rescale pulled inside. At $\varepsilon=s\sqrt2\,\sigma_p$ the per-kink miss
probability is pinned at $p=\exp(-s^2)=e^{-9}=1.2\times10^{-4}$, independent of
$\sigma_{\rm scatt}$ and $\sigma_{\rm res}$. Inverting,
$\varepsilon=\sigma_p\sqrt{2\ln(1/p)}$ sets the acceptance from a target
inclusion rate.

Implementation: `lhcb_velo_toy.analysis.compute_epsilon` (and the re-export
from `Toy_Characterisation._shared.helpers`); defaults $\Delta z=33$ mm,
$s=3$, $\theta_{\min}=1.5\times10^{-5}$ rad.

References: `docs/reports/segment_analysis/segment_level_report.pdf` §2;
detector-physics translation in
`Toy_Characterisation/Epsilon_study_2/outputs/detector_physics_report.md`.
