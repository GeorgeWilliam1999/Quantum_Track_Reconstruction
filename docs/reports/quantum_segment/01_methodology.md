# Methodology — Hamiltonian, cost function, parameter choices

<!-- STATUS: final -->
<!-- SOURCES: LHCb_VeLo_Toy_Model/src/lhcb_velo_toy/solvers/hamiltonians/fast.py; condor_obqf/run_event.py; segment_level_analysis.ipynb §1/§2/§14e -->

## Segments
Given $n_M = 5$ planar modules at $z_k = 33 + 33k$ mm, $k = 0,\dots,4$, a *segment* $\sigma_i = (h_a, h_b)$
is a pair of hits on consecutive modules. The total segment count is
$$ N_s \;=\; \sum_{k=0}^{n_M-2} n_k^{\text{hits}}\, n_{k+1}^{\text{hits}}. $$
For Poissonian truth-tracks with $n_M$ hits each, $N_s \sim T^2$ on average; the
**triplet** combinatorial pool of *pairs of segments sharing a middle hit* scales as $\sim T^3$
(see `segment_level_analysis.ipynb` §7 / §13a). Measured `n_seg_mean` from the §14e mirror:
| T | $\bar N_s$ | $\bar N_s / T^2$ |
|---:|---:|---:|
| 2 | 15.7 | 3.93 |
| 10 | 395 | 3.95 |
| 100 | 39 199 | 3.92 |
| 1000 | 3 907 530 | 3.91 |
[source: aggregate.csv `n_seg_mean`]

## Hopfield-style Hamiltonian
[source: `fast.py::construct_hamiltonian`, lines 268–410]
The library builds the symmetric SPD operator
$$ A_{ij} \;=\; \begin{cases}
\gamma + \delta, & i = j \\
-1, & i \ne j,\; \sigma_i,\sigma_j \text{ share a hit},\; \angle(\sigma_i,\sigma_j) < \varepsilon \\
0, & \text{otherwise}
\end{cases} $$
and $b_i = \delta$ for all $i$.

Implementation details (verified line-by-line in `fast.py`):
1. Off-diagonals are populated **per shared-middle hit**: for every adjacent module pair $(g,g+1)$,
   for every hit $h$ in module $g+1$, every "incoming" segment ending at $h$ is paired with every
   "outgoing" segment starting at $h$. This is the *triplet* / *shared-hit* enumeration that
   matches `segment_level_analysis.ipynb` §2b.
2. Angular compatibility: $\cos\angle(\sigma_i,\sigma_j) > \cos\varepsilon$, evaluated on the
   pre-computed unit segment vectors. The code calls this `convolution=False` (a hard step).
3. The library stores `self.A = -A_raw, self.b = b` so that the negated matrix has positive
   diagonal $\gamma+\delta$ and negative off-diagonals (sign convention used downstream).
4. Optional `convolution=True` path replaces the step by $1 + \mathrm{erf}((\varepsilon-\theta)/\sqrt 2\,\theta_d)$
   (range $[0,2]$); **not used** in the §14e mirror (`run_event.py` line 178 passes `convolution=False`).

## Cost function
The Hamiltonian implements the quadratic energy
$$ \mathcal{H}(s) \;=\; \tfrac{1}{2}\, s^\top A\, s \;-\; b^\top s
\;=\; \tfrac{1}{2}(\gamma+\delta)\|s\|^2 \;-\; \sum_{(i,j)\in E} s_i s_j \;-\; \delta \sum_i s_i, $$
where $E$ is the compatibility graph defined above. Minimising $\mathcal{H}$ in $\mathbb{R}^{N_s}$
gives the unique SPD fixed point $s^* = A^{-1} b$.

Hopfield fixed-point sanity checks (from §14b/§16c of the classical notebook):
- Isolated segment: $s^* = \delta/(\gamma+\delta) = 0.25$ at $(\gamma,\delta)=(3,1)$ — observed peak of the *false* segment population.
- Segment on the outer module of a 5-hit truth chain (one neighbour): $s^* = 0.3636$ — observed peak of the *true* outermost-segment population.

These two peaks bracket the activation threshold $\tau = 0.35$ used in the §14e operating point.

## Activation rule
Two thresholding modes appear in the codebase:
1. **Absolute** (used in §14 of the classical notebook and §14e of the quantum mirror):
   $a_i = \mathbb{1}[\, s_i > \tau\,]$ with $\tau = 0.35$.
2. **Relative** / per-event normalised (used in `get_tracks_layered`, threshold 0.45 default;
   also explored as a "fix" in §7c of the quantum notebook).

The notebook §7e τ-sweep (this work) explicitly varies the **absolute** $\tau \in \{0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70\}$.

## ε model
[source: `segment_level_analysis.ipynb` §1 `compute_epsilon`; mirrored §14e constant]
$$ \varepsilon \;=\; \sqrt{2\,(\mathrm{scale}\cdot\sigma_s)^2 \;+\; 12\,\arctan^2(\mathrm{scale}\cdot\sigma_r/\Delta z) \;+\; 2\,\theta_{\min}^2}. $$
For the §14e operating point ($\sigma_s = 1\times10^{-4}$ rad, $\sigma_r = 5$ µm, scale = 3, $\theta_{\min}=1.5\times10^{-5}$ rad, $\Delta z = 33$ mm) this formula gives $\varepsilon \approx 4.6\times10^{-4}$ rad.
**However**: `run_event.py` overrides this with a **fixed** $\varepsilon = 2\times10^{-3}$ rad = 2 mrad [source: lines 144, default argument]. This is the same fixed 2 mrad value used in §14–§17 of the classical notebook and `docs/reports/segment_analysis/04_consistency_audit.md`.

## §14e operating point (used by all of `seg14e_T1000_{statevector,sampling}`)
| Parameter | Value | Source |
|---|---|---|
| Particle | pion (m = 139.6 MeV/c², q = 1) | `run_event.py::_safe_generate` |
| Modules | 5 planar, half-width 40×40 mm² | id. |
| PV σ | (0, 0, 1) mm | id. |
| Cone | $\phi_{\max}=\theta_{\max}=0.2$ rad | id. |
| σ_scatt | $1\times10^{-4}$ rad (per-module direction noise) | id. |
| σ_res | $5\times10^{-6}$ m (hit smearing) | id. |
| Drop rate | 0.01 (1% hit drop applied **after** efficiency denominator is fixed) | id. |
| γ | 3.0 | `run_job` default |
| δ | 1.0 | id. |
| ε | $2\times10^{-3}$ rad | id. |
| τ | 0.35 | id. |
| Validator | purity≥0.70, hit_eff_min=0.0, min_rec_hits=3 | id. |

**Important caveat**: γ=3 puts the spectrum of $A$ across the QPE rejection notch of `OneBitHHL`
(see `02_onebit_hhl.md`). This was the choice in the §14e classical paper; we deliberately
keep it for the head-to-head comparison even though it is provably sub-optimal for the
quantum solver.

## Metric definitions
For each event, with truth mask $\mathcal{T}$ (segment truth-track agreement) and active set
$\mathcal{A}(\tau) = \{i : s_i > \tau\}$:

- $n_\text{true,active} = |\mathcal{T} \cap \mathcal{A}|$
- $n_\text{false,active} = |\mathcal{A}| - n_\text{true,active}$
- $n_\text{true,clean}$ = truth-segment count on the **un-dropped** event (efficiency denominator)
- **Segment efficiency**: $\mathrm{eff} = n_\text{true,active}\,/\,n_\text{true,clean}$
- **Segment purity**: $\mathrm{pur} = n_\text{true,active}\,/\,|\mathcal{A}|$
- **Segment false rate** (§14e convention used in this report): $\mathrm{FR} = 1 - \mathrm{pur} = n_\text{false,active}\,/\,|\mathcal{A}|$
- **False-acceptance rate** (`far_*` column in the CSV): $n_\text{false,active}\,/\,n_\text{false,all}$ — orders of magnitude smaller because $n_\text{false,all} \sim T^3$.

Both `pur_*_mean` (→ FR) and `far_*_mean` are reported.
