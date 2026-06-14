# Detector Physics of Segment Reconstruction
### How $\sigma_{\rm scatt}$ and $\sigma_{\rm res}$ map to momentum, material and resolution in the Quantum-VeLo toy

*Epsilon_study_2 — detector-physics translation layer. Companion to §7 (ε-sensitivity), §7.10 (T=400 + efficiency-first τ), §7.11 (sparsity), §7.12 (indefiniteness). Geometry: 5 planes at $z=33,66,99,132,165$ mm, $\Delta z=33$ mm, half-aperture 40 mm, slopes $\lesssim0.2$; Hamiltonian $\gamma=3,\delta=1$; acceptance scale $s=3$.*

---

## 0. Reading guide — the whole report in one equation and one number

The toy is driven by two noise knobs. The supervisor's three physical questions map onto them as

$$
\underbrace{\sigma_{\rm scatt}}_{\text{collision/scatter knob}}\;\longleftrightarrow\;\Big(\text{transverse momentum }p_T,\ \text{material }x/X_0\Big),
\qquad
\underbrace{\sigma_{\rm res}}_{\text{measurement knob}}\;\longleftrightarrow\;\Big(\text{pixel pitch / single-hit resolution}\Big).
$$

Both enter the reconstruction through a **single scalar**, the per-projection kink RMS

$$
\boxed{\;\sigma_p^2=\sigma_{\rm scatt}^2+6\,\frac{\sigma_{\rm res}^2}{\Delta z^2}
   \;=\;\underbrace{\Big[\tfrac{13.6\,\text{MeV}}{p_T}\sqrt{x/X_0}\,(1+0.038\ln x/X_0)\Big]^2}_{\text{momentum + material}}
   \;+\;\underbrace{\frac{6}{\Delta z^2}\Big(\tfrac{\text{pitch}}{\sqrt{12}}\Big)^2}_{\text{resolution}}\;}
$$

and the Hamiltonian acceptance is $\varepsilon=s\sqrt2\,\sigma_p$. **Everything downstream — efficiency, false rate, matrix sparsity, the solver break — is a function of $\sigma_p$ alone.** The detector physics is the job of writing $\sigma_p$ in terms of $p_T$, $x/X_0$ and pitch, which is what the two terms above do.

The **one number** that organises the discussion is the *crossover momentum*, where the two terms are equal:

$$
\boxed{\;p_{\rm cross}=\sqrt2\,\frac{\Delta z}{\text{pitch}}\,(13.6\,\text{MeV})\sqrt{x/X_0}\,(1+0.038\ln x/X_0)\;\approx\;0.95\ \text{GeV}\;}
$$

for VELO-like values (pitch $=55\,\mu$m, $\Delta z=33$ mm, $x/X_0=1\%$). Below $p_{\rm cross}$ the kink is **scattering-limited** (soft tracks, thick material); above it the kink is **resolution-limited** (stiff tracks, fine pixels).

**Three headline messages** follow, each derived in full below:

1. **Efficiency is detector-physics-independent.** Because $\varepsilon\propto\sigma_p$, the per-kink acceptance is pinned at a fixed miss probability $p=e^{-s^2}=e^{-9}$ (the chance a true kink is rejected; defined in §4) regardless of $p_T$, $x/X_0$ or pitch. Segment efficiency is *dial-set*, not noise-set (§5; confirmed flat from 11 GeV down to 0.36 GeV in the new soft-track run, DP4a).
2. **The detector quality is paid as a false-coupling tax** $\propto\sigma_p^2$. Soft tracks, thick material and coarse pixels each widen $\varepsilon$ and admit a quadratically larger combinatorial background (§6; DP4b, DP5).
3. **Where it breaks is a soft/coarse $\times$ high-occupancy corner.** The solver-level instability needs a wide $\varepsilon$ *and* many tracks, i.e. low $p_T$ or coarse pixels at HL-LHC pileup (§8).

> **Honest caveat up front.** The production grid $\sigma_{\rm scatt}\in\{1,3,5\}\times10^{-4}$ rad corresponds to $p_T\approx\{11,3.7,2.2\}$ GeV — **all above** $p_{\rm cross}$, i.e. entirely resolution-limited. We had never simulated a scattering-limited (sub-GeV) track. The new soft-track extension in this report (§5.4, §6.3) is the first look below 1 GeV.

---

## 1. The model and the two knobs

### 1.1 Geometry and segments
A track is a sequence of hits, one per plane, on $N_{\rm pl}=5$ equally spaced planes ($\Delta z=33$ mm). A **segment** is a directed pair of hits on adjacent planes; a track of 5 hits is a chain of 4 segments. A **triplet** (three consecutive hits) defines a **kink** at its middle hit — the angle between its two segments. With $T$ tracks the algorithm forms $\mathcal O(T^2)$ candidate segments (one per (in-hit, out-hit) pair on each plane gap); exactly $4T$ of them are true.

### 1.2 The segment Hamiltonian and the acceptance $\varepsilon$
The 1BQF solver works with the symmetric matrix
$$
A=(\gamma+\delta)\,\mathbb I - C,\qquad A\mathbf x=\delta\,\mathbf 1,\qquad \gamma=3,\ \delta=1,
$$
where $C$ is the **0/1 coupling (adjacency) matrix**: $C_{ij}=1$ iff segments $i,j$ share a hit *and* the triplet they form has kink angle $\theta_{ij}<\varepsilon$. So $\varepsilon$ is the single geometric acceptance: only kinks tighter than $\varepsilon$ become couplings. The solution amplitude $x_i$ scores each segment; a segment is **reconstructed** if $x_i>\tau$ (default $\tau=0.35$). Sections 5–8 show how $A$, $\mathbf x$ and the metrics depend on $\varepsilon$, hence on $\sigma_p(\sigma_{\rm scatt},\sigma_{\rm res})$.

### 1.3 The two noise knobs in the toy
- $\sigma_{\rm scatt}$ — the RMS of the random direction change ("collision noise") applied to the track at each plane, **per transverse projection**.
- $\sigma_{\rm res}$ — the RMS of the independent Gaussian measurement error added to each hit coordinate ("measurement error"), in millimetres.

The entire report is about giving these two knobs their detector meaning and tracing them through the algebra.

---

## 2. From detector physics to the two knobs (the translation)

### 2.1 Multiple scattering $\Rightarrow\ \sigma_{\rm scatt}$ (the $p_T$ + thickness knob)
A charged particle crossing material of thickness $x$ (in radiation lengths $X_0$) suffers many small-angle Coulomb deflections. The central part of the distribution of the **projected** scattering angle is Gaussian with RMS given by the Highland/Lynch–Dahl formula (natural units $c=1$, momentum in MeV; $\beta$ is the particle velocity and $z$ its charge — and note we write the momentum as $p_T$ throughout this report, *distinct* from the per-kink miss probability $p$ of §4):
$$
\theta_0=\frac{13.6\ \text{MeV}}{\beta\,p_T}\,z\,\sqrt{x/X_0}\,\Big[1+0.038\ln\!\big(x z^2/X_0\beta^2\big)\Big].
$$
For a relativistic singly charged particle ($\beta\!\approx\!1$, $z=1$),
$$
\boxed{\;\sigma_{\rm scatt}\equiv\theta_0=\frac{13.6\ \text{MeV}}{p_T}\,\sqrt{x/X_0}\,\big(1+0.038\ln x/X_0\big)\;}
$$
This single line answers two of the supervisor's three questions at once:
- **Momentum:** $\sigma_{\rm scatt}\propto1/p_T$. Soft tracks scatter more; stiff tracks barely kink.
- **Material/thickness:** $\sigma_{\rm scatt}\propto\sqrt{x/X_0}$. A thicker (more material) detector scatters more. (The logarithm is a slow correction; at $x/X_0=1\%$ it multiplies by $1+0.038\ln0.01=0.825$, a 17.5 % reduction.)

$\sigma_{\rm scatt}$ is therefore a **degenerate proxy** for $p_T$ and $x/X_0$: the same value can mean a soft track in a thin detector or a stiff track in a thick one. Holding one fixed converts the knob to the other. **DP3** plots exactly this — $p_T(\sigma_{\rm scatt})$ at three thicknesses $x/X_0=0.5,1,2\%$ — so a given $\sigma_{\rm scatt}$ reads off a different momentum on each curve.

### 2.2 Hit resolution $\Rightarrow\ \sigma_{\rm res}$ (the pixel knob)
For a pixel of pitch $w$ with binary (hit/no-hit) readout, the true position is uniform across the pixel, so the measurement error has variance $\int_{-w/2}^{w/2}\!u^2\,du/w=w^2/12$:
$$
\boxed{\;\sigma_{\rm res}=\frac{\text{pitch}}{\sqrt{12}}\;}
$$
Charge sharing / analog interpolation on inclined tracks improves on this, so $\text{pitch}/\sqrt{12}$ is a conservative ceiling. $\sigma_{\rm res}$ is the **pure resolution knob** — it does *not* depend on $p_T$. The VELO pixel ($w=55\,\mu$m) gives $\sigma_{\rm res}=15.9\,\mu\text{m}=0.0159$ mm, which sits between our grid cells $\sigma_{\rm res}=0.01$ and $0.02$ mm.

### 2.3 The production grid in physical units
| knob | toy values | physical meaning |
|---|---|---|
| $\sigma_{\rm scatt}$ | $1,3,5\times10^{-4}$ rad | $p_T\approx 11.2,\,3.7,\,2.2$ GeV at $x/X_0=1\%$ (DP3) |
| $\sigma_{\rm res}$ | $0,\,0.01,\,0.02,\,0.05$ mm | pitch $0,\,35,\,69,\,173\,\mu$m (binary) |

VELO reference: $\sigma_{\rm res}\approx0.0159$ mm (55 µm pixel); soft-track regime needs $\sigma_{\rm scatt}\gtrsim10^{-3}$ rad ($p_T\lesssim1$ GeV).

---

## 3. The kink scale $\sigma_p$ — where the two knobs combine

### 3.1 Derivation of $\sigma_p^2=\sigma_{\rm scatt}^2+6\sigma_{\rm res}^2/\Delta z^2$
The kink lives in 3D and has two independent transverse projections; we derive the variance of one projection (say $x$) and the same holds for $y$.

**Scattering contribution.** The middle-plane deflection adds $\Delta\theta\sim\mathcal N(0,\sigma_{\rm scatt}^2)$ directly to the projected kink → variance $\sigma_{\rm scatt}^2$.

**Resolution contribution.** With perfect directions the projected kink is the **discrete second difference** of the three measured coordinates,
$$
\theta_x=\frac{x_3-x_2}{\Delta z}-\frac{x_2-x_1}{\Delta z}=\frac{x_1-2x_2+x_3}{\Delta z}.
$$
Adding independent errors $x_i\to x_i+\mathcal N(0,\sigma_{\rm res}^2)$ and using the stencil weights $(1,-2,1)$,
$$
\mathrm{Var}(\theta_x)=\frac{1^2+(-2)^2+1^2}{\Delta z^2}\,\sigma_{\rm res}^2=\frac{6\,\sigma_{\rm res}^2}{\Delta z^2}.
$$
The famous **factor 6 is just $1+4+1$**. (For large lever angles replace $\sigma_{\rm res}/\Delta z$ by $\arctan(\sigma_{\rm res}/\Delta z)$; in our regime they agree to 1 part in $10^6$.)

Adding the two independent sources in quadrature gives the **per-projection kink variance**
$$
\boxed{\;\sigma_p^2=\sigma_{\rm scatt}^2+6\,\frac{\sigma_{\rm res}^2}{\Delta z^2}\;}
$$
**This is the central object.** $\sigma_{\rm scatt}$ and $\sigma_{\rm res}$ never appear separately again — only through $\sigma_p$.

> **Figure DP1 — `dp1_sigma_p_decomposition.png`.** How the per-projection kink RMS $\sigma_p$ (vertical axis) is assembled from the two knobs. *Panel (a), log–log:* $\sigma_p$ vs $\sigma_{\rm scatt}$, one curve per $\sigma_{\rm res}$. Each curve is **flat** at its resolution floor $\sqrt6\,\sigma_{\rm res}/\Delta z$ (horizontal dotted lines) until the diagonal scattering term $\sigma_{\rm scatt}$ climbs above it — the elbow is the crossover $\sigma^*_{\rm scatt}$. The grey band marks the production grid; the **top axis** re-labels $\sigma_{\rm scatt}$ as $p_T$ (at $x/X_0=1\%$), so the same plot reads as "kink blur vs momentum." *Panel (b), linear:* $\sigma_p$ vs $\sigma_{\rm res}$, one curve per $\sigma_{\rm scatt}$; $\sigma_p$ rises with $\sigma_{\rm res}$ (top axis = pixel pitch), the dashed line marking the VELO 55 µm pixel. **Read it as:** the two knobs combine only in quadrature — whichever term is larger sets $\sigma_p$, and the elbow is where scattering and resolution swap dominance.

### 3.2 The acceptance and the Rayleigh law
The 3D kink magnitude $\theta=\sqrt{\theta_x^2+\theta_y^2}$ with $\theta_{x,y}\sim\mathcal N(0,\sigma_p^2)$ is **Rayleigh-distributed**, pdf $f(\theta)=(\theta/\sigma_p^2)e^{-\theta^2/2\sigma_p^2}$, with
$$
\mathbb E[\theta^2]=2\sigma_p^2,\qquad P(\theta>\varepsilon)=e^{-\varepsilon^2/2\sigma_p^2}.
$$
Cutting at $s$ standard deviations means $\varepsilon^2=2(s\sigma_p)^2$ (the **factor 2** is the two projections / $\mathbb E[\theta^2]=2\sigma_p^2$ — *not* "two inter-plane gaps", which was a misstatement in earlier notes, now corrected in `theory.md`). Expanding $\sigma_p$ reproduces `compute_epsilon` exactly:
$$
\varepsilon=\sqrt{2(s\sigma_{\rm scatt})^2+12\arctan^2(s\sigma_{\rm res}/\Delta z)+2\theta_{\min}^2}=s\sqrt2\,\sigma_p,
$$
Here $\theta_{\min}=1.5\times10^{-5}$ rad is a tiny fixed pixel-pitch angular floor (utterly negligible in our regime); the $2$ on scattering and $12=2\times6$ on resolution are the per-projection coefficients ($1$ and $6$) times the projection factor $2$.

### 3.3 The crossover momentum — DP2, DP3
The scattering and resolution terms in $\sigma_p^2$ are equal when $\sigma_{\rm scatt}=\sqrt6\,\sigma_{\rm res}/\Delta z\equiv\sigma^*_{\rm scatt}$. At the VELO-like cell $\sigma_{\rm res}=0.01$ mm this is $\sigma^*_{\rm scatt}=7.4\times10^{-4}$ rad. Substituting $\sigma_{\rm scatt}=\theta_0(p_T)$ (§2.1) and $\sigma_{\rm res}=\text{pitch}/\sqrt{12}$ gives the crossover momentum
$$
p_{\rm cross}=\frac{13.6\,\text{MeV}\sqrt{x/X_0}\,(1+0.038\ln x/X_0)}{\sigma^*_{\rm scatt}}
=\sqrt2\,\frac{\Delta z}{\text{pitch}}\,13.6\,\text{MeV}\sqrt{x/X_0}\,(1+0.038\ln x/X_0)\approx0.95\ \text{GeV}.
$$
**DP2** is the two-dimensional version and **DP3** locates our grid on the momentum axis.

> **Figure DP2 — `dp2_crossover_map.png`.** Which kink source dominates, across detector phase space. Colour $=\log_{10}$ of the dominance ratio $\sigma_{\rm scatt}^2/(6\sigma_{\rm res}^2/\Delta z^2)$: **red = scattering-limited, blue = resolution-limited**; the heavy black contour (ratio $=1$) is the crossover $p_{\rm cross}$. *Panel (a)* spans $(p_T,\ \text{material }x/X_0)$ at fixed VELO resolution; *panel (b)* spans $(p_T,\ \text{pixel pitch})$ at fixed thickness. **Read it as:** lowering $p_T$, adding material, or coarsening the pixels pushes you up-left into the red (scattering) corner; the dotted reference lines mark the VELO operating point, where $p_{\rm cross}\approx0.95$ GeV.

> **Figure DP3 — `dp3_scatt_to_pT.png`.** The dictionary between the toy knob $\sigma_{\rm scatt}$ and physical momentum. Each curve is $p_T(\sigma_{\rm scatt})$ at a fixed material budget $x/X_0=0.5,1,2\%$; because $\sigma_{\rm scatt}\propto\sqrt{x/X_0}/p_T$, a single value of $\sigma_{\rm scatt}$ reads off a **different** $p_T$ on each curve — the same knob is a softer track in a thinner detector. The grey band is the production grid ($p_T\approx2$–11 GeV); the red dashed line is $p_{\rm cross}$. **Read it as:** the entire production grid sits *above* $p_{\rm cross}$ (resolution-limited) — which is precisely **why $\sigma_{\rm scatt}$ looked inert in every §7 heatmap**: at $\sigma_{\rm res}\ge0.01$ the resolution floor $\sqrt6\,\sigma_{\rm res}/\Delta z$ swamps it.

---

## 4. Setting the inclusion rate analytically (the inverse view)

The per-kink **miss probability** at the formula $\varepsilon$ is, from the Rayleigh survival function,
$$
p=P(\theta>\varepsilon)=\exp\!\Big(-\frac{\varepsilon^2-2\theta_{\min}^2}{2\sigma_p^2}\Big)=\exp(-s^2)=e^{-9}=1.23\times10^{-4},
$$
**independent of $\sigma_{\rm scatt}$ and $\sigma_{\rm res}$** — the noise cancels because $\varepsilon$ scales with $\sigma_p$. Inverting,
$$
\boxed{\;\varepsilon=\sigma_p\sqrt{2\ln(1/p)}\;}
$$
gives the acceptance needed for any target per-kink inclusion rate $1-p$. This is the precise sense in which *"the true-segment inclusion rate can be set analytically"*: choose $p$, read off $\varepsilon=\sigma_p\sqrt{2\ln(1/p)}$, and the segment efficiency follows in closed form (§5) — the detector physics enters only through $\sigma_p$, i.e. through $(p_T,x/X_0,\text{pitch})$.

---

## 5. Segment efficiency — closed form, and why it is detector-independent

### 5.1 The Hopfield motif ladder
For a *fragment* of $n$ consecutive surviving segments (a path motif $P_n$) the amplitudes solve $(4\mathbb I-\mathrm{Adj})\mathbf x=\mathbf1$. Solving the small tridiagonal systems exactly:

| motif | amplitudes | alive at $\tau=0.35$? |
|---|---|---|
| $P_1$ (isolated) | $1/4=0.250$ | dead |
| $P_2$ (pair) | $1/3=0.333$ | dead |
| $P_3$ | ends $5/14=0.357$, mid $3/7=0.429$ | **alive** |
| $P_4$ (full track) | ends $4/11=0.364$, mid $5/11=0.455$ | **alive** |

$\tau=0.35$ slices **between $P_2$ (0.333, dead) and $P_3$-ends (0.357, alive)**: isolated segments and pairs score below threshold, triples and quads above.

### 5.2 The fragmentation enumeration → $\text{eff}(p)$
A true track is a $P_4$ chain with **3 interior kinks**, each independently *kept* with probability $q=1-p$. Enumerate:
- **All 3 kept** ($q^3$): intact $P_4$ → all 4 segments alive → 4.
- **Exactly one lost** ($3$ ways, $q^2p$ each): losing an **end** kink (2 ways) splits off a dead $P_1$ and leaves a live $P_3$ → 3 alive; losing the **middle** kink (1 way) makes two $P_2$'s → both dead → 0. Total $= (2\cdot3+1\cdot0)\,q^2p=6q^2p$.
- **Two or more lost:** only $P_1/P_2$ fragments survive → 0.

With 4 true segments per track,
$$
\boxed{\;\text{eff}(p)=\frac{4q^3+6q^2p}{4}=(1-p)^2\Big(1+\tfrac p2\Big)\;}
$$
**zero free parameters.** It depends on $\sigma_{\rm scatt},\sigma_{\rm res}$ only through $p(\varepsilon)=e^{-(\varepsilon^2-2\theta_{\min}^2)/2\sigma_p^2}$.

### 5.3 Universal collapse (existing fig `eff_universal_collapse`)
Plotted against the *computed* $p$, all six noise cells (σ_p spanning 20×) fall on this one curve — the empirical proof that $\sigma_{\rm scatt}$ and $\sigma_{\rm res}$ act only through $p$.

### 5.4 At the formula $\varepsilon$: efficiency is detector-flat — DP4a
With $p$ pinned at $e^{-9}$, $\text{eff}=(1-p)^2(1+p/2)=0.99981$ — flat in $\sigma_{\rm scatt}$ and $\sigma_{\rm res}$, hence flat in $p_T$, $x/X_0$ and pitch. **The new soft-track run (DP4a) confirms this where it was never tested:** classical efficiency stays $1.000$ as $\sigma_{\rm scatt}$ is walked from $10^{-4}$ ($p_T=11$ GeV) up to $3\times10^{-3}$ ($p_T=0.36$ GeV), at both $\sigma_{\rm res}=0$ and $0.01$ mm. The acceptance formula does its job into the scattering-limited regime.

> **Figure DP4 — `dp4_softtrack_extension.png` (new data).** Walking $\sigma_{\rm scatt}$ from the production grid down to $p_T=0.36$ GeV at fixed $\sigma_{\rm res}\in\{0,\,0.01\}$ mm, with $\varepsilon$ set by the formula at every point ($T=30$, classical solid + 1BQF dashed; **top axis** $=p_T$; grey band = production grid, dash-dot line $=\sigma^*_{\rm scatt}$). *Panel (a):* segment **efficiency stays flat** — classical at $1.000$, 1BQF at its $0.75$ plateau — across the whole momentum range, confirming efficiency is dial-set, not noise-set. *Panel (b), log $y$:* segment **false rate rises** as $\sigma_{\rm scatt}$ grows past $\sigma^*_{\rm scatt}$, tracking the $c\,\varepsilon^2$ prediction (dotted, $c=267$ from §7). **Read it as:** the cost of a soft track (or thick material) is paid entirely in the false rate, never in efficiency — the first measurement of this below $p_{\rm cross}$.

### 5.5 The 1BQF 0.75 plateau
The 1-bit quantum filter ($f(\lambda)=\cos(\lambda t/2)$, with $\lambda$ a Hamiltonian eigenvalue and $t$ the evolution time) drops $P_4$-ends just below $\tau$ before the signal-support rescale, capping its efficiency at $\approx0.75$ — also flat in noise (DP4a). With the efficiency-first threshold $\tau_{\rm eff}\approx0.175$ (§7.10) this recovers to $\ge0.997$. The plateau is a *threshold* effect, not lost physics, and is detector-independent.

### 5.6 Consequence
**Efficiency is dial-set.** A thicker detector, a softer track or a coarser pixel all widen $\varepsilon\propto\sigma_p$ in lock-step with the broadened kink, so the same fraction of true kinks survives. The detector quality does **not** show up in efficiency — it shows up in the false rate.

---

## 6. The false rate — the detector tax

### 6.1 The dilute $c\varepsilon^2$ law
A *false* coupling is an accidental triplet (out-segment / in-segment sharing a hit, from different tracks) whose kink falls below $\varepsilon$. In the dilute limit the accepted fraction of the 2D kink phase space is $\propto\varepsilon^2$, so the segment false rate obeys
$$
\overline{\text{far}}\approx c\,\varepsilon^2,
$$
with $c$ a geometric coefficient. Empirically (§7) $c\approx267\ \text{rad}^{-2}$ at $T=30$, **nearly cell-independent** (246–290 across all six cells despite $\sigma_p$ varying 20×): the false background is set by acceptance *geometry*, not by noise.

### 6.2 The detector decomposition — DP5
Substituting $\varepsilon^2=2s^2\sigma_p^2$ exposes the two detector terms additively:
$$
\boxed{\;\overline{\text{far}}\approx c\,\varepsilon^2=c\,2s^2\Big(\underbrace{\sigma_{\rm scatt}^2}_{\propto(1/p_T^2)(x/X_0)}+\underbrace{6\,\sigma_{\rm res}^2/\Delta z^2}_{\propto\,\text{pitch}^2}\Big)\;}
$$
- **Scattering tax** $\propto\sigma_{\rm scatt}^2\propto(x/X_0)/p_T^2$: halving $p_T$ **quadruples** it; doubling material **doubles** it.
- **Resolution tax** $\propto\sigma_{\rm res}^2\propto\text{pitch}^2$: doubling the pitch **quadruples** it.

The two are equal at $\sigma^*_{\rm scatt}$ — the same crossover as $\sigma_p$, now in the *observable*. Because $c$ cancels in the ratio of the two parts, the crossover location is robust even though the absolute scale carries the ($T$-dependent) $c$.

> **Figure DP5 — `dp5_false_tax_decomposition.png`.** The false-rate "tax" split into its two detector contributions, $\overline{\text{far}}=c\,2s^2\big(\sigma_{\rm scatt}^2+6\sigma_{\rm res}^2/\Delta z^2\big)$, as a stacked area vs $\sigma_{\rm scatt}$. **Blue = resolution part** (flat, set by pixel pitch); **red = scattering part** ($\propto\sigma_{\rm scatt}^2$, set by $p_T$/material); the black line is the total. The two parts are equal at $\sigma^*_{\rm scatt}$ (dash-dot). The two panels are $\sigma_{\rm res}=0.01$ mm and the VELO 55 µm pixel; the **top axis** $=p_T$. **Read it as:** to the left of $\sigma^*_{\rm scatt}$ (stiff tracks) the false background is resolution-dominated — finer pixels help; to the right (soft tracks) it is scattering-dominated — only less material or higher $p_T$ helps.

### 6.3 Empirical confirmation in the soft regime — DP4b
The new soft-track run measures the tax directly: as $\sigma_{\rm scatt}$ rises past $\sigma^*_{\rm scatt}$, the classical false rate climbs from $\sim0$ toward $\sim0.04$ at $p_T=0.36$ GeV, tracking $c\varepsilon^2$ (overlaid). This is the empirical statement that **soft tracks (and thick material) drag in a quadratically larger combinatorial halo** — the core detector-physics result, and the first measurement of it below $p_{\rm cross}$.

### 6.4 Occupancy scaling
$c$ is geometric but **T-dependent**: at $T=400$ the §7.10 fit gives $c\approx1.3$–$1.5\times10^4\ \text{rad}^{-2}$, $\sim45$–$50\times$ the $T=30$ value, reflecting the $\mathcal O(T^3)$ candidate-pair pool against $\mathcal O(T)$ true segments. So the detector tax and the occupancy multiply: the same $\sigma_p$ costs far more false couplings at HL-LHC multiplicity.

---

## 7. Sparsity / circuit cost vs the detector

Because $A=(\gamma+\delta)\mathbb I-C$ has a full diagonal and symmetric $\pm$ structure, its nonzero count is
$$
A_{\rm nnz}(\varepsilon)=\underbrace{4T^2}_{n_{\rm seg}}+2\big[n_{\rm true}(\varepsilon)+n_{\rm false}(\varepsilon)\big],
$$
with $n_{\rm true}=3T\big(1-e^{-(\varepsilon^2-2\theta_{\min}^2)/2\sigma_p^2}\big)$ (Rayleigh CDF, saturating at exactly $3T$ by the formula $\varepsilon$) and $n_{\rm false}=\kappa\varepsilon^2$, $\kappa=3T^3/2\theta_{\rm ps}^2$ with $\theta_{\rm ps}\approx0.72$ rad (a geometric false-phase-space half-angle, $T$-independent; called $\theta_0$ in §7.11, but written $\theta_{\rm ps}$ here to keep it distinct from the Highland $\theta_0$ of §2.1). In detector terms: **a wider $\varepsilon$ — i.e. a softer track, more material or a coarser pixel — fills $A$ faster.** The dense wall $\varepsilon_{\rm dense}=\theta_{\rm ps}\sqrt{8/3T}\propto T^{-1/2}$ is why a $\varepsilon$ that is harmless at $T=30$ goes dense at $T=400$. Since $A_{\rm nnz}$ is also the 1BQF QRAM gate count, **detector quality sets circuit cost**: cleaner detectors (smaller $\sigma_p$) keep $A$ near its diagonal floor (figs `sparsity_components_vs_epsilon`, `sparsity_vs_epsilon_Tscan`).

---

## 8. Where it breaks — the indefiniteness corner in detector terms

The solver is well-posed only while $A\succ0$. Since $\mathrm{eig}(A)=(\gamma+\delta)-\mathrm{eig}(C)$ exactly,
$$
A\ \text{indefinite}\iff\lambda_{\max}(C)>\gamma+\delta=4.
$$
$\lambda_{\max}(C)$ grows with the **local coupling density**, which grows with both $\varepsilon$ (wider acceptance → more false couplings around a shared hit) and $T$ (more tracks → denser hubs). The result (§7.12) is a sharp $(\sigma_{\rm res},T)$ boundary: $\sigma_{\rm res}\le0.02$ mm stays positive-definite to $T=1000$; $\sigma_{\rm res}=0.05$ mm (pitch 173 µm) crosses $\lambda_{\max}=4$ at $T_c\approx290$ and the classical efficiency cliffs $0.98\to0.909\to0.786$ at $T=400/700/1000$. **In detector language the danger corner is coarse pixels (or soft tracks / thick material — anything that widens $\varepsilon$) at high pileup** — exactly the HL-LHC / Upgrade-II operating point. The 1BQF is *worse* here: its even filter $f(\lambda)=\cos(\lambda t/2)$ cannot distinguish $\pm\lambda$, so indefinite (negative) modes alias onto their positive mirror and the rescale cannot recover them.

---

## 9. Synthesis — the decoupling and the operating prescription

The system has **three independent control knobs plus occupancy**:

| knob | physical origin | sets |
|---|---|---|
| $\sigma_p=\sqrt{\sigma_{\rm scatt}^2+6\sigma_{\rm res}^2/\Delta z^2}$ | $p_T$, $x/X_0$, pitch | the kink blur |
| $\varepsilon=s\sqrt2\,\sigma_p$ | acceptance scale $s$ | per-kink survival $p=e^{-s^2}$ |
| $\tau$ | post-solve threshold | which Hopfield levels count |
| $T$ | pileup / multiplicity | confusable-pair pool $\propto T^2$–$T^3$ |

The confound the supervisor worried about — *"as the σ's grow we get both more confused segments and a wider acceptance"* — **factorises cleanly**:
- The wider acceptance is *deliberate* and *exactly compensating*: $\varepsilon\propto\sigma_p$ holds efficiency fixed (§5). This is the "more acceptance" effect, and it is benign.
- The extra confusion is the **false tax** $\propto\sigma_p^2$ (§6), the genuine cost of detector quality, paid at fixed efficiency.
- The two are only coupled because the *formula* slides $\varepsilon$ along the diagonal $\varepsilon\propto\sigma_p$. To break them apart, sweep $\varepsilon$ and $\sigma_p$ **independently** (a 2D grid), separating "noise blurs true kinks" (the $p(\varepsilon)$ axis) from "wide acceptance admits false couplings" (the $\varepsilon^2$ axis). This is the recommended decoupling experiment.

**Operating prescription — per-track adaptive $\varepsilon$.** Since $\varepsilon=\sigma_p\sqrt{2\ln(1/p)}$ and $\sigma_p$ is set by the track's own $p_T$ (and the local material), the optimal acceptance is **track-by-track**: wide for soft tracks, tight for stiff ones. This holds efficiency flat *and* minimises the false tax for every track, instead of paying the worst-case $\varepsilon$ globally.

**Design levers (at fixed efficiency).**

| lever | effect on $\sigma_p$ | false tax | regime where it pays |
|---|---|---|---|
| finer pitch | $\sigma_{\rm res}\downarrow$ linearly | $\propto\text{pitch}^2$ | $p_T>p_{\rm cross}$ (resolution-limited) |
| less material $x/X_0$ | $\sigma_{\rm scatt}\downarrow\propto\sqrt{x/X_0}$ | $\propto x/X_0$ | $p_T<p_{\rm cross}$ (scattering-limited) |
| longer lever $\Delta z$ | resolution kink $\propto1/\Delta z$ | $\propto1/\Delta z^2$ | resolution-limited (trades against occupancy/acceptance) |

For a **soft-track physics programme** reduce material; for a **stiff-track programme** reduce pitch (or lengthen $\Delta z$).

---

## 10. What we have, and the gap to close

- **Have:** the full §7 grid (resolution-limited, $p_T\!\ge\!2$ GeV); the closed-form efficiency; the $c\varepsilon^2$ false law; the sparsity decomposition; the indefiniteness boundary; and now the **soft-track extension** (DP4) — the first data at $p_T<1$ GeV, confirming flat efficiency and the rising scattering tax.
- **Gap / recommendations:**
  1. Extend the *production* grid to $\sigma_{\rm scatt}\le3\times10^{-3}$ (down to $p_T\approx0.36$ GeV) so the scattering-limited regime is characterised at full statistics and at $T\ge400$.
  2. Sweep $x/X_0$ explicitly (not just via $\sigma_{\rm scatt}$) to separate the momentum and material contributions to the scattering tax.
  3. Run the **independent $(\sigma_p,\varepsilon)$ 2D grid** to fully decouple noise-blur from acceptance-width.
  4. Replace the placeholder $x/X_0=1\%$ with the measured VELO Run-3 material budget (the RF foil dominates) to sharpen $p_{\rm cross}$ and the momentum labels.

---

## Appendix A — symbol table
| symbol | meaning | value / formula |
|---|---|---|
| $\sigma_{\rm scatt}$ | per-projection scattering RMS | $=\theta_0=\frac{13.6\,\text{MeV}}{p_T}\sqrt{x/X_0}(1+0.038\ln x/X_0)$ |
| $\sigma_{\rm res}$ | single-hit resolution | $=\text{pitch}/\sqrt{12}$ |
| $\Delta z$ | inter-plane spacing | 33 mm |
| $\sigma_p$ | per-projection kink RMS | $\sqrt{\sigma_{\rm scatt}^2+6\sigma_{\rm res}^2/\Delta z^2}$ |
| $s$ | acceptance scale | 3 |
| $\varepsilon$ | acceptance angle | $s\sqrt2\,\sigma_p$ |
| $\theta_{\min}$ | pixel angular floor | $1.5\times10^{-5}$ rad |
| $p$ | per-kink miss prob | $e^{-(\varepsilon^2-2\theta_{\min}^2)/2\sigma_p^2}=e^{-9}$ at formula $\varepsilon$ |
| $\tau$ | amplitude threshold | 0.35 (eff-first: classical 0.363, 1BQF 0.175) |
| $c$ | false-rate coeff | $\approx267\,\text{rad}^{-2}$ ($T=30$); $\sim1.4\times10^4$ ($T=400$) |
| $\theta_0$ | Highland scattering angle | $=\sigma_{\rm scatt}$ (row 1) |
| $\theta_{\rm ps}$ | false phase-space half-angle ($=\theta_0$ in §7.11) | $\approx0.72$ rad (geometric) |
| $p_{\rm cross}$ | scattering/resolution crossover | $\approx0.95$ GeV (VELO, $x/X_0=1\%$) |

## Appendix B — the detector anchor and its leverage
Every momentum label scales as $\sqrt{x/X_0}$. The placeholder $x/X_0=1\%$ enters only the $\sigma_{\rm scatt}\!\leftrightarrow\!p_T$ map and $p_{\rm cross}$, never the toy physics. Doubling the assumed material shifts all $p_T$ labels up by $\sqrt2$ and $p_{\rm cross}$ to $\approx1.35$ GeV. Replace with the measured budget when available.

## Appendix C — figure index
| fig | file | content |
|---|---|---|
| DP1 | `dp1_sigma_p_decomposition.png` | $\sigma_p$ vs $\sigma_{\rm scatt}$ (per $\sigma_{\rm res}$) and vs $\sigma_{\rm res}$ (per $\sigma_{\rm scatt}$), with $p_T$/pitch axes |
| DP2 | `dp2_crossover_map.png` | dominance map over $(p_T,x/X_0)$ and $(p_T,\text{pitch})$; $p_{\rm cross}$ contour |
| DP3 | `dp3_scatt_to_pT.png` | $p_T(\sigma_{\rm scatt})$ at three thicknesses; grid located at 2–11 GeV |
| DP4 | `dp4_softtrack_extension.png` | **new data:** efficiency (flat) + false rate (rising) vs $\sigma_{\rm scatt}$ to $p_T=0.36$ GeV |
| DP5 | `dp5_false_tax_decomposition.png` | false rate split into scattering + resolution parts, crossing at $\sigma^*_{\rm scatt}$ |
| — | `eff_universal_collapse.png` | all six cells collapse onto $(1-p)^2(1+p/2)$ |
| — | `sigma_scan_formula_eps.png` | eff/far vs $\sigma_{\rm scatt}$ and vs $\sigma_{\rm res}$ at formula $\varepsilon$ |
| — | `neg_eigs_vs_sigma_res_T.png` | indefiniteness boundary in $(\sigma_{\rm res},T)$ |

*Numbers: `outputs/detector_physics.json`. Generator: `gen_detector_physics.py`. Derivation: `theory.md`.*
