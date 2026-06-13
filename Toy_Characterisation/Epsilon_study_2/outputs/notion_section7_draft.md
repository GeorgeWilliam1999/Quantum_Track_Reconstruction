# 7. ε-sensitivity of the segment metrics — direct sweeps, σ-scans, T-scans, and the analytic laws (2026-06-12)

<callout icon="📐" color="blue_bg">
	**Source pointers.** Paired-event ε-scan: `Epsilon_study_2/gen_epsilon_sensitivity_scan.py` → `figures/epsilon_sensitivity/*.png` + `outputs/epsilon_sensitivity_scan.json` (raw solution vectors archived in `outputs/epsilon_sensitivity_{roc,vectors}.npz` — metrics stay a recomputable view). Store-backed T-scans: `gen_metrics_vs_T.py` over the qtrk-store metrics view (`$QTRK_STORE/manifest/metrics.csv`). Commit `f9163421`. This section is the **solver-level companion** to the ε-formula validity report (which treats the per-kink physics: Rayleigh tail, p_miss dial, √2): here the same dial is read out in the downstream observables — segment efficiency and false rate.
</callout>
<callout icon="📏" color="gray_bg">
	**Metric definitions (canonical, from 📐 Data & Metrics — The Source of Truth).** A segment is truth-**true** iff its two hits share a `track_id`. After a solve, segment $`i`$ is **active** iff $`x_i > \tau`$ with the **absolute** γ-aware threshold $`\tau = \delta/(\delta+\gamma) + 0.10 = 0.35`$ at $`\gamma=3,\ \delta=1`$ (never a relative $`\tau\cdot\max x`$). **Segment efficiency** $`= n_{\rm true\,active}/n_{\rm true\,all}`$ — the fraction of true segments recovered. **Segment false rate** $`= n_{\rm false\,active}/n_{\rm active}`$ — the contamination of the active set (denominator = all active segments, *not* all false segments; equals $`1-`$purity). **Quantum convention:** the 1BQF statevector solution is rescaled onto the classical amplitude scale **on the classical signal support** (`rescale_to_signal`, the confirmed 2026-06-09 convention), then cut at the *same* absolute τ. **ε vs τ:** ε is the kink-angle acceptance used to *build* $`A`$ (changing it changes the matrix and requires new solves); τ is the post-processing cut on the solution vector.
</callout>

## 7.1 Data & coverage

Two data sources feed this section, and they are deliberately different:

1. **Paired-event direct ε-scan** (fresh, this section): $`T=30`$, 3 reps per cell, ε swept over a 10-point log grid $`[10^{-4}, 4\times10^{-2}]`$ rad. The *same generated event* is reused at every ε — only $`A`$ changes — so the ε-dependence is exact (no event-to-event fluctuation enters the comparison along the ε axis). Six noise cells in two families: **Family A** fixes $`\sigma_{\rm res}=0.01`$ mm (small) and varies $`\sigma_{\rm scatt}\in\{10^{-4}, 5\times10^{-4}, 2\times10^{-3}\}`$; **Family B** fixes $`\sigma_{\rm scatt}=10^{-4}`$ and varies $`\sigma_{\rm res}\in\{0, 0.02, 0.05\}`$ mm (with the (0.01, 10⁻⁴) cell shared as reference).
2. **The qtrk store at the formula ε** (the study's production data): used for the T-scans (§7.6) and the high-T cross-check (§7.8).

**Actual store coverage for Epsilon_study_2** (metrics view, counted 2026-06-12 — this is the ground truth behind every store-backed figure; the Source-of-Truth §11 nominal is 20 classical / 3 quantum reps over $`T\in\{10\dots1000\}`$):

<table header-row="true">
<tr>
<td>Solver</td>
<td>T = 10–200</td>
<td>T = 400</td>
<td>T = 700</td>
<td>T = 1000</td>
</tr>
<tr>
<td>**classical**</td>
<td>**20 reps × all 12 cells** — complete</td>
<td>complete</td>
<td>complete</td>
<td>complete</td>
</tr>
<tr>
<td>**quantum (1BQF statevector)**</td>
<td>**3 reps × all 12 cells** — complete</td>
<td>partial: 0–3 reps per cell (σ_res = 0.05 row empty)</td>
<td>only (0, 10⁻⁴): 1 rep</td>
<td>(0, 10⁻⁴) and (0, 3×10⁻⁴): 1 rep each</td>
</tr>
</table>

Total 2125 metric rows (1920 classical + 205 quantum). The quantum gap at $`T\ge400`$ in the noisy cells is the known GPU-lane shortfall; quantum curves in §7.6/§7.8 are drawn only where ≥ 2 reps exist.

## 7.2 The direct view — efficiency and false rate against ε

The user-facing question of this study, asked directly: *what do the two segment metrics do as the Hamiltonian acceptance ε is turned?* Each figure shows efficiency (left) and false rate (right) on the shared log-ε grid; vertical lines mark the closed-form formula ε of each cell; dotted curves are the **analytic laws of §7.3–7.4**, not fits to these points (the only fitted number is the single constant $`c`$ of the false-rate phase-space law).

![Segment efficiency and false rate vs Hamiltonian epsilon at fixed sigma_res = 0.01 mm, three sigma_scatt values, classical and 1BQF, with analytic overlays](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/eps_scan_fixed_sres.png?v=f9163421)

Figure (ε-scan, Family A — σ_res = 0.01 mm fixed). Efficiency is a **knee curve**: zero while ε is below the kink scale $`\sigma_p`$, a sharp rise over roughly one decade, then a hard plateau at 1 (classical) / ≈ 0.75 (1BQF). The knee position moves right as $`\sigma_{\rm scatt}`$ grows — but only weakly, because at $`\sigma_{\rm res}=0.01`$ mm the resolution term dominates $`\sigma_p`$ (§7.5). The false rate is **zero until well past the formula point**, then climbs as the acceptance-area law $`c\,\varepsilon^2`$. The formula ε (vertical lines) lands just above the efficiency knee and below the false-rate take-off in every cell — the regime split the formula is designed to hit. Concretely, at the grid point nearest each cell's formula ε: $`\sigma_{\rm scatt}=10^{-4}`$ gives (effC, farC, effQ, farQ) = (1.000, 0.000, 0.750, 0.000); $`5\times10^{-4}`$ → (0.983, 0.000, 0.736, 0.000); $`2\times10^{-3}`$ → (1.000, 0.014, 0.742, 0.011). The formula ε itself shifts only 3.18 → 3.80 → 9.05 mrad across the three cells: the knee barely moves until σ_scatt clears the resolution floor.

![Segment efficiency and false rate vs Hamiltonian epsilon at fixed sigma_scatt = 1e-4, four sigma_res values, classical and 1BQF, with analytic overlays](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/eps_scan_fixed_sscatt.png?v=f9163421)

Figure (ε-scan, Family B — σ_scatt = 10⁻⁴ fixed). The reverse family: now the knee moves over a full decade as $`\sigma_{\rm res}`$ goes 0 → 0.05 mm, because $`\sigma_p`$ is resolution-dominated everywhere above $`\sigma_{\rm res}^* = \Delta z\tan(\sigma_{\rm scatt}/\sqrt6) = 1.3\times10^{-3}`$ mm. At $`\sigma_{\rm res}=0.05`$ mm the knee and the false-rate take-off have almost met: the clean plateau between them — the working room the formula needs — is closing. That is the geometric origin of the high-noise purity collapse seen everywhere in this report. At the formula ε of each cell: $`\sigma_{\rm res}=0`$ (formula ε = 0.42 mrad) gives (effC, farC, effQ) = (1.000, 0.000, 0.750); $`\sigma_{\rm res}=0.02`$ mm (6.31 mrad) → (1.000, 0.011, 0.744); $`\sigma_{\rm res}=0.05`$ mm (15.75 mrad) is the first cell where the false rate is already non-zero *at* the formula point (far ≈ 0.18 at the adjacent grid node) — the knee and the take-off have collided, and there is no longer a clean zero-false plateau to place ε in.

## 7.3 The analytics, efficiency side — exactly solvable, zero parameters

The question "can these sensitivities be understood analytically?" has a clean **yes** for efficiency. Three ingredients:

**(i) The per-kink miss probability.** From the validity report: the measured 3D kink at a shared hit is Rayleigh with per-projection scale $`\sigma_p^2 = \sigma_{\rm scatt}^2 + 6\arctan^2(\sigma_{\rm res}/\Delta z)`$, so the probability that a *true* coupling is rejected by the acceptance ε is

$$
p(\varepsilon) = \exp\!\Big(-\frac{\varepsilon^2 - 2\theta_{\min}^2}{2\sigma_p^2}\Big),
$$

the exact inverse of the formula (at the formula ε this is $`e^{-9}=1.2\times10^{-4}`$ by construction).

**(ii) Which fragments survive the threshold.** Losing a kink removes one off-diagonal coupling and fragments the track's 4-segment chain. Each fragment's amplitudes are the Hopfield levels of $`\big((\gamma+\delta)I - \mathrm{Adj}\big)\,x = \delta\mathbf{1}`$ on the fragment motif (γ = 3, δ = 1 → diagonal 4, couplings −1):

<table header-row="true">
<tr>
<td>Motif</td>
<td>Amplitudes</td>
<td>vs τ = 0.35</td>
</tr>
<tr>
<td>P1 (isolated segment)</td>
<td>1/4 = 0.250</td>
<td>**dead**</td>
</tr>
<tr>
<td>P2 (2-chain)</td>
<td>1/3 = 0.333</td>
<td>**dead**</td>
</tr>
<tr>
<td>P3 (3-chain)</td>
<td>ends 5/14 = 0.357, mid 3/7 = 0.429</td>
<td>**alive** (ends clear τ by only 0.007)</td>
</tr>
<tr>
<td>P4 (full track)</td>
<td>ends 4/11 = 0.364, interior 5/11 = 0.455</td>
<td>**alive**</td>
</tr>
</table>

**(iii) Enumerate the fragmentations.** A 5-hit track has 3 interior kinks, each kept independently with probability $`q = 1-p`$. All three kept (probability $`q^3`$): 4 active segments. One *end* kink lost (2 ways, $`q^2p`$ each): P3 + P1 → 3 active. One *middle* kink lost: P2 + P2 → 0. Two or more lost: nothing survives. Hence

$$
\mathrm{eff}(p) = \frac{4q^3 + 3\cdot 2\,q^2 p}{4} = (1-p)^2\Big(1+\frac{p}{2}\Big).
$$

This is a **zero-parameter prediction**: $`\sigma_p`$ comes from the noise settings, $`p`$ from the Rayleigh tail, the levels from linear algebra on path graphs, the combinatorics from counting. Replotting *every* measured efficiency point of both families against its computed $`p(\varepsilon)`$ collapses all six cells onto the single enumeration curve:

![Left: Hopfield motif amplitude ladder against the threshold; right: all six noise cells' measured efficiencies collapsing onto the analytic enumeration curve when plotted vs the computed per-kink miss probability](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/eff_universal_collapse.png?v=f9163421)

Figure (universal collapse). Left: the motif ladder — the entire efficiency story is "P1 and P2 are below τ, P3 and P4 above". Note how little margin the theory rests on: the P3 end level 5/14 clears τ = 0.35 by 0.007. A threshold of 0.36 would kill P3 fragments (their ends fall below the cut) and change the enumeration to the much steeper $`\mathrm{eff}=q^3`$ — the law is exact but **brittle to τ**. Right: measured classical efficiency vs computed $`p(\varepsilon)`$ for all cells and all ε; the black curve is $`(1-p)^2(1+p/2)`$ with nothing fitted. The formula default sits at $`p=e^{-9}`$ (green line), predicting eff = 0.99981 — the plateau. The collapse is the strongest single claim in this section: six noise cells spanning two decades in $`\sigma_p`$, ten ε each, all 60 points falling on one parameter-free curve confirms that the only thing ε does to efficiency is move $`p`$ along the Rayleigh tail. The half-decade scatter at fixed $`p`$ near the steep part is the 3-rep event statistics, not a systematic.

**The 1BQF inherits the same collapse scaled by its ladder plateau.** Under the signal-support rescale the 1-bit filter halves the outer true band (P4 ends 4/11 → below τ), settling at the known ≈ 0.75 plateau; its ε-response is the same knee at the same position — the filter changes *which rung* of the ladder survives τ, not *where the knee sits in ε* (the knee is pure Rayleigh geometry, upstream of any solver).

## 7.4 The analytics, false-rate side — mechanism-bounded, one fitted constant

The false side has no closed form, but the mechanism ladder is exact. Which *false*-segment motifs can activate at τ = 0.35?

<table header-row="true">
<tr>
<td>False motif</td>
<td>Amplitude</td>
<td>vs τ = 0.35</td>
</tr>
<tr>
<td>isolated false segment</td>
<td>0.250</td>
<td>dead</td>
</tr>
<tr>
<td>false–false pair (one accepted false kink between two false segments)</td>
<td>0.333</td>
<td>dead</td>
</tr>
<tr>
<td>**pendant**: false segment with one accepted kink onto a P4 *interior*</td>
<td>**0.392**</td>
<td>**ACTIVE**</td>
</tr>
<tr>
<td>pendant onto a P4 *end*</td>
<td>0.365</td>
<td>**ACTIVE**</td>
</tr>
<tr>
<td>false 3-chain (two accepted false kinks)</td>
<td>ends 0.357, mid 0.429</td>
<td>**ACTIVE**</td>
</tr>
<tr>
<td>star K(1,m) leaves (hub hit with m accepted kinks)</td>
<td>$`(4+m)/(16-m)`$ centre, leaves above 0.357</td>
<td>**ACTIVE**</td>
</tr>
</table>

So **a single accepted false kink onto a true chain is sufficient** to put one false segment over the threshold — false activation needs no conspiracies. The rate of such accepted kinks is pure phase space: near zero the false-kink density grows linearly in θ (2D measure), so the acceptance probability per candidate scales as $`\varepsilon^2`$, giving the **dilute law** $`\mathrm{far} \approx c\,\varepsilon^2`$ — the dotted overlays in §7.2, with $`c`$ the one fitted constant per cell (it absorbs the cell's false-kink density near zero and the per-event combinatorics). The fit delivers a sharp result: $`c`$ is **nearly cell-independent** — $`\{269, 271, 246, 257, 268, 290\}`$ rad⁻² across the six cells (mean ≈ 267, spread ±8%) despite $`\sigma_p`$ varying 20×. This is the quantitative statement that **the false-coupling load is set by the acceptance geometry, not the noise**: a false kink is a combinatorial-pair angle drawn from the near-uniform background, so the count accepted below ε depends on ε (the disc area $`\propto\varepsilon^2`$) and on T (the candidate density), but barely on how the noise was generated. The two σ-scan fits ($`c=401`$ and $`137`$ rad⁻²) differ more only because they average a moving cell along the noise axis rather than holding one fixed.

Beyond the dilute regime two higher-order mechanisms steepen the growth: **false chains** (P3 of false segments needs *two* accepted kinks → contributes $`\propto\varepsilon^4`$) and **hubs** (a hit collecting m accepted kinks; the star's top adjacency eigenvalue is $`\sqrt m`$, so for $`m > (\gamma+\delta)^2 = 16`$ the matrix $`A`$ is locally **indefinite** — the M-matrix/attractor picture breaks entirely). These are the same chain ≥ 3 + hub topologies identified in the FP-taxonomy write-up; here they explain two observed departures from the dilute law: the super-quadratic false-rate growth with T (§7.6), and the **classical efficiency break** at the (σ_res = 0.05, T ≥ 700) corner — eff 0.91 at T = 700, 0.79 at T = 1000 — where hub stars with m far above 16 make the solve oscillatory and push even *true* segments below τ (the mean coupling degree there reaches 0.30 vs 0.014 in the same cells at σ_res = 0.01, and the degree is concentrated on hub hits). Saturation closes the story: when essentially everything activates, $`\mathrm{far} \to n_{\rm false}/n_{\rm seg} = 1 - 1/T`$ (measured 0.994 at the corner, predicted 0.999).

## 7.5 The reverse view — σ on the x-axis at the formula ε

The same two metrics with the *noise* on the x-axis and ε set by the formula at every point — the operating curve of the study as actually run:

![Two-by-two panel: segment efficiency and false rate vs sigma_scatt at fixed sigma_res = 0.01 (left column) and vs sigma_res at fixed sigma_scatt = 1e-4 (right column), at the formula epsilon, with the analytic flat-efficiency line and the sigma_p-squared false-rate law](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/sigma_scan_formula_eps.png?v=f9163421)

Figure (σ-scans at the formula ε, T = 30). **Efficiency is flat** — the formula's defining property. Because ε rescales with $`\sigma_p`$, the per-kink miss probability is *pinned* at $`p=e^{-9}`$ regardless of the noise, so the analytic efficiency is the constant 0.99981 (black dotted): the classical points sit on it across two decades of $`\sigma_{\rm scatt}`$ and the full $`\sigma_{\rm res}`$ range; the 1BQF sits on its parallel ≈ 0.75 ladder plateau. **The false rate is where the noise actually goes**: $`\mathrm{far} \propto \varepsilon^2(\sigma) \propto \sigma_p^2`$ — flat in $`\sigma_{\rm scatt}`$ until the crossover $`\sigma^*_{\rm scatt} = \sqrt6\,\arctan(\sigma_{\rm res}/\Delta z) = 7.4\times10^{-4}`$ rad (grey line; the entire study grid σ_scatt ≤ 5×10⁻⁴ sits *below* it at σ_res = 0.01, which is why the scattering axis looks inert in every Epsilon_study_2 heatmap), then quadratic; and quadratic in $`\sigma_{\rm res}`$ from the start in the right column. **In one sentence: at the formula ε, noise does not cost efficiency — it costs purity, at the rate $`\sigma_p^2`$.** The numbers (T = 30, 3 reps): along $`\sigma_{\rm scatt}`$ classical efficiency is 1.000 at every point and the 1BQF holds 0.744–0.753, while the classical false rate stays ≤ 0.005 below the crossover and rises to 0.043 just past it ($`\sigma_{\rm scatt}=1.3\times10^{-3}`$); along $`\sigma_{\rm res}`$ efficiency is again flat (1.000 / 0.742–0.750) and the false rate climbs 0.000 → 0.011 → 0.035 over $`\sigma_{\rm res}=0.01\to0.05`$ mm. The 1BQF false rate sits on top of the classical one throughout (0.032 vs 0.035 at the worst point) — the signal-support rescale working exactly as §7.1 claims.

## 7.6 The standard view — efficiency and false rate vs track count T

The production-store view (formula ε, 20 classical / 3 quantum reps), the standard panels of every study in this programme, now with both family slices:

![Segment efficiency and false rate vs track count, curves per sigma_res at fixed sigma_scatt = 1e-4, classical and 1BQF from the qtrk store](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/eff_fr_vs_T_sres_family.png?v=f9163421)

Figure (vs T, σ_res family at σ_scatt = 10⁻⁴). Efficiency: classical pinned at 1.000 for every cell and every T **except** the hub-driven break at σ_res = 0.05 (0.984 at T = 200 → 0.786 at T = 1000 — §7.4); 1BQF flat on the 0.75 plateau with no T-dependence to T = 400 (and to T = 1000 in the clean row). False rate: a **sigmoid in log T** whose onset T scales inversely with ε² — σ_res = 0 never leaves zero (far = 0.002 even at T = 1000), σ_res = 0.01 takes off at T ≈ 400, σ_res = 0.02 at T ≈ 100–200, σ_res = 0.05 at T ≈ 50. The 1BQF false rate **tracks the classical curve** under the signal-support rescale (slightly above it in the mid-noise rows at T ≥ 100).

![Segment efficiency and false rate vs track count, curves per sigma_scatt at fixed sigma_res = 0.01 mm, classical and 1BQF from the qtrk store](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/eff_fr_vs_T_sscatt_family.png?v=f9163421)

Figure (vs T, σ_scatt family at σ_res = 0.01 mm). The reverse family is **almost degenerate, and the analytics says exactly how almost**: at σ_res = 0.01 the formula ε moves only from 3.18 to 3.80 mrad across σ_scatt = 1→5×10⁻⁴ (ratio 1.195, since the resolution term dominates $`\sigma_p`$ below the crossover), so the false-rate curves separate by just the factor $`\varepsilon^2`$ ratio ≈ 1.4 — visible only at the top of the grid (far = 0.59 / 0.65 / 0.73 at T = 1000). Efficiency is flat at 1.000 / 0.75 for all three curves. This panel is the quantitative proof that **σ_scatt is the subdominant axis** of the entire study at any σ_res ≥ 0.01 mm.

The T-growth itself: the dilute pendant-counting argument (one accepted kink onto a true chain; candidates ∝ T per true segment, 4T true segments) predicts active-false counts $`\propto T^2\varepsilon^2`$ hence a logistic $`\mathrm{far} = u/(1+u)`$ with $`u \propto T\varepsilon^2`$; the measured growth is steeper (u roughly ∝ T² between T = 200 and 1000 at σ_res = 0.01–0.02), the §7.4 chain-and-hub terms. We quote the mechanism and the bound, not a closed form.

## 7.7 Working points — the fixed-τ numbers are one point on a curve

Per the project doctrine (τ is per-solver, efficiency-first), the fixed-τ = 0.35 numbers above are *one operating point*, not the solvers' capability. Sweeping τ on the stored solution vectors at three ε values around the formula point:

![ROC-style efficiency vs false rate as tau sweeps, at three epsilon values around the formula point, classical and 1BQF, reference cell](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/roc_tau_sweep.png?v=f9163421)

Figure (τ-sweep working points, reference cell σ_res = 0.01, σ_scatt = 10⁻⁴, T = 30; formula ε = 3.18 mrad). Three ε are shown — 0.45×, 0.88× (≈ formula) and 3.32× the formula. The efficiency-first reading is decisive. **At the formula ε** (green, 0.88×): the classical solver reaches eff = 1.000 at far = 0.000 already at τ = 0.35, and the 1BQF can be pushed to eff ≥ 0.99 for a false-rate price of just **0.005**. **Tighten ε to 0.45×** (purple): true couplings are already cut, so the classical efficiency is *capped at 0.76* at τ = 0.35 and reaching eff ≥ 0.99 costs far = 0.97 (the curve runs flat along the top — useless); the 1BQF caps at 0.85 and **cannot reach 0.99 at any τ**. **Widen ε to 3.32×** (orange): efficiency is recoverable but the 1BQF now pays far = 0.082 for eff ≥ 0.99 — 16× the formula-ε price — because the extra false couplings have contaminated the amplitude distribution itself. The reading: the 1BQF's 0.75 plateau at τ = 0.35 is **the cut, not lost physics** — lowering its τ buys the efficiency back at a false-rate price, exactly as the per-solver working-point doctrine prescribes; and widening ε beyond the formula point degrades *both* solvers' curves (the extra false couplings contaminate the amplitude distributions themselves), while tightening below the formula point amputates efficiency at any τ. **The formula ε is where the best ROC lives, not just where a heuristic put it.**

## 7.8 High-T cross-check from the store

The paired ε-scan runs at T = 30 (statevector cost); the store confirms its conclusions carry to production multiplicities:

![Store cross-check: efficiency and false rate vs sigma_scatt and sigma_res at the formula epsilon for T = 50 to 400, classical and quantum](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/store_grid_highT.png?v=f9163421)

Figure (store grid, T = 50–400). Same axes as §7.5 from the production store (20 classical / 3 quantum reps): efficiency flat at 1.000 / 0.75 in every panel; false rate growing along both noise axes with the σ_res axis dominant and the T-ordering of §7.6. The T = 30 scan, the T = 50–400 store slices and the analytic laws tell one consistent story.

## 7.9 Verdict — what is analytic, what is empirical

<table header-row="true">
<tr>
<td>Sensitivity</td>
<td>Analytic status</td>
</tr>
<tr>
<td>efficiency vs ε (any cell)</td>
<td>**closed form, zero parameters**: $`\mathrm{eff} = (1-p)^2(1+p/2)`$ with $`p=e^{-(\varepsilon^2-2\theta_{\min}^2)/2\sigma_p^2}`$ — all cells collapse onto it</td>
</tr>
<tr>
<td>efficiency vs σ at formula ε</td>
<td>**closed form**: constant 0.99981 ($`p`$ pinned at $`e^{-9}`$); 1BQF parallel plateau ≈ 0.75 from the activation ladder</td>
</tr>
<tr>
<td>false rate vs ε (fixed cell, dilute)</td>
<td>**power law** $`c\,\varepsilon^2`$ — exponent analytic (2D kink phase space), one fitted constant $`c`$ per cell</td>
</tr>
<tr>
<td>false rate vs σ at formula ε</td>
<td>**power law** $`\propto\sigma_p^2`$ with the crossover $`\sigma^*_{\rm scatt}=\sqrt6\arctan(\sigma_{\rm res}/\Delta z)`$ — inherited from the ε² law</td>
</tr>
<tr>
<td>false rate vs T</td>
<td>**mechanism ladder, no closed form**: dilute pendant counting gives $`u\propto T\varepsilon^2`$, logistic saturation $`1-1/T`$; chains (ε⁴) and hubs steepen the middle</td>
</tr>
<tr>
<td>classical efficiency break (high σ_res · T)</td>
<td>**spectral criterion, qualitative**: hub stars K(1,m) with $`\sqrt m > \gamma+\delta`$ make A indefinite — same mechanism as the FP-taxonomy hubs</td>
</tr>
<tr>
<td>fragility of the whole picture</td>
<td>the enumeration rests on P3-ends at 5/14 = 0.357 clearing τ = 0.35 by **0.007** — the laws are exact but the threshold sits two levels deep in the ladder's gap structure</td>
</tr>
</table>

<callout icon="✅">
	**Bottom line.** The ε-sensitivity of the segment metrics is now understood end to end. Efficiency against ε is *exactly* solvable — a Rayleigh knee pushed through the Hopfield fragment ladder, $`(1-p)^2(1+p/2)`$, verified by a zero-parameter collapse of all six noise cells — and at the formula ε it is analytically pinned at 0.9998 (classical) / ≈ 0.75 (1BQF ladder plateau) independent of noise. The false rate is the dial that actually moves: $`\propto\varepsilon^2`$ in the dilute regime (one accepted false kink onto a true chain suffices to activate, pendant level 0.392 above τ), $`\propto\sigma_p^2`$ along the noise axes, sigmoid in T with hub/chain steepening, saturating at $`1-1/T`$. The formula ε sits where it should — above the efficiency knee, below the false-rate take-off — and the working-point view confirms it is ROC-optimal, not merely conventional. The remaining empirical content is exactly two things: the per-cell phase-space constant $`c`$, and the T-exponent in the crowded regime.
</callout>
