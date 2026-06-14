## 7.11 Sparsity of A against ε — analytic and empirical, contrasted

ε does not only move the metrics; it sets how many couplings the Hamiltonian carries, i.e. how sparse A is — and A_nnz **is** the 1BQF circuit cost (the QRAM block has O(A_nnz) gates). This sparsity is **fully analytic**, and the closed form is verified against direct nonzero counts here. Script: `gen_sparsity_vs_epsilon.py` → `figures/epsilon_sensitivity/sparsity_*.png` + `outputs/sparsity_vs_epsilon.json`, commit `6b2bd553`. (Building A needs no solve, so this sweep is cheap and reaches the dense regime even at larger T.)

### The closed form

A is symmetric with a full diagonal ($`A_{ii}=\gamma+\delta\neq0`$), so the nonzero count splits exactly:

$$
A_{\rm nnz}(\varepsilon) = n_{\rm seg} + 2\,n_{\rm coupling}(\varepsilon), \qquad n_{\rm coupling} = n_{\rm true}(\varepsilon) + n_{\rm false}(\varepsilon),
$$

with $`n_{\rm seg}=4T^2`$ the diagonal floor. The two coupling terms are both pure phase space — no solver, no noise model beyond $`\sigma_p`$:

- **True couplings — a saturating S-curve.** Each 5-plane track has exactly **3** interior shared-hit kinks, so $`N_{\rm true}=3T`$, and a true coupling is kept iff its Rayleigh kink is below ε:
$$
n_{\rm true}(\varepsilon) = 3T\Big(1 - e^{-(\varepsilon^2-2\theta_{\min}^2)/2\sigma_p^2}\Big).
$$
It rises through the §7.3 knee and **saturates at 3T by the formula ε** — past that, widening ε adds no true couplings.

- **False couplings — an unbounded ε² law.** A combinatorial pair sharing a hit has a kink drawn from the near-uniform background; near zero its density is linear in θ (2-D measure), so the accepted fraction is $`\varepsilon^2/2\theta_0^2`$. The candidate pool is the whole shared-hit combinatorics, $`\sim 3T^3`$ (3 interior planes × T middle hits × $`T^2`$ in/out segment pairs per hit), hence
$$
n_{\rm false}(\varepsilon) = \kappa\,\varepsilon^2, \qquad \kappa = \frac{3T^3}{2\theta_0^2},
$$
with $`\theta_0`$ a fixed geometric angular spread (T-independent). It never saturates.

The matrix stays sparse ($`A_{\rm nnz}\approx n_{\rm seg}`$) only while $`n_{\rm false}<n_{\rm seg}`$, i.e.

$$
\varepsilon < \varepsilon_{\rm dense} \;\sim\; \theta_0\sqrt{\tfrac{8}{3T}},
$$

and because $`\varepsilon_{\rm dense}\propto T^{-1/2}`$ the dense wall moves to **smaller** ε as T grows — this is exactly the blow-up that made the T = 400 ε-sweep go dense above ≈ 2× the formula ε (§7.10).

### Empirical vs analytic

![Sparsity decomposition vs epsilon at fixed T: measured A_nnz, true couplings (saturating) and false couplings (eps^2) against the analytic closed forms, with the n_seg floor, formula epsilon and dense-onset epsilon marked](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/sparsity_components_vs_epsilon.png?v=6b2bd553)

Figure (components). Measured nonzero counts (markers) against the closed forms (lines), no free parameter on the true side and one fitted $`\kappa`$ on the false side. At T = 120 the diagonal floor is $`n_{\rm seg}=4T^2=57\,600`$; the true couplings climb through the knee and **saturate at exactly $`2\cdot 3T=720`$** by the formula ε (3.18 mrad), where A is still only 1.5% above the floor ($`A_{\rm nnz}=58\,436`$, $`n_{\rm true}=360`$, $`n_{\rm false}=58`$). The picture is the whole story: below the knee A is **pure diagonal** ($`A_{\rm nnz}=n_{\rm seg}`$, no couplings survive); through the knee the **true** couplings switch on and saturate at 3T (green); then the **false** couplings (red, ε²) take over and, past $`\varepsilon_{\rm dense}`$, dominate A_nnz — the matrix densifies. The formula ε sits in the flat valley between true-saturation and false-takeoff, which is precisely why it keeps A sparse.

![A_nnz and fill multiplier vs epsilon for several track counts, showing the dense-onset epsilon moving to smaller values as T grows, with analytic totals overlaid](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/sparsity_vs_epsilon_Tscan.png?v=6b2bd553)

Figure (T-scan). Left: absolute $`A_{\rm nnz}`$ (= 1BQF QRAM gate count) vs ε for T = 30/60/120, three bands at $`4T^2`$, analytic totals as lines. Right: the fill multiplier $`A_{\rm nnz}/n_{\rm seg}`$; the dashed verticals are $`\varepsilon_{\rm dense}(T)= \{0.216, 0.153, 0.108\}`$ rad for T = 30/60/120 — moving left as $`T^{-1/2}`$. (Note the crossover in the valley: at fixed ε the *true*-coupling fill $`6T/4T^2=1.5/T`$ is larger for smaller T, so T = 30 sits slightly above T = 120 until the false term, $`\propto T^3`$, overtakes.) The dense wall marches left as T grows: at production multiplicity even a modestly-too-wide ε pushes the 1BQF gate count from $`O(T^2)`$ toward $`O(T^3)`$ — the sparse-A invariant is an ε constraint, not just a kernel choice. (The *classical* spsolve wall arrives even earlier than $`\varepsilon_{\rm dense}`$ — LU fill-in is superlinear in nnz, which is why the T = 400 solve already choked at ≈ 2× the formula ε in §7.10, well below the A-doubling point.)

![Fitted false-coupling prefactor kappa vs T against the analytic 3T^3 over 2 theta0 squared scaling, and the implied theta0 shown to be T-independent](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/epsilon_sensitivity/sparsity_kappa_vs_T.png?v=6b2bd553)

Figure (κ vs T — the decisive contrast). Left: the empirically fitted prefactor $`\kappa`$ against $`3T^3/2\theta_0^2`$. Right: the implied $`\theta_0=\sqrt{3T^3/2\kappa}`$, which collapses to a **T-independent constant** — the signature that the false-coupling growth is genuinely the $`T^3`$ combinatorial pool times a fixed geometric acceptance, not an empirical power fit. The fitted $`\kappa = \{8.8\times10^4,\ 5.5\times10^5,\ 4.9\times10^6\}`$ rad⁻² for T = 30/60/120 — a 56× rise over a 64× rise in $`T^3`$, so $`\kappa/T^3 = \{3.3, 2.5, 2.8\}`$ is constant to single-seed scatter — and the implied $`\theta_0 = \{0.68, 0.77, 0.73\}`$ rad averages **0.72 rad**, about 3.6× the cone half-angle (0.2 rad), the natural spread of the kink between two unrelated segments.

<callout icon="✅">
	**Bottom line (sparsity).** A_nnz against ε is closed-form and verified: $`A_{\rm nnz}=n_{\rm seg}+2\big[3T(1-e^{-\varepsilon^2/2\sigma_p^2}) + \kappa\varepsilon^2\big]`$ with $`\kappa=3T^3/2\theta_0^2`$. True couplings saturate at 3T (the signal is finite); false couplings grow as ε² with a $`T^3`$ prefactor (the combinatorial background is not). The formula ε lives in the sparse valley between them — at T = 120 it leaves A just 1.5% above the diagonal floor; the dense wall $`\varepsilon_{\rm dense}=\theta_0\sqrt{8/3T}\propto T^{-1/2}`$ is what the sparse-A invariant really protects, and it is why a fixed wide ε that is harmless at T = 30 ($`\varepsilon_{\rm dense}=0.22`$ rad) bites at T = 400 (≈ 0.06 rad, and the spsolve fill-in wall earlier still). The analytic and empirical curves agree with **one** fitted geometric constant, $`\theta_0\approx 0.72`$ rad, shared across all T — so the sparsity-vs-ε behaviour is understood analytically and confirmed empirically, the two agreeing wherever both were computed.
</callout>
