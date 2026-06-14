## 7.12 Local-vs-cooperative indefiniteness of A — the hub breakdown behind the classical efficiency cliff

§7.4 and §7.10 reported a *classical* efficiency cliff that the false-rate story cannot explain: at $`\sigma_{\rm res}=0.05`$ mm the segment efficiency is pinned at ~1.0 up to $`T\approx400`$ and then **breaks** — 0.984 @ T=200, 0.98 @ 400, 0.909 @ 700, 0.786 @ 1000 (qtrk store, absolute $`\tau=0.35`$, 20 reps). At T=400 the efficiency-first search of §7.10 bottomed out at $`\tau=0.061`$: the TRUE-segment amplitudes themselves fall below $`\tau`$ and go **negative**, so *no* threshold recovers ≥99%. This is not a cut problem; it is a **solver** problem, and it is spectral. Script: `gen_indefiniteness_study.py` → `figures/epsilon_sensitivity/{star_amplitude_divergence,hub_degree_vs_T,neg_eigs_vs_sigma_res_T,indefiniteness_boundary,critical_mode_localization,gamma_epsilon_fix}.png` + `outputs/indefiniteness_study.json`, commit `1b102d6f`. (The spectral part needs only A, no solve, so it is cheap; the localization/correlation part reuses the store classical vectors.)

<callout icon="📐">
**Definition (the order parameter).** With $`A=(\gamma+\delta)I-C`$ and $`C`$ the kink-acceptance coupling (an adjacency matrix: $`C_{ij}=1`$ for an accepted coupling, 0 otherwise, zero diagonal), the diagonal is the *constant* $`\gamma+\delta`$, so the spectra are related **exactly**:
$$
\lambda(A) = (\gamma+\delta) - \lambda(C) \;\Rightarrow\; \lambda_{\min}(A) = (\gamma+\delta) - \lambda_{\max}(C).
$$
Hence **A is indefinite iff $`\lambda_{\max}(C) > \gamma+\delta`$** ($`=4`$ at $`\gamma=3,\delta=1`$). Every result below is a statement about $`\lambda_{\max}(C)`$.
</callout>

### The idealised building block: a hub is a *bipartite* block (closed form)

`construct_hamiltonian` couples two segments only when they **share a middle hit** and their kink is below ε. We verified directly that the coupling is **strictly bipartite**: of all coupled pairs, 100% join an *in*-segment (ending at the shared hit $`h`$) to an *out*-segment (starting at $`h`$); **zero** pairs are same-side (in–in or out–out). So the local block at a hub is a restricted complete-bipartite graph $`K(m_{\rm in},m_{\rm out})`$, not a same-side star — the relevant size is the **product** $`m_{\rm in}m_{\rm out}`$, not a degree.

For a *complete* bipartite block, $`A=(\gamma+\delta)I-\mathrm{Adj}`$, $`Ax=\delta\mathbf 1`$ solves in closed form (verified to $`<10^{-13}`$):
$$
x_{\rm in}=\frac{(\gamma+\delta)+m_{\rm out}}{(\gamma+\delta)^2-m_{\rm in}m_{\rm out}},\qquad
x_{\rm out}=\frac{(\gamma+\delta)+m_{\rm in}}{(\gamma+\delta)^2-m_{\rm in}m_{\rm out}},
$$
$$
\lambda_{\min}(A)=(\gamma+\delta)-\sqrt{m_{\rm in}m_{\rm out}}.
$$
The **star $`K(1,m)`$** is the $`m_{\rm in}=1`$ case: $`x_c=(4+m)/(16-m)`$, $`x_\ell=5/(16-m)`$. The amplitudes **diverge** at the product $`(\gamma+\delta)^2=16`$ and **flip sign** beyond it — the textbook picture of a true segment driven negative. The critical product is $`(\gamma+\delta)^2`$: it is **$`\gamma`$-tunable** (9 at $`\gamma=2`$, 25 at $`\gamma=4`$).

### …but the product>16 criterion *over-predicts*

In the real Hamiltonian the hub blocks are **sparse**: at T=400, $`\sigma_r=0.05`$ the largest hub product reaches $`m_{\rm in}m_{\rm out}\approx 6\,000\gg16`$, yet the block carries only a few % of those possible edges, so its singular value $`\sigma_{\max}(B)`$ sits near ≈4.0 — only *just* at $`\gamma+\delta=4`$, and below it for most events. The complete-block formula is the wrong limit; the product$`>16`$ test predicts indefiniteness hundreds of T too early. The right per-block test is $`\sigma_{\max}(B)>\gamma+\delta`$, and at the onset it is only marginally met.

### The real mechanism is *cooperative*

The bipartite blocks **share segments** — every segment is an out-segment at the hub at its start hit and an in-segment at the hub at its end hit — so the blocks chain across the three interior planes into connected clusters. The instability is a property of the **global** $`\lambda_{\max}(C)`$, not of any one hub. We compute it **exactly**: $`C`$ is a nonnegative adjacency matrix and $`A`$ is block-diagonal over connected components, so a per-component dense eigensolve gives $`\lambda_{\max}(C)`$ and the count of unstable modes with no Lanczos-convergence ambiguity. At T=400, $`\sigma_r=0.05`$ the global $`\lambda_{\max}(C)=`$ 4.59 exceeds 4 even though the best isolated block is only ≈4.0 (the *cooperative gap*) — and some events are indefinite **purely cooperatively** (best block $`\sigma_{\max}<4<\lambda_{\max}(C)`$). The lowest mode of A is tightly localised: 90% of its weight sits on ~22 segments spanning ~15 distinct hits across all interior planes (a connected multi-hub cluster), not one hub. The true segments embedded in that cluster are exactly the ones driven below $`\tau`$ and negative. (At extreme occupancy — $`\sigma_r=0.05,\,T=1000`$ — these clusters merge into a single percolating component of thousands of segments with ~1600 unstable modes; the instability stops being local.)

### The (σ_res, T) boundary reproduces the efficiency cliff

Building A on a $`(\sigma_{\rm res},T)`$ grid (formula ε, $`\sigma_{\rm scatt}=10^{-4}`$, build-only, 3 reproducible reps) and reading $`\lambda_{\max}(C)`$ and the exact number of unstable modes $`n_{\rm neg}=\#\{\lambda(A)<0\}`$:

- $`\sigma_r=0.01`$: $`\lambda_{\max}(C)\le3.1`$ through T=700 → A stays **PD** ($`n_{\rm neg}=0`$), efficiency pinned at 1.000. ✓
- $`\sigma_r=0.02`$: $`\lambda_{\max}(C)`$ climbs but stays **below 4** ($`\le3.5`$) even at T=1000 → PD on average; efficiency essentially flat (0.999 @ 700, 0.997 @ 1000), the tiny erosion coming from the rare event that does cross. ✓
- $`\sigma_r=0.05`$: crosses 4 near $`T_c\approx`$ 290 → indefinite for $`T\gtrsim400`$. The unstable-mode count tracks the efficiency loss: $`n_{\rm neg}\approx`$ 3 @ T=400 (eff 0.98) → 118 @ T=700 (eff 0.909) → ~1600 @ T=1000 (eff 0.786). ✓

The predicted onset locus $`\lambda_{\max}(C)=\gamma+\delta`$ overlays the measured classical-efficiency cliff (Fig. boundary). This is **distinct from the purity collapse of §7.3–7.5**: purity falls far earlier (the false-coupling load $`\propto\kappa\varepsilon^2`$ activating spurious *isolated* segments), whereas efficiency only breaks once A actually goes indefinite and the *true* amplitudes are corrupted.

### Implications

- **Classical.** The store solver already uses **MINRES** (symmetric-indefinite Krylov), not CG, *because* A goes indefinite — CG is invalid here and stalls. The near-singular modes ($`|\lambda(A)|\to0`$) get amplified in $`x=A^{-1}b=\sum_k (v_k\!\cdot b/\lambda_k)v_k`$, pushing the true amplitudes on the critical cluster below $`\tau`$ and negative. This is the cliff.
- **1BQF.** The 1-bit filter response $`f(\lambda)=\cos(\lambda t/2)`$ is **even in $`\lambda`$**: it cannot distinguish $`+\lambda`$ from $`-\lambda`$. Negative eigenvalues of A — the indefinite modes — alias onto their positive mirror, so the sign information the inversion needs is destroyed on the critical cluster, and the signal-support rescale cannot restore it. The breakdown is therefore *worse* for the quantum filter than for MINRES.
- **The two fixes.** (i) **Raise $`\gamma`$** (lift the diagonal): PD is restored for $`\gamma>\lambda_{\max}(C)-\delta`$ — at the T=400, $`\sigma_r=0.05`$ event that means $`\gamma\gtrsim`$ 4.0, and the $`\gamma`$-aware $`\tau=\delta/(\delta+\gamma)+0.10`$ simply shifts down with it. (ii) **Tighten ε** (thin the coupling): shrinking ε to ≈0.6× the formula value drops $`\lambda_{\max}(C)`$ below 4, but at the cost of true-coupling efficiency (the §7.3 knee / §7.7 ROC) — the same trade as everywhere. At fixed efficiency the $`\gamma`$-bump is the cleaner lever, because it moves the *threshold* rather than the *signal*.

<callout icon="✅">
**Bottom line (indefiniteness).** $`A=(\gamma+\delta)I-C`$ is indefinite **iff $`\lambda_{\max}(C)>\gamma+\delta`$**. The coupling at a hub is bipartite $`K(m_{\rm in},m_{\rm out})`$, whose *complete*-block limit goes singular at product $`(\gamma+\delta)^2=16`$ — but real blocks are ~3% filled, so the product>16 criterion over-predicts and no isolated hub breaks. The true instability is **cooperative**: chained bipartite blocks push the *global* $`\lambda_{\max}(C)`$ past $`\gamma+\delta`$ on a connected multi-hub cluster, driving the true amplitudes there negative. The crossing locus in $`(\sigma_{\rm res},T)`$ reproduces the measured classical efficiency cliff (PD at $`\sigma_r\le0.01`$; breaks at $`\sigma_r=0.05,\,T\gtrsim400`$). Distinct from the purity collapse. Fix by raising $`\gamma`$ above $`\lambda_{\max}(C)-\delta`$ (cleaner) or tightening ε (costs efficiency).
</callout>
