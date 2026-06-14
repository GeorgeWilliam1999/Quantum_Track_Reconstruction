<callout icon="🧭">
**What this is.** A hands-on companion to **Epsilon_study_2 §7.12** — a runnable notebook, `Toy_Characterisation/Epsilon_study_2/indefiniteness_exploration.ipynb`, that re-derives and plots, from scratch, why the *classical* segment efficiency collapses at high $`\sigma_{\rm res}\times T`$. §7.12 holds the full production treatment (the $T\le1000$ grid, the store-backed localisation, the 1BQF analysis); this page walks the notebook's figures. Builder: `make_indefiniteness_notebook.py`; figures: `figures/indefiniteness/`. Commit `3243dfe2`.
</callout>

## The puzzle, and the order parameter

At $`\sigma_{\rm res}=0.05`$ mm the **classical** segment efficiency is pinned at $`\sim1.0`$ up to $`T\approx400`$ and then **breaks**: 0.984 → 0.98 → 0.909 → 0.786 at T = 200/400/700/1000 (qtrk store, absolute $`\tau=0.35`$). A false-rate story cannot lose *true* segments; the efficiency-first search of §7.10 bottomed at $`\tau=0.061`$ because the true-segment **amplitudes themselves go below $`\tau`$ and negative**. It is a **solver** problem, and it is spectral.

The segment Hamiltonian is $`A=(\gamma+\delta)I-C`$, $`Ax=\delta\mathbf 1`$, with $`C`$ the kink-acceptance coupling — an **adjacency matrix** (0/1, zero diagonal). Since the diagonal of $`A`$ is the *constant* $`\gamma+\delta`$, $`\lambda(A)=(\gamma+\delta)-\lambda(C)`$, so

$$\lambda_{\min}(A) = (\gamma+\delta)-\lambda_{\max}(C)\quad\Longrightarrow\quad A\ \text{indefinite}\iff \lambda_{\max}(C)>\gamma+\delta\ (=4\ \text{at}\ \gamma=3,\delta=1).$$

## 1 — The idealised hub is a complete bipartite block (closed form)

A hub is a **bipartite** block $`K(m_{\rm in},m_{\rm out})`$ (in-segments ending at the shared hit × out-segments starting there). For a *complete* block $`Ax=\delta\mathbf1`$ solves in closed form (verified numerically to $`<10^{-13}`$): $`x_{\rm in}=((\gamma+\delta)+m_{\rm out})/((\gamma+\delta)^2-m_{\rm in}m_{\rm out})`$, and $`\lambda_{\min}(A)=(\gamma+\delta)-\sqrt{m_{\rm in}m_{\rm out}}`$. The amplitudes **diverge and flip sign** at the product $`(\gamma+\delta)^2=16`$; the threshold is $`\gamma`$-tunable (9 at $`\gamma=2`$, 25 at $`\gamma=4`$).

![Star K(1,m) amplitudes diverge and flip sign at product 16; lambda_min(A) crosses zero; the critical product is gamma-tunable](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/01_closed_forms.png?v=3243dfe2)

## 2 — A real hub *is* bipartite

Building a real event and classifying every coupled pair: **100% are in→out, zero same-side** (verified live). The densest hub is drawn as a biadjacency matrix $`B`$ — and it is only a few % filled, so $`\sigma_{\max}(B)`$ is far below $`\sqrt{\rm product}`$.

![Real hub biadjacency B (a few percent filled) and its isolated-block spectrum; sigma_max(B) well below sqrt(product)](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/02_real_hub_bipartite.png?v=3243dfe2)

## 3 — The `product > 16` criterion over-predicts

Because real blocks are sparse, the hub product blows past 16 by $`T\sim100`$ while $`\sigma_{\max}(B)`$ barely reaches $`\gamma+\delta=4`$. The complete-block criterion predicts the breakdown hundreds of T too early; the right per-block test is $`\sigma_{\max}(B)>\gamma+\delta`$, and it is only marginally met.

![Max hub product crosses 16 by T~100 but max isolated-block sigma_max barely reaches 4 - product criterion over-predicts](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/03_product_overpredicts.png?v=3243dfe2)

## 4 — The instability is cooperative, localised, and computed *exactly*

Bipartite blocks **share segments** (each segment is an out-segment at its start-hub and an in-segment at its end-hub), so they chain across the interior planes; the order parameter is the **global** $`\lambda_{\max}(C)`$. We compute it **exactly** by connected-component decomposition (C nonnegative, A block-diagonal) — a whole-matrix `eigsh` *under-converges* on the clustered top spectrum at high $`T`$ (in the §7.12 grid the same matrix scored 4.47 / 4.96 / 5.03 / 6.44 on repeated calls), which the per-component dense eig sidesteps. At $`T=400,\,\sigma_r=0.05`$ the global $`\lambda_{\max}(C)\approx4.1`$ exceeds 4 while the best isolated block is only $`\approx3.5`$ (the **cooperative gap**); the lowest mode of A is **localised** on ~20 segments across ~15 hits and the interior planes.

![Critical mode participation (90% on ~20 of 640k segments) and the cooperative gap: global lambda_max(C) > 4 > best isolated block](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/04_cooperative_localization.png?v=3243dfe2)

## 5 — ... and the classical solve drives the true amplitudes negative

Solving $`Ax=\delta\mathbf1`$ with MINRES (the symmetric-indefinite method the store solver uses — CG is invalid), the true segments sitting on the critical cluster are pushed **below $`\tau`$ and negative** (≈1.5% of true amplitudes go negative here). That is the efficiency loss, and no threshold can recover it.

![Classical amplitude vs weight on the critical mode: true segments on the cluster are driven below tau and negative](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/05_amplitudes_corrupted.png?v=3243dfe2)

## 6 — The (σ_res, T) boundary reproduces the efficiency cliff

Sweeping $`\lambda_{\max}(C)`$ over a small reproducible grid: only $`\sigma_{\rm res}=0.05`$ pierces $`\gamma+\delta=4`$ (near $`T_c\approx290`$), exactly where the measured classical efficiency cliffs; $`\sigma_{\rm res}\le0.02`$ stays PD and efficiency stays $`\approx1`$. (§7.12 runs this to $`T=1000`$ over 3 reps, overlays the store efficiency heatmap, and shows the exact unstable-mode count $`n_{\rm neg}`$ rising $`3\to118\to1600`$ to track the cliff depth.)

![lambda_max(C) vs T per sigma_res: only sigma_res=0.05 crosses gamma+delta=4 near T_c~290](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/06_boundary.png?v=3243dfe2)

## 7 — The two fixes

$`A`$ went indefinite because $`\lambda_{\max}(C)`$ overtook the diagonal $`\gamma+\delta`$. Two levers: **raise $`\gamma`$** (PD restored for $`\gamma>\lambda_{\max}(C)-\delta`$; the $`\gamma`$-aware $`\tau=\delta/(\delta+\gamma)+0.10`$ shifts down with it), or **tighten $`\varepsilon`$** to $`\approx0.6\times`$ the formula value (at the cost of true-coupling efficiency — the §7.3 knee / §7.7 ROC). At fixed efficiency the $`\gamma`$-bump is cleaner: it moves the *threshold*, not the *signal*.

![Two fixes: lambda_min(A) vs gamma crosses zero at gamma*, and lambda_max(C) vs epsilon drops below 4 as epsilon shrinks](https://raw.githubusercontent.com/GeorgeWilliam1999/Quantum_Track_Reconstruction/main/Toy_Characterisation/Epsilon_study_2/figures/indefiniteness/07_fixes.png?v=3243dfe2)

<callout icon="✅">
**Bottom line.** $`A=(\gamma+\delta)I-C`$ is indefinite iff $`\lambda_{\max}(C)>\gamma+\delta`$. A hub is bipartite $`K(m_{\rm in},m_{\rm out})`$ whose complete-block limit goes singular at product $`(\gamma+\delta)^2=16`$ — but real blocks are sparse, so that criterion over-predicts. The true breakdown is **cooperative**: chained bipartite blocks push the global $`\lambda_{\max}(C)`$ past $`\gamma+\delta`$ on a localised multi-hub cluster, driving the true amplitudes there negative. The $`(\sigma_{\rm res},T)`$ crossing reproduces the measured classical efficiency cliff. The full production detail — exact $`n_{\rm neg}`$ tracking ($`3\to118\to1600`$), store-backed localisation (100% of lost-true on indefinite clusters, $`\sim4660\times`$ lift), and the 1BQF $`f(\lambda)=\cos(\lambda t/2)`$ sign-blindness — is in **Epsilon_study_2 §7.12**.
</callout>
