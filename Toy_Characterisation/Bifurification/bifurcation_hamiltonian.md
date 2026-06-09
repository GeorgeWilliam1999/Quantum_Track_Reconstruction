# Adding a bifurcation (fork) term to the segment Hamiltonian — the mathematics

**Scope.** This note derives, from the matrix, what happens when we add a
Denby–Peterson **bifurcation penalty** to the segment-activation Hamiltonian, in
two forms (off-diagonal-only and full). It sets up the eigenvalue picture, the
expected effect on the segment-level metrics, and the differences between the
classical solve and the 1-bit quantum filter (1BQF). It is the analytical primer
for the notebooks in this folder.

Background (the base model and the eigenvalue/notch picture it produces) is in the
companion `Segment_level_studies/` write-ups; we reuse that language: **hit, track,
segment, true/false segment, compatibility, cluster, eigenvalue, the notch, τ**.

Operating point unless stated: γ = 3, δ = 1, so s ≡ γ+δ = 4 and the (base)
threshold τ = δ/(δ+γ) + 0.10 = 0.35.

---

## 1. The base Hamiltonian (recap)

Segment activations **x** solve the sparse linear system

$$
A_0\,\mathbf{x} = \mathbf{b}_0,\qquad \mathbf{b}_0=\delta\mathbf{1},\qquad
(A_0)_{ii} = \gamma+\delta,\qquad (A_0)_{ij} = -\,[\,\text{$i,j$ are a compatible continuation}\,].
$$

Two segments are a **compatible continuation** when the end-hit of one is the
start-hit of the other (they share a *middle* hit) **and** their kink angle is
below ε. Write $C$ for this 0/1 continuation adjacency, so

$$
A_0 = (\gamma+\delta)\,I - C .
$$

A segment is declared **active** iff $x_i>\tau$. The base model has the Hopfield
levels: an isolated/false segment sits at $\delta/(\delta+\gamma)=0.25$; a true
4-segment track-chain sits at $\tfrac4{11},\tfrac5{11}$; $\tau=0.35$ separates them.
Its weakness (established in the companion work): **false positives are accidental
cross-track clusters** — forks and bridges where one hit is shared by segments of
different tracks. The bifurcation term targets exactly those.

---

## 2. The bifurcation penalty (Denby–Peterson)

A real track uses each hit **once**: one segment in, one segment out. A
**bifurcation** is a violation of this — one hit feeding (or fed by) *several*
active segments. Two segments **fork** when they share a hit *on the same side*:

- **out-fork:** they share a start-hit (one hit → two outgoing segments),
- **in-fork:** they share an end-hit (two incoming segments → one hit).

Let the **fork graph** be the 0/1 adjacency

$$
B_{ij}=\big[\,i\neq j\ \text{and}\ \big(\operatorname{start}(i)=\operatorname{start}(j)\ \text{or}\ \operatorname{end}(i)=\operatorname{end}(j)\big)\,\big].
$$

(Note $B$ couples *same-side* hit-sharing; the continuation $C$ couples
*opposite-side* hit-sharing. They act on disjoint pairs.)

**Two choices for the fork graph:**
- **Full occupancy** $B$ — *all* co-hit pairs, regardless of angle. This is the
  naïve Denby–Peterson term; it is **dense** ($\mathrm{nnz}=O(T^3)$, since every
  hit starts $O(T)$ segments) and is analysed in §3–§8 as the worst case.
- **ε-windowed** $B_\varepsilon$ — only co-hit pairs whose **mutual angle is $<\varepsilon$**
  (inside the acceptance window), i.e. the *genuinely competing* near-collinear
  continuations: *of all the segments at a hit that lie within $\varepsilon$, you are
  punished for choosing more than one*. $B_\varepsilon$ is **sparse**
  ($\mathrm{nnz}=O(n_{\rm seg})$ or less), preserves the sparse-A invariant, and is
  the **production choice** — see §9, where it removes false positives *and* keeps
  the 1BQF intact.

The Denby–Peterson **occupancy penalty** charges $\tfrac{\beta}{2}N_h(N_h-1)$ for
each hit-side group, where $N_h=\sum_{i\in h}x_i$ is the total activation using
that group. Summed over the groups $g\in\mathcal G$ (the set of "segments starting
at $h$" and "segments ending at $h$" over all hits $h$):

$$
E_{\rm bif}(\mathbf{x})=\frac{\beta}{2}\sum_{g\in\mathcal G}\Big[\big(\textstyle\sum_{i\in g}x_i\big)^2-\sum_{i\in g}x_i\Big].
$$

The base solve is the stationary point of $E_0=\tfrac12\mathbf{x}^\top A_0\mathbf{x}-\mathbf{b}_0^\top\mathbf{x}$.
Adding $E_{\rm bif}$ and setting $\nabla(E_0+E_{\rm bif})=0$ gives the modified
linear system. The gradient of the penalty is (each segment lies in exactly two
groups — its start-group and its end-group):

$$
\frac{\partial E_{\rm bif}}{\partial x_k}=\beta\!\!\sum_{g\ni k}\sum_{i\in g}x_i-\beta
=\beta\big(2x_k+(B\mathbf{x})_k\big)-\beta .
$$

---

## 3. The two forms

### 3a. Full Denby–Peterson
Using the complete gradient above, $\big(A_0+2\beta I+\beta B\big)\mathbf{x}=\mathbf{b}_0+\beta\mathbf{1}$, i.e.

$$
\boxed{\,A''=(\gamma+\delta+2\beta)\,I-C+\beta B,\qquad \mathbf{b}''=(\delta+\beta)\mathbf{1}\,.}
$$

### 3b. Off-diagonal only
Keep only the **repulsive pairwise** part $\tfrac{\beta}{2}\mathbf{x}^\top B\mathbf{x}$ (drop the
diagonal/linear occupancy pieces). Then $\nabla=\beta B\mathbf{x}$ and

$$
\boxed{\,A'=(\gamma+\delta)\,I-C+\beta B,\qquad \mathbf{b}'=\delta\mathbf{1}\,.}
$$

**Both forms have a *constant* diagonal.** This is special to a segment model where
**every segment touches exactly two hits** (one per adjacent plane): the occupancy
diagonal is the uniform shift $+2\beta$, never a per-segment (degree-dependent)
cost. Consequently **both $A'$ and $A''$ satisfy the 1BQF constant-diagonal
requirement and can be solved on the quantum filter** — the fork-degree enters only
the *off-diagonal* $\beta B$. (The diagonal moves the notch: $A'$ keeps it at
$\gamma+\delta$; $A''$ moves it to $\gamma+\delta+2\beta$.)

Off-diagonal entries of the new Hamiltonian: $-1$ on continuations, $+\beta$ on
forks. The fork coupling is **repulsive** (positive), so co-active forking
segments push each other *down*.

---

## 4. Worked example — a real track with a false fork

Five segments: a clean track chain $s_1\!-\!s_2\!-\!s_3\!-\!s_4$ (continuations,
$-1$) plus a false segment $f$ that shares the **start-hit** of $s_1$ (an out-fork,
so $B_{s_1 f}=1$). In the base model $f$ is continuation-isolated (eigenvalue on the
notch, $x_f=0.25$). Solving $A\mathbf{x}=\mathbf{b}$ for a β-sweep (γ=3, δ=1, s=4):

**Off-diagonal form** $A'=(s)I-C+\beta B$, $\mathbf b=\delta\mathbf 1$ (attractor fixed at $\delta/s=0.25$):

| β | $x_{s_1},x_{s_2},x_{s_3},x_{s_4}$ (true) | $x_f$ (false fork) | eigenvalues of $A'$ |
|---|---|---|---|
| 0.00 | 0.364, 0.455, 0.455, 0.364 | **0.250** | 2.382, 3.382, **4**, 4.618, 5.618 |
| 0.50 | 0.336, 0.447, 0.453, 0.363 | **0.208** | 2.359, 3.254, **4**, 4.746, 5.641 |
| 1.00 | 0.318, 0.442, 0.451, 0.363 | **0.171** | 2.268, 3.000, **4**, 5.000, 5.732 |
| 2.00 | 0.314, 0.441, 0.451, 0.363 | **0.093** | 1.697, 2.697, **4**, 5.303, 6.303 |

The false fork is driven **down** ($0.25\to0.09$) while the true track is almost
untouched — exactly the intended effect: the repulsion from the active true
segment $s_1$ suppresses the competing fork. Analytically, $f$'s row reads
$(\gamma+\delta)x_f+\beta x_{s_1}=\delta$, so

$$
x_f=\frac{\delta-\beta\,x_{s_1}}{\gamma+\delta}=\frac{1-\beta\,x_{s_1}}{4}\ \xrightarrow{\ \beta\uparrow\ }\ \text{below the }0.25\text{ attractor and toward }0 .
$$

**Full form** $A''=(s+2\beta)I-C+\beta B$, $\mathbf b=(\delta+\beta)\mathbf 1$ — the
attractor **rises** with β, $(\delta+\beta)/(\gamma+\delta+2\beta)$ (0.25, 0.30,
0.333, 0.375 at β = 0, 0.5, 1, 2), and $x_f$ only drifts (0.25→0.28); suppression
is *relative to a moving floor*. The bare τ = 0.35 stops being the right cut — the
full form needs a **β-aware threshold**

$$
\tau(\beta)=\frac{\delta+\beta}{\gamma+\delta+2\beta}+0.10 .
$$

> **Take-away (isolated fork).** With a *single* fork partner the off-diagonal
> form is a clean fork-suppressor at fixed threshold; the full form rescales the
> spectrum and needs a β-aware τ.

### 4b. The dense-event reality — equal fork degree

The clean suppression above assumes $f$ has **one** fork partner. In a real event
this is false: every hit on an interior plane starts $O(T)$ candidate segments,
all mutually forked, so **every** segment — true or false — has a large fork
degree, and (measured on stored events) **the median fork degree of true and false
segments is essentially equal** (≈ 38 at T = 20, ≈ 98 at T = 50). The penalty
therefore pushes true and false activations down at *the same rate*. Concretely
(off-diagonal form, stored T = 50 event):

| β | median $x_{\rm true}$ | median $x_{\rm false}$ | ratio | min $x_{\rm true}$ | max $x_{\rm false}$ | AUC(true:false) |
|---|---|---|---|---|---|---|
| 0.00 | 0.423 | 0.250 | 1.69 | 0.364 | 0.392 | 1.000 |
| 0.01 | 0.340 | 0.200 | 1.70 | 0.292 | 0.314 | 1.000 |
| 0.05 | 0.191 | 0.111 | 1.72 | 0.164 | 0.177 | 1.000 |
| 0.10 | 0.125 | 0.072 | 1.74 | 0.107 | 0.115 | 1.000 |

So $\beta$ acts as a near-**uniform down-scaling** of the whole solution: the
true/false ratio and the ranking (AUC = 1.000) are essentially unchanged; only the
absolute scale shrinks. What differentiates true from false is **not** the fork
degree (equal) but the **continuation attraction** (the $-1$ chain coupling) that
only true segments enjoy — and that buffer is already what the base model uses.

**Consequence:** at a *fixed* τ the fork term looks like it "kills" segments (a
false at $0.25$ and a true outer at $0.364$ both cross below $0.35$ together as the
scale shrinks), so efficiency and false-rate both collapse. This is a
**threshold artefact**, not a real loss of separability — exactly the
fixed-absolute-threshold pitfall documented in the base-model write-ups. The
honest readouts are (i) a **β-aware threshold** and (ii) the **separation / AUC**.

---

## 5. Spectrum and the notch

Write $A=\text{(diag)}\,I-M$ with $M=C-\beta B$. Eigenvalues are
$\lambda=\text{(diag)}-\mu(M)$, and a **bad (notch) eigenvalue** is still
$\lambda=\text{diag}\iff\mu(M)=0$ (diag $=\gamma+\delta$ for $A'$,
$\gamma+\delta+2\beta$ for $A''$). The 1BQF filter $f(\lambda)=\cos(\lambda t/2)$,
$t=\pi/\text{diag}$, again erases the notch eigenspace.

What the fork term changes:

- In the base model the **false bulk** (isolated segments) is one giant notch
  degeneracy. Adding $\beta B$ couples it — **but the fork graph has a large null
  space**. A repulsive fork star $K_{1,m}$ (one segment forked by $m$ others) gives
  $A$-eigenvalues $\text{diag}-\{-\sqrt m\,\beta,\,0^{(m-1)},\,+\sqrt m\,\beta\}$:
  **$m-1$ modes stay exactly on the notch**, only two split off
  ($\text{diag}\mp\sqrt m\,\beta$). So the notch degeneracy **largely survives**;
  only the symmetric fork combinations move (the 5×5 example opens a $\{3,5\}$ pair
  at β = 1 while a notch mode persists). On a stored T = 20 event the giant notch
  spike is barely dented by β.
- The visible *classical* effect is therefore on the **solution scale**, not the
  spectrum: β down-scales every activation roughly uniformly (§4b).

---

## 6. Effect on the segment-level metrics (what to expect)

With efficiency $=n_{\rm TA}/n_{\rm true}$, purity $=n_{\rm TA}/n_{\rm act}$,
false-rate $=n_{\rm FA}/n_{\rm act}$, the §4b result drives everything:

- **At a fixed τ = 0.35:** because β down-scales true and false together, the
  outer true plateau ($0.364$, barely above τ) crosses below τ at almost the same
  β as the false bulk ($0.25$). So both efficiency **and** false-rate fall steeply
  (e.g. T = 50: at β = 0.01, far $0.02\to0$ but efficiency $1.0\to0.5$ as the outer
  true segments are lost; by β ≈ 0.05 everything is inactive). This is the
  threshold artefact, not a genuine purity win.
- **At a β-aware / separation-based readout:** AUC stays $\approx 1$ and the
  true/false ratio is essentially flat, so the fork term buys **little** extra
  separability over the base model on these clean low-T events.
- **It is not a targeted false-positive fix.** The actual false positives are
  cross-track **bridges**, which are *coupled* clusters carrying the same
  continuation + fork structure as a real track — so the fork penalty cannot tell
  them apart from true segments and does not preferentially suppress them (the
  max-false bridge tracks the min-true segment down in lock-step in §4b).

So the central questions for the notebooks are sharper than "does purity improve":
**(a)** quantify the uniform down-scaling and confirm AUC is preserved;
**(b)** check whether the *full* form + β-aware τ, or a higher-density regime (more
bridges), ever yields a real separability gain; **(c)** see whether the quantum
1BQF reproduces the classical (down-scaled) ranking or distorts it.

---

## 7. Classical vs quantum — the key difference

1. **Both forms run on the 1BQF** (constant diagonal). The notch sits at the
   diagonal: fixed at $\gamma+\delta$ for $A'$, shifting to $\gamma+\delta+2\beta$
   for $A''$ (so the quantum evolution time $t=\pi/\text{diag}$ and the whole
   filter shift with β in the full form).
2. **Sparsity — the real cost.** The continuation graph $C$ is sparse,
   $\mathrm{nnz}(C)=O(n_{\rm seg})=O(T^2)$, which is what made the 1BQF feasible.
   The fork graph $B$ is **dense**: every hit starts $O(T)$ segments, all mutually
   forked, so $\mathrm{nnz}(B)=O(T^3)$. The 1BQF circuit cost scales as
   $O(\mathrm{nnz}(A)\,2^{n_{\rm sys}})$, so the bifurcation term makes the quantum
   solve **much** more expensive — feasible only at small T, while the classical
   (MINRES) solve handles $T=100$ easily. **This asymmetry is itself a headline
   result:** the fork penalty is cheap classically but breaks the quantum
   sparsity that the 1BQF relies on.
3. **The 1BQF discrimination collapses — confirmed.** This is the headline
   classical/quantum difference. The 1BQF returns the *exact* one-bit-filtered
   solution $\mathbf{x}_Q\propto\sum_j\beta_j\cos(\lambda_j t/2)\,\mathbf{u}_j$ — so
   this is a property of the modified solution, not a circuit-error artefact. In
   the base model the source projects onto modes that are either on the notch
   (erased, the false bulk) or on the clean true-chain satellites. The fork term
   **redistributes that projection** onto the few modes that split off the notch
   (§5), where the coarse, sign-changing one-bit filter mis-weights them and mixes
   true and false amplitudes. The result, on a stored T = 10 event:

   | | β = 0 | β = 0.02 (off-diag) |
   |---|---|---|
   | classical AUC(true:false) | 1.000 | **1.000** |
   | quantum AUC(true:false) | 1.000 | **0.547 (≈ random)** |
   | quantum false-rate (fixed τ) | 0.000 | 0.92 |
   | quantum solve time | 4.4 s | 86 s |

   So even a tiny β leaves the **classical** ranking perfect but **destroys the
   quantum** one (AUC → random) and slows the solve ~20× (the dense fork circuit).
   **The bifurcation term is essentially incompatible with the 1-bit quantum
   filter** as it stands: the very thing that makes the false bulk easy for the
   1BQF (its degeneracy on the notch) is what the fork term removes.

> **Punchline (full, dense $B$).** The naïve fork penalty over *all* co-hit pairs
> is classically blunt (uniform down-scaling, not targeted) and quantumly harmful
> (breaks sparse-A → ≈20× cost; redistributes weight onto off-notch modes the
> one-bit inversion mis-weights → AUC ≈ 0.5). **§9 shows the fix: restrict to the
> ε acceptance window.** Both problems are caused by coupling the whole dense
> co-hit population; the ε-window couples only the few genuinely-competing forks.

---

## 8. What the notebooks do (T ≤ 100)

- **Construction & spectrum:** build $A'$ and $A''$ on stored events (reuse the
  events; the Hamiltonian is rebuilt with $\beta B$, so **metrics are recomputed
  here, not read from the base store**), show the fork graph density, and the
  spectrum lifting off the notch as β grows; with the small analytic examples
  above as anchors.
- **Metrics vs β:** segment efficiency / purity / false-rate vs β for both forms,
  **classical to T = 100** and **quantum (1BQF) at the small T that stays
  tractable**, against the β = 0 baseline — quantifying the purity gain and the
  classical/quantum difference.

**Conventions carried in:** truth and the metric definitions are unchanged
(`qtrk_pipeline` truth mask + segment metrics); only the *solution vectors* change
because $A$ changed. Threshold: bare τ = 0.35 for $A'$; the β-aware τ(β) for $A''$
(both reported).

---

## 9. The ε-windowed fork — sparse, targeted, and 1BQF-safe (the fix)

Restrict the penalty to the acceptance window: $A=(\gamma+\delta)I-C+\beta B_\varepsilon$,
where $B_\varepsilon$ couples only co-hit pairs of mutual angle $<\varepsilon$ — the genuinely
competing (near-collinear) continuations. This couples a *handful* of segment
pairs instead of the whole $O(T^3)$ co-hit population, which fixes every problem of
the dense form.

**Sparsity (aligns with the to-do).** $\mathrm{nnz}(B_\varepsilon)$ is $O(n_{\rm seg})$ or
less — at $\varepsilon=2$ mrad on clean events it is *empty* at T = 20 (tracks
well-separated), ≈ 552 at T = 50, ≈ 46 at T = 100 (vs the dense $\mathrm{nnz}(B)$
≈ 60 800 and 3 960 000). The acceptance window only fires where two tracks are
actually within ε at a hit.

**Classical — targeted, no collateral.** On a stored T = 50 event (4 false
positives at β = 0):

| β | efficiency | false-rate | # false-active | median true | median false |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.020 | 4 | 0.423 | 0.250 |
| 0.5 | 0.980 | **0.000** | **0** | 0.412 | 0.250 |
| 2.0 | 0.980 | 0.000 | 0 | 0.410 | 0.250 |

The false positives are **removed** (false-rate → 0) at a 2 % efficiency cost and
with **no down-scaling** (the medians are unchanged) — the opposite of the dense
form, which collapses everything together.

**Quantum — the 1BQF is preserved.** Because $B_\varepsilon$ is sparse, the false bulk
stays isolated on the notch and the circuit stays small. Adding sparse fork edges
to a T = 10 event keeps the quantum AUC at **0.96–0.99** (vs the dense fork's
**0.55**) and the solve at **≈ 1 s** (vs **86 s**). The mechanism that broke the
1BQF for the dense fork — lifting the whole false population off the notch and
inflating the circuit — simply does not occur when only a few genuinely-competing
pairs are coupled.

**Summary.**

| property | dense fork (all co-hit) | **ε-windowed $B_\varepsilon$** |
|---|---|---|
| sparsity | $O(T^3)$, breaks sparse-A | **$O(n_{\rm seg})$, sparse** |
| classical | uniform down-scaling (eff & far collapse) | **false-rate → 0, no collateral** |
| quantum 1BQF | AUC → 0.55, ≈ 20× slower | **AUC ≈ 1, fast** |

The ε-windowed bifurcation is the version to carry forward: it suppresses exactly
the near-collinear false bridges (the genuine bifurcation ambiguities catalogued in
`../Segment_level_studies/07_segment_amplitude_atlas.ipynb`, mode **F4/F3**) while
leaving the rest of the spectrum — and the quantum solver — intact. See
`03_epsilon_windowed_bifurcation.ipynb`.
