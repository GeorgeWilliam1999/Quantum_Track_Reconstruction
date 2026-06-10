# QSVT — can a polynomial filter map *all* the failure modes to zero?

> **STATUS — Steps A–E executed (2026-06-10).** Notebooks `00`–`04` + `qsvt_helpers.py`.
> **Verdict: YES** (down to a provable floor). Step A: true tracks are all P4; the
> false dominant modes sit ≥ 0.2 from every true eigenvalue; the same-length-bridge
> degeneracy is the only irreducible piece (0 % at T≤200, 0.6 % at T=400/ε=0.004).
> Step B: a degree-30 band-limited-inverse polynomial, $\max|p|=0.38<1$ (realizable).
> Step C: it lifts the true/false **AUC from 0.49 (classical) / 0.64 (1-bit) to 0.88**,
> nulling **all** off-length bridges and ~95 % of hubs (at τ=0.35: eff 0.84 / far 0.04
> vs 1-bit 0.54 / 0.34). Step D: cost ≈ 30 block-encoding calls × a ~2× amplitude-
> amplification overhead (success prob 0.08 vs 0.34) ≈ tens× the 1-bit depth, two
> extra ancillas, sparse-$A$ preserved; phase-angle synthesis (pyqsp) deferred — it
> failed to build here. Step E: the **AUC advantage is robust** (0.87–0.93 across
> scattering ≤ 8e-4 and inefficiency ≤ 5 %); the fixed-τ eff/far is a knife-edge
> (true outer segments sit at x≈0.38≈τ) and hit inefficiency feeds the P3-bridge
> notch (gate it on track-length). *Remaining: minimax/Lin–Tong design + pyqsp angles.*

**Question (G. Scriven).** The 1-bit cosine notch of the 1BQF can only erase the
*isolated* false segments (a single degenerate eigenvalue at $\lambda=\gamma+\delta$).
The T=200 deep-dive
(`../Toy_Characterisation/Bifurification/04_failure_types_and_phase.ipynb`) showed a
*second* notch can't reach the coupled false. Could **QSVT** instead build a
**polynomial** $p(A)$ that maps *every* failure-mode eigenvalue to ~0 while keeping
the true tracks?

**One-line answer (to be verified by this plan).** QSVT is exactly the right
generalisation — it replaces the single cosine notch by an (almost) arbitrary
bounded polynomial $p(\lambda)$, i.e. *many* sharp notches. Whether it can null all
failure modes reduces to one measurable question: **are the false eigenvalues
separated from the true ones, or coincident?** A polynomial is a *function* of
$\lambda$ — it can separate *near* spectra but never *coincident* ones. A quick
census on real T=200 events (below) says the false modes sit **≥ 0.2 away** from
every true eigenvalue and **0 %** are spectrally degenerate with a true track — so
there *is* a gap to exploit. The cost is a degree-~25–30 polynomial (≈ that many
block-encoding calls), and the only irreducible residue is a false bridge of the
*same length* as a true track (rare here, grows with density).

---

## 0. TL;DR feasibility numbers (probe, 20 × 200-track events, classical)

| quantity | value | meaning |
|---|---|---|
| FP dominant-mode min gap to nearest true eigenvalue | **median 0.228**, 10th pct 0.206 | there is a real spectral gap |
| FP dominant within 0.20 of a true eigenvalue | **0.0 %** | no false dominant mode hides under a true one |
| FP clusters spectrally identical to true P4 (irreducible) | **0 / 123 = 0 %** | no same-length degeneracy in this sample |
| smallest exploitable gap (bridge 2.586 vs true 2.382) | **0.20** → rescaled 0.075 | sets the polynomial degree |
| → approx QSVT degree to resolve it | **~ 27** (hubs cheaper, gap ≥ 0.38) | ≈ 27 controlled-$A$ calls vs 1 for 1BQF |

**Green light, with caveats** (full spectral support, the renormalisation
interaction, and the same-length-bridge floor — see §3–§4).

---

## 1. Background — from the 1-bit cosine to QSVT

The 1BQF is a Hadamard test on a single ancilla controlling $e^{-iAt}$: it realises
the filter $f(\lambda)=\cos(\lambda t/2)=\cos(\pi\varphi)$ — a degree-1
(in $\cos$) response with **one** in-band zero, fixed at $\lambda=\pi/t$. Tuning $t$
moves that single zero; it cannot add a second (nb04 §4).

**QSVT** (Gilyén–Su–Low–Wiebe 2019) takes a block-encoding $U_A$ of (rescaled) $A$
and a sequence of $d$ single-qubit phase rotations $\phi_1,\dots,\phi_d$, and
realises $p(A)$ for **any** real polynomial $p$ of degree $d$ with definite parity
and $|p(x)|\le 1$ on $[-1,1]$. For Hermitian PSD $A$ this is QSP/QET on the
eigenvalues. So:

> **1BQF = QSVT restricted to $p=\cos$ (one notch). QSVT lets us engineer the whole
> response** — a multi-notch / band-pass / band-limited-inverse polynomial.

The natural target is a **band-limited matrix inverse**: $p(\lambda)\approx 1/\lambda$
on the true-track eigenvalues (so the true solution is preserved, like HHL), and
$p(\lambda)\approx 0$ on every false eigenvalue.

## 2. The design spec — what maps where

Rescale the spectrum $[\lambda_{\min},\lambda_{\max}]\to[-1,1]$ (here roughly
$[0.7, 6.0]$ on coupled clusters). Operating point $\gamma=3,\delta=1$, $s=4$,
$\tau=0.35$.

**Keep (pass, ideally $\approx 1/\lambda$) — the true-track spectrum.**
Detector geometry pins true tracks to a fixed length (5 hits → 4 segments → **P4**),
whose path-graph eigenvalues are
$$\lambda_k=(\gamma+\delta)-2\cos\!\frac{k\pi}{5}=\{2.382,\;3.382,\;4.618,\;5.618\},\quad k=1..4.$$
(Edge/short tracks add a few discrete lines; the census in Step A characterises the
real true support cleanly.)

**Null (→ 0) — the failure modes** (from the atlas + nb04):
- **Isolated false:** $1\times1$ block at $\lambda=s=4$ — sits in the P4 *gap*
  between 3.382 and 4.618. (Already handled by the cosine; trivially by $p$.)
- **Hubs** (77 % of FPs): star $K_{1,m}$ symmetric modes at $(\gamma+\delta)\pm\sqrt m$,
  $m\ge3$ → low modes $\{2.27, 2.0, 1.76, \dots\}$ (below the true band) and high
  modes $\{5.73, 6.0, 6.24,\dots\}$ (above it). The $m-1$ degenerate modes at
  $\lambda=s$ are already on the isolated notch.
- **Off-length bridges** (23 %): a false chain $P_m$ with $m\neq4$. The data's
  bridges are $P_3$: $\{2.586, 4.0, 5.414\}$ — which **interleave** the P4 lines with
  ~0.2 gaps and so are nullable.

**Irreducible:** a false chain of length **exactly 4** ($P_4$ bridge) — identical
spectrum *and* identical uniform-$b$ projection to a true track → $p(A)$ acts on it
identically → **no polynomial can remove it** (§4).

## 3. The hinge — spectral overlap (Step A is the decisive measurement)

A polynomial separates the false from the true **iff their eigenvalue supports do
not overlap**. The §0 probe used the *dominant* mode of each false segment; that is
the optimistic metric. Step A must do the honest version:

1. **Clean true census.** Isolate *pure* true clusters (all members true) and
   histogram their full eigenvalue spectra → the exact "keep" set (don't contaminate
   with mixed tangles, which polluted the quick probe's true set with spurious 4.0 / 6.0 lines).
2. **Full false support.** For each false segment, its spectral weight is spread over
   *all* its cluster modes ($x_i=\sum_k c_k u_k(i)$, $c_k=\beta_k/\lambda_k$). To null
   it, $p$ must be ~0 at **every** mode carrying its weight — not just the dominant
   one. Compute the weighted support and check each mode's gap to the true set. The
   tightest secondary modes (e.g. hub high mode $5.73$ vs true $5.618$, gap 0.11)
   set the real degree.
3. **Overlap map + minimal exploitable gap** → the per-failure-mode verdict
   (green/amber/red) and the polynomial degree.

## 4. The irreducible floor — the one thing QSVT cannot fix

A false bridge $P_4$ is graph-isomorphic to a true track $P_4$: same Laplacian-type
block, same eigenvalues, and with uniform $b=\delta\mathbf 1$ the isomorphism maps
the solution onto itself → **identical amplitudes**. Any $p(A)$ — QSVT, multi-bit
QPE, or a classical eigenfilter — is a function of the spectrum and therefore treats
them identically. This is the **topological degeneracy** (atlas F3) and is an
information-theoretic floor: it can only be broken with *geometry / track-level*
information (occupancy: one segment per hit), never with a spectral filter.
**Quantify this floor vs $T\in\{100,200,400\}$ and $\varepsilon$** — it is ~0 % here
but must grow with track density, and it is the true ceiling on what *any* filter
(QSVT included) can achieve.

## 5. Staged checks

**Step A — Spectral census** (`00_spectral_census.ipynb`, classical, numpy only).
Pure-true vs false full-spectrum histograms; overlap map; minimal exploitable gap;
irreducible same-length-bridge fraction vs $T$ and $\varepsilon$. *Output:* the
feasibility table + the "keep/null" eigenvalue lists for the design.

**Step B — Polynomial design** (`01_polynomial_design.ipynb`).
Construct $p(\lambda)$: target = band-limited inverse ($\approx 1/\lambda$ on the
keep-set, 0 on the null-set), bounded $|p|\le1$, definite parity, minimal degree.
Approaches: (i) product of notch factors × an inversion polynomial; (ii) Remez /
Chebyshev least-squares to the target response; (iii) convex design if `cvxpy` is
added. *Phase angles* for the actual circuit come from **`pyqsp`** (pip-installable;
not needed for Steps A/C). Verify the realised $p$ against the target.

**Step C — Classical efficacy test** (`02_classical_filter_test.ipynb`) — **the
decisive plot, needs only numpy.** Apply $x_p=p(A)\,\mathbf b$ per cluster via
eigendecomposition (block-diagonal, exact), renormalise to the true signal support,
threshold at $\tau$, and measure **efficiency vs false-rate** — the *same harness as
nb04's two-notch scan*. Compare: {exact classical inverse, 1-bit cosine, 2-notch,
QSVT degree $d$} and sweep $d$ → the **efficacy-vs-degree (cost) curve**. This
answers "can a polynomial null the failure modes?" with a number, before any circuit.
*Crucially, it also re-tests the renormalisation backfire that killed the 2-notch
scan* — a proper $1/\lambda$-on-true design should avoid it.

**Step D — Quantum cost & realizability** (`03_quantum_cost.ipynb`).
Degree $d$ → $d$ controlled-$A$ (block-encoding) calls + 1 QSVT ancilla + the sparse
block-encoding/LCU oracle for $A$. Circuit depth vs the 1BQF; success probability
$\propto\|p(A)\mathbf b\|^2$; conditioning. Small-$T$ qiskit demo (qiskit 2.4.1 is
installed) to confirm $p(A)$ on a toy cluster.

**Step E — Robustness.** Noise ($\sigma_{\rm scatt}$, hit inefficiency) broadens the
eigenvalue lines → shrinks the gap → raises the required degree. Quantify
degree-vs-noise; this sets whether the scheme survives realistic conditions.

## 6. Predicted outcome (hypothesis to falsify)

- **Hubs (77 %) and off-length bridges (most of 23 %): nullable** by a degree
  ~20–30 QSVT polynomial — a *strict* improvement over the 1-bit cosine, which
  cannot. The high-amplitude deep hubs are the *easiest* (largest gap).
- **Same-length ($P_4$) bridges: irreducible** — floor ~0 % at T=200, rising with
  density; the residue that genuinely needs track-level/occupancy logic.
- **Net:** QSVT plausibly moves the false-positive story from *"only the isolated
  bulk is removed"* to *"all-but-the-degenerate are removed,"* at a circuit-depth
  cost ≈ $d\times$ the 1BQF. Step C will turn "plausibly" into an efficiency number
  and an achievable false-rate floor.

## 7. Layout & dependencies

```
QSVT/
  PLAN.md                       <- this file
  qsvt_helpers.py               <- events/solve/cluster reuse (mirror Bifurification/bif.py)
  00_spectral_census.ipynb      <- Step A  (numpy)
  01_polynomial_design.ipynb    <- Step B  (numpy; pyqsp for angles)
  02_classical_filter_test.ipynb<- Step C  (numpy) — the decisive plot
  03_quantum_cost.ipynb         <- Step D  (qiskit)
```

Reuse `Toy_Characterisation/_shared/qtrk_pipeline` (events/$A$/truth) and the
cluster-eigendecomposition pattern from `Bifurification/bif.py`. **Q_env has**
numpy/scipy/qiskit 2.4.1; **add** `pyqsp` (phase angles, Step B/D) and optionally
`cvxpy` (Step B design) via pip when those steps start. Steps A and C — the
feasibility verdict — need nothing beyond the current env.

## 8. References
- Gilyén, Su, Low, Wiebe, *Quantum singular value transformation…*, STOC 2019
  (arXiv:1806.01838).
- Martyn, Rossi, Tan, Chuang, *Grand unification of quantum algorithms*,
  PRX Quantum 2021 (arXiv:2105.02859) — QSVT inversion / filtering tutorial.
- Lin, Tong, *Optimal polynomial based quantum eigenstate filtering* (arXiv:1910.14596)
  — band-pass eigenvalue filters, the design template for the null/keep response.
- `pyqsp` (Chuang group) — QSP phase-angle synthesis.
- Internal: `../Toy_Characterisation/Bifurification/04_failure_types_and_phase.ipynb`
  (the failure-mode spectra this plan builds on) and
  `../Toy_Characterisation/Segment_level_studies/07_segment_amplitude_atlas.ipynb`.
