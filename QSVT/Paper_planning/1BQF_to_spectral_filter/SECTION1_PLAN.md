# Paper 1, section 1 — plan, figure designs and draft text (16 Sep 2026)

Goal for today: finish the introduction. Message: this is the problem; this is how the Hamiltonian
is built from a noisy event; this is what the one-bit filter does and why; this is what it leaves on
the table; this is the one change that fixes it (the QSVT).

Open decisions are marked **[Q1]** to **[Q7]** and listed at the end. Numbers marked `[measure]`
come from the new 15 µm events and must not be filled in by hand.

---

## Structure

Communications Physics puts figures in Results, not in the Introduction. So "section 1" becomes two
pieces:

1. **Introduction** (prose, about 700 words, no figures): the problem, the lineage, the gap, what
   this paper does.
2. **Results 1: "From hits to a filter"** (about 900 words, three figures): the toy and its noise,
   the ε window, the Hamiltonian, the one-bit filter and why it acts as it does, what it leaves,
   and the difference the QSVT makes.

Dropped from this section: the chain ladder (no added benefit; its one point, endpoint halving,
is a row of the new atlas), the HHL circuit (one sentence and one number instead), the
QSVT response functions and degree scans (section 3).

---

## Event configuration for every plot in this section

- Resolution σ_res = 15 µm (generator units: 0.015 mm).
- Scattering σ_scatt: **[Q1]**. The two readings of "comparable scale":

| reading | σ_scatt | kink width per projection σ_p | acceptance ε (κ = 3) |
|---|---|---|---|
| the stored value | 0.10 mrad | 1.12 mrad | 4.74 mrad |
| same displacement over one gap, 15 µm / 33 mm | 0.45 mrad | 1.20 mrad | 5.10 mrad |
| same kink width as resolution, √6 · 15 µm / 33 mm | 1.11 mrad | 1.57 mrad | 6.68 mrad |

  A rough Highland estimate for a track crossing about 1 % of a radiation length gives 0.56, 0.22
  and 0.11 mrad at 2, 5 and 10 GeV. The displacement-matched 0.45 mrad is physically realistic;
  the kink-matched 1.1 mrad corresponds to a soft track near 1 GeV. Recommendation: 0.45 mrad.
- Hit inefficiency: **[Q2]** (1 % shows broken tracks and fragments; 0 keeps them out).
- Acceptance: the formula ε of the table, step window (the comb is not defined for the erf window).
- γ = 3, δ = 1, s = 4. Multiplicity T = 200 for the illustration **[Q3]**; five repetitions for any
  quoted fraction.
- One-bit filter amplitudes: exact readout of the correctly wired circuit **[Q4]**.
- Every solver read at its own efficiency-first working point on the same event.

---

## Figure 1 — From hits to a Hamiltonian (new)

Message: *a track-finding event is a matrix, and the detector's noise decides which entries are on.*

- **(a) The toy event.** Five planes at z = 33 to 165 mm, a handful of tracks from the vertex
  drawn in grey, one highlighted. A zoom box on the highlighted track shows the three noise
  sources, exaggerated and labelled with their true sizes: a smeared hit (σ_res = 15 µm), a
  scattering kink (σ_scatt), a lost hit (hollow marker) if [Q2] is yes.
- **(b) The ε window.** Two segments sharing a hit and the kink angle θ between them. Beneath:
  the measured kink distribution on the event, true continuations (Rayleigh, width σ_p) against
  false continuations (flat at small angle), the step window f(θ) = 1 for θ < ε, with ε marked at
  3√2 σ_p. Annotation: true lost e⁻⁹ ≈ 10⁻⁴; false admitted ∝ ε².
- **(c) Graph and matrix.** A small cut-out of the event: one clean track (a chain), one false pair,
  one hub, one isolated false segment. Their compatibility graph, and the matrix A = sI − C
  drawn as blocks (reuse the graph_correspondence layout).
- **(d) The classical answer.** Histogram (log count) of the classical activations on the whole
  event, true and false stacked, with the efficiency-first threshold. Annotation of the three
  levels a reader should see: isolated false at 1/4, track ends at 4/11, track middles at 5/11,
  and the false mass above threshold.

Source: new script in this folder's future `scripts/` (event from the shared pipeline at the
configuration above); panel (c) adapted from the chapter's figure factory.

## Figure 2 — The one-bit filter, and the one change

Message: *the one-bit filter applies one cosine of the Hamiltonian; the QSVT applies any
polynomial by repeating the same kind of call.*

- **(a) The one-bit filter circuit**, correctly wired (true inverse uncompute), states |Ψ₁⟩ to
  |Ψ₄⟩ marked, from the chapter's `circuits.tex` panel (b). Call count written on it: 2.
- **(b) Its response** |cos(λt/2)|, t = π/s, on the eigenvalue axis, zero at λ = s. The bright
  lines of each error type (from Figure 3's rows) marked on the axis with the kept factor
  |cos(λt/2)| printed: isolated 0; false pair 0.38; false triple 0.53; true track 0.59 on its
  dominant line (2.382) and 0.24 on its weak one (4.618); hub K(1,3) 0.63; hub K(1,5) 0.77.
  The punch: every hub keeps more of itself than a true track does.
- **(c) The QSVT circuit** **[Q5]**, drawn to the same conventions, with the difference boxed: the
  single controlled evolution becomes d calls with ancilla rotations (or walk steps) between them.
  Call count written on it: d = 40.
- A one-line strip under (a) and (c): "one call pair, one zero" against "d calls, any bounded
  polynomial of degree d, up to d zeros".

The QSVT *response* is not drawn here; it arrives in section 3 on the spectrum. HHL appears only in
the text.

## Figure 3 — What the one-bit filter leaves on the table (the atlas, remade)

Message: *noise creates a small set of error types; the one zero removes exactly one of them and
lets the rest through, and it does so because of where each type's eigenvalues sit.*

One row per error type, ordered by its lowest bright eigenvalue so the cosine envelope of
Figure 2(b) can be read down the rows:

| row | error type | graph | bright lines (s = 4) |
|---|---|---|---|
| 1 | complete true track | chain of 4 | 2.382, 4.618 |
| 2 | broken track (true fragment) [Q2] | chain of 1 to 3 | 3, 5 (pair); 2.586, 5.414 (triple) |
| 3 | contaminated track (true chain with a false segment attached) | tree of 5 | [measure] |
| 4 | isolated false segment | single | 4 |
| 5 | false pair | chain of 2 | 3, 5 |
| 6 | false triple or bridge | chain of 3 | 2.586, 5.414 |
| 7 | hub (one hit, several continuations) | star | 4 ± √m |

Columns:

- **Left: the type**, a cartoon from the motif gallery and its count on the event
  (e.g. "false pairs: [measure] segments, [measure] % of false above the classical threshold").
- **Middle: classical activation**, a strip of every segment of that type (true green, false red),
  with the classical efficiency-first threshold.
- **Right: one-bit activation**, the same strip after the filter, with the one-bit
  efficiency-first threshold; red shading where false segments sit above it; the row's surviving
  count printed at the right edge.

What a reader should see without the caption: row 4 collapses to zero; rows 5 to 7 stay above the
threshold; row 1's ends drop to the level of row 5, which is why the threshold has to fall onto
the false pairs.

Source: the census classes of `section2_graphs/01_cluster_census.py` (isolated, path, star, tree,
cyclic) extended with the truth composition per component; exact classical and one-bit amplitudes
per component.

---

## Draft text

### Introduction

At the High-Luminosity LHC a single bunch crossing will leave thousands of charged-particle hits in
the LHCb vertex detector, and the first task of reconstruction is to decide which hits belong
together. The number of candidate connections grows as the square of the number of particles while
the number of real ones grows linearly, so the problem is a search for a thin signal in a
quadratically growing background. It is also one of the few reconstruction tasks that can be
written exactly as linear algebra: Denby and Peterson cast it as the minimisation of a quadratic
energy over candidate segments, and relaxing the segment variables to real numbers turns that
minimisation into a linear system.

That observation is the starting point of a short line of work. Nicotra et al. solved the system
with the Harrow–Hassidim–Lloyd algorithm and found tracking performance comparable to the
classical solver, but a circuit whose depth and readout made it impractical. Chiotopoulos et al.
then noticed that the task never needs the solution's precise values, only which segments are
switched on, and replaced the inversion by the one-bit quantum filter: a single-ancilla circuit that
applies one cosine of the Hamiltonian, with its zero placed on the eigenvalue shared by every
isolated false segment. The filter needs two calls to the time evolution where phase estimation
needs hundreds, and it was validated against the noise models of trapped-ion and superconducting
processors.

What the single zero removes, and what it leaves, has not been measured. On events with realistic
detector noise the false candidates are not only isolated segments: accidental pairs, bridges and
hubs appear, and each has its own eigenvalues. A cosine has one zero. This paper asks what happens
with more. The quantum singular value transformation applies any bounded polynomial of a
Hamiltonian at a cost linear in its degree, and the one-bit filter is its degree-one member. We
show that the true-track spectrum of the toy detector is discrete, that a polynomial placed on it
removes the error types the one-bit filter lets through, and what that costs in calls, gates and
qubits.

[One paragraph of results, written last, once sections 4 and 5 are settled.]

### Results 1 — From hits to a filter

**The toy detector and its noise.** We use the simplified vertex detector of refs [Nicotra, TrackHHL]:
five planes 33 mm apart, T tracks per event from a common vertex inside a cone of half-opening
0.2 rad, one hit per track per plane. Three detector effects are applied: Gaussian smearing of each
hit (σ_res = 15 µm), a multiple-scattering kink at each plane (σ_scatt = [Q1]), and [Q2]
(Figure 1a).

**The acceptance window.** Two segments are compatible when they share a hit and the kink between
them is small. Resolution and scattering give a true continuation a kink whose projections are
Gaussian of width σ_p² = σ_scatt² + 6 (σ_res/Δz)², so its magnitude is Rayleigh-distributed; we
accept kinks below ε = 3√2 σ_p ([measure] mrad), which loses a fraction e⁻⁹ of true continuations
whatever the noise (Figure 1b). The window is the physics of the problem: every false connection
that passes it becomes an entry of the matrix.

**The Hamiltonian.** With C the adjacency of compatible segment pairs, the energy
E(x) = ½ xᵀAx − δ1ᵀx with A = (γ+δ)I − C is minimised by Ax = δ1 (Figure 1c). The matrix breaks
into blocks, one per connected group of compatible segments. A segment with no compatible
neighbour, the fate of nearly every false candidate, solves to δ/(γ+δ) = 1/4; a complete five-hit
track solves to 4/11 at its ends and 5/11 in the middle. On a T = 200 event [measure] % of the
false segments are isolated, and the classical solver's threshold separates them cleanly; what
it lets through are the false segments that do have neighbours (Figure 1d).

**The one-bit filter.** The one-bit filter (Figure 2a) prepares the uniform state, applies one bit of
phase estimation to e^{−iAt}, and keeps the branch that returns the ancilla, so it multiplies each
eigencomponent of the input by cos(λt/2). With t = π/(γ+δ) the zero lands on λ = γ+δ, the
eigenvalue of every isolated segment (Figure 2b), and those segments are removed exactly.

**What it leaves.** Every other configuration keeps a nonzero factor fixed by where its eigenvalues
sit. A false pair has eigenvalues s ± 1 and keeps 38 % of its amplitude, a false triple 53 %, a three-pronged hub 63 %
and a five-pronged hub 77 %; a true track keeps 59 % on the line that carries almost all of its
weight, so every hub keeps more of itself than a real track does, and the track's end segments fall
to the level of the false pairs.
Read at the threshold that keeps 99 % of true segments, the filter's cut must therefore drop onto
the false pairs, and on the 15 µm event [measure] % of the selected segments are false against
[measure] % for the classical solver (Figure 3). The one zero is spent on the population that was
already easy.

**The one change.** Every error type in Figure 3 sits on known eigenvalues, so what is needed is not
a different primitive but a response with more than one zero. The quantum singular value
transformation provides one: d calls to the same kind of operator realise any bounded polynomial of
degree d (Figure 2c). Section 2 shows where the eigenvalues of each type lie, and section 3 which
polynomial to apply.

---

## Decisions for George

- **[Q1] Scattering.** Displacement-matched 0.45 mrad (recommended, realistic for a few-GeV track),
  or kink-matched 1.1 mrad, or keep 0.10 mrad?
- **[Q2] Hit inefficiency.** 1 % (shows broken tracks, brings in the fragment twins) or none?
- **[Q3] Multiplicity** for the illustration: T = 200 (recommended; every error type present,
  plots readable) or 400?
- **[Q4] One-bit amplitudes.** Exact readout of the correctly wired circuit (recommended; matches the
  circuit we draw) or the package circuit's stored values?
- **[Q5] Which QSVT circuit** in Figure 2(c): the trigonometric comb over e^{−iAt} (smallest visible
  difference from the one-bit filter, and what direct structural synthesis builds), or the walk over
  a block encoding (what the simulations run and what the hit-priced cost assumes)?
- **[Q6] Scope.** If section 1 shows 15 µm events, the reader will expect the headline in section 4
  at the same noise. Rerun the headline at this configuration, or keep section 4 on clean events
  and say so?
- **[Q7] Names** of the error types: complete track, broken track, contaminated track, isolated
  false segment, false pair, false triple or bridge, hub. Keep, rename, or merge?
