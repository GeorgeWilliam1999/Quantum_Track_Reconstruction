# OneBitHHL / OneBQF algorithm — full mathematical description

<!-- STATUS: final -->
<!-- SOURCES: LHCb_VeLo_Toy_Model/src/lhcb_velo_toy/solvers/quantum/OneBQF.py; OneBQF_parameter_audit.tex -->

## Inputs & padding
Given the SPD pair $(A, b)$ with $A \in \mathbb{R}^{N_s \times N_s}$ and constant diagonal $c \equiv A_{00} = \gamma + \delta$:

1. Pad to the next power of two $N = 2^n$, $n = \lceil \log_2 N_s\rceil$. The dense path
   stores zero off-diagonals on the padding rows; the *sparse* path used in the
   §14e workflow never materialises the padded matrix and instead just tracks
   $c$, $t$, the off-diagonal index list, and the padded $b$.
2. $b$ is padded with ones and $L_2$-normalised:
   $\;\;\ket{b} = b_\text{pad}/\|b_\text{pad}\|_2.$
3. The sparse path requires the diagonal to be strictly constant. This is satisfied
   for `SimpleHamiltonianFast` since all diagonal entries equal $\gamma+\delta$.

[source: `OneBQF.py::__init__`, lines 13–80]

## Hamiltonian simulation
The algorithm encodes $A$ as the Hamiltonian of a controlled-$U$ where $U = e^{-iAt}$ with
$$ t \;=\; \pi/c \;=\; \pi/(\gamma+\delta). $$
Because $A = c\,\mathbb{1} - B$ with $B$ an *off-diagonal* matrix whose only non-zero entries are
$B_{ij} = +1$ on edges $(i,j) \in E$, the evolution factorises as
$$ e^{-iAt} \;=\; e^{-ict}\, e^{+iBt}, $$
and the off-diagonal part is implemented exactly via two-level (Givens-style) decomposition:
for every interaction pair $(i,j) \in E$, the algorithm applies a multi-controlled RX($\theta$)
with $\theta = 2t$ that mixes basis states $\ket{i}$ and $\ket{j}$, and a global phase
$\exp(-i c t)$ on the control. The constant-diagonal assumption is what makes this
exact (no Trotter step) for the SPD `SimpleHamiltonianFast` matrix.

[source: `OneBQF.py::_apply_direct_controlled_u`, lines 96–145]

## Quantum phase estimation (one time qubit)
The standard QPE layout is used with $n_t = 1$ time qubit (default in `run_event.py`):
$$ \ket{0}_t \otimes \ket{b}_s \;\xrightarrow{H}\; \ket{+}_t\ket{b}_s \;\xrightarrow{\text{CU}}\; \frac{1}{\sqrt2}\big(\ket{0}_t U^{0} + \ket{1}_t U^{1}\big)\ket{b}_s \;\xrightarrow{\text{IQFT}}\; \ldots $$
Decomposed in the eigenbasis $A\ket{\lambda_k} = \lambda_k\ket{\lambda_k}$ with $\ket{b} = \sum_k \beta_k \ket{\lambda_k}$:
$$ \ket\Psi \;=\; \sum_k \beta_k\, \frac{1+e^{-i\lambda_k t}}{2}\ket{0}_t \ket{\lambda_k} \;+\; \beta_k\, \frac{1-e^{-i\lambda_k t}}{2}\ket{1}_t \ket{\lambda_k}. $$
With $\varphi_k \equiv \lambda_k t/(2\pi) = \lambda_k/(2c)$ this rewrites as
$$ \ket\Psi \;=\; \sum_k \beta_k \big[\,\cos(\pi\varphi_k)\,\ket{0}_t - i\sin(\pi\varphi_k)\,\ket{1}_t\,\big]\ket{\lambda_k}. $$

## Inversion + ancilla rotation
The standard HHL ancilla rotation $R_y(2\arcsin(C/\lambda))$ is replaced by an even simpler
"X·CNOT·X" gadget acting on the time qubit and ancilla [source: `OneBQF.py::build_circuit`, lines 159–170]:
$$ X_t \cdot \mathrm{CX}_{t\to a} \cdot X_t \;:\; \ket{0}_t\ket{0}_a \mapsto \ket{0}_t\ket{1}_a, \quad \ket{1}_t\ket{0}_a \mapsto \ket{1}_t\ket{0}_a. $$
That is: the ancilla is flipped iff the time qubit reads $\ket{0}$. Post-selecting on
ancilla $=1$ therefore keeps exactly the time-qubit-$\ket{0}$ amplitude. The remaining
QFT-uncompute restores the system register to:
$$ \ket{s}_\text{post-sel} \;\propto\; \sum_k \beta_k \cos(\pi \varphi_k)\,\ket{\lambda_k}. $$
This is **the** core difference from standard HHL: the eigenvalues are not inverted ($1/\lambda$);
they are *filtered* by a cosine envelope.

## Effective linear map (the kept-weight kernel)
The kept-weight per eigenmode is
$$ \boxed{\; w(\lambda_k) \;=\; \cos^2\!\Big(\tfrac{\pi\lambda_k}{2c}\Big) \quad\text{with}\quad c = \gamma+\delta\,.\;} $$
On the system register, the (unnormalised) output is therefore
$$ \tilde s \;=\; f(A)\,b, \qquad f(\lambda) = \cos\!\Big(\tfrac{\pi\lambda}{2c}\Big), $$
**not** $f(\lambda) = 1/\lambda$. This is a *spectral filter* rather than a matrix-inverse.
The total ancilla-$1$ probability is
$$ P(\mathrm{anc}=1) \;=\; \sum_k |\beta_k|^2 \cos^2(\pi\varphi_k). $$

Key points:
- Eigenvalues with $\lambda_k = c$ ($\varphi_k = 0.5$) are **completely wiped** — this is the
  *rejection notch*. The kernel of $B = c\,\mathbb{1} - A$ is mapped to $\ket{0}_t$ in QPE
  with probability one, and the ancilla never flips.
- Eigenvalues near $0$ or $2c$ are preserved with full weight.
- Because `SimpleHamiltonianFast` has spectrum bounded by Gershgorin on $[\delta + \gamma - r,
  \delta + \gamma + r]$ where $r$ is the maximum off-diagonal row-sum, the spectrum *straddles* $c$
  symmetrically for any value of $\gamma$. **The position of the spectrum relative to the
  notch is set entirely by $\gamma$.**

## Readout
[source: `OneBQF.py::get_solution` (sampling) and `get_solution_from_statevector` (E1)]

**E2 — sampling.** Shots are drawn; for each outcome with ancilla bit $=1$, the basis index
read from the system register is binned. The empirical probability $p_i$ is converted to
amplitude $s_i^Q = \sqrt{p_i}/\|\sqrt p\|_2$. This loses all sign information.

**E1 — statevector.** The full state is reshaped as `psi3 = (2, N_sys, N_time)` (Qiskit
little-endian: time slowest, system middle, ancilla fastest), the ancilla=1 branch is
selected, $|amp|^2$ is summed over time qubits, and the same $\sqrt{p}/\|\sqrt p\|_2$
extraction is applied. This is **exact** modulo floating-point.

## Norm scaling
The classical comparison vector $s_C$ has a calibrated $L_2$ norm. The raw quantum vector
$s_Q^{\text{raw}}$ has $\|s_Q^{\text{raw}}\|_2 = 1$ by construction. The pipeline rescales:
$$ s_Q^\text{scaled} \;=\; s_Q^\text{raw} \cdot \|s_C\|_2 / \|s_Q^\text{raw}\|_2, $$
and the activation rule $s_Q^\text{scaled} > \tau$ uses the same absolute $\tau$ as
classical [source: `run_event.py` lines 218–224].

## Why $\gamma=3$ is sub-optimal for OneBitHHL
[source: `OneBQF_parameter_audit.tex` §3, §4, §5]

At $\gamma=3, \delta=1$: $c = 4$, Gershgorin radius $r \le 4$ (each row has up to 4 unit
off-diagonals because each segment shares an endpoint hit with up to 4 others). Spectrum
$\lambda(A) \subset [c-r, c+r] = [0, 8]$; observed numerically:

| $\gamma$ | $c=A_{00}$ | $\lambda_\min$ | $\lambda_\max$ | $\varphi_\min$ | $\varphi_\max$ | "at notch" |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.38 | 3.62 | 0.095 | 0.905 | 8/16 (kernel of $B$) |
| **3** | **4** | **2.38** | **5.62** | **0.298** | **0.703** | **8/16** |
| 5 | 6 | 4.38 | 7.62 | 0.365 | 0.635 | 8/16 |
[source: `outputs/quantum_segment_analysis/gamma_sweep_ntrk2.csv` at $T=2$]

At $\gamma=3$ the $\lambda_\max/\lambda_\min$ envelope straddles $\varphi=0.5$; the
$\cos^2$ kernel falls to $\cos^2(0.703\pi) \approx 0.295$ on the upper end and
$\cos^2(0.298\pi) \approx 0.295$ on the lower end. The 8 eigenvectors that *do not*
sit at the notch are still suppressed in amplitude by up to a factor $\sqrt{0.295} \approx 0.54$.
For $T=2$ this gives the well-known $|s_Q|^2 \approx 0.75$ "75% floor" reported in the audit.

At $\gamma=1$ the spectrum sits inside $[0.38, 3.62]$ → $\varphi \in [0.095, 0.905]$;
the $\cos^2$ kernel is $\ge \cos^2(0.405\pi) \approx 0.094$ but never zero away from the
notch eigenvectors, and the algorithm reproduces the classical answer with
$\cos(s_Q, s_C) > 0.84$ for $T \in [2,10]$ [source: `sweep_statevector_gamma1.csv`].
