# Circuit diagrams for the direct-vertexing write-up (Notion section 7)

C1 to C6 draw each operator on its own; C7 to C9 draw the three full computations.

| File | Content |
|---|---|
| C1 | State preparation: `H^{⊗n}` (1BQF), `PREP_b` (QSVT), `PREP_L` and the sign diagonal |
| C2 | Segment controlled evolution `c-G` as a Givens product, one pair block, and the inverse |
| C3 | Hit-register controlled evolution `c-G_r` with `r` product steps, one pair block |
| C4 | One-bit blocks: `QPE_1`, flag, uncompute, readout |
| C5 | QSVT blocks: dilation block encoding, walk, SELECT, LCU sandwich |
| C6 | Phase sequence: projector-controlled phase, one sequence, real part and parity |
| C7 | Full computation 1: segment register, 1BQF |
| C8 | Full computation 2: hit register per vertex (proposal; not in the package) |
| C9 | Full computation 3: segment register, QSVT (LCU build and phase-sequence variant) |

Build: `./build.sh C1 C2 ...` (pdflatex, then pdftoppm). Styles come from the paper's `figures/circuit_styles.tex`.
Circuits follow `LHCb_VeLo_Toy_Model` on branch centralise-2.1 (`one_bqf.py`, `QSVT.py`, `qsp_phases.py`).
