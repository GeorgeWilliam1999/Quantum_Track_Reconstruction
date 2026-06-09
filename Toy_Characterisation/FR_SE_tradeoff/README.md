# FR_SE_tradeoff — segment-efficiency ↔ false-rate trade-off is driven by ε

Todo 1 of the Quantum LHCb Toy project. Contrasts two 400-track events solved with
**both** solvers (classical exact `A⁻¹b` and the 1BQF):

- **Event A — clean / isolated-only:** σ_scatt=0, σ_res=0, ε=1e-6. True segments are
  exactly collinear → couple into 4-chains; cross-track false segments stay **isolated**.
  Generated on demand (`qp.ensure_event` + `build_hamiltonian`); quantum = real per-block
  1BQF over the block-diagonal A (isolated → notch → 0).
- **Event B — noisy / fixed-ε scan:** σ_scatt=1e-4, σ_res=0, ε=2 mrad
  (`ev_591b8b9b4b66`, fixed-ε `eps_provenance='set'` solves). Scattering + the wide
  acceptance produce **coupled** false bridges/hubs.

**Result** (`clean_vs_noisy_400.ipynb`, `outputs/`):

| case | eff | far | false-positives | min(true)−max(false) margin |
|---|---|---|---|---|
| A classical | 100% | 0 | 0 | +0.114 |
| A quantum   | 75%  | 0 | 0 | +0.180 |
| B classical | 100% | 2.0% | 33 | +0.030 |
| B quantum   | 75%  | 1.7% | 21 | **−0.127** |

When the only false segments are isolated (A), both solvers have **far=0** and a clean
positive separation margin (the 1BQF erases the isolated false bulk at the notch). Noise
+ a wider ε create **coupled** false (1289 vs 0), the activation spectra grow a false tail
into the true band, far rises, and the quantum separation margin goes **negative** —
the ε-driven trade-off. ε must widen to keep scattered true segments coupled (efficiency),
but that same widening couples cross-track false segments (false rate).

Figures: `degree_structure.png`, `activation_spectra.png`, `summary.csv`.
