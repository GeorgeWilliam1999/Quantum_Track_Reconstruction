#!/usr/bin/env python3
"""γ-validation: does γ = γ* actually restore the 1BQF under the DP terms?

The γ map (dp_matrix_spectrum.csv) predicts, at heavy T=200: fork β=1 needs
γ ≥ 4.31, β=2 needs γ ≥ 6.29; occupancy α=0.3 needs γ ≈ 235 (window) at the
price of collapsed filter contrast. Beyond the window there is a second,
γ-suppressible mechanism: a coupling of strength w shifts eigenvalues off the
notch by ~±w, so the revived amplitude scales like sin(πw/2s') — raising γ
grows s' and squeezes revived pairs back into the notch. This script measures
both effects directly: rebuild A0 at each γ (γ enters the diagonal only),
re-solve + re-emulate, and report wp99/eff-1 separation per class.

Output: results/dp_gamma_validation.csv (+ printed summary).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.linalg import minres

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dp_matrix_characterisation as dmc  # noqa: E402
import dp_terms  # noqa: E402
import qtrk_pipeline as qp  # noqa: E402

HERE = Path(__file__).resolve().parent
T, NOISE = 200, "heavy"
CASES = [("base", 0.0, 0.0, [3.0, 5.0]),
         ("fork b=1", 1.0, 0.0, [3.0, 4.31, 5.0, 6.0, 8.0]),
         ("fork b=2", 2.0, 0.0, [3.0, 6.29, 8.0]),
         ("occ a=0.3", 0.0, 0.3, [3.0, 236.0])]


def main():
    ham3, truth, eps = dmc.get_ham(T, NOISE)
    cls, _ = dmc.classify(ham3, truth, eps)
    ev, _ = qp.ensure_event(n_trk=T, rep=0, **dmc.NOISES[NOISE])
    rows = []
    for name, beta, alpha, gammas in CASES:
        for g in gammas:
            ham = qp.build_hamiltonian(ev, epsilon=eps, kernel="step",
                                       gamma=g, delta=1.0)
            A, b, _, info = dp_terms.dp_system(ham, beta=beta, alpha=alpha,
                                               gamma=g, delta=1.0)
            x, _ = minres(A, b, rtol=1e-8, maxiter=8000)
            x = np.asarray(x).ravel()
            xq, p_anc, s_p, t_emu = dmc.emulate_1bqf(A, ham.n_segments)
            new = []
            for solver, v in [("classical", x), ("1bqf_emu", xq)]:
                new += dmc.class_rows(v, truth, cls,
                                      dict(config=name, beta=beta, alpha=alpha,
                                           gamma=g, s_prime=s_p, solver=solver,
                                           p_anc=p_anc))
            rows += new
            aq = next(r for r in new if r["solver"] == "1bqf_emu" and r["cls"] == "ALL")
            ac = next(r for r in new if r["solver"] == "classical" and r["cls"] == "ALL")
            coh = [r for r in new if r["solver"] == "1bqf_emu" and r["cls"] == "cohit-true"]
            ct = [r for r in new if r["solver"] == "1bqf_emu" and r["cls"] == "C-true"]
            print(f"{name} γ={g:g}: CLS far_wp={ac['far_wp']:.3f} | "
                  f"1BQF eff_wp={aq['eff_wp']:.3f} far_wp={aq['far_wp']:.3f} "
                  f"gap={aq['gap']:+.2e} P_anc={p_anc:.3e} "
                  f"cohit-adm={coh[0]['frac_wp'] if coh else float('nan'):.4f} "
                  f"C-true-adm={ct[0]['frac_wp'] if ct else float('nan'):.3f} "
                  f"[emu {t_emu:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "dp_gamma_validation.csv",
                              index=False)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
