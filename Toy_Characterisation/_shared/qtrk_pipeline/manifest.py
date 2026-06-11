"""
Study specs -> a deduplicated manifest of events and planned solves.

Each study is declared as a :class:`StudySpec` over a few axes.  ``build_manifest``
expands the specs, takes the UNION of unique events and unique solves (so shared
baseline points are generated/solved once), and writes two CSVs:

    manifest/events.csv      one row per unique event   (Layer 0)
    manifest/solutions.csv   one row per planned solve   (Layer 3 plan)

``standard_specs()`` encodes the five segment-level studies, so the manifest is
also the executable proof that the schema covers them all.
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_SHARED = Path(__file__).resolve().parent.parent
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
sys.path.insert(0, "/data/bfys/gscriven/LHCb_VeLo_Toy_Model/src")

from .keys import event_key, ham_key, sol_key  # noqa: E402
from .store import manifest_dir  # noqa: E402

__all__ = ["StudySpec", "expand_spec", "build_manifest", "standard_specs",
           "DEFAULT_TRACK_GRID"]

DEFAULT_TRACK_GRID = [10, 20, 50, 100, 200, 400, 700, 1000]


def _const(v):
    return lambda T: v


@dataclass
class StudySpec:
    """One study = a small Cartesian sweep over event + ham + solver axes."""
    name: str
    track_grid: list = field(default_factory=lambda: list(DEFAULT_TRACK_GRID))
    reps: int = 20                       # classical reps per (cell, T)
    # event axes. (sigma_scatt, sigma_res) are taken as a full cross-product
    # UNLESS noise_pairs is given, in which case those explicit (ss, sr) pairs
    # are used (ERF uses paired noise, not a product).
    sigma_scatt: list = field(default_factory=lambda: [1e-4])
    sigma_res: list = field(default_factory=lambda: [0.0])
    noise_pairs: Optional[list] = None   # e.g. [(1e-4, 0.0), (3e-4, 0.01)]
    phi_max: list = field(default_factory=lambda: [0.2])
    hit_ineff: list = field(default_factory=lambda: [0.0])
    ghost_rate: list = field(default_factory=lambda: [0.0])
    # ham axes
    kernel: str = "step"
    erf_sigma: list = field(default_factory=lambda: [1e-4])
    gamma: list = field(default_factory=lambda: [3.0])
    delta: float = 1.0
    eps_provenance: str = "formula"      # "formula" -> compute_epsilon(sr,ss); "set"
    eps_value: Optional[float] = None    # single fixed eps (eps_provenance == "set")
    eps_values: Optional[list] = None    # OR a list of fixed eps to sweep ("set")
    # solver axes
    run_classical: bool = True
    quantum_Tmax: int = 1000             # solve quantum for T <= this
    # Multi-threaded AerSimulator CPU is ~3-7x slower than GPU at T<=200 but runs
    # on abundant CPU slots (vs scarce GPUs).  T=1000 on CPU is ~18h/solve, so it
    # stays on GPU (88 min); everything below goes to CPU.
    quantum_device: Callable[[int], str] = field(default=lambda T: "GPU" if T >= 1000 else "CPU")
    # DEFAULT = statevector only (exact, no shot noise -> 3 reps).  Sampling is
    # opt-in for explicit statevector-vs-sampling comparison (20 reps, shots~1e5);
    # both share the SAME sparse A and O(A_nnz * 2^n_sys) circuit-sim cost, so
    # sampling is NOT cheaper per solve -> keep it to low-moderate T.
    quantum_readouts: list = field(default_factory=lambda: ["statevector"])
    quantum_reps_by_readout: dict = field(
        default_factory=lambda: {"statevector": 3, "sampling": 20})
    quantum_shots: int = 100_000         # shots for the sampling readout
    # Statevector is EXACT (no shot noise), so high-T reps only add event-to-event
    # variance already sampled at lower T. Cut to 1 rep at T>=700 to tame the GPU
    # tail (each T=1000 statevector solve is ~88 min). Sampling unaffected.
    quantum_sv_hiT: int = 700            # T threshold for the high-T rep cut
    quantum_sv_hiT_reps: int = 1
    # Large-noise (large-ε) cells have a HUGE 1BQF circuit (gates ∝ ε²): e.g.
    # σ_res=0.05 (ε≈0.016) at T=400 needs ~96 GB, and worse at higher T — heavy AND
    # marginal (classical recon already collapses at that noise). Cap quantum at
    # T<=400 for ε >= quantum_hieps. (user, 2026-06-09)
    quantum_hieps: float = 0.012
    quantum_hieps_Tmax: int = 400

    def quantum_for(self, eps: float, T: int) -> bool:
        """Whether to run quantum for this (epsilon, T)."""
        if T > self.quantum_Tmax:
            return False
        if eps >= self.quantum_hieps and T > self.quantum_hieps_Tmax:
            return False
        return True

    def quantum_reps(self, readout: str, T: int) -> int:
        if readout == "statevector" and int(T) >= self.quantum_sv_hiT:
            return int(self.quantum_sv_hiT_reps)
        return int(self.quantum_reps_by_readout.get(readout, self.reps))


def _epsilons(spec: StudySpec, sr: float, ss: float) -> list:
    """Epsilon value(s) for a (sigma_res, sigma_scatt) cell.

    Formula provenance -> the single noise-matched epsilon.  Set provenance ->
    one or more hand-fixed epsilons (eps_values, or eps_value) swept as separate
    Hamiltonians (different ham_key each).
    """
    if spec.eps_provenance == "set":
        vals = spec.eps_values
        if vals is None:
            vals = [spec.eps_value] if spec.eps_value is not None else None
        if not vals:
            raise ValueError(f"{spec.name}: eps_provenance='set' needs eps_value(s)")
        return [float(v) for v in vals]
    from lhcb_velo_toy.analysis import compute_epsilon
    return [float(compute_epsilon(sr, ss))]


def expand_spec(spec: StudySpec) -> tuple[list[dict], list[dict]]:
    """Return (event_rows, solve_rows) for one study (not yet deduplicated)."""
    event_rows: list[dict] = []
    solve_rows: list[dict] = []

    erf_list = spec.erf_sigma if spec.kernel == "erf" else [spec.erf_sigma[0]]
    noise = (list(spec.noise_pairs) if spec.noise_pairs is not None
             else list(itertools.product(spec.sigma_scatt, spec.sigma_res)))
    cells = itertools.product(noise, spec.phi_max, spec.hit_ineff, spec.ghost_rate)
    for (ss, sr), phi, drop, ghost in cells:
      for eps in _epsilons(spec, sr, ss):
        for g in spec.gamma:
            hkey = {erf: ham_key(eps, spec.kernel, erf, g, spec.delta,
                                 spec.eps_provenance) for erf in erf_list}
            for T in spec.track_grid:
                # solver list for this T: (solver, device, readout, shots, nrep)
                solvers = []
                if spec.run_classical:
                    solvers.append(("classical", "CPU", "na", 0, int(spec.reps)))
                if spec.quantum_for(eps, T):
                    dev = spec.quantum_device(T)
                    for rdt in spec.quantum_readouts:
                        shots = spec.quantum_shots if rdt == "sampling" else 0
                        solvers.append(("quantum", dev, rdt, shots,
                                        spec.quantum_reps(rdt, T)))
                for solver, device, readout, shots, nrep in solvers:
                    for rep in range(nrep):
                        ekey = event_key(T, rep, ss, sr, phi, drop, ghost)
                        event_rows.append(dict(
                            event_key=ekey, n_trk=T, rep=rep,
                            sigma_scatt=ss, sigma_res=sr, phi_max=phi,
                            hit_ineff=drop, ghost_rate=ghost,
                        ))
                        for erf in erf_list:
                            hk = hkey[erf]
                            skey = sol_key(ekey, hk, solver, device, readout)
                            solve_rows.append(dict(
                                sol_key=skey, event_key=ekey, ham_key=hk,
                                study=spec.name, n_trk=T, rep=rep,
                                sigma_scatt=ss, sigma_res=sr, phi_max=phi,
                                hit_ineff=drop, ghost_rate=ghost,
                                epsilon=eps, eps_provenance=spec.eps_provenance,
                                kernel=spec.kernel,
                                erf_sigma=(erf if spec.kernel == "erf" else 0.0),
                                gamma=g, delta=spec.delta,
                                solver=solver, device=device,
                                readout=readout, shots=shots,
                            ))
    return event_rows, solve_rows


def build_manifest(specs: list[StudySpec], write: bool = True):
    """Union + dedup all specs; optionally write events.csv / solutions.csv."""
    import pandas as pd

    all_events, all_solves = [], []
    for spec in specs:
        ev, so = expand_spec(spec)
        all_events.extend(ev)
        all_solves.extend(so)

    events = pd.DataFrame(all_events).drop_duplicates("event_key").reset_index(drop=True)
    # A solve is unique by sol_key; keep the first study that requested it but
    # record all requesters for traceability.
    solves = pd.DataFrame(all_solves)
    req = (solves.groupby("sol_key")["study"]
           .apply(lambda s: ",".join(sorted(set(s)))).rename("studies"))
    solves = solves.drop_duplicates("sol_key").merge(req, on="sol_key").reset_index(drop=True)

    if write:
        md = manifest_dir()
        events.to_csv(md / "events.csv", index=False)
        solves.to_csv(md / "solutions.csv", index=False)
        print(f"[manifest] {len(events)} unique events -> {md/'events.csv'}")
        print(f"[manifest] {len(solves)} unique solves -> {md/'solutions.csv'}")
        _print_breakdown(solves)
    return events, solves


def _print_breakdown(solves) -> None:
    q = solves[solves.solver == "quantum"]
    print(f"[manifest]   classical solves:      {(solves.solver=='classical').sum()}")
    print(f"[manifest]   quantum statevector:   {((solves.solver=='quantum') & (solves.readout=='statevector')).sum()}")
    print(f"[manifest]   quantum sampling:      {((solves.solver=='quantum') & (solves.readout=='sampling')).sum()}")
    if len(q):
        for lane, sub in (("CPU quantum (T<=200)", q[q.device == "CPU"]),
                          ("GPU quantum (T>200) ", q[q.device == "GPU"])):
            print(f"[manifest]   {lane}: {len(sub)}")
        hi = q[(q.device == "GPU") & (q.n_trk >= 700)]
        print(f"[manifest]   GPU quantum T>=700:    {len(hi)}  (heaviest)")


# ---------------------------------------------------------------------------
# The five segment-level studies, encoded as specs (coverage proof).
# ---------------------------------------------------------------------------

def standard_specs() -> list[StudySpec]:
    # Quantum policy (user, 2026-06-08): both readouts, statevector=3 reps
    # (exact, no shot noise), sampling=20 reps (characterise shot distribution).
    # Classical = 20 reps everywhere.  Defaults on StudySpec already encode this.
    eps2 = StudySpec(
        name="Epsilon_study_2",
        sigma_scatt=[1e-4, 3e-4, 5e-4], sigma_res=[0.0, 0.01, 0.02, 0.05],
        kernel="step",
    )
    larger_scatter = StudySpec(
        name="Larger_Scatter",
        sigma_scatt=[1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
        hit_ineff=[0.0, 0.01, 0.02, 0.05, 0.10],
        kernel="step",
    )
    lsd = StudySpec(
        name="Larger_Scatter_Density",
        phi_max=[0.2, 0.1, 0.05, 0.02, 0.01], sigma_scatt=[1e-4, 3e-4],
        kernel="step",
    )
    erf = StudySpec(
        name="ERF",
        noise_pairs=[(1e-4, 0.0), (3e-4, 0.01), (5e-4, 0.02)],   # paired, not product
        kernel="erf", erf_sigma=[1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    )
    # Verify: baseline (formula) + a gamma sweep, drop in {0, 0.01}.
    verify = StudySpec(
        name="Verify_new_results",
        sigma_scatt=[1e-4], hit_ineff=[0.0, 0.01],
        gamma=[1.0, 2.0, 3.0], kernel="step",
    )
    # Verify FIXED epsilon = 0.002 (2 mrad): the actual Hamiltonian acceptance
    # used by segment_level_analysis.ipynb (FIXED_EPSILON) and
    # Quantum_segment_level_analysis.ipynb (EPS_LOC). Distinct ham_key from the
    # formula runs; reuses the same events. (NB: the [0.34..0.906] "eps" sweep in
    # the quantum notebook is a SOLUTION-VECTOR THRESHOLD sweep, done in
    # analysis from stored vectors — not a Hamiltonian epsilon, no jobs needed.)
    verify_fixed = StudySpec(
        name="Verify_new_results",
        sigma_scatt=[1e-4], hit_ineff=[0.0, 0.01],
        gamma=[1.0, 2.0, 3.0], kernel="step",
        eps_provenance="set", eps_values=[0.002],
    )
    return [eps2, larger_scatter, lsd, erf, verify, verify_fixed]
