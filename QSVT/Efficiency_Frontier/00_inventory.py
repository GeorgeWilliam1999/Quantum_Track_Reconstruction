#!/usr/bin/env python3
"""Stage 0 — inventory + baseline frontier (no compute).

Harvests every existing (eff, far, degree, set-out) cell relevant to the
Efficiency_Frontier question from:

  S1  qtrk_store metrics.csv        — all qsvt rows (+ classical/1BQF partners in
                                      the same studies): Verify_new_results (QSVT II
                                      campaign, production d=40 comb), QSVT_robustness,
                                      QSVT_codesign (domain-scaled d), plus the ERF /
                                      Larger_Scatter / Noisy_Density qsvt rows.
  S2  Segment_level_studies/outputs/depth_and_qubits/degree_sweep.csv
                                      — the direct degree axis (clean, wp99).
  S3  Codesign/outputs/fork_noisy/fit_comb.csv
                                      — heavy T=200 fitted response + gate columns
                                      (far at matched eff 0.95–0.99).
  S4  Codesign/outputs/fork_noisy/occupancy_fitted_check.csv
                                      — occupancy alpha x fitted degree (heavy T=200).
  S5  Block_Encoding/outputs/03_normalized_walk_metrics.csv
                                      — normalized +-1/2 comb, step+erf, clean.
  S6  Toy_Characterisation/Bifurification/results/dp_postselect_uniqueness.csv
                                      — composed (uniqueness-gate) points.

Outputs:
  outputs/00_inventory.csv     — unified long-format inventory (one row per
                                 harvested operating point / frontier cell)
  outputs/00_gap_list.csv      — the frozen config matrix: every PLAN §3 cell
                                 with covered/partial/missing status; Stages 1-3
                                 compute ONLY rows with status != covered.
  stdout                       — coverage pivots + the gap summary.

No solves, no store writes. PLAN.md §4 Stage 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # Quantum_Track_Reconstruction
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

STORE_METRICS = Path("/data/bfys/gscriven/qtrk_store/manifest/metrics.csv")
DEGREE_SWEEP = ROOT / "QSVT/Segment_level_studies/outputs/depth_and_qubits/degree_sweep.csv"
FIT_COMB = ROOT / "QSVT/Codesign/outputs/fork_noisy/fit_comb.csv"
OCC_FITTED = ROOT / "QSVT/Codesign/outputs/fork_noisy/occupancy_fitted_check.csv"
BE03 = ROOT / "Block_Encoding/outputs/03_normalized_walk_metrics.csv"
POSTSEL = ROOT / "Toy_Characterisation/Bifurification/results/dp_postselect_uniqueness.csv"

ROWS: list[dict] = []


def add(source, provenance, *, regime, T, rep, response_family, degree,
        metric, eff, far, solver="qsvt", kernel="step", gamma=3.0,
        eps_setout="formula", fork_beta=0.0, occ_alpha=0.0,
        normalization="raw", composed="none", tau=np.nan, target_eff=np.nan,
        notes=""):
    ROWS.append(dict(
        source=source, provenance=provenance, regime=regime, T=T, rep=rep,
        solver=solver, response_family=response_family, degree=degree,
        kernel=kernel, gamma=gamma, eps_setout=eps_setout,
        fork_beta=fork_beta, occ_alpha=occ_alpha, normalization=normalization,
        composed=composed, metric=metric, target_eff=target_eff, tau=tau,
        eff=eff, far=far, notes=notes))


# ── regime classification ────────────────────────────────────────────────────
# House triples per Bifurification/dp_matrix_characterisation.NOISES (the DP/
# XII/postselect operating points): sigma_scatt=1e-4 everywhere;
# clean res=0 drop=0 · moderate res=0.01 drop=0.01 · heavy res=0.02 drop=0.01.
# NB PLAN.md §3C says moderate drop=0.5% — the study code uses 1%; the code wins
# (all existing moderate cells were solved at 1%). Flagged in the worklog.
def classify_regime(scatt, res, drop):
    scatt = float(scatt or 0); res = float(res or 0); drop = float(drop or 0)
    base_scatt = scatt <= 1.5e-4          # 0 or the 1e-4 house baseline
    if base_scatt and abs(res - 0.02) < 1e-6 and abs(drop - 0.01) < 1e-6:
        return "heavy"
    if base_scatt and abs(res - 0.01) < 1e-6 and abs(drop - 0.01) < 1e-6:
        return "moderate"
    if base_scatt and res == 0.0 and drop == 0.0:
        return "clean"
    if base_scatt and res == 0.0 and abs(drop - 0.01) < 1e-6:
        return "clean+drop1%"
    return f"other(sc={scatt:g},res={res:g},drop={drop:g})"


# ── S1: store metrics ────────────────────────────────────────────────────────
def harvest_store():
    m = pd.read_csv(STORE_METRICS, low_memory=False)
    m = m[m["solver"].isin(["qsvt", "quantum", "classical"])].copy()
    keep_studies = {"Verify_new_results", "QSVT_robustness", "QSVT_codesign",
                    "QSVT_fork_noisy", "ERF", "Larger_Scatter", "Noisy_Density"}
    m = m[m["study"].isin(keep_studies)]
    # classical/1BQF rows only matter as baselines where a qsvt row shares the study
    qsvt_studies = set(m.loc[m["solver"] == "qsvt", "study"].unique())
    m = m[m["study"].isin(qsvt_studies)]

    fam = {"qsvt": "comb_prod", "quantum": "1bqf_cos", "classical": "classical_invA"}
    deg = {"qsvt": 40.0, "quantum": 1.0, "classical": np.nan}

    for r in m.itertuples(index=False):
        regime = classify_regime(r.sigma_scatt, r.sigma_res, r.hit_ineff)
        degree = deg[r.solver]
        note = ""
        if r.solver == "qsvt":
            if r.study == "QSVT_codesign":
                degree, note = np.nan, "domain-scaled d=40-127 (codesign campaign)"
            elif r.study == "QSVT_fork_noisy":
                degree, note = np.nan, "see 03_fork_quantum_noisy.py"
            else:
                note = "campaign default d=40 line-comb"
        common = dict(
            regime=regime, T=int(r.n_trk), rep=int(r.rep),
            solver=r.solver, response_family=fam[r.solver], degree=degree,
            kernel=str(r.kernel), gamma=float(r.gamma),
            fork_beta=float(r.fork_beta) if pd.notna(r.fork_beta) else 0.0,
            notes=note)
        add("store", f"metrics.csv:{r.study}:{r.sol_key}",
            metric="fixed_tau", tau=float(r.threshold),
            eff=float(r.segment_efficiency), far=float(r.segment_false_rate),
            **common)
        if pd.notna(r.segment_efficiency_wp):
            add("store", f"metrics.csv:{r.study}:{r.sol_key}",
                metric="wp", target_eff=float(r.target_eff),
                tau=float(r.tau_wp), eff=float(r.segment_efficiency_wp),
                far=float(r.segment_false_rate_wp), **common)
    return len(m)


# ── S2: degree sweep (clean, production comb family) ─────────────────────────
def harvest_degree_sweep():
    d = pd.read_csv(DEGREE_SWEEP)
    for r in d.itertuples(index=False):
        base = dict(regime="clean", T=int(r.T), rep=int(r.rep),
                    response_family="comb_prod", degree=float(r.degree),
                    notes=f"P_succ={r.P:.3g}")
        add("degree_sweep", "depth_and_qubits/degree_sweep.csv",
            metric="wp", target_eff=0.99, eff=0.99, far=float(r.far99), **base)
        add("degree_sweep", "depth_and_qubits/degree_sweep.csv",
            metric="fixed_tau", tau=0.35, eff=float(r.eff_fixed),
            far=float(r.far_fixed), **base)
    return len(d)


# ── S3: fitted response + gates, heavy T=200 ─────────────────────────────────
FIT_FAMS = {
    "cls": ("classical_invA", np.nan, "classical", "none"),
    "1b": ("1bqf_cos", 1.0, "quantum", "none"),
    "comb": ("comb_prod", 40.0, "qsvt", "none"),
    "opt": ("fitted_ridge", np.nan, "qsvt", "none"),
    "gen": ("fitted_ridge_xval", np.nan, "qsvt", "none"),
    "cheb40": ("fitted_cheb", 40.0, "qsvt", "none"),
    "cheb80": ("fitted_cheb", 80.0, "qsvt", "none"),
    "cheb160": ("fitted_cheb", 160.0, "qsvt", "none"),
    "cheb240": ("fitted_cheb", 240.0, "qsvt", "none"),
    "chebd40": ("fitted_cheb", 40.0, "qsvt", "none"),
    "chebd80": ("fitted_cheb", 80.0, "qsvt", "none"),
    "chebd160": ("fitted_cheb", 160.0, "qsvt", "none"),
    "chebd240": ("fitted_cheb", 240.0, "qsvt", "none"),
    "optgate": ("fitted_ridge", np.nan, "qsvt", "uniqueness-greedy"),
    "clsgate": ("classical_invA", np.nan, "classical", "uniqueness-greedy"),
}
BETA = {"beta0": 0.0, "b0.5-gstar": 0.5, "b1-gstar": 1.0}


def harvest_fit_comb():
    d = pd.read_csv(FIT_COMB)
    n = 0
    for r in d.itertuples(index=False):
        rd = r._asdict()
        for col, v in rd.items():
            if not col.startswith("far_") or pd.isna(v):
                continue
            stem, etag = col[4:].rsplit("_e", 1)
            if stem not in FIT_FAMS:
                continue
            fam, degree, solver, gate = FIT_FAMS[stem]
            eff = int(etag) / 1000.0
            add("fit_comb", "Codesign/outputs/fork_noisy/fit_comb.csv",
                regime="heavy", T=200, rep=int(rd["rep"]), solver=solver,
                response_family=fam, degree=degree, gamma=float(rd["gamma"]),
                fork_beta=BETA[rd["config"]], composed=gate,
                metric="far_at_eff", target_eff=eff, eff=eff, far=float(v),
                notes=f"floor_frac={rd['floor_frac']:.4f}")
            n += 1
    return n


# ── S4: occupancy alpha x fitted degree, heavy T=200 ─────────────────────────
def harvest_occ():
    d = pd.read_csv(OCC_FITTED)
    n = 0
    alpha = {"base": 0.0, "occ_a0.05": 0.05, "occ_a0.3": 0.3}
    for r in d.itertuples(index=False):
        rd = r._asdict()
        for col, v in rd.items():
            if not col.startswith("far_") or pd.isna(v):
                continue
            stem, etag = col[4:].rsplit("_e", 1)
            eff = int(etag) / 1000.0
            if stem == "cls":
                fam, degree, solver = "classical_invA", np.nan, "classical"
            elif stem.startswith("fit"):
                fam, degree, solver = "fitted_ridge", float(stem[3:]), "qsvt"
            else:
                continue
            add("occ_fitted", "Codesign/outputs/fork_noisy/occupancy_fitted_check.csv",
                regime="heavy", T=200, rep=int(rd["rep"]), solver=solver,
                response_family=fam, degree=degree,
                occ_alpha=alpha[rd["system"]],
                metric="far_at_eff", target_eff=eff, eff=eff, far=float(v),
                notes=f"span={rd['span']:.2f}")
            n += 1
    return n


# ── S5: normalized +-1/2 comb (Block_Encoding 03), clean ─────────────────────
def harvest_be03():
    d = pd.read_csv(BE03)
    n = 0
    for r in d.itertuples(index=False):
        s = str(r.solver)
        if s == "classical":
            fam, degree, norm, solver = "classical_invA", np.nan, "raw", "classical"
        elif s.startswith("szegedy_comb_d"):
            fam, degree, norm, solver = "normalized_half_comb", float(s.rsplit("d", 1)[1]), "normalized", "qsvt"
        elif s.startswith("qsvt_comb_d"):
            fam, degree, norm, solver = "comb_prod", float(s.rsplit("d", 1)[1]), "raw", "qsvt"
        else:
            continue
        base = dict(regime="clean", T=int(r.T), rep=0, solver=solver,
                    response_family=fam, degree=degree, kernel=str(r.kernel),
                    normalization=norm, notes=f"p_succ={r.p_succ:.3g}")
        add("be03", "Block_Encoding/outputs/03_normalized_walk_metrics.csv",
            metric="fixed_tau", tau=0.35, eff=float(r.eff), far=float(r.far), **base)
        add("be03", "Block_Encoding/outputs/03_normalized_walk_metrics.csv",
            metric="wp", target_eff=0.99, tau=float(r.tau_wp),
            eff=float(r.eff_wp), far=float(r.far_wp), **base)
        n += 1
    return n


# ── S6: postselect composed points ───────────────────────────────────────────
def harvest_postselect():
    d = pd.read_csv(POSTSEL)
    fam = {"classical": ("classical_invA", np.nan),
           "1bqf_emu": ("1bqf_cos", 1.0)}
    n = 0
    for r in d.itertuples(index=False):
        if r.solver not in fam:
            continue
        f, degree = fam[r.solver]
        sel = str(r.selector)
        composed = sel if ("composed" in sel or "gate" in sel or
                           sel.startswith("filter-") or "occ" in sel) else "none"
        add("postselect", "Bifurification/results/dp_postselect_uniqueness.csv",
            regime=str(r.noise), T=int(r.T), rep=int(r.rep), solver=r.solver,
            response_family=f, degree=degree, composed=composed,
            occ_alpha=0.05 if "inmatrix-occ" in sel else 0.0,
            metric="composed" if composed != "none" else "wp",
            tau=float(r.tau) if pd.notna(r.tau) else np.nan,
            eff=float(r.eff), far=float(r.far), notes=sel)
        n += 1
    return n


# ── gap analysis: PLAN §3 target matrix vs harvested coverage ────────────────
def coverage_key(row):
    return (row["regime"], int(row["T"]), row["response_family"],
            None if pd.isna(row["degree"]) else float(row["degree"]))


def build_gaps(inv: pd.DataFrame):
    """Emit the frozen config matrix: PLAN Stage 1-3 cells with coverage status."""
    have = {}
    for _, row in inv.iterrows():
        k = coverage_key(row)
        have.setdefault(k, set()).add(int(row["rep"]))

    def status(regime, T, fam, degrees, min_reps=3, extra=""):
        out = []
        for dg in degrees:
            reps = have.get((regime, T, fam, dg), set())
            st = ("covered" if len(reps) >= min_reps
                  else "partial" if reps else "missing")
            out.append(dict(regime=regime, T=T, response_family=fam,
                            degree=dg, reps_have=len(reps),
                            reps_needed=min_reps, status=st, notes=extra))
        return out

    gaps = []
    COMB_D = [8, 12, 16, 20, 24, 28, 32, 40, 48, 64]
    # Stage 1: response x degree at baseline set-out, clean + moderate, T=200
    for regime in ("clean", "moderate"):
        gaps += status(regime, 200, "comb_prod", COMB_D,
                       extra="S1 production comb")
        gaps += status(regime, 200, "band_inverse", COMB_D,
                       extra="S1 band-limited inverse")
        gaps += status(regime, 200, "sharp_comb", [80, 120],
                       extra="S1 sharp high-purity (VIII)")
        gaps += status(regime, 200, "fitted_ridge", [20, 28, 40],
                       extra="S1 fitted response (refit per regime)")
        gaps += status(regime, 200, "normalized_half_comb", [12, 20, 28, 32, 40],
                       extra="S1 normalized +-1/2 comb")
    # Stage 2 knobs (moderate T=200, carried responses; one knob at a time)
    gaps += status("moderate", 200, "comb_prod_gamma1", [40], extra="S2 gamma=1")
    gaps += status("moderate", 200, "comb_prod_gamma2", [40], extra="S2 gamma=2")
    gaps += status("moderate", 200, "comb_erf_kernel", [40],
                   extra="S2 erf kernel (raw comb known to collapse; confirm on frontier)")
    gaps += status("moderate", 200, "normalized_half_comb_erf", [20, 40],
                   extra="S2 erf kernel, normalized comb")
    for s in (1.5, 2.12, 3.0):
        gaps += status("moderate", 200, f"comb_prod_eps{s}", [40],
                       extra=f"S2 epsilon scale s={s} (suspected L3 driver)")
    for a in (0.05, 0.10):
        gaps += status("moderate", 200, f"fitted_ridge_occ{a}", [40],
                       extra=f"S2 occupancy alpha={a}, refit per alpha")
    gaps += status("clean", 200, "comb_prod_fork0.5", [40],
                   extra="S2 fork beta=0.5 clean only (heavy NO-GO stands)")
    # Stage 3: heavy composition on carried responses
    for fam, dgs in [("fitted_ridge", [40, 160]),
                     ("normalized_half_comb", [20, 40]),
                     ("fitted_ridge_occ0.05_normalized", [40])]:
        gaps += status("heavy", 200, fam, dgs, extra="S3 composed (gate on/off) — H4 cell")

    g = pd.DataFrame(gaps)
    g.to_csv(OUT / "00_gap_list.csv", index=False)
    return g


def main():
    counts = {
        "store_rows": harvest_store(),
        "degree_sweep_rows": harvest_degree_sweep(),
        "fit_comb_cells": harvest_fit_comb(),
        "occ_cells": harvest_occ(),
        "be03_rows": harvest_be03(),
        "postselect_rows": harvest_postselect(),
    }
    inv = pd.DataFrame(ROWS)
    inv.to_csv(OUT / "00_inventory.csv", index=False)

    print("== harvested ==")
    for k, v in counts.items():
        print(f"  {k:>20}: {v}")
    print(f"  inventory rows: {len(inv)} -> outputs/00_inventory.csv")

    print("\n== coverage: rows per (regime, response_family) ==")
    inv2 = inv.copy()
    inv2["regime_c"] = inv2["regime"].where(
        ~inv2["regime"].str.startswith("other("), "other(robustness axes)")
    pv = inv2.pivot_table(index="response_family", columns="regime_c",
                          values="eff", aggfunc="count", fill_value=0)
    print(pv.to_string())

    print("\n== existing frontier: best far at eff>=0.99, T in {200,400}, "
          "standard regimes ==")
    ok = inv[(inv["eff"] >= 0.99) & inv["far"].notna()
             & inv["T"].isin([200, 400])
             & inv["regime"].isin(["clean", "clean+drop1%", "moderate", "heavy"])]
    best = (ok.sort_values("far")
              .groupby(["regime", "response_family"], as_index=False)
              .first())[["regime", "response_family", "degree", "composed",
                         "T", "eff", "far", "source", "notes"]]
    print(best.to_string(index=False))

    g = build_gaps(inv)
    print("\n== frozen config matrix (gap list) ==")
    print(g.groupby(["status"]).size().to_string())
    print("\n-- covered / partial cells (no recompute) --")
    print(g[g["status"] != "missing"][["regime", "T", "response_family",
          "degree", "reps_have", "status", "notes"]].to_string(index=False))
    todo = g[g["status"] != "covered"]
    print(f"\ncells to compute in Stages 1-3: {len(todo)} "
          f"({(g['status'] == 'missing').sum()} missing, "
          f"{(g['status'] == 'partial').sum()} partial) -> outputs/00_gap_list.csv")
    print("\n== missing cells by stage note ==")
    print(todo.groupby("notes").size().to_string())


if __name__ == "__main__":
    sys.exit(main())
