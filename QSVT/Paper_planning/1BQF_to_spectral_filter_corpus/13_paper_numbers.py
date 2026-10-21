#!/usr/bin/env python3
"""Write every number quoted in paper 1's Introduction and Results 1 as a LaTeX macro.

Reads only generated files of the 5 um corpus and the figure scripts:
  outputs_5um/checks.csv, outputs_5um/summary_by_event.csv,
  figures/outputs_5um/fig1_numbers.txt, figures/fig2_onebit_response_factors.csv,
  figures/fig3_error_atlas_T1000_with_qsvt_counts.csv
Writes ../1BQF_to_spectral_filter/numbers.tex (and a copy here). No number is typed by hand.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "1BQF_to_spectral_filter"
C = "outputs_5um"
mac = {}


def put(name, value):
    assert name.isalpha(), name
    mac[name] = value


chk = pd.read_csv(HERE / C / "checks.csv")
assert chk.failed.isna().all(), "a corpus event failed its checks"
put("NEvents", f"{int(chk.groupby('T').size().min())}")
put("NEventsTotal", f"{len(chk)}")
big = chk[chk["T"] >= 100]
put("HitLossMean", f"{100 * big.C6_hit_loss_fraction.mean():.1f}")
put("HitLossMin", f"{100 * big.C6_hit_loss_fraction.min():.1f}")
put("HitLossMax", f"{100 * big.C6_hit_loss_fraction.max():.1f}")
put("SigmaPMeas", f"{1e3 * big.C6_sigma_p_measured.mean():.2f}")
put("SigmaPPred", f"{1e3 * chk.C6_sigma_p_expected.iloc[0]:.2f}")
put("CheckMaxDiff", f"{max(chk.C1_minres_maxabs.max(), chk.C2_cos2_maxabs.max(), chk.C2_sin_maxabs.max(), chk.C3_recursion_maxabs.max()):.0e}".replace("e-0", "e-"))

f1 = dict(l.strip().split("=") for l in open(HERE / "figures" / C / "fig1_numbers.txt") if "=" in l)
put("EpsMrad", f"{float(f1['epsilon_mrad']):.2f}")
_th = float(f1["epsilon_mrad"]) * 1e-3
_ec = 1.0 - np.cos(_th)
_m, _e = f"{_ec:.2e}".split("e")
put("EpsCos", f"{_m}\\times10^{{{int(_e)}}}")
put("ResidualRms", f"{float(f1['residual_rms_um_T400']):.1f}")
tout = float(f1["true_outside_window"])
m, e = f"{tout:.1e}".split("e")
put("TrueOutside", f"{m}\\times10^{{{int(e)}}}")

s = pd.read_csv(HERE / C / "summary_by_event.csv")
names = {200: "TwoHundred", 400: "FourHundred", 700: "SevenHundred", 1000: "Thousand"}
for T, w in names.items():
    g = s[s["T"] == T].groupby("solver")
    put(f"ClassFar{w}", f"{g.far99.median()['classical']:.2f}")
    put(f"OneBitFar{w}", f"{g.far99.median()['1BQF']:.2f}")
    put(f"QsvtEff{w}", f"{g.eff_at_far1.median()['QSVT']:.3f}")
    put(f"QsvtEffMin{w}", f"{g.eff_at_far1.min()['QSVT']:.3f}")
    put(f"QsvtFar{w}", f"{g.far99.median()['QSVT']:.2f}")
    put(f"ClassEffAtOne{w}", f"{g.eff_at_far1.median()['classical']:.2f}")
    put(f"OneBitEffAtOne{w}", f"{g.eff_at_far1.median()['1BQF']:.2f}")

fac = pd.read_csv(HERE / "figures" / "fig2_onebit_response_factors.csv")
kf = fac.groupby("configuration").kept_factor.max()
put("KeptPair", f"{kf['false pair']:.2f}")
put("KeptTriple", f"{kf['false triple']:.2f}")
put("KeptTrack", f"{kf['complete track']:.2f}")
put("KeptHubThree", f"{kf['hub with 3 prongs']:.2f}")
put("KeptHubFive", f"{kf['hub with 5 prongs']:.2f}")

cnt = pd.read_csv(HERE / "figures" / "fig3_error_atlas_T1000_with_qsvt_counts.csv").set_index(["solver", "error_type"])


def n(v):
    return f"{int(v):,}".replace(",", "{,}")


put("AtlasPairs", n(cnt.loc[("classical", "false pair"), "n_false"]))
put("AtlasPairsKeptClass", n(cnt.loc[("classical", "false pair"), "false_kept"]))
put("AtlasPairsKeptQsvt", n(cnt.loc[("QSVT", "false pair"), "false_kept"]))
put("AtlasTriples", n(cnt.loc[("classical", "false triple or bridge"), "n_false"]))
put("AtlasTriplesKeptQsvt", n(cnt.loc[("QSVT", "false triple or bridge"), "false_kept"]))
put("AtlasComplete", n(cnt.loc[("QSVT", "complete track"), "n_true"]))
put("AtlasCompleteKeptQsvt", n(cnt.loc[("QSVT", "complete track"), "true_kept"]))
put("AtlasBroken", n(cnt.loc[("QSVT", "broken track"), "n_true"]))
put("AtlasBrokenLostQsvt", n(cnt.loc[("QSVT", "broken track"), "n_true"] - cnt.loc[("QSVT", "broken track"), "true_kept"]))
put("AtlasContamLostQsvt", n(cnt.loc[("QSVT", "contaminated track"), "n_true"] - cnt.loc[("QSVT", "contaminated track"), "true_kept"]))
iso = cnt.loc[("classical", "isolated false segment"), "n_false"]
allfalse = cnt.xs("classical", level="solver").n_false.sum()
put("IsolatedShare", f"{100 * iso / allfalse:.1f}")
tot_true = cnt.xs("QSVT", level="solver").n_true.sum()
kept_true = cnt.xs("QSVT", level="solver").true_kept.sum()
put("AtlasQsvtEff", f"{kept_true / tot_true:.3f}")
kept_false_q = cnt.xs("QSVT", level="solver").false_kept.sum()
put("AtlasQsvtFar", f"{kept_false_q / (kept_true + kept_false_q):.3f}")
kt_c = cnt.xs("classical", level="solver").true_kept.sum()
kf_c = cnt.xs("classical", level="solver").false_kept.sum()
put("AtlasClassFar", f"{kf_c / (kt_c + kf_c):.2f}")
put("AtlasClassEff", f"{kt_c / tot_true:.3f}")


# ---- Results 2: spectrum figure and teeth scan ----
f4 = pd.read_csv(HERE / "figures" / "fig4_spectrum_numbers.csv").set_index("quantity")["value"]
g = lambda q: float(f4[q])
put("TrackInputWeightLow", f"{100 * g('complete track: input weight on lambda=2.382'):.1f}")
put("TrackSolutionWeightLow", f"{100 * g('complete track: solution weight on lambda=2.382'):.1f}")
put("ContamNearTrue", f"{100 * g('bright weight within 0.18 of a bright true line: contaminated track'):.0f}")
put("HubNearTrue", f"{100 * g('bright weight within 0.18 of a bright true line: hub'):.0f}")
put("TripleNearTrue", f"{100 * g('bright weight within 0.18 of a bright true line: false triple or bridge'):.1f}")
for key, q in (("QTrackLow", "bright true line 2.382"), ("QTrackHigh", "bright true line 4.618"),
               ("QIso", "isolated line 4"), ("QPairLow", "pair line 3"), ("QPairHigh", "pair line 5"),
               ("QTripleLow", "triple line 2.586"), ("QTripleHigh", "triple line 5.414"),
               ("QHubLow", "hub line 2.268"), ("QHubHigh", "hub line 5.732")):
    put(key, f"{g('QSVT kept factor at ' + q):.3f}")
lo_d, hi_d = g("comb domain T1000 rep0 low"), g("comb domain T1000 rep0 high")
W = 0.5 * (hi_d - lo_d)
_q_ratio = g("QSVT kept factor at pair line 3") / g("QSVT kept factor at bright true line 2.382")
_o_ratio = g("one-bit kept factor at pair line 3") / g("one-bit kept factor at bright true line 2.382")
put("PairSuppression", f"{_o_ratio / _q_ratio:.0f}")
put("DomainLow", f"{lo_d:.2f}")
put("DomainHigh", f"{hi_d:.2f}")
put("DomainHalfWidth", f"{W:.2f}")
put("DomainLength", f"{2 * W:.2f}")
put("ResolutionForty", f"{2 * W / 40:.2f}")
put("ResolutionSixty", f"{2 * W / 60:.2f}")
put("NIsolatedThousand", n(g("isolated false segments, T=1000, ten events")))

ts = pd.read_csv(HERE / "outputs_teeth" / "teeth_scan.csv")
held = ts[ts.rep >= 5]
def teeth(tee, dg, hw, T):
    sel = held[(held.teeth == tee) & (held.degree == dg) & (np.isclose(held.half_width, hw)) & (held["T"] == T)]
    return f"{sel['eff_at_far0.01'].median():.3f}"
put("TeethFourSeven", teeth("four", 40, 0.18, 700))
put("TeethFourThousand", teeth("four", 40, 0.18, 1000))
put("TeethBrightThousand", teeth("two bright", 40, 0.18, 1000))
put("TeethDarkTwoHundred", teeth("two dark", 40, 0.18, 200))
put("TeethDarkThousand", teeth("two dark", 40, 0.18, 1000))
put("TeethBestSeven", teeth("two bright", 60, 0.12, 700))
put("TeethBestThousand", teeth("two bright", 60, 0.12, 1000))
put("TeethFourSixtySeven", teeth("four", 60, 0.12, 700))
put("TeethFourSixtyThousand", teeth("four", 60, 0.12, 1000))
put("TeethNDesigns", f"{ts.groupby(['teeth', 'degree', 'half_width']).ngroups}")

# ---- Results 2.3 and 2.4: multiplicity and cost figures ----
f5 = pd.read_csv(HERE / "figures" / C / "fig5_numbers.csv").set_index("quantity")["value"]
put("ClassCloseUpToT", f"{int(f5['classical close to polynomial up to T'])}")
put("ClassDropT", f"{int(f5['classical first T more than 0.05 below polynomial'])}")
put("OneBitPlateau", f"{float(f5['eff_at_far1 median 1BQF T=1000']):.2f}")
f7 = pd.read_csv(HERE / "figures" / C / "fig7_numbers.csv").set_index("quantity")["value"]
def sci(x):
    m, e = f"{float(x):.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"
RN = {"1bqf": "OneBit", "phase": "Phase", "lcu": "Walk", "hhl8": "Hhl"}
TN = {400: "FourHundred", 1000: "Thousand"}
for T, tn in TN.items():
    put(f"NSys{tn}", f"{int(float(f7[f'n_sys T={T}']))}")
    put(f"CxEvo{tn}", sci(f7[f"cx_evo T={T}"]))
    put(f"CxHit{tn}", sci(f7[f"cx_hit T={T}"]))
    for k, rn in RN.items():
        put(f"Width{rn}{tn}", f"{int(float(f7[f'width {k} T={T}']))}")
        put(f"P{rn}{tn}", sci(f7[f"P {k} T={T}"]))
        put(f"Rounds{rn}{tn}", f"{float(f7[f'rounds {k} T={T}']):.0f}")
        put(f"TotalCalls{rn}{tn}", f"{float(f7[f'total_calls {k} T={T}']):.0f}")
        put(f"GatesEvo{rn}{tn}", sci(f7[f"gates_evo {k} T={T}"]))
        put(f"GatesHit{rn}{tn}", sci(f7[f"gates_hit {k} T={T}"]))
for k, rn in RN.items():
    put(f"Calls{rn}", f"{int(float(f7[f'calls {k} T=400']))}")
put("PPhasePrefactor", f"{float(f7['P phase corpus fit prefactor']):.2f}")
put("PPhaseExponent", f"{-float(f7['P phase corpus fit exponent']):.2f}")
put("POneBitPrefactor", f"{float(f7['P 1bqf corpus fit prefactor']):.2f}")
put("POneBitExponent", f"{-float(f7['P 1bqf corpus fit exponent']):.2f}")
put("CxEvoPrefactor", f"{float(f7['givens_cx_per_call prefactor']):.0f}")
put("CxEvoExponent", f"{float(f7['givens_cx_per_call exponent']):.2f}")
put("CxHitPrefactor", f"{float(f7['hit_cx_per_call prefactor']):.0f}")
put("CxHitExponent", f"{float(f7['hit_cx_per_call exponent']):.2f}")
put("MeasuredTMax", f"{int(float(f7['measured T max']))}")
put("HitOracleAncillas", f"{int(float(f7['hit oracle ancillas']))}")

lines = ["% Generated by 1BQF_to_spectral_filter_corpus/13_paper_numbers.py -- do not edit by hand."]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
text = "\n".join(lines) + "\n"
(PAPER / "numbers.tex").write_text(text)
(HERE / "numbers.tex").write_text(text)
print(text)
