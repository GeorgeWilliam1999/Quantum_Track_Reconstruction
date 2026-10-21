#!/usr/bin/env python3
"""Paper 1, figure 7 and table 2: the cost of the three algorithms in one currency, on the
corpus multiplicities T = 10 ... 1000.

(a) logical qubits against T: the system register ceil(log2 4T^2) plus the ancillas of each
    route (one-bit filter +2, phase-sequence QSVT +4, Chebyshev-walk QSVT +7, eight-bit HHL +9)
(b) two-qubit gates per solved event against T: calls per shot x amplification rounds
    pi/(4 sqrt P) x two-qubit gates per call. The controlled evolution is priced by the measured
    Givens transpilation (filled markers, T in {10, 50, 100, 200, 400}) and its power-law
    extrapolation (open markers); the hit-priced block encoding is a model (dashed). HHL is a
    structure-level estimate (dotted).

The post-selection probability P of the one-bit filter and of the polynomial is measured on the
paper corpus from the stored amplitudes (outputs_5um/amplitudes): P = ||p(A)u||^2 with u the
uniform state, isolated segments included; for the walk route it is divided by the squared
one-norm of the Chebyshev coefficients stored with each event. HHL's P is the estimate of the
resource model. Widths, calls per shot, the transpilation fits and the hit-oracle price come from
../section4_qsvt/outputs/02_resource_model.csv and 02_resource_fits.json (02_resource_model.py).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C = "outputs_5um"
FIG = HERE / "figures" / C
AMP = HERE / C / "amplitudes"
r = pd.read_csv(HERE.parent / "section4_qsvt" / "outputs" / "02_resource_model.csv").set_index("T")
fits = json.load(open(HERE.parent / "section4_qsvt" / "outputs" / "02_resource_fits.json"))
TS = [10, 20, 50, 100, 200, 400, 700, 1000]
ROUTES = {"1bqf": ("one-bit filter (2 calls)", "#1f77b4"), "phase": ("QSVT, phase sequence (160 calls)", "#d95f02"),
          "lcu": ("QSVT, Chebyshev walk (63 calls)", "#7570b3"), "hhl8": ("HHL, eight bits (510 calls)", "0.45")}
ANC = {"1bqf": 2, "phase": 4, "lcu": 7, "hhl8": 9}
MEASURED = set(r.index[r.givens_measured.astype(bool)])


def P_corpus(T):
    pq, pb, l1 = [], [], []
    for rep in range(10):
        z = np.load(AMP / f"T{T}_rep{rep}.npz")
        nt, nf = int(z["n_isolated_true"]), int(z["n_isolated_false"])
        N = len(z["x_qsvt"]) + nt + nf
        pq.append((np.sum(z["x_qsvt"] ** 2) + (nt + nf) * float(z["iso_qsvt"]) ** 2) / N)
        pb.append((np.sum(z["x_1bqf"] ** 2) + (nt + nf) * float(z["iso_1bqf"]) ** 2) / N)
        l1.append(float(np.sum(np.abs(z["comb_coef"]))))
    return float(np.median(pq)), float(np.median(pb)), float(np.median(l1))


rows = []
for T in TS:
    pq, pb, l1 = P_corpus(T)
    n_sys = int(np.ceil(np.log2(4 * T * T)))
    assert n_sys == int(r.loc[T, "n_sys"]), (T, n_sys)
    cx_evo = fits["givens_cx_per_call"]["prefactor"] * T ** fits["givens_cx_per_call"]["exponent"]
    if T in MEASURED: cx_evo = float(r.loc[T, "givens_cx_per_call"])
    cx_hit = fits["hit_cx_per_call"]["prefactor"] * T ** fits["hit_cx_per_call"]["exponent"]
    P = {"1bqf": pb, "phase": pq, "lcu": pq / l1 ** 2, "hhl8": float(r.loc[T, "P_hhl8"])}
    calls = {k: int(r.loc[T, f"calls_{k}"]) for k in ROUTES}
    for k in ROUTES:
        rounds = np.pi / (4 * np.sqrt(P[k]))
        rows.append(dict(T=T, route=k, n_sys=n_sys, width=n_sys + ANC[k], calls=calls[k], P=P[k], rounds=rounds,
                         total_calls=calls[k] * rounds, cx_evo=cx_evo, cx_evo_measured=T in MEASURED, cx_hit=cx_hit,
                         gates_evo=calls[k] * rounds * cx_evo, gates_hit=calls[k] * rounds * cx_hit, l1=l1))
d = pd.DataFrame(rows)
d.to_csv(FIG / "fig7_cost_table.csv", index=False)
assert (d.set_index(["T", "route"]).width == r.loc[TS][[f"w_{k}" for k in ROUTES]].stack().values).all()

fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
ax = axs[0]
for k, (lab, c) in ROUTES.items():
    g = d[d.route == k]
    ax.plot(g["T"], g.width, color=c, lw=1.5, marker="o", ms=3, label=lab)
ax.set_xscale("log"); ax.set_xlabel("tracks per event $T$"); ax.set_ylabel("logical qubits")
ax.set_title("(a) width", loc="left", fontsize=9); ax.grid(alpha=0.25); ax.legend(fontsize=6.5, loc="upper left")
ax = axs[1]
for k, (lab, c) in ROUTES.items():
    g = d[d.route == k]; m = g.cx_evo_measured.values
    ax.plot(g["T"], g.gates_evo, color=c, lw=1.3, ls=":" if k == "hhl8" else "-")
    ax.plot(g["T"][m], g.gates_evo[m], "o", color=c, ms=4); ax.plot(g["T"][~m], g.gates_evo[~m], "o", color=c, ms=4, mfc="w")
    if k in ("phase", "1bqf"): ax.plot(g["T"], g.gates_hit, color=c, lw=1.1, ls="--")
ax.plot([], [], "o", color="k", ms=4, label="controlled evolution, measured transpilation")
ax.plot([], [], "o", color="k", ms=4, mfc="w", label=f"controlled evolution, extrapolated $\\propto T^{{{fits['givens_cx_per_call']['exponent']:.2f}}}$")
ax.plot([], [], color="k", ls="--", lw=1.1, label=f"hit-priced block encoding, model $\\propto T^{{{fits['hit_cx_per_call']['exponent']:.2f}}}$")
ax.plot([], [], color="0.45", ls=":", lw=1.3, label="HHL, structure-level estimate")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("tracks per event $T$"); ax.set_ylabel("two-qubit gates per solved event")
ax.set_title("(b) gates per event, amplification included", loc="left", fontsize=9); ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=6, loc="upper left")
fig.tight_layout(w_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(FIG / f"fig7_cost.{ext}", dpi=200)

# scalar numbers for numbers.tex
out = {}
for T in (400, 1000):
    for k in ROUTES:
        g = d[(d["T"] == T) & (d.route == k)].iloc[0]
        for col in ("width", "calls", "P", "rounds", "total_calls", "gates_evo", "gates_hit"):
            out[f"{col} {k} T={T}"] = g[col]
    g = d[(d["T"] == T)].iloc[0]
    out[f"n_sys T={T}"] = g.n_sys; out[f"cx_evo T={T}"] = g.cx_evo; out[f"cx_hit T={T}"] = g.cx_hit; out[f"cx_evo_measured T={T}"] = g.cx_evo_measured
big = d[(d.route == "phase") & (d["T"] >= 100)]
b, a = np.polyfit(np.log(big["T"]), np.log(big.P), 1)
out["P phase corpus fit prefactor"] = float(np.exp(a)); out["P phase corpus fit exponent"] = float(b)
big = d[(d.route == "1bqf") & (d["T"] >= 100)]
b, a = np.polyfit(np.log(big["T"]), np.log(big.P), 1)
out["P 1bqf corpus fit prefactor"] = float(np.exp(a)); out["P 1bqf corpus fit exponent"] = float(b)
for key in ("givens_cx_per_call", "hit_cx_per_call"):
    out[f"{key} prefactor"] = fits[key]["prefactor"]; out[f"{key} exponent"] = fits[key]["exponent"]
out["measured T max"] = fits["givens_extrapolation"]["measured_T_max"]
out["hit oracle ancillas"] = float(r.hit_oracle_ancillas.dropna().iloc[0])
pd.Series(out, name="value").rename_axis("quantity").to_csv(FIG / "fig7_numbers.csv")
print("wrote", FIG / "fig7_cost.pdf"); print(d[d["T"].isin([400, 1000])].to_string()); print(pd.Series(out).tail(10).to_string())
