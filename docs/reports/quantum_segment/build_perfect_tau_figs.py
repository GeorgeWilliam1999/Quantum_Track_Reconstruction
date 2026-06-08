#!/usr/bin/env python3
"""Generate the full set of segment-level analysis plots at the "perfect" tau.

The "perfect" tau is the F1-optimal *relative* threshold from
best_tau_rel_f1.csv (per-T choice).  For every event pickle we recompute:
    active = sol_Q_scaled > tau*(T) * max(sol_Q_scaled)
    classical active = sol_C > tau*(T) * max(sol_C)
and derive segment efficiency, false rate, purity, F1, Jaccard and activation
counts.  Fidelity quantities (cos sim, rel-L2, success prob) are tau-independent
and come straight from the pickle.

Outputs (PDF + PNG into BOTH report figs/ dirs):
    fig1_seg_metrics_tauopt   — eff / FR / purity / F1 vs T
    fig1b_activations_tauopt  — true-active / false-active vs T
    fig2_fidelity_tauopt      — cos / rel-L2 / Jaccard / P_anc vs T
    fig5_seg14e_mirror_tauopt — 2×2 §14e mirror at tau*
Also writes CSVs   perfect_tau_agg_{sv,sm,classical}.csv  next to this script.
"""
from __future__ import annotations
import pickle, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).resolve().parent
REPO   = HERE.parents[2]
WORK   = REPO / 'Toy_Characterisation' / 'Verify_new_results'
OUTBASE= WORK / 'outputs' / 'quantum_segment_analysis'
PIK_SV = OUTBASE / 'seg14e_T1000_statevector' / 'pickles'
PIK_SM = OUTBASE / 'seg14e_T1000_sampling'    / 'pickles'

FIGDIRS = [
    HERE / 'figs',
    HERE.parent / 'quantum_vs_classical_segment' / 'figs',
]
for d in FIGDIRS:
    d.mkdir(parents=True, exist_ok=True)

# ── perfect-τ table (per-T F1 optimum, relative rule) ────────────────────────
BEST = pd.read_csv(HERE / 'best_tau_rel_f1.csv')
TAU_BY_T = dict(zip(BEST['n_trk'].astype(int), BEST['tau'].astype(float)))
print(f'perfect τ_rel(T) = {TAU_BY_T}')

# ── plot style (mirror §14e) ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.size':12, 'axes.labelsize':13, 'axes.titlesize':13,
    'xtick.labelsize':11, 'ytick.labelsize':11, 'legend.fontsize':10,
    'figure.titlesize':14, 'axes.linewidth':1.2,
    'xtick.major.width':1.0, 'ytick.major.width':1.0,
    'lines.linewidth':1.8, 'lines.markersize':6,
    'axes.grid':True, 'grid.alpha':0.3,
})
cC, cQsv, cQsm = '#444444', '#2166ac', '#e08214'

def _style(ax):
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, zorder=1)

# ── per-pickle recompute ─────────────────────────────────────────────────────
def _seg_truth_from(sC, n_true_all):
    """Reconstruct truth mask: top n_true_all by |sol_C|."""
    truth = np.zeros_like(sC, dtype=bool)
    if n_true_all > 0:
        idx = np.argsort(-sC)[: int(n_true_all)]
        truth[idx] = True
    return truth

def _seg_metrics(sol, tau_rel, truth, n_true_clean):
    qmax = float(np.max(sol)) if len(sol) else 0.0
    tau  = tau_rel * qmax
    active = sol > tau
    n_active = int(active.sum())
    n_true_act  = int((active & truth).sum())
    n_false_act = n_active - n_true_act
    eff = 100.0 * n_true_act / n_true_clean if n_true_clean > 0 else np.nan
    fr  = 100.0 * n_false_act / n_active if n_active > 0 else 0.0
    pur = 100.0 * n_true_act / n_active if n_active > 0 else np.nan
    f1  = 2*eff*pur/(eff+pur) if (eff+pur) > 0 else np.nan
    return dict(eff=eff, fr=fr, pur=pur, f1=f1,
                n_active=n_active, n_true_act=n_true_act,
                n_false_act=n_false_act, qmax=qmax)

def _load(pdir):
    rows = []
    for fp in sorted(Path(pdir).glob('event_*.pkl')):
        try:    r = pickle.load(open(fp,'rb'))
        except Exception: continue
        T   = int(r['n_trk']); rep = int(r['rep'])
        sC  = np.asarray(r['sol_C'], dtype=float)
        sQ  = np.asarray(r['sol_Q_scaled'], dtype=float)
        seg = r.get('seg_C', {})
        n_true_all   = int(seg.get('n_true_all', 0))
        n_true_clean = int(seg.get('n_true_clean') or n_true_all)
        truth = _seg_truth_from(sC, n_true_all)
        tau_rel = TAU_BY_T.get(T, 0.30)
        mC = _seg_metrics(sC, tau_rel, truth, n_true_clean)
        mQ = _seg_metrics(sQ, tau_rel, truth, n_true_clean)
        # Jaccard at τ*
        aC = sC > tau_rel * mC['qmax']
        aQ = sQ > tau_rel * mQ['qmax']
        union = int((aC | aQ).sum())
        jacc  = int((aC & aQ).sum()) / union if union > 0 else 1.0
        rows.append(dict(
            n_trk=T, rep=rep, tau_rel=tau_rel,
            n_true_all=n_true_all, n_true_clean=n_true_clean,
            n_false_all=int(seg.get('n_false_all', 0)),
            eff_C=mC['eff'], eff_Q=mQ['eff'],
            fr_C=mC['fr'], fr_Q=mQ['fr'],
            pur_C=mC['pur'], pur_Q=mQ['pur'],
            f1_C=mC['f1'], f1_Q=mQ['f1'],
            true_act_C=mC['n_true_act'], false_act_C=mC['n_false_act'],
            true_act_Q=mQ['n_true_act'], false_act_Q=mQ['n_false_act'],
            n_active_C=mC['n_active'], n_active_Q=mQ['n_active'],
            cos=float(r.get('fidelity', {}).get('cos', np.nan)),
            rel_l2=float(r.get('fidelity', {}).get('rel_l2', np.nan)),
            jaccard=jacc,
            p_anc=float(r.get('success_prob', np.nan)),
        ))
    return pd.DataFrame(rows)

def _agg(df):
    if df.empty: return df
    sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    cols = [c for c in df.columns if c not in ('n_trk', 'rep', 'tau_rel')]
    spec = {f'{c}_mean': (c, 'mean') for c in cols}
    spec.update({f'{c}_sem': (c, sem) for c in cols})
    spec['n_reps'] = ('rep', 'count')
    spec['tau_rel'] = ('tau_rel', 'first')
    return df.groupby('n_trk').agg(**spec).reset_index().sort_values('n_trk')

print('loading E1 statevector pickles …')
df_sv = _load(PIK_SV)
print(f'  {len(df_sv)} events')
print('loading E2 sampling pickles …')
df_sm = _load(PIK_SM)
print(f'  {len(df_sm)} events')

g_sv, g_sm = _agg(df_sv), _agg(df_sm)
g_sv.to_csv(HERE / 'perfect_tau_agg_sv.csv', index=False)
g_sm.to_csv(HERE / 'perfect_tau_agg_sm.csv', index=False)
print(f"wrote {HERE/'perfect_tau_agg_sv.csv'}")
print(f"wrote {HERE/'perfect_tau_agg_sm.csv'}")

# ── helpers ──────────────────────────────────────────────────────────────────
def _err(ax, g, col, fmt, color, label, mfc=None):
    if g.empty: return
    kw = dict(fmt=fmt, color=color, capsize=4, capthick=1.2,
              markeredgecolor='black', markeredgewidth=0.6, zorder=3,
              label=label)
    if mfc is not None: kw['mfc'] = mfc
    return ax.errorbar(g['n_trk'], g[f'{col}_mean'], yerr=g[f'{col}_sem'], **kw)

def _logfloor(y, e, floor=0.5):
    y = np.asarray(y, float); e = np.asarray(e, float)
    lo = np.minimum(e, np.maximum(y - floor, 0.0))
    return np.vstack([lo, e])

def _save(fig, stem):
    for d in FIGDIRS:
        fig.savefig(d / f'{stem}.pdf', format='pdf', dpi=600,
                    bbox_inches='tight', facecolor='white')
        fig.savefig(d / f'{stem}.png', dpi=200,
                    bbox_inches='tight', facecolor='white')
    print(f'  saved {stem}.{{pdf,png}} in {len(FIGDIRS)} dirs')

# ───────────────────────── Figure 1 — seg metrics at τ* ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
_err(ax, g_sv, 'eff_C', 'D--', cC,  'Classical', mfc='white')
_err(ax, g_sv, 'eff_Q', 'o-',  cQsv, 'Quantum E1 (statevector)')
_err(ax, g_sm, 'eff_Q', 's-',  cQsm, 'Quantum E2 (sampling)')
ax.axhline(100, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Segment efficiency (%)')
ax.set_title(r'(a) Efficiency $= N_{\rm true\,act}/N_{\rm true\,all}$')
ax.legend(loc='lower left', framealpha=0.95)

ax = axes[0, 1]
_err(ax, g_sv, 'fr_C', 'D--', cC,  'Classical', mfc='white')
_err(ax, g_sv, 'fr_Q', 'o-',  cQsv, 'Quantum E1')
_err(ax, g_sm, 'fr_Q', 's-',  cQsm, 'Quantum E2')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Segment false rate (%)')
ax.set_title(r'(b) False rate $= N_{\rm false\,act}/N_{\rm active}$')
ax.legend(loc='upper left', framealpha=0.95)

ax = axes[1, 0]
_err(ax, g_sv, 'pur_C', 'D--', cC,  'Classical', mfc='white')
_err(ax, g_sv, 'pur_Q', 'o-',  cQsv, 'Quantum E1')
_err(ax, g_sm, 'pur_Q', 's-',  cQsm, 'Quantum E2')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Segment purity (%)')
ax.set_title('(c) Purity = 100 - false rate')
ax.legend(loc='lower left', framealpha=0.95)

ax = axes[1, 1]
_err(ax, g_sv, 'f1_C', 'D--', cC,  'Classical', mfc='white')
_err(ax, g_sv, 'f1_Q', 'o-',  cQsv, 'Quantum E1')
_err(ax, g_sm, 'f1_Q', 's-',  cQsm, 'Quantum E2')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel(r'$F_1$ (%)')
ax.set_title(r'(d) $F_1 = 2\,\mathrm{eff}\cdot\mathrm{pur}/(\mathrm{eff}+\mathrm{pur})$')
ax.legend(loc='lower left', framealpha=0.95)

for ax in axes.flat:
    ax.set_xlabel('Number of tracks $T$')

fig.suptitle(r"Segment-level metrics at perfect $\tau$ (relative, per-$T$ F$_1$-optimum)",
             y=1.02, fontweight='bold')
fig.tight_layout()
_save(fig, 'fig1_seg_metrics_tauopt')
plt.close(fig)

# ───────────────────────── Figure 1b — activations at τ* ─────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
_err(ax, g_sv, 'true_act_C',  'D--', cC,   'true active — classical',     mfc='white')
_err(ax, g_sv, 'false_act_C', 'D:',  cC,   'false active — classical',    mfc='white')
_err(ax, g_sv, 'true_act_Q',  'o-',  cQsv, 'true active — Q E1')
_err(ax, g_sv, 'false_act_Q', 'o:',  cQsv, 'false active — Q E1')
_err(ax, g_sm, 'true_act_Q',  's-',  cQsm, 'true active — Q E2')
_err(ax, g_sm, 'false_act_Q', 's:',  cQsm, 'false active — Q E2')
ax.set_xscale('log'); ax.set_yscale('log'); _style(ax)
ax.set_xlabel('Number of tracks $T$')
ax.set_ylabel('Number of segments')
ax.set_title(r"Active segment counts at perfect $\tau$")
ax.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2)
fig.tight_layout()
_save(fig, 'fig1b_activations_tauopt')
plt.close(fig)

# ───────────────────────── Figure 2 — fidelity ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
_err(ax, g_sv, 'cos', 'o-',  cQsv, 'Quantum E1')
_err(ax, g_sm, 'cos', 's-',  cQsm, 'Quantum E2')
ax.axhline(1, color='gray', ls='--', lw=1)
ax.set_xscale('log'); _style(ax)
ax.set_ylabel(r'$\cos(s_Q, s_C)$')
ax.set_title('(a) Solution-vector cosine similarity')
ax.legend(loc='lower left', framealpha=0.95)

ax = axes[0, 1]
_err(ax, g_sv, 'rel_l2', 'o-', cQsv, 'Quantum E1')
_err(ax, g_sm, 'rel_l2', 's-', cQsm, 'Quantum E2')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel(r'$\|s_Q - s_C\|_2 / \|s_C\|_2$')
ax.set_title('(b) Relative $L_2$ error')
ax.legend(loc='upper left', framealpha=0.95)

ax = axes[1, 0]
_err(ax, g_sv, 'jaccard', 'o-', cQsv, 'Quantum E1')
_err(ax, g_sm, 'jaccard', 's-', cQsm, 'Quantum E2')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Jaccard($\\tau$-set$_Q$, $\\tau$-set$_C$)')
ax.set_title(r'(c) Activation overlap at perfect $\tau$')
ax.legend(loc='lower left', framealpha=0.95)

ax = axes[1, 1]
_err(ax, g_sv, 'p_anc', 'o-', cQsv, 'Quantum E1')
_err(ax, g_sm, 'p_anc', 's-', cQsm, 'Quantum E2')
ax.set_xscale('log'); ax.set_yscale('log'); _style(ax)
ax.set_ylabel(r'$P(\mathrm{anc} = 1)$')
ax.set_title('(d) Ancilla success probability')
ax.legend(loc='upper right', framealpha=0.95)

for ax in axes.flat:
    ax.set_xlabel('Number of tracks $T$')

fig.suptitle(r"Solution-vector fidelity (τ-independent except panel c)",
             y=1.02, fontweight='bold')
fig.tight_layout()
_save(fig, 'fig2_fidelity_tauopt')
plt.close(fig)

# ───────────────────────── Figure 5 — §14e mirror at τ* ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

ax = axes[0, 0]
_err(ax, g_sv, 'eff_C', 'D--', cC,  'Classical solver', mfc='white')
_err(ax, g_sv, 'eff_Q', 'o-',  cQsv, 'OneBitHHL (E1 SV)')
_err(ax, g_sm, 'eff_Q', 's-',  cQsm, 'OneBitHHL (E2 SM)')
ax.axhline(100, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_ylabel('Segment efficiency (%)')
ax.set_title(r'i) Efficiency at perfect $\tau$', fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='lower left', framealpha=0.95); _style(ax)

ax = axes[0, 1]
_err(ax, g_sv, 'fr_C', 'D--', cC,  'Classical solver', mfc='white')
_err(ax, g_sv, 'fr_Q', 'o-',  cQsv, 'OneBitHHL (E1 SV)')
_err(ax, g_sm, 'fr_Q', 's-',  cQsm, 'OneBitHHL (E2 SM)')
ax.set_ylabel('Segment false rate (%)')
ax.set_title(r'ii) False rate at perfect $\tau$', fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='upper left', framealpha=0.95); _style(ax)

ax = axes[1, 0]
if not g_sv.empty:
    ax.errorbar(g_sv['n_trk'], g_sv['n_true_all_mean'],
                yerr=_logfloor(g_sv['n_true_all_mean'], g_sv['n_true_all_sem']),
                fmt='o-', color='#1b7837', capsize=4, capthick=1.2,
                markeredgecolor='black', markeredgewidth=0.6,
                label='True segments', zorder=3)
    ax.errorbar(g_sv['n_trk'], g_sv['n_false_all_mean'],
                yerr=_logfloor(g_sv['n_false_all_mean'], g_sv['n_false_all_sem']),
                fmt='s-', color='#c51b7d', capsize=4, capthick=1.2,
                markeredgecolor='black', markeredgewidth=0.6,
                label='False segments', zorder=3)
ax.set_yscale('log')
ax.set_ylabel('Number of segment pairs')
ax.set_title('iii) Segment pair counts', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95); _style(ax)

ax = axes[1, 1]
def _eb_act(ax, g, col, fmt, color, label, mfc=None):
    if g.empty: return
    y = g[f'{col}_mean'].values; e = g[f'{col}_sem'].values
    yp = np.maximum(y, 0.5)
    ep = np.where(y >= 0.5, e, 0.0)
    kw = dict(fmt=fmt, color=color, capsize=4, capthick=1.2,
              markeredgecolor='black', markeredgewidth=0.6, zorder=3, label=label)
    if mfc is not None: kw['mfc'] = mfc
    ax.errorbar(g['n_trk'], yp, yerr=_logfloor(yp, ep), **kw)

_eb_act(ax, g_sv, 'true_act_C',  'D--', cC,   'true act — classical', mfc='white')
_eb_act(ax, g_sv, 'true_act_Q',  'o-',  cQsv, 'true act — E1')
_eb_act(ax, g_sm, 'true_act_Q',  's-',  cQsm, 'true act — E2')
_eb_act(ax, g_sv, 'false_act_C', 'D:',  '#c51b7d', 'false act — classical', mfc='white')
_eb_act(ax, g_sv, 'false_act_Q', 'o:',  '#7b3294', 'false act — E1')
_eb_act(ax, g_sm, 'false_act_Q', 's:',  '#bf812d', 'false act — E2')
ax.set_yscale('log')
ax.set_ylim(0.35, None)
ax.set_ylabel('Number of active segments')
ax.set_title(r'iv) Active segments at perfect $\tau$', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=8, ncol=2); _style(ax)

for ax in axes.flat:
    ax.set_xlabel('Number of tracks $T$')
for ax in axes[1, :]:
    ax.set_xscale('log')

fig.suptitle(r"§14e mirror at perfect $\tau$ (relative rule, per-$T$ F$_1$-optimum)",
             y=1.00, fontweight='bold')
fig.tight_layout()
_save(fig, 'fig5_seg14e_mirror_tauopt')
plt.close(fig)

print('done.')
