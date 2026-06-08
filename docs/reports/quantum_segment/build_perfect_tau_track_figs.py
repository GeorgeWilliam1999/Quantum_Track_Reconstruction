#!/usr/bin/env python3
"""Aggregate rethreshold-condor pickles and build fig3_tracking_tauopt.

Reads `outputs/quantum_segment_analysis/seg14e_T1000_{readout}_tauopt/pickles/`
for both readouts, aggregates trk_C_lay / trk_Q_lay (layered tracker)
metrics by T, and produces:
    fig3_tracking_tauopt   — efficiency, ghost_rate, clone_fraction, n_rec
    fig4_timing_tauopt     — copy of original timing (τ-independent)
into both report figs/ dirs.

Also writes perfect_tau_trk_agg_{sv,sm}.csv next to this script.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE   = Path(__file__).resolve().parent
REPO   = HERE.parents[2]
WORK   = REPO / 'Toy_Characterisation' / 'Verify_new_results'
OUTBASE= WORK / 'outputs' / 'quantum_segment_analysis'

SRC = {
    'sv': OUTBASE / 'seg14e_T1000_statevector_tauopt' / 'pickles',
    'sm': OUTBASE / 'seg14e_T1000_sampling_tauopt'    / 'pickles',
}
SRC_ORIG = {
    'sv': OUTBASE / 'seg14e_T1000_statevector' / 'pickles',
    'sm': OUTBASE / 'seg14e_T1000_sampling'    / 'pickles',
}

FIGDIRS = [
    HERE / 'figs',
    HERE.parent / 'quantum_vs_classical_segment' / 'figs',
]

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

def _load_trk(pdir):
    rows = []
    for fp in sorted(Path(pdir).glob('event_*.pkl')):
        try: r = pickle.load(open(fp,'rb'))
        except Exception: continue
        T = int(r['n_trk']); rep = int(r['rep'])
        rec = dict(n_trk=T, rep=rep, tau_rel=float(r.get('tau_rel', np.nan)))
        for key in ('trk_C_cc','trk_C_lay','trk_Q_cc','trk_Q_lay'):
            d = r.get(key, {}) or {}
            for m in ('efficiency','ghost_rate','clone_fraction',
                      'mean_purity','hit_efficiency','n_rec_tracks'):
                rec[f'{key}_{m}'] = float(d.get(m, np.nan)) \
                    if m != 'n_rec_tracks' else float(d.get(m, np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)

def _load_timing(pdir):
    rows = []
    for fp in sorted(Path(pdir).glob('event_*.pkl')):
        try: r = pickle.load(open(fp,'rb'))
        except Exception: continue
        rows.append(dict(n_trk=int(r['n_trk']), rep=int(r['rep']),
                         t_classical=float(r.get('t_classical', np.nan)),
                         t_quantum=float(r.get('t_quantum', np.nan)),
                         wall_total=float(r.get('wall_total', np.nan))))
    return pd.DataFrame(rows)

def _agg(df):
    if df.empty: return df
    sem = lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x) > 1 else 0.0
    cols = [c for c in df.columns if c not in ('n_trk','rep','tau_rel')]
    spec = {f'{c}_mean':(c,'mean') for c in cols}
    spec.update({f'{c}_sem':(c,sem) for c in cols})
    spec['n_reps'] = ('rep','count')
    if 'tau_rel' in df.columns:
        spec['tau_rel'] = ('tau_rel','first')
    return df.groupby('n_trk').agg(**spec).reset_index().sort_values('n_trk')

print('loading SV tauopt pickles …'); df_sv = _load_trk(SRC['sv']); print(f"  {len(df_sv)}")
print('loading SM tauopt pickles …'); df_sm = _load_trk(SRC['sm']); print(f"  {len(df_sm)}")
g_sv, g_sm = _agg(df_sv), _agg(df_sm)
g_sv.to_csv(HERE/'perfect_tau_trk_agg_sv.csv', index=False)
g_sm.to_csv(HERE/'perfect_tau_trk_agg_sm.csv', index=False)

def _err(ax, g, col, fmt, color, label, mfc=None):
    if g.empty or f'{col}_mean' not in g.columns: return
    kw = dict(fmt=fmt, color=color, capsize=4, capthick=1.2,
              markeredgecolor='black', markeredgewidth=0.6, zorder=3, label=label)
    if mfc is not None: kw['mfc'] = mfc
    ax.errorbar(g['n_trk'], g[f'{col}_mean'], yerr=g[f'{col}_sem'], **kw)

def _save(fig, stem):
    for d in FIGDIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d/f'{stem}.pdf', format='pdf', dpi=600, bbox_inches='tight', facecolor='white')
        fig.savefig(d/f'{stem}.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  saved {stem}.{{pdf,png}}')

# ── Figure 3 — tracking at τ* (layered tracker) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ax = axes[0, 0]
_err(ax, g_sv, 'trk_C_lay_efficiency', 'D--', cC,   'Classical (τ*)', mfc='white')
_err(ax, g_sv, 'trk_Q_lay_efficiency', 'o-',  cQsv, 'Quantum E1 (τ*)')
_err(ax, g_sm, 'trk_Q_lay_efficiency', 's-',  cQsm, 'Quantum E2 (τ*)')
ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Tracking efficiency')
ax.set_title('(a) Reconstruction efficiency')
ax.legend(loc='lower left', framealpha=0.95)

ax = axes[0, 1]
_err(ax, g_sv, 'trk_C_lay_ghost_rate', 'D--', cC,   'Classical (τ*)', mfc='white')
_err(ax, g_sv, 'trk_Q_lay_ghost_rate', 'o-',  cQsv, 'Quantum E1 (τ*)')
_err(ax, g_sm, 'trk_Q_lay_ghost_rate', 's-',  cQsm, 'Quantum E2 (τ*)')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Ghost rate')
ax.set_title('(b) Ghost-track fraction')
ax.legend(loc='upper left', framealpha=0.95)

ax = axes[1, 0]
_err(ax, g_sv, 'trk_C_lay_clone_fraction', 'D--', cC,   'Classical (τ*)', mfc='white')
_err(ax, g_sv, 'trk_Q_lay_clone_fraction', 'o-',  cQsv, 'Quantum E1 (τ*)')
_err(ax, g_sm, 'trk_Q_lay_clone_fraction', 's-',  cQsm, 'Quantum E2 (τ*)')
ax.set_xscale('log'); _style(ax)
ax.set_ylabel('Clone fraction')
ax.set_title('(c) Clone-track fraction')
ax.legend(loc='upper left', framealpha=0.95)

ax = axes[1, 1]
_err(ax, g_sv, 'trk_C_lay_n_rec_tracks', 'D--', cC,   'Classical (τ*)', mfc='white')
_err(ax, g_sv, 'trk_Q_lay_n_rec_tracks', 'o-',  cQsv, 'Quantum E1 (τ*)')
_err(ax, g_sm, 'trk_Q_lay_n_rec_tracks', 's-',  cQsm, 'Quantum E2 (τ*)')
ax.set_xscale('log'); ax.set_yscale('log'); _style(ax)
ax.set_ylabel('Reconstructed tracks per event')
ax.set_title('(d) Track multiplicity')
ax.legend(loc='upper left', framealpha=0.95)

for ax in axes.flat: ax.set_xlabel('Number of tracks $T$')
fig.suptitle(r"Tracking metrics at perfect $\tau$ (layered tracker)",
             y=1.02, fontweight='bold')
fig.tight_layout()
_save(fig, 'fig3_tracking_tauopt')
plt.close(fig)

# ── Figure 4 — timing (τ-independent) ────────────────────────────────────────
print('loading timing from original pickles …')
t_sv = _agg(_load_timing(SRC_ORIG['sv']))
t_sm = _agg(_load_timing(SRC_ORIG['sm']))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
_err(ax, t_sv, 't_classical', 'D--', cC,   'Classical solve', mfc='white')
_err(ax, t_sv, 't_quantum',   'o-',  cQsv, 'Quantum E1 solve (statevector)')
_err(ax, t_sm, 't_quantum',   's-',  cQsm, 'Quantum E2 solve (sampling)')
ax.set_xscale('log'); ax.set_yscale('log'); _style(ax)
ax.set_xlabel('Number of tracks $T$'); ax.set_ylabel('Wall time per event (s)')
ax.set_title(r'Solver wall time (τ-independent)')
ax.legend(loc='upper left', framealpha=0.95)
fig.tight_layout()
_save(fig, 'fig4_timing_tauopt')
plt.close(fig)
print('done.')
