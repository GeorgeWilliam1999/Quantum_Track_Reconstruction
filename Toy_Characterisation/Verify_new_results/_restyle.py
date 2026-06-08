"""Restyle Quantum_segment_level_analysis.ipynb to match the §14e paper style
from segment_level_analysis.ipynb cells 62 + 76."""
import json
from pathlib import Path

P = Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation'
         '/Verify_new_results/Quantum_segment_level_analysis.ipynb')
nb = json.loads(P.read_text())

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

SETUP = r"""# §1 Setup — imports, paths, §14e constants, and §14e paper plot style
import os, sys, glob, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO     = Path('/data/bfys/gscriven/Quantum_Track_Reconstruction')
WORK     = REPO / 'Toy_Characterisation' / 'Verify_new_results'
OUT_BASE = WORK / 'outputs' / 'quantum_segment_analysis'
DIR_SV   = OUT_BASE / 'seg14e_T1000_statevector'
DIR_SM   = OUT_BASE / 'seg14e_T1000_sampling'
PIK_SV   = DIR_SV / 'pickles'
PIK_SM   = DIR_SM / 'pickles'
for d in (DIR_SV, DIR_SM, PIK_SV, PIK_SM):
    d.mkdir(parents=True, exist_ok=True)

# §14e physics constants
GAMMA, DELTA, EPSILON = 3.0, 1.0, 2e-3
SIGMA_RES, SIGMA_SCT  = 5e-6, 1e-4
TAU, DROP_RATE        = 0.35, 0.01

# ---- §14e paper plot settings (mirrors segment_level_analysis.ipynb cell 62) ----
plt.rcParams.update({
    'font.size':         12,
    'axes.labelsize':    14,
    'axes.titlesize':    14,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   10,
    'figure.titlesize':  16,
    'axes.linewidth':    1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'lines.linewidth':   2,
    'lines.markersize':  7,
    'axes.grid':         True,
    'grid.alpha':        0.3,
})

cC, cQ_sv, cQ_sm = '#444444', '#2166ac', '#e08214'
c_true, c_false  = '#1b7837', '#c51b7d'

def _style_axes(ax):
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, zorder=1)

def _err(ax, agg, col, fmt='o-', color=None, label=None, mfc=None,
         capsize=5, capthick=1.5):
    if agg is None or len(agg) == 0:
        return
    kw = dict(fmt=fmt, color=color, capsize=capsize, capthick=capthick,
              markeredgecolor='black', markeredgewidth=0.8, zorder=3,
              label=label)
    if mfc is not None:
        kw['mfc'] = mfc
    return ax.errorbar(agg['n_trk'], agg[f'{col}_mean'],
                       yerr=agg[f'{col}_sem'], **kw)

def _save_paper(fig, stem, outdir=None):
    outdir = Path(outdir) if outdir else DIR_SV
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{stem}.pdf', format='pdf', dpi=600,
                bbox_inches='tight', facecolor='white')
    fig.savefig(outdir / f'{stem}.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f'  saved -> {outdir}/{stem}.{{pdf,png}}')

print(f'DIR_SV = {DIR_SV}')
print(f'DIR_SM = {DIR_SM}')
"""
nb['cells'][1] = code(SETUP)

FIG1 = r"""# §7 Figure 1 — segment metrics (E1 + E2 overlaid), §14e paper style
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
_err(ax, agg_sv, 'eff_C', fmt='o--', color=cC,    label='classical', mfc='white')
_err(ax, agg_sv, 'eff_Q', fmt='o-',  color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'eff_Q', fmt='s-',  color=cQ_sm, label='E2 sampling')
ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2)
ax.set_ylim(0, 1.05); ax.set_ylabel('Segment Efficiency')
ax.set_title('i) Segment Efficiency', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95); _style_axes(ax)

ax = axes[0, 1]
_err(ax, agg_sv, 'pur_C', fmt='o--', color=cC,    label='classical', mfc='white')
_err(ax, agg_sv, 'pur_Q', fmt='o-',  color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'pur_Q', fmt='s-',  color=cQ_sm, label='E2 sampling')
ax.set_ylim(0, 1.05); ax.set_ylabel('Segment Purity')
ax.set_title('ii) Segment Purity', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95); _style_axes(ax)

ax = axes[1, 0]
_err(ax, agg_sv, 'far_C', fmt='o--', color=cC,    label='classical', mfc='white')
_err(ax, agg_sv, 'far_Q', fmt='o-',  color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'far_Q', fmt='s-',  color=cQ_sm, label='E2 sampling')
ax.set_yscale('log'); ax.set_ylabel('Segment False-Activation Rate')
ax.set_title('iii) False-Activation Rate', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95); _style_axes(ax)

ax = axes[1, 1]
_err(ax, agg_sv, 'jaccard', fmt='o-', color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'jaccard', fmt='s-', color=cQ_sm, label='E2 sampling')
ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2)
ax.set_ylim(0, 1.05); ax.set_ylabel(r'Jaccard$(A_Q, A_C)$')
ax.set_title('iv) Active-Set Overlap', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95); _style_axes(ax)

for ax in axes.flat:
    ax.set_xlabel('Number of Tracks'); ax.set_xscale('log')

fig.tight_layout()
_save_paper(fig, 'fig1_segment_metrics')
plt.show()
"""
nb['cells'][8] = code(FIG1)

FIG1B = r"""# §7b Per-segment amplitude distributions vs T (§14e paper style)
def _pool_amps(pdir, ts, max_files_per_t=5):
    out = {}
    for t in ts:
        files = sorted(Path(pdir).glob(f'event_n{t:04d}_rep*.pkl'))[:max_files_per_t]
        sC_t, sC_f, sQ_t, sQ_f = [], [], [], []
        for fp in files:
            try: r = pickle.load(open(fp, 'rb'))
            except Exception: continue
            sC = np.asarray(r['sol_C']); sQ = np.asarray(r['sol_Q_scaled'])
            n_true = int(r['seg_C']['n_true_all'])
            idx_sorted = np.argsort(-sC)
            mask = np.zeros_like(sC, dtype=bool); mask[idx_sorted[:n_true]] = True
            sC_t.append(sC[mask]); sC_f.append(sC[~mask])
            sQ_t.append(sQ[mask]); sQ_f.append(sQ[~mask])
        if sC_t:
            out[t] = dict(sC_true=np.concatenate(sC_t), sC_false=np.concatenate(sC_f),
                          sQ_true=np.concatenate(sQ_t), sQ_false=np.concatenate(sQ_f))
    return out

T_DIAG = [10, 50, 100, 200, 500]
pools  = _pool_amps(PIK_SV, T_DIAG, max_files_per_t=5)

fig, axes = plt.subplots(2, len(T_DIAG), figsize=(3.2*len(T_DIAG), 6.8),
                         sharex=False, sharey='row')
for j, t in enumerate(T_DIAG):
    if t not in pools:
        for ax in axes[:, j]: ax.set_visible(False)
        continue
    p = pools[t]
    ax = axes[0, j]
    bins = np.linspace(0, max(0.8, p['sC_true'].max()*1.05), 60)
    ax.hist(p['sC_false'], bins=bins, color='0.75', alpha=0.85, label='false')
    ax.hist(p['sC_true'],  bins=bins, color=cC, histtype='step', lw=2.0, label='true')
    ax.axvline(TAU, color=c_false, ls='--', lw=1.5, alpha=0.85,
               label=fr'$\tau={TAU}$')
    if p['sC_true'].size:
        ax.axvline(TAU*p['sC_true'].max(), color=cQ_sv, ls=':', lw=1.8,
                   alpha=0.9, label=r'$\tau\cdot\max$')
    ax.set_yscale('log')
    ax.set_title(f'i) Classical, T={t}', fontweight='bold')
    if j == 0: ax.set_ylabel('Count')
    if j == len(T_DIAG)-1: ax.legend(fontsize=8, loc='upper right', framealpha=0.95)
    _style_axes(ax)

    ax = axes[1, j]
    qmax = p['sQ_true'].max() if p['sQ_true'].size else 0
    bins = np.linspace(0, max(0.8, qmax*1.05), 60)
    ax.hist(p['sQ_false'], bins=bins, color='0.75', alpha=0.85)
    ax.hist(p['sQ_true'],  bins=bins, color=cQ_sv, histtype='step', lw=2.0)
    ax.axvline(TAU,      color=c_false, ls='--', lw=1.5, alpha=0.85)
    ax.axvline(TAU*qmax, color=cQ_sv,   ls=':',  lw=1.8, alpha=0.9)
    ax.set_yscale('log'); ax.set_xlabel(r'$s_i$')
    ax.set_title(f'ii) Quantum (scaled), T={t}', fontweight='bold')
    if j == 0: ax.set_ylabel('Count')
    _style_axes(ax)

fig.suptitle(r'Per-segment amplitude distributions vs $T$ (E1 statevector)',
             fontweight='bold')
fig.tight_layout()
_save_paper(fig, 'fig1b_activation_distributions')
plt.show()
"""
nb['cells'][10] = code(FIG1B)

FIG1C = r"""# §7c Relative-threshold comparison (§14e paper style)
def _rel_metrics_from_pkl(p):
    try: r = pickle.load(open(p, 'rb'))
    except Exception: return None
    sC = np.asarray(r['sol_C']); sQ = np.asarray(r['sol_Q_scaled'])
    n_true   = int(r['seg_C']['n_true_all'])
    n_true_c = int(r['seg_C'].get('n_true_clean', n_true))
    n_total  = sC.size
    n_false  = n_total - n_true
    idx_sorted = np.argsort(-sC)
    truth = np.zeros(n_total, dtype=bool); truth[idx_sorted[:n_true]] = True
    out = dict(n_trk=r['n_trk'], rep=r['rep'], readout=r.get('readout',''))
    for tag, s in (('C', sC), ('Q', sQ)):
        if s.max() <= 0:
            out[f'eff_{tag}_rel']=np.nan; out[f'pur_{tag}_rel']=np.nan; out[f'far_{tag}_rel']=np.nan
            continue
        act = s > TAU * s.max()
        n_act  = int(act.sum())
        n_tact = int((act & truth).sum())
        n_fact = n_act - n_tact
        out[f'eff_{tag}_rel'] = n_tact / max(n_true_c, 1)
        out[f'pur_{tag}_rel'] = n_tact / max(n_act, 1)
        out[f'far_{tag}_rel'] = n_fact / max(n_false, 1)
    return out

def load_rel(pdir):
    rows = [_rel_metrics_from_pkl(p) for p in sorted(Path(pdir).glob('event_*.pkl'))]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)

rel_sv = load_rel(PIK_SV)
rel_sm = load_rel(PIK_SM)
agg_rel_sv = aggregate(rel_sv, ['eff_C_rel','eff_Q_rel','pur_C_rel','pur_Q_rel','far_C_rel','far_Q_rel'])
agg_rel_sm = aggregate(rel_sm, ['eff_C_rel','eff_Q_rel','pur_C_rel','pur_Q_rel','far_C_rel','far_Q_rel'])
with pd.option_context('display.float_format','{:0.4f}'.format,'display.width',200):
    print('Relative-threshold metrics, E1:')
    print(agg_rel_sv[['n_trk','n_reps','eff_C_rel_mean','eff_Q_rel_mean','pur_Q_rel_mean','far_Q_rel_mean']].to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
_err(ax, agg_sv,     'eff_Q',     fmt='o--', color=cQ_sv, label=r'E1, $\tau$ absolute', mfc='white')
_err(ax, agg_rel_sv, 'eff_Q_rel', fmt='o-',  color=cQ_sv, label=r'E1, $\tau\cdot\max$')
_err(ax, agg_sm,     'eff_Q',     fmt='s--', color=cQ_sm, label=r'E2, $\tau$ absolute', mfc='white')
_err(ax, agg_rel_sm, 'eff_Q_rel', fmt='s-',  color=cQ_sm, label=r'E2, $\tau\cdot\max$')
_err(ax, agg_rel_sv, 'eff_C_rel', fmt='D:',  color=cC,    label='classical (rel)', mfc='white')
ax.set_ylim(0, 1.05); ax.set_ylabel('Segment Efficiency')
ax.set_title('i) Efficiency: absolute vs relative', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95, fontsize=9); _style_axes(ax)

ax = axes[0, 1]
_err(ax, agg_sv,     'pur_Q',     fmt='o--', color=cQ_sv, label='E1 abs', mfc='white')
_err(ax, agg_rel_sv, 'pur_Q_rel', fmt='o-',  color=cQ_sv, label='E1 rel')
_err(ax, agg_sm,     'pur_Q',     fmt='s--', color=cQ_sm, label='E2 abs', mfc='white')
_err(ax, agg_rel_sm, 'pur_Q_rel', fmt='s-',  color=cQ_sm, label='E2 rel')
_err(ax, agg_rel_sv, 'pur_C_rel', fmt='D:',  color=cC,    label='classical', mfc='white')
ax.set_ylim(0, 1.05); ax.set_ylabel('Segment Purity')
ax.set_title('ii) Purity', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95, fontsize=9); _style_axes(ax)

ax = axes[1, 0]
_err(ax, agg_sv,     'far_Q',     fmt='o--', color=cQ_sv, label='E1 abs', mfc='white')
_err(ax, agg_rel_sv, 'far_Q_rel', fmt='o-',  color=cQ_sv, label='E1 rel')
_err(ax, agg_sm,     'far_Q',     fmt='s--', color=cQ_sm, label='E2 abs', mfc='white')
_err(ax, agg_rel_sm, 'far_Q_rel', fmt='s-',  color=cQ_sm, label='E2 rel')
ax.set_yscale('log'); ax.set_ylabel('Segment False-Activation Rate')
ax.set_title('iii) FAR', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=9); _style_axes(ax)

ax = axes[1, 1]
_err(ax, agg_sv, 'jaccard', fmt='o-', color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'jaccard', fmt='s-', color=cQ_sm, label='E2 sampling')
ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2)
ax.set_ylim(0, 1.05); ax.set_ylabel(r'Jaccard$(A_Q, A_C)$')
ax.set_title('iv) Active-set overlap (abs)', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95, fontsize=9); _style_axes(ax)

for ax in axes.flat:
    ax.set_xlabel('Number of Tracks'); ax.set_xscale('log')

fig.tight_layout()
_save_paper(fig, 'fig1c_threshold_fix')
plt.show()
"""
nb['cells'][12] = code(FIG1C)

FIG2 = r"""# §8 Figure 2 — solution-vector fidelity (§14e paper style)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
_err(ax, agg_sv, 'cosine', fmt='o-', color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'cosine', fmt='s-', color=cQ_sm, label='E2 sampling')
ax.axhline(0.95, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2,
           label='0.95 reference')
ax.set_ylim(0, 1.05); ax.set_ylabel(r'$\cos(s_Q, s_C)$')
ax.set_title('i) Cosine similarity', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95); _style_axes(ax)

ax = axes[1]
_err(ax, agg_sv, 'rel_l2', fmt='o-', color=cQ_sv, label='E1 statevector')
_err(ax, agg_sm, 'rel_l2', fmt='s-', color=cQ_sm, label='E2 sampling')
ax.set_ylabel(r'$\|s_Q - s_C\|_2 / \|s_C\|_2$')
ax.set_title(r'ii) Relative $L_2$ error', fontweight='bold')
ax.legend(framealpha=0.95); _style_axes(ax)

ax = axes[2]
_err(ax, agg_sv, 'P_anc', fmt='o-', color=cQ_sv, label='measured (E1)')
_err(ax, agg_sm, 'P_anc', fmt='s-', color=cQ_sm, label='measured (E2)')
if not agg_sv.empty:
    ax.plot(agg_sv['n_trk'], 1.0/agg_sv['n_seg_mean'],
            color='gray', ls=':', lw=1.5, label=r'$1/n_\mathrm{seg}$')
ax.set_yscale('log'); ax.set_ylabel(r'$P(\mathrm{anc}{=}1)$')
ax.set_title('iii) Ancilla post-selection', fontweight='bold')
ax.legend(framealpha=0.95); _style_axes(ax)

for ax in axes:
    ax.set_xscale('log'); ax.set_xlabel('Number of Tracks')

fig.tight_layout()
_save_paper(fig, 'fig2_solution_fidelity')
plt.show()
"""
nb['cells'][14] = code(FIG2)

FIG3 = r"""# §9 Figure 3 — downstream track reconstruction (§14e paper style)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
_err(ax, agg_sv, 'trk_Clay_efficiency', fmt='D--', color=cC,    label='classical (layered)', mfc='white')
_err(ax, agg_sv, 'trk_Qlay_efficiency', fmt='o-',  color=cQ_sv, label='E1 layered')
_err(ax, agg_sm, 'trk_Qlay_efficiency', fmt='s-',  color=cQ_sm, label='E2 layered')
_err(ax, agg_sv, 'trk_Qcc_efficiency',  fmt='o:',  color=cQ_sv, label='E1 cc', mfc='white')
_err(ax, agg_sm, 'trk_Qcc_efficiency',  fmt='s:',  color=cQ_sm, label='E2 cc', mfc='white')
ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2)
ax.set_ylim(0, 1.05); ax.set_ylabel('Track Efficiency')
ax.set_title('i) Track Efficiency', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95, fontsize=9, ncol=2); _style_axes(ax)

ax = axes[0, 1]
_err(ax, agg_sv, 'trk_Clay_ghost_rate', fmt='D--', color=cC,    label='classical (layered)', mfc='white')
_err(ax, agg_sv, 'trk_Qlay_ghost_rate', fmt='o-',  color=cQ_sv, label='E1 layered')
_err(ax, agg_sm, 'trk_Qlay_ghost_rate', fmt='s-',  color=cQ_sm, label='E2 layered')
_err(ax, agg_sv, 'trk_Qcc_ghost_rate',  fmt='o:',  color=cQ_sv, label='E1 cc', mfc='white')
_err(ax, agg_sm, 'trk_Qcc_ghost_rate',  fmt='s:',  color=cQ_sm, label='E2 cc', mfc='white')
ax.set_ylabel('Ghost Rate')
ax.set_title('ii) Ghost Rate', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2); _style_axes(ax)

ax = axes[1, 0]
_err(ax, agg_sv, 'trk_Clay_hit_efficiency', fmt='D--', color=cC,    label='classical (layered)', mfc='white')
_err(ax, agg_sv, 'trk_Qlay_hit_efficiency', fmt='o-',  color=cQ_sv, label='E1 layered')
_err(ax, agg_sm, 'trk_Qlay_hit_efficiency', fmt='s-',  color=cQ_sm, label='E2 layered')
ax.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2)
ax.set_ylim(0, 1.05); ax.set_ylabel('Hit Efficiency')
ax.set_title('iii) Hit Efficiency', fontweight='bold')
ax.legend(loc='lower left', framealpha=0.95, fontsize=9); _style_axes(ax)

ax = axes[1, 1]
if not agg_sv.empty:
    ax.plot(agg_sv['n_trk'], agg_sv['n_trk'], color='gray', ls=':', lw=1.5,
            label=r'ideal $n_\mathrm{rec}=n_\mathrm{trk}$')
_err(ax, agg_sv, 'trk_Clay_n_rec_tracks', fmt='D--', color=cC,    label='classical (layered)', mfc='white')
_err(ax, agg_sv, 'trk_Qlay_n_rec_tracks', fmt='o-',  color=cQ_sv, label='E1 layered')
_err(ax, agg_sm, 'trk_Qlay_n_rec_tracks', fmt='s-',  color=cQ_sm, label='E2 layered')
ax.set_yscale('log'); ax.set_ylabel(r'$n_\mathrm{rec\,tracks}$')
ax.set_title('iv) Reconstructed-Track Count', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95, fontsize=9); _style_axes(ax)

for ax in axes.flat:
    ax.set_xlabel('Number of Tracks'); ax.set_xscale('log')

fig.tight_layout()
_save_paper(fig, 'fig3_tracking')
plt.show()
"""
nb['cells'][16] = code(FIG3)

FIG4 = r"""# §10 Figure 4 — timing and problem-size scaling (§14e paper style)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
for agg, lbl, fmt, col, mfc in [
        (agg_sv, 'classical (per-job mean)',  'D--', cC,    'white'),
        (agg_sv, 'E1 statevector',            'o-',  cQ_sv, None),
        (agg_sm, 'E2 sampling',               's-',  cQ_sm, None)]:
    if agg.empty: continue
    xs = agg['n_seg_mean']
    if 'classical' in lbl:
        ys, ye = agg['t_classical_mean'], agg['t_classical_sem']
    else:
        ys, ye = agg['t_quantum_mean'],   agg['t_quantum_sem']
    kw = dict(fmt=fmt, color=col, capsize=5, capthick=1.5,
              markeredgecolor='black', markeredgewidth=0.8, zorder=3, label=lbl)
    if mfc is not None: kw['mfc'] = mfc
    ax.errorbar(xs, ys, yerr=ye, **kw)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$n_\mathrm{seg}$'); ax.set_ylabel('Wall Time (s)')
ax.set_title('i) Solve time vs problem size', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95); _style_axes(ax)

ax = axes[1]
if not agg_sv.empty:
    T  = agg_sv['n_trk'].values
    ns = agg_sv['n_seg_mean'].values
    ax.plot(T, ns, 'o-', color=cQ_sv, markeredgecolor='black',
            markeredgewidth=0.8, zorder=3, label=r'measured $n_\mathrm{seg}$')
    ref = ns[0] * (T / T[0])**2
    ax.plot(T, ref, color='gray', ls=':', lw=1.5, label=r'$\propto T^2$ reference')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Number of Tracks'); ax.set_ylabel(r'$n_\mathrm{seg}$')
ax.set_title('ii) Problem-size scaling', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.95); _style_axes(ax)

fig.tight_layout()
_save_paper(fig, 'fig4_timing_scaling')
plt.show()
"""
nb['cells'][18] = code(FIG4)

P.write_text(json.dumps(nb, indent=1))
print(f'restyled {len(nb["cells"])} cells to §14e paper style')
