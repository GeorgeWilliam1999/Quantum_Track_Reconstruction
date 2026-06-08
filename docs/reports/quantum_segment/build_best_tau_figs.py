"""Best-tau analysis: absolute AND relative tau conventions side by side.

Generates into both report figs/ directories:
  fig7_abs_vs_rel_tau.pdf      -- FR vs T for absolute tau vs relative tau (panels)
  fig8_pareto_per_T.pdf        -- Pareto (eff, purity) per T, with default & best markers
  fig9_best_tau_vs_T.pdf       -- best relative tau as a function of T
  fig10_best_tau_metrics.pdf   -- eff and FR at best-tau, head-to-head with classical

Conventions:
  * ABSOLUTE: act = sol_Q_scaled > tau                  (= operational default at tau=0.35)
  * RELATIVE: act = sol_Q_scaled > tau * sol_Q_scaled.max()  (= notebook §7e sweep)
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker

BASE   = Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/Toy_Characterisation/Verify_new_results/outputs/quantum_segment_analysis')
PIK_SV = BASE / 'seg14e_T1000_statevector' / 'pickles'

OUT_DIRS = [
    Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/docs/reports/quantum_segment/figs'),
    Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/docs/reports/quantum_vs_classical_segment/figs'),
]
for d in OUT_DIRS: d.mkdir(parents=True, exist_ok=True)

TAU_REL = np.round(np.arange(0.05, 0.96, 0.025), 4)
TAU_ABS = np.round(np.concatenate([np.arange(0.10, 1.01, 0.05),
                                   np.arange(1.20, 5.01, 0.20)]), 3)

def per_event(p, taus_abs, taus_rel):
    try: r = pickle.load(open(p, 'rb'))
    except Exception: return []
    sC = np.asarray(r['sol_C']); sQ = np.asarray(r['sol_Q_scaled'])
    n_true   = int(r['seg_C']['n_true_all'])
    n_true_c = int(r['seg_C'].get('n_true_clean', n_true))
    n_total  = sC.size
    truth = np.zeros(n_total, dtype=bool)
    truth[np.argsort(-sC)[:n_true]] = True
    qmax = float(sQ.max()) if sQ.size else 0.0
    n_trk, rep = int(r['n_trk']), int(r['rep'])
    out = []
    for tau in taus_abs:
        act = sQ > tau
        nA = int(act.sum()); nT = int((act & truth).sum()); nF = nA - nT
        out.append(dict(kind='abs', n_trk=n_trk, rep=rep, tau=float(tau), qmax=qmax,
                        eff=nT/max(n_true_c,1), pur=nT/max(nA,1),
                        fr=nF/max(nA,1), n_act=nA))
    if qmax > 0:
        for tau in taus_rel:
            act = sQ > tau * qmax
            nA = int(act.sum()); nT = int((act & truth).sum()); nF = nA - nT
            out.append(dict(kind='rel', n_trk=n_trk, rep=rep, tau=float(tau), qmax=qmax,
                            eff=nT/max(n_true_c,1), pur=nT/max(nA,1),
                            fr=nF/max(nA,1), n_act=nA))
    return out

print('[*] Scanning pickles ...')
rows = []
for p in sorted(PIK_SV.glob('event_*.pkl')):
    rows.extend(per_event(p, TAU_ABS, TAU_REL))
df = pd.DataFrame(rows)
print(f'    rows: {len(df)}')

def agg(df):
    sem = lambda x: (x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else 0.0
    g = (df.groupby(['kind','n_trk','tau'])
           .agg(eff=('eff','mean'), eff_s=('eff', sem),
                pur=('pur','mean'), pur_s=('pur', sem),
                fr =('fr','mean'),  fr_s =('fr', sem),
                qmax=('qmax','mean'), n_reps=('rep','count'))
           .reset_index())
    g['f1'] = 2*g['eff']*g['pur'] / (g['eff']+g['pur']).clip(lower=1e-12)
    return g
g = agg(df)
g_abs = g[g['kind']=='abs'].drop(columns='kind').copy()
g_rel = g[g['kind']=='rel'].drop(columns='kind').copy()

agg_csv = pd.read_csv(BASE / 'seg14e_T1000_statevector' / 'aggregate.csv')
ref_C = agg_csv[['n_trk','eff_C_mean','eff_C_sem','pur_C_mean','pur_C_sem']].copy()
ref_C['fr_C']     = 1 - ref_C['pur_C_mean']
ref_C['fr_C_sem'] = ref_C['pur_C_sem']
ref_C = ref_C.sort_values('n_trk').reset_index(drop=True)

def best_tau(g, criterion, **kw):
    out = []
    for t, sub in g.groupby('n_trk'):
        if criterion == 'f1':
            best = sub.iloc[sub['f1'].argmax()]
        elif criterion == 'eff_floor':
            ok = sub[sub['eff'] >= kw['eff_floor']]
            best = ok.iloc[ok['fr'].argmin()] if len(ok) else sub.iloc[sub['eff'].argmax()]
        elif criterion == 'dom_classical':
            tgt = float(kw['ref'].loc[kw['ref']['n_trk']==t, 'fr_C'].iloc[0])
            ok = sub[sub['fr'] <= tgt]
            best = ok.iloc[ok['eff'].argmax()] if len(ok) else sub.iloc[sub['fr'].argmin()]
        else: raise ValueError(criterion)
        out.append(dict(n_trk=int(t), tau=float(best['tau']),
                        eff=float(best['eff']), eff_s=float(best['eff_s']),
                        fr=float(best['fr']),   fr_s=float(best['fr_s']),
                        pur=float(best['pur']), f1=float(best['f1']),
                        qmax=float(best['qmax'])))
    return pd.DataFrame(out).sort_values('n_trk').reset_index(drop=True)

opt_rel_f1  = best_tau(g_rel, 'f1')
opt_rel_eff = best_tau(g_rel, 'eff_floor', eff_floor=0.90)
opt_rel_dom = best_tau(g_rel, 'dom_classical', ref=ref_C)

print('\n=== Best RELATIVE tau (max F1) ==='); print(opt_rel_f1.to_string(index=False, float_format='%.3f'))
print('\n=== Best RELATIVE tau (eff >= 0.90) ==='); print(opt_rel_eff.to_string(index=False, float_format='%.3f'))
print('\n=== Best RELATIVE tau (dominate classical FR) ==='); print(opt_rel_dom.to_string(index=False, float_format='%.3f'))

RPT = Path('/data/bfys/gscriven/Quantum_Track_Reconstruction/docs/reports/quantum_segment')
g_abs.to_csv(RPT/'tau_sweep_abs.csv', index=False)
g_rel.to_csv(RPT/'tau_sweep_rel.csv', index=False)
opt_rel_f1 .to_csv(RPT/'best_tau_rel_f1.csv',  index=False)
opt_rel_eff.to_csv(RPT/'best_tau_rel_eff.csv', index=False)
opt_rel_dom.to_csv(RPT/'best_tau_rel_dom.csv', index=False)

TAU_R_SHOW = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]
def piv(g, col, taus):
    return (g[g['tau'].isin(taus)].pivot(index='n_trk', columns='tau', values=col)*100).round(2)
print('\n=== RELATIVE tau, False Rate (%) ===')
print(piv(g_rel, 'fr', TAU_R_SHOW).to_string())
print('\n=== RELATIVE tau, Efficiency (%) ===')
print(piv(g_rel, 'eff', TAU_R_SHOW).to_string())
print('\n=== ABSOLUTE tau, False Rate (%) ===')
print(piv(g_abs, 'fr', [0.20, 0.35, 0.50, 1.00, 2.00, 3.00, 4.00, 5.00]).to_string())
print('\n=== mean qmax(sol_Q_scaled) per T ===')
print(g_rel.groupby('n_trk')['qmax'].mean().round(3).to_string())

plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
                     'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False})
cC = '#1f77b4'
def save_all(fig, stem):
    for d in OUT_DIRS:
        fig.savefig(d/f'{stem}.pdf', bbox_inches='tight')
        fig.savefig(d/f'{stem}.png', bbox_inches='tight', dpi=160)

# Fig 7
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
TAU_A_PLOT = [0.35, 0.50, 1.00, 2.00, 3.00, 4.00]
TAU_R_PLOT = [0.20, 0.30, 0.35, 0.40, 0.50, 0.60]
cA = cm.plasma(np.linspace(0.08, 0.9, len(TAU_A_PLOT)))
cR = cm.viridis(np.linspace(0.08, 0.9, len(TAU_R_PLOT)))
for ax, taulist, colors, lbl, ttl in [
    (axes[0], TAU_A_PLOT, cA, r'\tau_{\mathrm{abs}}', r'(a) Absolute $\tau$ on $\mathrm{sol\_Q\_scaled}$'),
    (axes[1], TAU_R_PLOT, cR, r'\tau_{\mathrm{rel}}', r'(b) Relative $\tau \cdot q_{\max}$ on $\mathrm{sol\_Q\_scaled}$'),
]:
    src = g_abs if ax is axes[0] else g_rel
    for tau, col in zip(taulist, colors):
        sub = src[np.isclose(src['tau'], tau)].sort_values('n_trk')
        if sub.empty: continue
        ax.errorbar(sub['n_trk'], sub['fr']*100, yerr=sub['fr_s']*100,
                    fmt='o-', color=col, lw=1.5, ms=6, capsize=3,
                    markeredgecolor='black', markeredgewidth=0.5,
                    label=fr'${lbl}={tau:g}$', zorder=3)
    ax.errorbar(ref_C['n_trk'], ref_C['fr_C']*100, yerr=ref_C['fr_C_sem']*100,
                fmt='D--', color=cC, mfc='white', capsize=4,
                markeredgecolor='black', markeredgewidth=0.8,
                label='Classical', zorder=4)
    ax.set_xscale('log'); ax.set_xlabel('Number of Tracks $T$')
    ax.set_ylabel('Segment False Rate (%)')
    ax.set_title(ttl, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.legend(loc='upper left', ncol=2, fontsize=8, framealpha=0.95); ax.grid(alpha=0.3)
fig.suptitle('Absolute vs relative threshold conventions on the quantum amplitudes',
             y=1.02, fontweight='bold')
fig.tight_layout()
save_all(fig, 'fig7_abs_vs_rel_tau'); plt.close(fig)

# Fig 8 - Pareto
T_LIST = [10, 50, 100, 200, 500, 1000]
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
norm = plt.Normalize(vmin=float(TAU_REL.min()), vmax=float(TAU_REL.max()))
sm_  = cm.ScalarMappable(cmap='viridis', norm=norm); sm_.set_array([])
for ax, T in zip(axes.flat, T_LIST):
    sub = g_rel[g_rel['n_trk']==T].sort_values('tau')
    if sub.empty:
        ax.set_title(f'$T={T}$ (no data)'); continue
    ax.plot(sub['eff']*100, sub['pur']*100, '-', color='k', lw=0.7, alpha=0.45, zorder=2)
    ax.scatter(sub['eff']*100, sub['pur']*100, c=sub['tau'],
               cmap='viridis', vmin=norm.vmin, vmax=norm.vmax,
               s=40, edgecolor='black', linewidth=0.4, zorder=3)
    i35 = int((sub['tau']-0.35).abs().values.argmin())
    ax.scatter(sub['eff'].iloc[i35]*100, sub['pur'].iloc[i35]*100, marker='s',
               facecolor='none', edgecolor='#d62728', linewidth=2.0, s=170, zorder=5,
               label=r'default $\tau_{\mathrm{rel}}=0.35$')
    bf1 = opt_rel_f1[opt_rel_f1['n_trk']==T]
    if not bf1.empty:
        ax.scatter(bf1['eff']*100, (1-bf1['fr'])*100, marker='*',
                   color='gold', edgecolor='black', linewidth=0.8, s=320, zorder=6,
                   label=fr'$\tau^\star_{{F_1}}={float(bf1["tau"].iloc[0]):.2f}$')
    rc = ref_C[ref_C['n_trk']==T]
    if not rc.empty:
        ax.scatter(rc['eff_C_mean']*100, (1-rc['fr_C'])*100, marker='D',
                   facecolor='white', edgecolor=cC, linewidth=1.8, s=110, zorder=5,
                   label='Classical')
    ax.set_title(f'$T = {T}$', fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.95); ax.grid(alpha=0.3)
for ax in axes[-1, :]: ax.set_xlabel('Segment Efficiency (%)')
for ax in axes[:, 0]:  ax.set_ylabel('Segment Purity = $1-\\mathrm{FR}$ (%)')
fig.colorbar(sm_, ax=axes, fraction=0.025, pad=0.02, label=r'relative $\tau$')
fig.suptitle(r'Per-$T$ trade-off (eff vs purity); colour = relative $\tau$', y=0.998, fontweight='bold')
save_all(fig, 'fig8_pareto_per_T'); plt.close(fig)

# Fig 9
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(opt_rel_f1 ['n_trk'], opt_rel_f1 ['tau'], 'o-', color='#2ca02c', lw=1.8, ms=8,
        markeredgecolor='black', markeredgewidth=0.7,
        label=r'$\tau^\star_{F_1}$ (max $F_1$ of eff & purity)')
ax.plot(opt_rel_eff['n_trk'], opt_rel_eff['tau'], 's-', color='#9467bd', lw=1.8, ms=8,
        markeredgecolor='black', markeredgewidth=0.7,
        label=r'$\tau^\star_{\mathrm{eff}\geq 0.90}$ (min FR s.t.\ eff $\geq$ 0.90)')
ax.plot(opt_rel_dom['n_trk'], opt_rel_dom['tau'], '^-', color='#d62728', lw=1.8, ms=8,
        markeredgecolor='black', markeredgewidth=0.7,
        label=r'$\tau^\star_{\mathrm{dom\;C}}$ (FR $\leq$ classical, max eff)')
ax.axhline(0.35, color='k', ls=':', lw=1.2, alpha=0.7, label=r'default $\tau_{\mathrm{rel}}=0.35$')
ax.set_xscale('log'); ax.set_xlabel('Number of Tracks $T$')
ax.set_ylabel(r'best relative $\tau$')
ax.set_title(r'Optimal relative $\tau$ as a function of $T$ (E1 statevector)', fontweight='bold')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
save_all(fig, 'fig9_best_tau_vs_T'); plt.close(fig)

# Fig 10
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sub35 = g_rel[np.isclose(g_rel['tau'], 0.35)].sort_values('n_trk')
ax = axes[0]
ax.errorbar(sub35['n_trk'], sub35['eff']*100, yerr=sub35['eff_s']*100,
            fmt='x:', color='gray', capsize=3, ms=7, lw=1.2,
            label=r'Quantum @ default $\tau_{\mathrm{rel}}=0.35$')
ax.errorbar(opt_rel_f1['n_trk'], opt_rel_f1['eff']*100, yerr=opt_rel_f1['eff_s']*100,
            fmt='o-', color='#2ca02c', capsize=4, ms=7, lw=1.8,
            markeredgecolor='black', markeredgewidth=0.7,
            label=r'Quantum @ $\tau^\star_{F_1}$')
ax.errorbar(opt_rel_dom['n_trk'], opt_rel_dom['eff']*100, yerr=opt_rel_dom['eff_s']*100,
            fmt='^-', color='#d62728', capsize=4, ms=7, lw=1.8,
            markeredgecolor='black', markeredgewidth=0.7,
            label=r'Quantum @ $\tau^\star_{\mathrm{dom\,C}}$')
ax.errorbar(ref_C['n_trk'], ref_C['eff_C_mean']*100, yerr=ref_C['eff_C_sem']*100,
            fmt='D--', color=cC, mfc='white', capsize=5, ms=8,
            markeredgecolor='black', markeredgewidth=0.8, label='Classical')
ax.set_xscale('log'); ax.set_xlabel('Number of Tracks $T$')
ax.set_ylabel('Segment Efficiency (%)')
ax.set_title(r'(a) Efficiency at best $\tau_{\mathrm{rel}}$', fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='lower left', fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(sub35['n_trk'], sub35['fr']*100, yerr=sub35['fr_s']*100,
            fmt='x:', color='gray', capsize=3, ms=7, lw=1.2,
            label=r'Quantum @ default $\tau_{\mathrm{rel}}=0.35$')
ax.errorbar(opt_rel_f1['n_trk'], opt_rel_f1['fr']*100, yerr=opt_rel_f1['fr_s']*100,
            fmt='o-', color='#2ca02c', capsize=4, ms=7, lw=1.8,
            markeredgecolor='black', markeredgewidth=0.7,
            label=r'Quantum @ $\tau^\star_{F_1}$')
ax.errorbar(opt_rel_dom['n_trk'], opt_rel_dom['fr']*100, yerr=opt_rel_dom['fr_s']*100,
            fmt='^-', color='#d62728', capsize=4, ms=7, lw=1.8,
            markeredgecolor='black', markeredgewidth=0.7,
            label=r'Quantum @ $\tau^\star_{\mathrm{dom\,C}}$')
ax.errorbar(ref_C['n_trk'], ref_C['fr_C']*100, yerr=ref_C['fr_C_sem']*100,
            fmt='D--', color=cC, mfc='white', capsize=5, ms=8,
            markeredgecolor='black', markeredgewidth=0.8, label='Classical')
ax.set_xscale('log'); ax.set_xlabel('Number of Tracks $T$')
ax.set_ylabel('Segment False Rate (%)')
ax.set_title(r'(b) False rate at best $\tau_{\mathrm{rel}}$', fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)

fig.tight_layout()
save_all(fig, 'fig10_best_tau_metrics'); plt.close(fig)

print('\n[*] Wrote figures fig7–fig10 to:')
for d in OUT_DIRS: print('   ', d)
