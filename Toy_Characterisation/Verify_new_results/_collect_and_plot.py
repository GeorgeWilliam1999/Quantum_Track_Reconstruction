"""Aggregate seg14e_T1000 results and produce the E1/E2 plots."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DROP_RATE = 0.01
GAMMA = 3.0
cC, cQ = '#1f77b4', '#d62728'
OUT  = Path('outputs/quantum_segment_analysis')

def _eff(seg):
    n = seg.get('n_true_clean') or seg.get('n_true_all')
    return float(seg['n_true_active']) / float(n) if n else np.nan

def _pur(seg):
    n = seg.get('n_active')
    return float(seg['n_true_active']) / float(n) if n else np.nan

def _far(seg):
    n = seg.get('n_false_all')
    return float(seg['n_false_active']) / float(n) if n else np.nan

def _seg_metrics_from_pkl(p):
    try:
        with open(p, 'rb') as f:
            r = pickle.load(f)
    except Exception:
        return None
    sC, sQ = r.get('seg_C'), r.get('seg_Q')
    if not sC or not sQ:
        return None
    fid = r.get('fidelity', {}) or {}
    return dict(
        n_trk=r['n_trk'], rep=r['rep'], shots=r.get('shots', 0),
        readout=r.get('readout', ''), drop_rate=r.get('drop_rate', DROP_RATE),
        n_seg=r.get('n_seg', np.nan),
        cosine=float(fid.get('cos', np.nan)),
        rel_l2=float(fid.get('rel_l2', np.nan)),
        jaccard=float(fid.get('jaccard', np.nan)),
        P_anc=float(r.get('success_prob', np.nan)),
        eff_C=_eff(sC), eff_Q=_eff(sQ),
        pur_C=_pur(sC), pur_Q=_pur(sQ),
        far_C=_far(sC), far_Q=_far(sQ),
        t_q=float(r.get('t_quantum', np.nan)),
    )

def load_results(d):
    rows = [_seg_metrics_from_pkl(p) for p in sorted(Path(d).glob('*.pkl'))]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows).sort_values(['n_trk','rep']).reset_index(drop=True) if rows else pd.DataFrame()

def aggregate(df):
    if df.empty: return df
    sem = lambda x: x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else 0.0
    return df.groupby('n_trk').agg(
        n_reps=('rep','count'),
        n_seg_mean=('n_seg','mean'),
        eff_C_mean=('eff_C','mean'), eff_C_sem=('eff_C', sem),
        eff_Q_mean=('eff_Q','mean'), eff_Q_sem=('eff_Q', sem),
        pur_C_mean=('pur_C','mean'), pur_C_sem=('pur_C', sem),
        pur_Q_mean=('pur_Q','mean'), pur_Q_sem=('pur_Q', sem),
        far_Q_mean=('far_Q','mean'), far_Q_sem=('far_Q', sem),
        cos_mean=('cosine','mean'),  cos_sem=('cosine', sem),
        P_mean=('P_anc','mean'),     P_sem=('P_anc', sem),
        shots_mean=('shots','mean'),
    ).reset_index().sort_values('n_trk')

def plot_2x2(agg, *, title, save_to):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    T = agg['n_trk'].values
    ax = axes[0,0]
    ax.errorbar(T, agg.eff_C_mean, yerr=agg.eff_C_sem, fmt='s--', color=cC, capsize=3, label='classical')
    ax.errorbar(T, agg.eff_Q_mean, yerr=agg.eff_Q_sem, fmt='o-',  color=cQ, capsize=3, label='OneBitHHL')
    ax.axhline(1.0, color='k', ls=':', lw=0.7, alpha=0.5)
    ax.set_ylabel('segment efficiency'); ax.set_ylim(0,1.05); ax.set_title('(a) Segment efficiency')
    ax.grid(alpha=0.3); ax.legend()
    ax = axes[0,1]
    ax.errorbar(T, agg.pur_C_mean, yerr=agg.pur_C_sem, fmt='s--', color=cC, capsize=3, label='classical')
    ax.errorbar(T, agg.pur_Q_mean, yerr=agg.pur_Q_sem, fmt='o-',  color=cQ, capsize=3, label='OneBitHHL')
    ax.set_ylabel('segment purity'); ax.set_ylim(0,1.05); ax.set_title('(b) Segment purity')
    ax.grid(alpha=0.3); ax.legend()
    ax = axes[1,0]
    ax.errorbar(T, agg.cos_mean, yerr=agg.cos_sem, fmt='o-', color=cQ, capsize=3)
    ax.axhline(0.95, color='k', ls=':', lw=0.7, alpha=0.5, label='fidelity gate (0.95)')
    ax.set_xlabel(r'$n_\mathrm{trk}$'); ax.set_ylabel(r'$\cos(s_Q,s_C)$')
    ax.set_ylim(0,1.05); ax.set_title('(c) Solution agreement')
    ax.grid(alpha=0.3, which='both'); ax.legend()
    ax = axes[1,1]
    ax.errorbar(T, agg.P_mean, yerr=agg.P_sem, fmt='o-', color=cQ, capsize=3, label='measured')
    ax.plot(T, 1.0/agg.n_seg_mean, 'k:', alpha=0.6, label=r'$1/n_\mathrm{seg}$')
    ax.set_xlabel(r'$n_\mathrm{trk}$'); ax.set_ylabel(r'$P(\mathrm{anc}{=}1)$')
    ax.set_yscale('log'); ax.set_title('(d) Post-selection probability')
    ax.grid(alpha=0.3, which='both'); ax.legend()
    for ax in axes.flat: ax.set_xscale('log')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_to)+'.pdf', bbox_inches='tight')
    fig.savefig(str(save_to)+'.png', dpi=200, bbox_inches='tight')
    print(f'  saved -> {save_to}.pdf / .png')
    plt.close(fig)

for label, sub in [('E1 statevector', 'seg14e_T1000_statevector'),
                   ('E2 sampling',    'seg14e_T1000_sampling')]:
    pdir = OUT / sub / 'pickles'
    df = load_results(pdir)
    print(f'\n=== {label} : {len(df)} pickles loaded from {pdir} ===')
    if df.empty:
        print('  (empty)'); continue
    print('per-T counts :')
    print(df.groupby('n_trk').size().rename('n').to_string())
    agg = aggregate(df)
    agg_csv = OUT / sub / 'aggregate.csv'
    agg.to_csv(agg_csv, index=False)
    print(f'  aggregate -> {agg_csv}')
    with pd.option_context('display.float_format','{:0.4f}'.format,'display.width',200):
        cols = ['n_trk','n_reps','n_seg_mean','eff_C_mean','eff_Q_mean','pur_Q_mean','cos_mean','P_mean','shots_mean']
        print(agg[cols].to_string(index=False))
    fig_stem = OUT / sub / f'fig_{sub}'
    title = (rf'{label} - $\gamma{{=}}3$, drop $=1\%$, $\varepsilon{{=}}2$ mrad, '
             rf'conv$=$False; $n_\mathrm{{trk}}\leq 1000$')
    plot_2x2(agg, title=title, save_to=fig_stem)
