"""Insert §7e threshold-sweep markdown + code cells before §11."""
import json, sys
from pathlib import Path

NB = Path('Quantum_segment_level_analysis.ipynb')
nb = json.load(open(NB))

MD = r"""## §7e Threshold sweep — does tuning $\tau$ recover classical-like False Rate?

The data above shows the false-rate gap is **not** a sampling artefact:
E1 statevector and E2 sampling track each other to within $\sim 1\%$
at every $T$, so 10x shots will not close the gap.

What may help is the activation threshold itself. We already use a
relative rule $s_i^{\rm active} \Leftrightarrow s_i > \tau\cdot\max_j s_j$
inside §7c, but only at the single fixed value $\tau = 0.35$. Below we
sweep $\tau \in \{0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70\}$
directly on the saved $s_Q^{\rm scaled}$ arrays for both readouts and plot:

- **i)** Quantum Segment Efficiency vs $T$ (one curve per $\tau$)
- **ii)** Quantum Segment **False Rate** ($N_\mathrm{false\,act}/N_\mathrm{accepted}$) vs $T$
- **iii)** ROC-style trade-off: False Rate vs Efficiency at fixed $T$
- **iv)** Optimal $\tau^\star(T)$ — argmin of false rate subject to eff $\geq 0.9$

Classical (at the default $\tau=0.35$) is overlaid for reference.
"""

PY1 = r'''# §7e Threshold sweep on existing E1 / E2 pickles (no condor needed)
TAU_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

def _tau_sweep_one(p, taus=TAU_GRID):
    try: r = pickle.load(open(p, 'rb'))
    except Exception: return None
    sC = np.asarray(r['sol_C']); sQ = np.asarray(r['sol_Q_scaled'])
    n_true   = int(r['seg_C']['n_true_all'])
    n_true_c = int(r['seg_C'].get('n_true_clean', n_true))
    n_total  = sC.size
    n_false  = n_total - n_true
    truth = np.zeros(n_total, dtype=bool)
    truth[np.argsort(-sC)[:n_true]] = True
    rows = []
    if sQ.max() <= 0:
        return rows
    qmax = sQ.max()
    for tau in taus:
        act    = sQ > tau * qmax
        n_act  = int(act.sum())
        n_tact = int((act & truth).sum())
        n_fact = n_act - n_tact
        rows.append(dict(
            n_trk=r['n_trk'], rep=r['rep'], tau=tau,
            eff = n_tact / max(n_true_c, 1),
            fr  = n_fact / max(n_act,    1),
            far = n_fact / max(n_false,  1),
            n_act=n_act, n_tact=n_tact, n_fact=n_fact,
        ))
    return rows

def load_tau_sweep(pdir):
    out = []
    for p in sorted(Path(pdir).glob('event_*.pkl')):
        rs = _tau_sweep_one(p)
        if rs: out.extend(rs)
    return pd.DataFrame(out)

tau_sv = load_tau_sweep(PIK_SV); print(f'E1 tau-sweep rows: {len(tau_sv)}')
tau_sm = load_tau_sweep(PIK_SM); print(f'E2 tau-sweep rows: {len(tau_sm)}')

def _agg_tau(df):
    sem = lambda x: (x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else 0.0
    g = df.groupby(['n_trk','tau']).agg(
        eff_mean=('eff','mean'), eff_sem=('eff', sem),
        fr_mean =('fr', 'mean'), fr_sem =('fr',  sem),
        far_mean=('far','mean'), far_sem=('far', sem),
        n_reps  =('rep','count'),
    ).reset_index()
    return g

g_tau_sv = _agg_tau(tau_sv)
g_tau_sm = _agg_tau(tau_sm)
print('\n=== E1 statevector — Segment False Rate (%) vs (T, tau) ===')
piv = (g_tau_sv.pivot(index='n_trk', columns='tau', values='fr_mean') * 100).round(2)
print(piv.to_string())
print('\n=== E1 statevector — Segment Efficiency (%) vs (T, tau) ===')
piv = (g_tau_sv.pivot(index='n_trk', columns='tau', values='eff_mean') * 100).round(2)
print(piv.to_string())
'''

PY2 = r'''# §7e Plots — efficiency / false-rate vs T per tau, ROC, optimal tau(T)
import matplotlib.cm as cm
TAU_COLORS = cm.viridis(np.linspace(0.05, 0.95, len(TAU_GRID)))

ref_eff_C = agg_sv[['n_trk','eff_C_mean','eff_C_sem']].copy()
ref_fr_C  = agg_sv[['n_trk','pur_C_mean','pur_C_sem']].copy()
ref_fr_C['fr_C_mean'] = 1.0 - ref_fr_C['pur_C_mean']
ref_fr_C['fr_C_sem']  = ref_fr_C['pur_C_sem']

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# ----- i) Efficiency vs T, one curve per tau (E1) -----
ax = axes[0, 0]
for tau, col in zip(TAU_GRID, TAU_COLORS):
    sub = g_tau_sv[g_tau_sv['tau'] == tau].sort_values('n_trk')
    ax.errorbar(sub['n_trk'], sub['eff_mean']*100, yerr=sub['eff_sem']*100,
                fmt='o-', color=col, capsize=3, capthick=1.0, lw=1.5, ms=6,
                markeredgecolor='black', markeredgewidth=0.6, zorder=3,
                label=fr'$\tau={tau:.2f}$')
ax.errorbar(ref_eff_C['n_trk'], ref_eff_C['eff_C_mean']*100,
            yerr=ref_eff_C['eff_C_sem']*100, fmt='D--', color=cC, mfc='white',
            capsize=4, capthick=1.2, markeredgecolor='black', markeredgewidth=0.8,
            zorder=4, label='Classical')
ax.axhline(100, color='gray', ls='--', lw=1.2, alpha=0.7, zorder=2)
ax.set_xscale('log'); ax.set_xlabel('Number of Tracks')
ax.set_ylabel('Segment Efficiency (%)')
ax.set_title(r'i) Efficiency vs $T$ — E1 statevector, $\tau$-sweep',
             fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='lower left', framealpha=0.95, fontsize=8, ncol=2); _style_axes(ax)

# ----- ii) False Rate vs T, one curve per tau (E1) -----
ax = axes[0, 1]
for tau, col in zip(TAU_GRID, TAU_COLORS):
    sub = g_tau_sv[g_tau_sv['tau'] == tau].sort_values('n_trk')
    ax.errorbar(sub['n_trk'], sub['fr_mean']*100, yerr=sub['fr_sem']*100,
                fmt='o-', color=col, capsize=3, capthick=1.0, lw=1.5, ms=6,
                markeredgecolor='black', markeredgewidth=0.6, zorder=3,
                label=fr'$\tau={tau:.2f}$')
ax.errorbar(ref_fr_C['n_trk'], ref_fr_C['fr_C_mean']*100,
            yerr=ref_fr_C['fr_C_sem']*100, fmt='D--', color=cC, mfc='white',
            capsize=4, capthick=1.2, markeredgecolor='black', markeredgewidth=0.8,
            zorder=4, label='Classical')
ax.set_xscale('log'); ax.set_xlabel('Number of Tracks')
ax.set_ylabel('Segment False Rate (%)')
ax.set_title(r'ii) False Rate vs $T$ — E1 statevector, $\tau$-sweep',
             fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='upper left', framealpha=0.95, fontsize=8, ncol=2); _style_axes(ax)

# ----- iii) ROC: false rate vs efficiency, marker per T -----
ax = axes[1, 0]
T_MARK   = [50, 100, 200, 500, 1000]
T_MARKERS = {50:'o', 100:'s', 200:'^', 500:'D', 1000:'P'}
T_COLORS  = cm.plasma(np.linspace(0.1, 0.85, len(T_MARK)))
for t, tcol in zip(T_MARK, T_COLORS):
    sub = g_tau_sv[g_tau_sv['n_trk'] == t].sort_values('tau')
    if sub.empty: continue
    ax.plot(sub['eff_mean']*100, sub['fr_mean']*100, '-', color=tcol,
            lw=1.5, zorder=2, alpha=0.85)
    ax.scatter(sub['eff_mean']*100, sub['fr_mean']*100,
               marker=T_MARKERS[t], c=sub['tau'], cmap='viridis',
               vmin=min(TAU_GRID), vmax=max(TAU_GRID),
               edgecolor='black', linewidth=0.7, s=70, zorder=3,
               label=f'T={t}')
    rc = agg_sv[agg_sv['n_trk'] == t]
    if not rc.empty:
        eC = float(rc['eff_C_mean'].iloc[0]) * 100
        fC = (1 - float(rc['pur_C_mean'].iloc[0])) * 100
        ax.plot(eC, fC, marker=T_MARKERS[t], color=cC, mfc='white',
                markeredgewidth=1.4, ms=11, zorder=4)
ax.set_xlabel('Segment Efficiency (%)')
ax.set_ylabel('Segment False Rate (%)')
ax.set_title(r'iii) ROC — quantum (E1, colour = $\tau$), open = classical',
             fontweight='bold')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='upper left', framealpha=0.95, fontsize=9); _style_axes(ax)

# ----- iv) Optimal tau(T): argmin false-rate subject to eff >= 0.9 -----
ax = axes[1, 1]
EFF_FLOOR = 0.90
def _opt(g_tau):
    rows = []
    for t, sub in g_tau.groupby('n_trk'):
        ok = sub[sub['eff_mean'] >= EFF_FLOOR]
        if ok.empty:
            ok = sub.sort_values('eff_mean', ascending=False).head(1)
        best = ok.sort_values('fr_mean').iloc[0]
        rows.append(dict(n_trk=t, tau_star=best['tau'],
                         eff=best['eff_mean'], fr=best['fr_mean'],
                         fr_sem=best['fr_sem'], eff_sem=best['eff_sem']))
    return pd.DataFrame(rows).sort_values('n_trk')

opt_sv = _opt(g_tau_sv)
opt_sm = _opt(g_tau_sm)
print('Optimal tau (eff >= 0.90), E1:')
print(opt_sv.to_string(index=False, float_format='%.3f'))
print('Optimal tau (eff >= 0.90), E2:')
print(opt_sm.to_string(index=False, float_format='%.3f'))

ax.errorbar(opt_sv['n_trk'], opt_sv['fr']*100, yerr=opt_sv['fr_sem']*100,
            fmt='o-', color=cQ_sv, capsize=5, capthick=1.5,
            markeredgecolor='black', markeredgewidth=0.8, zorder=3,
            label=r'E1, FR at $\tau^\star$')
ax.errorbar(opt_sm['n_trk'], opt_sm['fr']*100, yerr=opt_sm['fr_sem']*100,
            fmt='s-', color=cQ_sm, capsize=5, capthick=1.5,
            markeredgecolor='black', markeredgewidth=0.8, zorder=3,
            label=r'E2, FR at $\tau^\star$')
ax.errorbar(ref_fr_C['n_trk'], ref_fr_C['fr_C_mean']*100,
            yerr=ref_fr_C['fr_C_sem']*100, fmt='D--', color=cC, mfc='white',
            capsize=5, capthick=1.5, markeredgecolor='black', markeredgewidth=0.8,
            zorder=3, label='Classical')
for _, row in opt_sv.iterrows():
    ax.annotate(fr'$\tau={row["tau_star"]:.2f}$',
                (row['n_trk'], row['fr']*100),
                textcoords='offset points', xytext=(6, 6),
                fontsize=8, color=cQ_sv)
ax.set_xscale('log'); ax.set_xlabel('Number of Tracks')
ax.set_ylabel('Segment False Rate (%)')
ax.set_title(r'iv) FR at $\tau^\star$ (efficiency $\geq 90\%$)',
             fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
ax.legend(loc='upper left', framealpha=0.95, fontsize=9); _style_axes(ax)

fig.tight_layout()
_save_paper(fig, 'fig6_threshold_sweep')
plt.show()
'''

def _md(src):
    return {"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}
def _py(src):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
            "source":src.splitlines(keepends=True)}

# Insert before the §11 cell (currently last cell at index 21)
idx = next(i for i,c in enumerate(nb['cells'])
           if c['cell_type']=='markdown'
           and ''.join(c['source']).lstrip().startswith('## §11'))
new_cells = [_md(MD), _py(PY1), _py(PY2)]
nb['cells'] = nb['cells'][:idx] + new_cells + nb['cells'][idx:]
json.dump(nb, open(NB,'w'), indent=1)
print(f'inserted §7e at index {idx}; total cells now {len(nb["cells"])}')
