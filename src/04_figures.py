"""Manuscript figures for Paper 3. Usage: python3 -I figures.py <results.pkl> <outdir>"""
import sys, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

R = pickle.load(open(sys.argv[1], 'rb')); OUT = sys.argv[2]
for f in fm.findSystemFonts():
    if 'Inter' in f and 'Display' not in f: fm.fontManager.addfont(f)
plt.rcParams.update({
    'font.family': 'Inter', 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': '#c9c8c2', 'axes.linewidth': 0.8,
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b', 'text.color': '#0b0b0b',
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'savefig.dpi': 600, 'figure.dpi': 150, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
SURF, GRID, MUTED, TXT2 = '#ffffff', '#e9e8e4', '#898781', '#52514e'
RC = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}           # validated: blue / orange / aqua
RLAB = {1: 'Round 1 · ivermectin', 2: 'Round 2 · moxidectin', 3: 'Round 3 · moxidectin'}
GRAY = '#b9b7b0'

def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.8); ax.set_axisbelow(True)

def save(fig, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight', facecolor=SURF)
    plt.close(fig)

# ------------------------------------------------------------------ Figure 1: treatment cascade
c = R['cascade']
stages = ['Surveyed', 'Recalled MDA', 'Participated\n(received drug)', 'Swallowed\nall tablets']
fig, ax = plt.subplots(figsize=(6.3, 3.4))
grid(ax)
for k in (1, 2, 3):
    r = c[c['round'] == k].iloc[0]
    vals = 100 * np.array([r.surveyed, r.recalled, r.received, r.swallowed]) / r.surveyed
    x = np.arange(4) + (k - 2) * 0.06
    ax.plot(x, vals, color=RC[k], lw=2, solid_capstyle='round', zorder=3)
    ax.scatter(x, vals, s=42, color=RC[k], edgecolor=SURF, linewidth=1.6, zorder=4)
    ax.annotate(f'{vals[-1]:.1f}%  {RLAB[k]}'.replace('.', '·', 1), (x[-1], vals[-1]), xytext=(10, {1: -2, 2: 6, 3: -6}[k]),
                textcoords='offset points', va='center', fontsize=8, color='#0b0b0b')
    if k == 1:
        ax.annotate('', xy=(2, vals[2]), xytext=(1, vals[1]))
ax.axhline(80, color=MUTED, lw=1, zorder=2)
ax.text(-0.25, 79.2, 'WHO minimum therapeutic coverage, 80%', fontsize=7.5, color=TXT2, va='top', ha='left')
ax.set_xticks(range(4)); ax.set_xticklabels(stages)
ax.set_ylim(50, 102); ax.set_xlim(-0.3, 3.15)
ax.set_ylabel('Percentage of surveyed respondents')
ax.spines['left'].set_visible(False)
# drop annotations: largest loss step
for k, dy in ((1, -14), ):
    pass
save(fig, 'Figure1_treatment_cascade')

# ------------------------------------------------------------------ Figure 2: community-level receipt
import json
CODE = json.load(open(__import__('os').environ.get('COMMUNITY_KEY', 'data/community_codes.json')))
comm = R['community'].copy(); comm['community'] = comm.community.map(CODE)
order = sorted(comm.community.unique(), reverse=True)
ypos = {c: i for i, c in enumerate(order)}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.4), sharey=True)
for ax, k in zip(axes, (1, 2, 3)):
    g = comm[comm['round'] == k]
    ax.axvline(80, color=MUTED, lw=1, zorder=1)
    for _, r in g.iterrows():
        i = ypos[r.community]
        col = RC[2] if r.p < 0.8 else GRAY
        if r.n < 5:
            ax.scatter(100 * r.p, i, s=26, facecolor='white', edgecolor=MUTED, linewidth=1.0, zorder=3)
            ax.text(100 * r.p + 4, i, f'n={r.n}', fontsize=6.3, color=MUTED, va='center'); continue
        ax.plot([100 * r.lo, 100 * r.hi], [i, i], color=col, lw=1.4, solid_capstyle='round', zorder=2, alpha=0.9)
        ax.scatter(100 * r.p, i, s=26, color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
    for c in order:
        if c not in set(g.community):
            ax.text(50, ypos[c], 'not sampled', fontsize=6.5, color=MUTED, ha='center', va='center', style='italic')
    c8 = R['rev']['comm_80_n5'][k]
    ax.set_title(f'{RLAB[k]}\n{c8[0]} of {c8[1]} communities ≥80%', loc='left', fontsize=8.3, color='#0b0b0b')
    ax.set_xlim(-2, 103); ax.set_xticks([0, 25, 50, 75, 100])
    grid(ax, 'x'); ax.spines['left'].set_visible(False)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=7.2)
axes[0].set_ylim(-0.7, len(order) - 0.3)
axes[1].set_xlabel('Reported community participation among respondents recalling the MDA (%)')
leg = [Line2D([0], [0], marker='o', color=RC[2], lw=1.4, markersize=5, markeredgecolor=SURF, label='Below 80%'),
       Line2D([0], [0], marker='o', color=GRAY, lw=1.4, markersize=5, markeredgecolor=SURF, label='80% or above'),
       Line2D([0], [0], color=MUTED, lw=1, label='80% reference'),
       Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=MUTED, markersize=5, label='Fewer than 5 respondents')]
fig.legend(handles=leg, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.56, -0.035), fontsize=7.5)
fig.tight_layout(w_pad=1.0)
save(fig, 'Figure2_community_receipt')

# ------------------------------------------------------------------ Figure 3: standardised receipt by age and round
emp = R['em_pd']['age3']
ages = {1: '14–24 years', 2: '25–44 years', 3: '≥45 years'}
fig, ax = plt.subplots(figsize=(6.0, 3.4))
grid(ax)
for k in (1, 2, 3):
    xs, ys, lo, hi = [], [], [], []
    for a in (1, 2, 3):
        est, l, h = emp[a][k - 1]
        xs.append(a + (k - 2) * 0.2); ys.append(100 * est); lo.append(100 * l); hi.append(100 * h)
    for x, l, h in zip(xs, lo, hi):
        ax.plot([x, x], [l, h], color=RC[k], lw=1.6, solid_capstyle='round', zorder=2)
    ax.scatter(xs, ys, s=44, color=RC[k], edgecolor=SURF, linewidth=1.6, zorder=3, label=RLAB[k])
ax.axhline(80, color=MUTED, lw=1, zorder=1)
ax.text(3.42, 80.6, '80%', fontsize=7.5, color=TXT2, va='bottom', ha='right')
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([ages[a] for a in (1, 2, 3)])
ax.set_ylim(35, 100); ax.set_xlim(0.55, 3.45)
ax.set_ylabel('Standardised participation (%)'); ax.spines['left'].set_visible(False)
gap1 = 100 * (emp[3][0][0] - emp[1][0][0])
ax.text(0.86, 100 * emp[1][0][0], f'Round 1: {gap1:.0f} points below\nthose aged ≥45', fontsize=7.5, color=TXT2, va='center', ha='left')
ax.legend(frameon=False, loc='lower right', fontsize=7.5, handletextpad=0.3)
save(fig, 'Figure3_receipt_by_age_round')

# ------------------------------------------------------------------ Figure 4: forest plot, Model C
mc = R['modelC'].set_index('term')
rows = [('Age (ref: 14–24 years)', None),
        ('25–34', 'age_grp=2'), ('35–44', 'age_grp=3'), ('45–54', 'age_grp=4'), ('55–64', 'age_grp=5'), ('≥65', 'age_grp=6'),
        ('Female (ref: male)', 'female'),
        ('Education (ref: none)', None), ('Primary', 'educ4=1'), ('JHS/middle', 'educ4=2'), ('Secondary or higher', 'educ4=3'),
        ('Wealth quintile (ref: Q1, poorest)', None), ('Q2', 'ses_q=2'), ('Q3', 'ses_q=3'), ('Q4', 'ses_q=4'), ('Q5', 'ses_q=5'),
        ('Farming household', 'farming_hh'), ('Permanent resident', 'permanent_res'),
        ('Lived <12 months in community', 'months_lt12'), ('Functional disability', 'wg_disability'),
        ('Time to river (ref: ≤10 min)', None), ('11–30 min', 'river_dist_grp=2'), ('31–60 min', 'river_dist_grp=3'), ('>60 min', 'river_dist_grp=4'),
        ('Information and knowledge (measured after MDA)', None),
        ('Radio', 'info_radio'), ('Health worker', 'info_healthworker'), ('CDD / volunteer', 'info_cdd'),
        ('Community information centre', 'info_cic'), ('Knowledge score (per point)', 'knowledge_score'),
        ('Concerned about onchocerciasis', 'concern_yes'), ('Black flies perceived present', 'blackflies_yes')]
fig, ax = plt.subplots(figsize=(6.6, 7.6))
n = len(rows); ys = np.arange(n)[::-1]
ax.axvline(1, color=MUTED, lw=1, zorder=1)
for y, (lab, term) in zip(ys, rows):
    if term is None:
        ax.text(-1.08, y, lab, fontsize=7.8, fontweight='semibold', va='center', ha='left', transform=ax.get_yaxis_transform())
        continue
    r = mc.loc[term]
    col = RC[3] if term.startswith(('info_', 'knowledge', 'concern', 'blackflies')) else RC[1]
    ax.plot([r.lo, r.hi], [y, y], color=col, lw=1.6, solid_capstyle='round', zorder=2)
    ax.scatter(r.est, y, s=30, marker='s', color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
    ax.text(1.02, y, f'{r.est:.2f} ({r.lo:.2f}–{r.hi:.2f})'.replace('.', '·'), transform=ax.get_yaxis_transform(), fontsize=7.4, va='center', color='#0b0b0b')
    ax.text(-1.08, y, '   ' + lab, transform=ax.get_yaxis_transform(), fontsize=7.6, va='center', ha='left', color='#0b0b0b')
ax.set_xscale('log'); ax.set_xlim(0.6, 1.7)
ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.5]); ax.set_xticklabels(['0·6', '0·8', '1·0', '1·25', '1·5'])
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_ylim(-0.8, n - 0.2)
grid(ax, 'x')
ax.text(1.02, n - 0.2, 'aPR (95% CI)', transform=ax.get_yaxis_transform(), fontsize=7.6, fontweight='semibold', va='bottom')
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel('← Lower participation        Adjusted prevalence ratio (log scale)        Higher participation →')
fig.subplots_adjust(left=0.42, right=0.78)
# move the plotting area right so labels have room
pos = ax.get_position(); ax.set_position([0.40, pos.y0, 0.36, pos.height])
for t in ax.texts:
    pass
save(fig, 'Figure4_forest_modelC')
print('figures written')
