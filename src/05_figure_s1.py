"""Figure S1: tipping-point heatmaps. Usage: python3 -I tipping_fig.py <tipping.pkl> <outdir>"""
import sys, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
T = pickle.load(open(sys.argv[1], 'rb')); OUT = sys.argv[2]
for f in fm.findSystemFonts():
    if 'Inter' in f and 'Display' not in f: fm.fontManager.addfont(f)
plt.rcParams.update({'font.family': 'Inter', 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': '#c9c8c2', 'axes.linewidth': 0.8, 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b',
    'xtick.major.size': 0, 'ytick.major.size': 0, 'savefig.dpi': 600, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
df = pd.DataFrame(T['rows']); qs = sorted(df.q1.unique())
cmap = LinearSegmentedColormap.from_list('seq', ['#f7fafe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), constrained_layout=True)
for ax, key, title in [(axes[0], 'PD_R2_R1', 'A  Round 2 vs Round 1'), (axes[1], 'PD_R3_R1', 'B  Round 3 vs Round 1')]:
    M = np.zeros((len(qs), len(qs))); L = np.zeros_like(M)
    for _, r in df.iterrows():
        i = qs.index(r.q23); j = qs.index(r.q1); M[i, j] = 100 * r[key][0]; L[i, j] = r[key][1]
    im = ax.imshow(M, origin='lower', cmap=cmap, vmin=0, vmax=25, extent=(-0.05, 1.05, -0.05, 1.05), aspect='equal')
    for i, a in enumerate(qs):
        for j, b in enumerate(qs):
            nul = L[i, j] <= 0
            if nul:
                ax.add_patch(Rectangle((b - 0.05, a - 0.05), 0.1, 0.1, fill=False, hatch='///', edgecolor='#a8a6a0', linewidth=0))
            ax.text(b, a, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=5.6, color='#ffffff' if M[i, j] > 15 else '#0b0b0b')
    ax.plot([-0.05, 1.05], [-0.05, 1.05], color='#ffffff', lw=1.0, ls=(0, (2, 2)))
    ax.set_title(title, loc='left', fontweight='bold'); ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xticklabels([f'{int(x*100)}' for x in [0, .2, .4, .6, .8, 1]]); ax.set_yticklabels([f'{int(x*100)}' for x in [0, .2, .4, .6, .8, 1]])
    ax.set_xlabel('Assumed participation of Round 1 non-recallers (%)'); ax.set_ylabel('Assumed participation of Rounds 2–3 non-recallers (%)')
    for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02); cb.set_label('Standardised difference (percentage points)'); cb.outline.set_visible(False)
fig.text(0.0, -0.04, 'Hatched cells: 95% CI includes 0. Dashed line: equal assumed participation in all rounds. Numbers: point estimate.', fontsize=7.5, color='#52514e')
for ext in ('png', 'pdf', 'svg'): fig.savefig(f'{OUT}/FigureS1_tipping_point.{ext}', bbox_inches='tight', facecolor='#ffffff')
print('ok')
