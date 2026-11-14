"""Tipping-point analysis for non-recallers (Paper 3). Usage: python3 -I tipping.py <paper3_uptake.csv> <analysis.py> <out.pkl>"""
import sys, pickle
src = open(sys.argv[2]).read()
src = src[:src.index('# ================================================================== 2. ROUND CONTRASTS')]
sys.argv = [sys.argv[0], sys.argv[1], '/tmp', sys.argv[3]]
exec(src)
import itertools
OUTP = sys.argv[3]
grid = np.round(np.arange(0, 1.0001, 0.1), 1)
rows = []
for q1, q23 in itertools.product(grid, grid):
    dd = d.copy()
    dd['y_tp'] = np.where(dd.recall_mda == 1, dd.received, np.where(dd['round'] == 1, q1, q23))
    f = Fit(dd, MODEL_B, y='y_tp'); g = gcomp(f, f.df)
    rows.append(dict(q1=q1, q23=q23, n=f.n, **{k: g[k] for k in ['cov_R1', 'cov_R2', 'cov_R3', 'PD_R2_R1', 'PD_R3_R1', 'PR_R2_R1', 'PR_R3_R1']}))
    print(q1, q23, {k: tuple(round(x, 3) for x in g[k]) for k in ['PD_R2_R1', 'PD_R3_R1']}, flush=True)
nonrec = d[d.recall_mda == 0].groupby('round').size().to_dict()
pickle.dump(dict(rows=rows, nonrec=nonrec, n_all=len(d)), open(OUTP, 'wb'))
