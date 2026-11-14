"""
Step 3 - Tipping-point analysis for respondents who did not recall the MDA (S1 Fig of Paper 3).

Non-recallers (54, 17 and 38 respondents in Rounds 1-3) are assigned a probability of having participated, from 0% to
100% in steps of 10 percentage points, separately for Round 1 and for Rounds 2-3. Model B is refitted to the
resulting fractional outcomes and the standardised Round 2 - Round 1 and Round 3 - Round 1 differences are recorded.
The grid shows how different the assumptions would have to be before the confidence interval includes zero.

The script reuses sections 0-1 of 02_analysis.py (data preparation and model functions) by executing them.

Usage
    python -I 03_tipping_point.py <paper3_uptake.csv> <02_analysis.py> <tipping.pkl>
"""

import sys, pickle

src = open(sys.argv[2]).read()
src = src[
    : src.index("# ================================================================== 2. ROUND CONTRASTS")
]
sys.argv = [sys.argv[0], sys.argv[1], "/tmp", sys.argv[3]]
exec(src)
import itertools

OUTP = sys.argv[3]
grid = np.round(np.arange(0, 1.0001, 0.1), 1)
rows = []
for q1, q23 in itertools.product(grid, grid):
    dd = d.copy()
    dd["y_tp"] = np.where(dd.recall_mda == 1, dd.received, np.where(dd["round"] == 1, q1, q23))
    f = Fit(dd, MODEL_B, y="y_tp")
    g = gcomp(f, f.df)
    rows.append(
        dict(
            q1=q1,
            q23=q23,
            n=f.n,
            **{
                k: g[k]
                for k in ["cov_R1", "cov_R2", "cov_R3", "PD_R2_R1", "PD_R3_R1", "PR_R2_R1", "PR_R3_R1"]
            },
        )
    )
    print(q1, q23, {k: tuple(round(x, 3) for x in g[k]) for k in ["PD_R2_R1", "PD_R3_R1"]}, flush=True)
nonrec = d[d.recall_mda == 0].groupby("round").size().to_dict()
pickle.dump(dict(rows=rows, nonrec=nonrec, n_all=len(d)), open(OUTP, "wb"))
