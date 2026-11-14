"""Paper 3 – MDA treatment uptake across three rounds. Implements the prespecified analysis plan.
Usage: python3 -I analysis.py <paper3_uptake.csv> <outdir>
"""
import sys, json, pickle, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData
np.random.seed(20261103)  # global seed so permutation tests (esda) and multiple imputation are reproducible
warnings.filterwarnings('ignore')
rng = np.random.default_rng(20261103)

SRC, OUT = sys.argv[1], sys.argv[2]
raw = pd.read_csv(SRC)
R = {}  # results

# ------------------------------------------------------------------ analysis variables
d = raw.copy()
d['months_lt12'] = np.where(d.months_in_comm.isna(), np.nan, (d.months_in_comm < 12).astype(float))
d['concern_yes'] = d.concern_oncho.map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 98: np.nan})
d.loc[d.heard_oncho == 0, 'concern_yes'] = 0          # not asked if never heard of oncho -> not concerned
d['eligible'] = (d.reason_not_received != 3).astype(float)
d['received_upper'] = d.received.where(d.recall_mda == 1, 1.0)
A = d[d.recall_mda == 1].copy()                          # analytic sample, n = 1,091
assert len(A) == 1091 and A.received.notna().all()

CAT = {'round': [1, 2, 3], 'age_grp': [1, 2, 3, 4, 5, 6], 'educ4': [0, 1, 2, 3], 'ses_q': [1, 2, 3, 4, 5],
       'subdistrict': [1, 2, 3, 4, 5], 'river_dist_grp': [1, 2, 3, 4], 'interviewer': [1, 2, 3, 4, 5, 6],
       'community': sorted(d.community.unique())}
REF = {'round': 1, 'age_grp': 1, 'educ4': 0, 'ses_q': 1, 'subdistrict': 1, 'river_dist_grp': 1, 'interviewer': 1}
TIER1 = ['age_grp', 'female', 'educ4', 'ses_q', 'farming_hh', 'permanent_res', 'months_lt12', 'wg_disability']
TIER3 = ['info_radio', 'info_healthworker', 'info_cdd', 'info_cic', 'knowledge_score', 'concern_yes', 'blackflies_yes']
MODEL_A = ['round']
MODEL_B = ['round'] + TIER1 + ['subdistrict']
MODEL_C = MODEL_B + ['river_dist_grp'] + TIER3

def design(df, terms, interact=None):
    cols = {}
    for t in terms:
        if t in CAT:
            for lv in CAT[t]:
                if lv == REF.get(t, CAT[t][0]): continue
                cols[f'{t}={lv}'] = (df[t] == lv).astype(float)
        else:
            cols[t] = df[t].astype(float)
    if interact:
        a, b = interact
        for la in CAT['round'][1:]:
            for lb in (CAT[b][1:] if b in CAT else [None]):
                if lb is None:
                    cols[f'round={la}:{b}'] = (df['round'] == la) * df[b].astype(float)
                else:
                    cols[f'round={la}:{b}={lb}'] = ((df['round'] == la) & (df[b] == lb)).astype(float)
    X = pd.DataFrame(cols, index=df.index)
    X.insert(0, 'const', 1.0)
    return X

def cr2_vcov(X, y, mu, clusters, family='poisson', w=None):
    """CR2 (Bell–McCaffrey) cluster-robust covariance for a GLM with canonical link."""
    X = np.asarray(X); y = np.asarray(y); mu = np.asarray(mu)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    if family == 'poisson': vf = mu
    elif family == 'binomial': vf = mu * (1 - mu)
    else: vf = np.ones_like(mu)                         # gaussian identity
    Wd = w * vf
    bread = np.linalg.inv(X.T @ (X * Wd[:, None]))
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(clusters):
        i = clusters == g
        Xg, Wg = X[i], Wd[i]
        sw = np.sqrt(Wg)
        H = (Xg * sw[:, None]) @ bread @ (Xg * sw[:, None]).T
        ev, evec = np.linalg.eigh(np.eye(len(Wg)) - H)
        ev = np.clip(ev, 1e-12, None)
        Ag = evec @ np.diag(ev ** -0.5) @ evec.T
        eg = w[i] * (y[i] - mu[i]) / sw                 # Pearson-type residual scaled
        ug = (Xg * sw[:, None]).T @ (Ag @ eg)
        meat += np.outer(ug, ug)
    return bread @ meat @ bread

class Fit:
    def __init__(self, df, terms, y='received', family='poisson', weights=None, interact=None, cluster='community'):
        df = df.dropna(subset=[y] + [t for t in terms if t in df] + ([interact[1]] if interact else []))
        if weights is not None: weights = weights.loc[df.index]
        X = design(df, terms, interact)
        X = X.loc[:, X.std() > 0].copy(); X['const'] = 1.0
        fam = {'poisson': sm.families.Poisson(), 'binomial': sm.families.Binomial(),
               'gaussian': sm.families.Gaussian()}[family]
        m = sm.GLM(df[y].astype(float), X, family=fam, var_weights=weights).fit()
        self.df, self.X, self.m, self.family, self.y = df, X, m, family, y
        self.G = df[cluster].nunique()
        self.V = cr2_vcov(X, df[y], m.mu, df[cluster].values, family, weights)
        self.b = m.params.values
        self.se = np.sqrt(np.diag(self.V))
        self.tcrit = stats.t.ppf(0.975, self.G - 1)
        self.n = len(df); self.events = int((df[y] == 0).sum())
        self.terms, self.interact = terms, interact
        self._wts = (np.ones(len(df)) if weights is None else np.asarray(weights)) * (m.mu if family == 'poisson' else (m.mu * (1 - m.mu) if family == 'binomial' else np.ones(len(df))))
        self._cl = df[cluster].values; self._Q = None
    def _bm_setup(self):
        Xt = self.X.values * np.sqrt(self._wts)[:, None]
        B = np.linalg.inv(Xt.T @ Xt); H = Xt @ B @ Xt.T; M = np.eye(len(Xt)) - H
        parts = []
        for g in np.unique(self._cl):
            i = np.where(self._cl == g)[0]
            ev, evec = np.linalg.eigh(M[np.ix_(i, i)]); ev = np.clip(ev, 1e-12, None)
            A = evec @ np.diag(ev ** -0.5) @ evec.T
            parts.append((i, M[i, :].T @ A @ Xt[i] @ B))      # n x k block: q_g = parts @ c
        self._Q = parts
    def bm_df(self, c):
        """Bell–McCaffrey (Satterthwaite) degrees of freedom for contrast c'b under CR2 (Pustejovsky & Tipton 2018)."""
        if self._Q is None: self._bm_setup()
        c = np.asarray(c, float)
        Q = np.column_stack([P @ c for _, P in self._Q]); L = Q.T @ Q
        return float(np.trace(L) ** 2 / np.sum(L ** 2))
    def df_coef(self, j):
        e = np.zeros(len(self.b)); e[j] = 1.0; return self.bm_df(e)
    def coef(self, name):
        j = list(self.X.columns).index(name)
        b, s = self.b[j], self.se[j]; dfj = self.df_coef(j)
        p = 2 * stats.t.sf(abs(b / s), dfj)
        return b, s, p
    def coef_ci(self, name):
        j = list(self.X.columns).index(name); dfj = self.df_coef(j); t = stats.t.ppf(0.975, dfj)
        b, s = self.b[j], self.se[j]
        return np.exp(b), np.exp(b - t * s), np.exp(b + t * s), 2 * stats.t.sf(abs(b / s), dfj), dfj
    def ratio_table(self):
        rows = []
        for j, c in enumerate(self.X.columns):
            if c == 'const': continue
            b, s = self.b[j], self.se[j]; dfj = self.df_coef(j); t = stats.t.ppf(0.975, dfj)
            rows.append(dict(term=c, est=np.exp(b), lo=np.exp(b - t * s), hi=np.exp(b + t * s),
                             p=2 * stats.t.sf(abs(b / s), dfj), df=dfj))
        return pd.DataFrame(rows)
    def predict(self, df, b=None):
        X = design(df, self.terms, self.interact).reindex(columns=self.X.columns, fill_value=0.0)
        X['const'] = 1.0
        eta = X.values @ (self.b if b is None else b)
        return np.exp(eta) if self.family == 'poisson' else (1 / (1 + np.exp(-eta)) if self.family == 'binomial' else eta)
    def wald(self, names):
        idx = [list(self.X.columns).index(n) for n in names if n in self.X.columns]
        b = self.b[idx]; V = self.V[np.ix_(idx, idx)]
        F = float(b @ np.linalg.solve(V, b)) / len(idx)
        return F, len(idx), float(stats.f.sf(F, len(idx), self.G - 1))

def standardise(fit, pop, levels=(1, 2, 3), var='round', weights=None, b=None):
    """Marginal standardisation: mean predicted outcome with everyone set to each level."""
    out = {}
    for lv in levels:
        p2 = pop.copy(); p2[var] = lv
        pr = fit.predict(p2, b)
        out[lv] = np.average(pr, weights=weights)
    return out

BMDF = True
def gcomp(fit, pop, weights=None):
    """Adjusted coverage by round plus PD and PR contrasts, delta-method CIs using CR2 vcov."""
    pop = pop.dropna(subset=[t for t in fit.terms if t in pop])
    base = standardise(fit, pop, weights=weights)
    eps = 1e-6; k = len(fit.b)
    def contrasts(b):
        s = standardise(fit, pop, weights=weights, b=b)
        return np.array([s[1], s[2], s[3], s[2] - s[1], s[3] - s[1], s[3] - s[2],
                         np.log(s[2] / s[1]), np.log(s[3] / s[1]), np.log(s[3] / s[2])])
    f0 = contrasts(fit.b)
    J = np.zeros((len(f0), k))
    for j in range(k):
        bb = fit.b.copy(); bb[j] += eps
        J[:, j] = (contrasts(bb) - f0) / eps
    se = np.sqrt(np.diag(J @ fit.V @ J.T))
    names = ['cov_R1', 'cov_R2', 'cov_R3', 'PD_R2_R1', 'PD_R3_R1', 'PD_R3_R2', 'PR_R2_R1', 'PR_R3_R1', 'PR_R3_R2']
    res = {}; dfs = {}
    for i, nm in enumerate(names):
        dfi = fit.bm_df(J[i]) if BMDF else fit.G - 1; dfs[nm] = dfi
        t = stats.t.ppf(0.975, dfi)
        if nm.startswith('PR'):
            res[nm] = (np.exp(f0[i]), np.exp(f0[i] - t * se[i]), np.exp(f0[i] + t * se[i]),
                       2 * stats.t.sf(abs(f0[i] / se[i]), dfi))
        else:
            res[nm] = (f0[i], f0[i] - t * se[i], f0[i] + t * se[i],
                       2 * stats.t.sf(abs(f0[i] / se[i]), dfi) if nm.startswith('PD') else np.nan)
    res['df'] = dfs
    return res

def cluster_prop(df, y, cluster='community'):
    """Proportion with cluster-robust (ratio-estimator) logit CI, t(G-1)."""
    df = df.dropna(subset=[y]); n = len(df); p = df[y].mean()
    g = df.groupby(cluster)[y].agg(['sum', 'size'])
    G = len(g)
    var = G / (G - 1) * (((g['sum'] - p * g['size']) ** 2).sum()) / n ** 2
    se_logit = np.sqrt(var) / (p * (1 - p))
    t = stats.t.ppf(0.975, G - 1)
    lg = np.log(p / (1 - p))
    inv = lambda x: 1 / (1 + np.exp(-x))
    return dict(n=n, k=int(df[y].sum()), p=p, lo=inv(lg - t * se_logit), hi=inv(lg + t * se_logit), G=G)

# ================================================================== 1. DESCRIPTIVES
LAB = {
 'age_grp': ('Age group, years', {1: '14–24', 2: '25–34', 3: '35–44', 4: '45–54', 5: '55–64', 6: '≥65'}),
 'female': ('Sex', {0: 'Male', 1: 'Female'}),
 'educ4': ('Education', {0: 'None', 1: 'Primary', 2: 'JHS/middle', 3: 'Secondary or higher'}),
 'ses_q': ('Household wealth quintile', {1: 'Q1 (poorest)', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Q5 (least poor)'}),
 'farming_hh': ('Main household income', {1: 'Farming', 0: 'Other'}),
 'permanent_res': ('Permanent resident', {1: 'Yes', 0: 'No'}),
 'months_lt12': ('Lived in community <12 of past 12 months', {1: 'Yes', 0: 'No'}),
 'wg_disability': ('Functional disability (WG-SS)', {1: 'Yes', 0: 'No'}),
 'river_dist_grp': ('Round-trip time to river', {1: '≤10 min', 2: '11–30 min', 3: '31–60 min', 4: '>60 min'}),
 'info_radio': ('Health information from radio', {1: 'Yes', 0: 'No'}),
 'info_healthworker': ('Health information from health worker', {1: 'Yes', 0: 'No'}),
 'info_cdd': ('Health information from CDD/volunteer', {1: 'Yes', 0: 'No'}),
 'info_cic': ('Health information from community information centre', {1: 'Yes', 0: 'No'}),
 'heard_oncho': ('Ever heard of onchocerciasis', {1: 'Yes', 0: 'No'}),
 'knowledge_score': ('Knowledge score (0–3)', None),
 'concern_yes': ('Personally concerned about onchocerciasis', {1: 'Yes', 0: 'No'}),
 'blackflies_yes': ('Black flies present in community', {1: 'Yes', 0: 'No'}),
}

def smd_cat(a, b):
    """Multi-category SMD (Yang & Dalton 2012) between two groups."""
    cats = sorted(set(a.dropna()) | set(b.dropna()))
    if len(cats) <= 1: return 0.0
    pa = np.array([(a == c).mean() for c in cats])[1:] / max(a.notna().mean(), 1e-9)
    pb = np.array([(b == c).mean() for c in cats])[1:] / max(b.notna().mean(), 1e-9)
    pa = np.array([(a.dropna() == c).mean() for c in cats])[1:]
    pb = np.array([(b.dropna() == c).mean() for c in cats])[1:]
    S = (np.diag(pa) - np.outer(pa, pa) + np.diag(pb) - np.outer(pb, pb)) / 2
    diff = pa - pb
    try: return float(np.sqrt(diff @ np.linalg.pinv(S) @ diff))
    except Exception: return np.nan

def smd_num(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))

t1 = []
for v, (lab, levels) in LAB.items():
    r1, r2, r3 = [A.loc[A['round'] == k, v] for k in (1, 2, 3)]
    if levels is None:
        row = dict(var=lab, level='Median (IQR)')
        for k, s in zip((1, 2, 3, 0), (r1, r2, r3, A[v])):
            row[f'R{k}'] = f'{s.median():.0f} ({s.quantile(.25):.0f}–{s.quantile(.75):.0f})'
        row['SMD21'] = smd_num(r2, r1); row['SMD31'] = smd_num(r3, r1)
        t1.append(row)
        row2 = dict(var='', level='Mean (SD)')
        for k, s in zip((1, 2, 3, 0), (r1, r2, r3, A[v])):
            row2[f'R{k}'] = f'{s.mean():.2f} ({s.std():.2f})'
        t1.append(row2)
        continue
    first = True
    s21, s31 = smd_cat(r2, r1), smd_cat(r3, r1)
    show = list(levels.items())
    if set(levels) == {0, 1} and v not in ('female', 'farming_hh'):
        show = [(1, levels[1])]
    for code, txt in show:
        row = dict(var=lab if first else '', level=txt)
        for k, s in zip((1, 2, 3, 0), (r1, r2, r3, A[v])):
            n = int((s == code).sum()); den = int(s.notna().sum())
            row[f'R{k}'] = f'{n} ({100 * n / den:.1f})'
        row['SMD21'] = s21 if first else np.nan; row['SMD31'] = s31 if first else np.nan
        t1.append(row); first = False
    miss = int(A[v].isna().sum())
    if miss:
        row = dict(var='', level='Missing', SMD21=np.nan, SMD31=np.nan)
        for k, s in zip((1, 2, 3, 0), (r1, r2, r3, A[v])):
            row[f'R{k}'] = f'{int(s.isna().sum())}'
        t1.append(row)
R['table1'] = pd.DataFrame(t1)
R['n_by_round'] = {k: int((A['round'] == k).sum()) for k in (1, 2, 3)}

# coverage table
cov = []
for lab, frame, y in [('Reported receipt among those recalling the MDA (primary)', A, 'received'),
                      ('Eligible coverage (self-reported ineligible excluded)', A[A.eligible == 1], 'received'),
                      ('Survey coverage (non-recall counted as not received)', d, 'received_itt'),
                      ('Swallowed all tablets, among recipients', A[A.received == 1], 'swallowed_all')]:
    row = dict(measure=lab)
    for k in (1, 2, 3):
        row[k] = cluster_prop(frame[frame['round'] == k], y)
    row['all'] = cluster_prop(frame, y)
    cov.append(row)
r3o = cluster_prop(A[(A['round'] == 3) & (A.received == 1)], 'swallowed_observed')
cov.append({'measure': 'Swallowed in front of distributor, among recipients (Round 3 only)', 1: None, 2: None, 3: r3o, 'all': None})
R['coverage'] = cov

# cascade
casc = []
for k in (1, 2, 3):
    s = d[d['round'] == k]
    casc.append(dict(round=k, surveyed=len(s), recalled=int(s.recall_mda.sum()), eligible=int(((s.recall_mda == 1) & (s.reason_not_received != 3)).sum()),
                     received=int(s.received.sum()), swallowed=int(s.swallowed_all.sum()),
                     observed=int(s.swallowed_observed.sum()) if k == 3 else None))
R['cascade'] = pd.DataFrame(casc)

# reasons
reasons = {1: "Not at home", 4: "Distributor never came", 2: "Refused", 3: "Not eligible", 5: "Other"}
nr = A[A.received == 0]
rt = []
for code, lab in reasons.items():
    row = dict(reason=lab, side='Programme-side' if code in (1, 4) else ('Person-side' if code in (2, 3) else 'Other'))
    for k in (1, 2, 3):
        s = nr[nr['round'] == k].reason_not_received
        row[k] = (int((s == code).sum()), 100 * (s == code).mean())
    s = nr.reason_not_received; row['all'] = (int((s == code).sum()), 100 * (s == code).mean())
    rt.append(row)
R['reasons'] = pd.DataFrame(rt)
R['reasons_n'] = {k: int((nr['round'] == k).sum()) for k in (1, 2, 3)}
side = nr.reason_not_received.map({1: 'prog', 4: 'prog', 2: 'person', 3: 'person', 5: 'other'})
ct = pd.crosstab(side, nr['round'])
chi = stats.chi2_contingency(ct.loc[['prog', 'person']])
R['reasons_test'] = dict(table=ct.to_dict(), chi2=chi[0], df=chi[2], p=chi[1])
# refusal themes (free text)
R['refusal_themes'] = nr.loc[nr.reason_not_received == 2].shape[0]

# community-level
comm = []
for (k, c), g in A.groupby(['round', 'community_name']):
    n, x = len(g), int(g.received.sum())
    lo = stats.beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = stats.beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    comm.append(dict(round=k, community=c, n=n, x=x, p=x / n, lo=lo, hi=hi))
comm = pd.DataFrame(comm)
R['community'] = comm
R['comm_80'] = {k: (int((comm[comm['round'] == k].p >= 0.8).sum()), int((comm['round'] == k).sum())) for k in (1, 2, 3)}

# ================================================================== 2. ROUND CONTRASTS
fA = Fit(A, MODEL_A)
fB = Fit(A, MODEL_B)
R['modelA'] = gcomp(fA, fA.df); R['modelB'] = gcomp(fB, fB.df)
R['modelA_n'], R['modelB_n'] = fA.n, fB.n
R['modelB_round_wald'] = fB.wald(['round=2', 'round=3'])

# cluster bootstrap of Model B g-computation (resample communities within sub-district)
boot = []
comms_by_sd = fB.df.groupby('subdistrict').community.unique()
for rep in range(999):
    parts = []
    for sd, cs in comms_by_sd.items():
        pick = rng.choice(cs, size=len(cs), replace=True)
        for j, c in enumerate(pick):
            g = fB.df[fB.df.community == c].copy(); g['community'] = f'{c}_{j}'; parts.append(g)
    bd = pd.concat(parts)
    try:
        X = design(bd, MODEL_B).reindex(columns=fB.X.columns, fill_value=0.0); X['const'] = 1.0
        m = sm.GLM(bd.received, X, family=sm.families.Poisson()).fit()
        class _F: pass
        f = _F(); f.terms = MODEL_B; f.interact = None; f.X = fB.X; f.family = 'poisson'
        f.predict = lambda df, b=None, _m=m: np.exp(design(df, MODEL_B).reindex(columns=fB.X.columns, fill_value=0.0).assign(const=1.0).values @ _m.params.values)
        s = standardise(f, bd)
        boot.append([s[1], s[2], s[3], s[2] - s[1], s[3] - s[1], s[3] - s[2], s[2] / s[1], s[3] / s[1], s[3] / s[2]])
    except Exception:
        continue
boot = np.array(boot)
names = ['cov_R1', 'cov_R2', 'cov_R3', 'PD_R2_R1', 'PD_R3_R1', 'PD_R3_R2', 'PR_R2_R1', 'PR_R3_R1', 'PR_R3_R2']
R['modelB_boot'] = {nm: (np.percentile(boot[:, i], 2.5), np.percentile(boot[:, i], 97.5)) for i, nm in enumerate(names)}
R['boot_reps'] = len(boot)

# ================================================================== 3. FACTORS (Model C)
fC = Fit(A, MODEL_C)
R['modelC'] = fC.ratio_table(); R['modelC_n'] = fC.n; R['modelC_events'] = fC.events; R['modelC_k'] = fC.X.shape[1] - 1
R['modelB_table'] = fB.ratio_table(); R['modelB_events'] = fB.events; R['modelB_k'] = fB.X.shape[1] - 1
# unadjusted PR for each variable (round-adjusted only, for the factors table)
unadj = {}
for v in TIER1 + ['river_dist_grp'] + TIER3:
    f = Fit(A, ['round', v]); tb = f.ratio_table(); unadj[v] = tb[~tb.term.str.startswith('round')]
R['unadj'] = unadj
# global Wald p for multi-category variables in Model C
glob = {}
for v in ['age_grp', 'educ4', 'ses_q', 'river_dist_grp', 'round', 'subdistrict']:
    glob[v] = fC.wald([c for c in fC.X.columns if c.startswith(v + '=')])
R['modelC_global'] = glob
# VIF
Xc = fC.X.drop(columns='const')
vif = {}
for c in Xc.columns:
    others = Xc.drop(columns=c)
    r2 = sm.OLS(Xc[c], sm.add_constant(others)).fit().rsquared
    vif[c] = 1 / (1 - r2) if r2 < 1 else np.inf
R['vif_max'] = max(vif.items(), key=lambda x: x[1])
# receipt % by level for factors table
lvl = {}
for v in TIER1 + ['river_dist_grp'] + TIER3:
    lvl[v] = A.groupby(v).received.agg(['sum', 'size']).to_dict('index')
R['levels'] = lvl

# ================================================================== 4. EFFECT MODIFICATION
A['age3'] = A.age_grp.map({1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3}); CAT['age3'] = [1, 2, 3]; REF['age3'] = 1
A['ses3'] = A.ses_q.map({1: 1, 2: 1, 3: 2, 4: 3, 5: 3}); CAT['ses3'] = [1, 2, 3]; REF['ses3'] = 1
cells = {}
for v in ['age_grp', 'age3', 'female', 'ses_q', 'ses3', 'permanent_res']:
    cells[v] = pd.crosstab(A[v], A['round'], values=(A.received == 0).astype(int), aggfunc='sum')
R['em_cells'] = {k: v.to_dict() for k, v in cells.items()}
em = {}
for v, lab in [('age3', 'Age (14–24, 25–44, ≥45)'), ('female', 'Sex'), ('ses3', 'Wealth (Q1–2, Q3, Q4–5)'), ('permanent_res', 'Permanent residence')]:
    minc = int(cells[v].min().min())
    base_terms = [t for t in MODEL_B if t not in ('age_grp', 'ses_q', v)] + [v] + (['age_grp'] if v != 'age3' else []) + (['ses_q'] if v != 'ses3' else [])
    if v == 'age3': base_terms = [t for t in base_terms if t != 'age_grp']
    if v == 'ses3': base_terms = [t for t in base_terms if t != 'ses_q']
    f = Fit(A, base_terms, interact=('round', v))
    W = f.wald([c for c in f.X.columns if c.startswith('round=') and ':' in c])
    # round-specific standardised coverage within each level of v
    out = {}
    levels = CAT.get(v, [0, 1])
    for lv in levels:
        sub = f.df.copy(); sub[v] = lv
        s = standardise(f, sub)
        out[lv] = s
    # delta-method CI for PD (R3 - R1) within each level and difference in PD between extreme levels
    em[v] = dict(label=lab, wald=W, min_cell=minc, std=out)
R['em'] = em
# per-level PD CIs via delta method
def level_pd(v, lv, f):
    pop = f.df.copy(); pop[v] = lv
    eps = 1e-6
    def fn(b):
        s = standardise(f, pop, b=b); return np.array([s[1], s[2], s[3], s[3] - s[1], s[2] - s[1]])
    f0 = fn(f.b); J = np.zeros((5, len(f.b)))
    for j in range(len(f.b)):
        bb = f.b.copy(); bb[j] += eps; J[:, j] = (fn(bb) - f0) / eps
    se = np.sqrt(np.diag(J @ f.V @ J.T))
    return [(f0[i], f0[i] - f.tcrit * se[i], f0[i] + f.tcrit * se[i]) for i in range(5)]
em_pd = {}
for v in ['age3', 'female', 'ses3', 'permanent_res']:
    base_terms = [t for t in MODEL_B if t not in ('age_grp', 'ses_q', v)] + [v]
    if v != 'age3': base_terms.append('age_grp')
    if v != 'ses3': base_terms.append('ses_q')
    f = Fit(A, base_terms, interact=('round', v))
    em_pd[v] = {lv: level_pd(v, lv, f) for lv in CAT.get(v, [0, 1])}
R['em_pd'] = em_pd

# ================================================================== 5. EQUITY (SII / RII on wealth ridit; age gap)
def ridit(s):
    r = s.rank(method='average'); return (r - 0.5) / len(s)
eq = {}
for k in (1, 2, 3):
    g = A[A['round'] == k].copy(); g['ridit'] = ridit(g.ses_index)
    fl = Fit(g, ['ridit'], family='gaussian'); b, s, p = fl.coef('ridit')
    fp = Fit(g, ['ridit']); bp, sp, pp = fp.coef('ridit')
    t1_ = stats.t.ppf(0.975, fl.df_coef(list(fl.X.columns).index('ridit'))); t2_ = stats.t.ppf(0.975, fp.df_coef(list(fp.X.columns).index('ridit')))
    eq[k] = dict(SII=(b, b - t1_ * s, b + t1_ * s, p), RII=(np.exp(bp), np.exp(bp - t2_ * sp), np.exp(bp + t2_ * sp), pp))
A['ridit'] = A.groupby('round').ses_index.transform(ridit)
CAT_ = None
fint = Fit(A, ['round', 'ridit'], family='gaussian', interact=('round', 'ridit'))
eq['change_wald'] = fint.wald(['round=2:ridit', 'round=3:ridit'])
# age gap: youngest (14–24) vs oldest (≥45) absolute difference by round
for k in (1, 2, 3):
    g = A[A['round'] == k]
    y = g[g.age3 == 1].received; o = g[g.age3 == 3].received
    eq[f'agegap_{k}'] = (o.mean() - y.mean(), len(y), len(o))
R['equity'] = eq

# ================================================================== 6. BETWEEN-COMMUNITY VARIATION (random-intercept logistic, Laplace)
def ri_logit(df, X, y, cl):
    """Random-intercept logistic regression by adaptive Gauss–Hermite (15 nodes)."""
    from scipy.optimize import minimize
    Xv, yv = X.values, df[y].values; groups = df[cl].values; ug = np.unique(groups)
    idx = [np.where(groups == g)[0] for g in ug]
    nodes, wts = np.polynomial.hermite.hermgauss(40)
    def nll(theta):
        b, ls = theta[:-1], theta[-1]; s = np.exp(ls); eta = Xv @ b; tot = 0.0
        for ii in idx:
            u = np.sqrt(2) * s * nodes
            e = eta[ii][:, None] + u[None, :]
            ll = (yv[ii][:, None] * e - np.log1p(np.exp(e))).sum(0)
            m = ll.max(); tot += m + np.log((wts * np.exp(ll - m)).sum() / np.sqrt(np.pi))
        return -tot
    b0 = sm.GLM(yv, Xv, family=sm.families.Binomial()).fit().params
    best = None
    for ls0 in (np.log(0.3), np.log(0.8), np.log(1.5)):
        r = minimize(nll, np.r_[b0, ls0], method='L-BFGS-B')
        r = minimize(nll, r.x, method='Powell', options=dict(maxiter=20000, xtol=1e-6, ftol=1e-9))
        if best is None or r.fun < best.fun: best = r
    return np.exp(best.x[-1]) ** 2
ml = {}
for k in (1, 2, 3):
    g = A[A['round'] == k]
    for lab, terms in [('null', []), ('adj', [t for t in MODEL_B if t not in ('round', 'subdistrict')])]:
        gg = g.dropna(subset=[t for t in terms])
        X = design(gg, terms); X = X.loc[:, X.std() > 0]; X['const'] = 1.0
        s2 = ri_logit(gg, X, 'received', 'community')
        ml[(k, lab)] = dict(var=s2, ICC=s2 / (s2 + np.pi ** 2 / 3), MOR=np.exp(np.sqrt(2 * s2) * stats.norm.ppf(0.75)))
R['multilevel'] = ml

# ================================================================== 7. SENSITIVITY
S = {}
S['main'] = R['modelB']
# 7a recall bounds (all 1,200)
dd = d.copy()
for nm, y in [('lower', 'received_itt'), ('upper', 'received_upper')]:
    f = Fit(dd, MODEL_B, y=y); S[f'bound_{nm}'] = gcomp(f, f.df)
# 7b IPW for recall
fr = Fit(d, ['round'] + TIER1 + ['subdistrict', 'river_dist_grp'], y='recall_mda', family='binomial')
pr = pd.Series(fr.predict(fr.df), index=fr.df.index)
wts = 1 / pr.loc[pr.index.intersection(A.index)]
Aw = A.loc[wts.index]
fw = Fit(Aw, MODEL_B, weights=wts)
S['ipw'] = gcomp(fw, fw.df, weights=wts.loc[fw.df.index])
S['ipw_wt_range'] = (float(wts.min()), float(wts.max()))
# 7c eligible coverage
fe = Fit(A[A.eligible == 1], MODEL_B); S['eligible'] = gcomp(fe, fe.df)
# 7d communities in all three rounds
allc = A.groupby('community_name')['round'].nunique(); keep = allc[allc == 3].index
f19 = Fit(A[A.community_name.isin(keep)], MODEL_B); S['comm19'] = gcomp(f19, f19.df); S['comm19_k'] = len(keep)
# 7e interviewer FE
fi = Fit(A, MODEL_B + ['interviewer']); S['interviewer'] = gcomp(fi, fi.df)
# 7f logistic
fl = Fit(A, MODEL_B, family='binomial'); S['logistic'] = gcomp(fl, fl.df)
# 7g multiple imputation (covariates), Model B and C
imp_cols = ['received', 'round', 'age_grp', 'female', 'educ4', 'ses_q', 'farming_hh', 'permanent_res', 'months_lt12',
            'wg_disability', 'subdistrict', 'river_dist_grp', 'info_radio', 'info_healthworker', 'info_cdd', 'info_cic',
            'knowledge_score', 'concern_yes', 'blackflies_yes', 'community']
Ai = A[imp_cols].copy().reset_index(drop=True)
BMDF = False
mice = MICEData(Ai.drop(columns="community").astype(float), rng=20261103)  # seeded for reproducibility
estB, estC = [], []
for m_ in range(20):
    mice.update_all(5)
    imp = mice.data.copy(); imp['community'] = Ai.community.values
    for c in ['age_grp', 'educ4', 'ses_q', 'river_dist_grp']:
        imp[c] = imp[c].round().clip(min(CAT[c]), max(CAT[c]))
    for c in ['female', 'farming_hh', 'permanent_res', 'months_lt12', 'wg_disability', 'concern_yes', 'blackflies_yes']:
        imp[c] = (imp[c] > 0.5).astype(float)
    imp['knowledge_score'] = imp.knowledge_score.round().clip(0, 3)
    f = Fit(imp, MODEL_B); g = gcomp(f, f.df); estB.append(g)
    fc = Fit(imp, MODEL_C); estC.append((fc.b, fc.V, list(fc.X.columns), fc.tcrit))
def rubin_g(lst, key, log=False):
    q = np.array([np.log(x[key][0]) if log else x[key][0] for x in lst])
    # within-variance from CI width
    t = stats.t.ppf(0.975, 21)
    u = np.array([((np.log(x[key][2]) - np.log(x[key][1])) if log else (x[key][2] - x[key][1])) / (2 * t) for x in lst]) ** 2
    qb = q.mean(); W = u.mean(); B = q.var(ddof=1); T = W + (1 + 1 / len(q)) * B
    lo, hi = qb - t * np.sqrt(T), qb + t * np.sqrt(T)
    return (np.exp(qb), np.exp(lo), np.exp(hi)) if log else (qb, lo, hi)
S['mi'] = {k: rubin_g(estB, k, log=k.startswith('PR')) for k in names}
BMDF = True
# Model C pooled under MI
cols = estC[0][2]; Q = np.array([e[0] for e in estC]); U = np.array([e[1] for e in estC])
qb = Q.mean(0); T = U.mean(0) + (1 + 1 / 20) * np.cov(Q.T, ddof=1)
se = np.sqrt(np.diag(T)); t = estC[0][3]
R['modelC_mi'] = pd.DataFrame(dict(term=cols, est=np.exp(qb), lo=np.exp(qb - t * se), hi=np.exp(qb + t * se))).query("term != 'const'")
# 7h E-values (for PR > 1)
def evalue(pr, lo):
    e = lambda r: r + np.sqrt(r * (r - 1)) if r > 1 else 1.0
    return e(pr), e(lo) if lo > 1 else 1.0
S['evalue'] = {k: evalue(R['modelB'][k][0], R['modelB'][k][1]) for k in ['PR_R2_R1', 'PR_R3_R1', 'PR_R3_R2']}
R['sens'] = S

# ================================================================== 8. REVISION ADDITIONS (peer review)
REV = {}
# co-primary: survey coverage among all respondents (non-recall = not received), Model B
REV['survey_cov'] = S['bound_lower']
REV['survey_crude'] = {k: (int(d.loc[d['round'] == k, 'received_itt'].sum()), int((d['round'] == k).sum())) for k in (1, 2, 3)}
# recall by age group
d['_y'] = d.age.between(14, 24)
REV['recall_age'] = {g: (int(x.recall_mda.sum()), len(x)) for g, x in d.groupby('_y')}
# full reasons-by-round table: chi2 with Monte Carlo p (10,000 permutations of round labels)
nr_ = nr.dropna(subset=['reason_not_received'])
ctf = pd.crosstab(nr_.reason_not_received, nr_['round'])
chi_obs = stats.chi2_contingency(ctf)[0]; rr = nr_['round'].values.copy(); rs_ = nr_.reason_not_received.values; cnt = 0
for _ in range(10000):
    rng.shuffle(rr); cnt += stats.chi2_contingency(pd.crosstab(rs_, rr))[0] >= chi_obs
REV['reasons_full'] = dict(chi2=chi_obs, df=(ctf.shape[0] - 1) * (ctf.shape[1] - 1), p_mc=(cnt + 1) / 10001, table=ctf)
# not-eligible non-participants: who were they
ne = nr_[nr_.reason_not_received == 3]
REV['not_eligible'] = dict(n=len(ne), by_round=ne['round'].value_counts().sort_index().to_dict(),
                           age_14_17=int(ne.age.between(14, 17).sum()), age_18_24=int(ne.age.between(18, 24).sum()), age_25_44=int(ne.age.between(25, 44).sum()),
                           age_45=int((ne.age >= 45).sum()), female=int(ne.female.sum()), female_15_49=int(((ne.female == 1) & ne.age.between(15, 49)).sum()))
# design effects for round-specific primary proportions (cluster ratio estimator vs SRS)
deff = {}
for k in (1, 2, 3):
    g = A[A['round'] == k]; p = g.received.mean(); n = len(g)
    cg = g.groupby('community').received.agg(['sum', 'size']); G = len(cg)
    var_cl = G / (G - 1) * (((cg['sum'] - p * cg['size']) ** 2).sum()) / n ** 2
    var_srs = p * (1 - p) / n
    deff[k] = dict(deff=var_cl / var_srs, n_eff=n * var_srs / var_cl, G=G)
REV['deff'] = deff
# observed-scale ICC (one-way ANOVA estimator) per round
icc = {}
for k in (1, 2, 3):
    g = A[A['round'] == k]; grp = g.groupby('community').received; a = grp.ngroups; N = len(g)
    n0 = (N - (grp.size() ** 2).sum() / N) / (a - 1)
    msb = (grp.size() * (grp.mean() - g.received.mean()) ** 2).sum() / (a - 1)
    msw = ((g.received - grp.transform('mean')) ** 2).sum() / (N - a)
    icc[k] = (msb - msw) / (msb + (n0 - 1) * msw)
REV['icc_obs'] = icc
# community fixed effects for round contrasts (sensitivity; replaces sub-district)
Acf = A.copy(); cnames = sorted(Acf.community.unique())
for i_, c in enumerate(cnames[1:]): Acf[f'cfe{i_}'] = (Acf.community == c).astype(float)
cfe_terms = ['round'] + TIER1 + [f'cfe{i_}' for i_ in range(len(cnames) - 1)]
fcf = Fit(Acf, cfe_terms); S['community_fe'] = gcomp(fcf, fcf.df)
# interviewer range of crude participation
iv = A.groupby('interviewer').received.agg(['mean', 'size']); REV['interviewer_range'] = (iv['mean'].min(), iv['mean'].max(), len(iv))
# communities with fewer than 5 respondents; counts at >=80% excluding them
REV['comm_small'] = comm[comm.n < 5][['round', 'community', 'n', 'x']].to_dict('records')
REV['comm_80_n5'] = {k: (int(((comm['round'] == k) & (comm.n >= 5) & (comm.p >= 0.8)).sum()), int(((comm['round'] == k) & (comm.n >= 5)).sum())) for k in (1, 2, 3)}
# persistent low: communities <80% in >=2 of the rounds in which surveyed (n>=5)
cc = comm[comm.n >= 5].copy(); cc['low'] = cc.p < 0.8
pl = cc.groupby('community').agg(rounds=('round', 'nunique'), low=('low', 'sum'))
REV['persistent_low'] = pl[(pl.low >= 2)].index.tolist(); REV['always_low'] = pl[(pl.low == pl.rounds) & (pl.rounds == 3)].index.tolist()
REV['bm_df_main'] = R['modelB']['df']
REV['sii_change_df'] = (2, fint.G - 1)
R['rev'] = REV
print('REV', {k: v for k, v in REV.items() if k not in ('reasons_full',)}, REV['reasons_full']['chi2'], REV['reasons_full']['p_mc'])
pickle.dump(R, open(f'{OUT}/p3_results.pkl', 'wb'))
print('Model A', {k: tuple(round(x, 3) for x in v[:3]) for k, v in R['modelA'].items() if k != 'df'})
print('Model B', {k: tuple(round(x, 3) for x in v[:3]) for k, v in R['modelB'].items() if k != 'df'})
print('boot', {k: tuple(round(x, 3) for x in v) for k, v in R['modelB_boot'].items()}, R['boot_reps'])
print('n', fA.n, fB.n, fC.n, 'events C', fC.events, 'k C', R['modelC_k'], 'k B', R['modelB_k'], 'VIF max', R['vif_max'])
print(R['modelC'].round(3).to_string())
print('global', {k: tuple(round(x, 3) for x in v) for k, v in glob.items()})
print('comm80', R['comm_80'])
print('reasons test', R['reasons_test'])
print('EM', {k: (v['wald'], v['min_cell']) for k, v in em.items()})
print('equity', {k: v for k, v in eq.items()})
print('multilevel', ml)
for k, v in S.items():
    if isinstance(v, dict) and 'PD_R3_R1' in v:
        print('SENS', k, {kk: tuple(round(x, 3) for x in v[kk][:3]) for kk in ['cov_R1', 'cov_R2', 'cov_R3', 'PD_R2_R1', 'PD_R3_R1', 'PD_R3_R2']})
print('evalue', S['evalue'], 'ipw range', S['ipw_wt_range'])
print(R['em_cells'])
