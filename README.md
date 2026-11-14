# Community participation in onchocerciasis mass drug administration before and after the introduction of moxidectin in Ghana: Three repeated cross-sectional surveys

Analysis code for the article:

> [FILL: authors]. Community participation in onchocerciasis mass drug administration before and after the introduction of moxidectin in Ghana: Three repeated cross-sectional surveys. PLOS Neglected Tropical Diseases. [FILL: year, DOI].

The code reproduces every number, table and figure in the article from the de-identified survey file. It does
**not** contain any survey data (see [Data access](#data-access)).

## What the study did

Three independent household surveys (400 residents aged 14 years or older per round) were carried out in Twifo
Atti-Morkwa District, Ghana, after one ivermectin round (survey September 2024) and the first two community-wide
moxidectin rounds (surveys April 2025 and February 2026). The analysis estimates:

- **survey coverage** - of all respondents, the share who reported taking the medicine;
- **participation** - the same share among respondents who remembered the distribution;
- how participation differed between rounds after accounting for differences in who was sampled;
- reasons for missing treatment, differences between communities, and characteristics of people who took part.

Each round used one drug, so the code compares rounds, not drugs.

## How the code is organised

Each script starts with a header that explains its purpose, inputs, outputs and the methods it uses. Inside the
scripts, every analysis step is a separate block preceded by a comment explaining the method and why it was chosen;
blocks are separated by one blank line. Scripts are numbered in the order they run.

## Repository layout

```
run_all.sh                      runs the whole pipeline
src/01_build_datasets.py        builds the analysis file from the raw survey export (cleaning, coding, wealth index)
src/02_analysis.py              all estimates: coverage, standardised round differences, models A-C,
                                subgroups, equity, between-community variation, sensitivity analyses
src/03_tipping_point.py         tipping-point analysis for respondents who did not remember the MDA
src/04_figures.py               Figures 1-4
src/05_figure_s1.py             Figure S1
src/06_results_report.py        Word report with all APA-style tables and figures
docs/codebook.md                definitions of the variables used
data/README.md                  which data files are needed and how to request them
```

## Methods in brief

| Step | Method | Where |
|---|---|---|
| Percentages and 95% CIs | Ratio estimator allowing for clustering within communities (logit scale, t with G-1 df) | `02_analysis.py` |
| Comparing rounds | Modified Poisson regression (log link) with CR2 cluster-robust variance and Bell-McCaffrey degrees of freedom (22 communities); marginal standardisation to the pooled sample | `02_analysis.py` |
| Check of the variance code | Matches R `clubSandwich` 0.5.10 (standard errors within 0.5%, df to two decimals) | - |
| Between-community variation | Random-intercept logistic models (adaptive Gauss-Hermite quadrature, 40 nodes); ICC and median odds ratio | `02_analysis.py` |
| Equity | Slope and relative index of inequality on the within-round wealth ridit | `02_analysis.py` |
| Missing recall | Lower/upper bounds, inverse probability of recall weights, tipping-point grid (0-100%) | `02_analysis.py`, `03_tipping_point.py` |
| Missing covariates | Multiple imputation by chained equations (20 imputations) | `02_analysis.py` |

## How to reproduce

Requirements: Python 3.13 and the packages in `requirements.txt` (exact versions used for the article).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put paper3_uptake.csv and community_codes.json in data/ (see data/README.md)
bash run_all.sh                    # about 2-3 minutes on a laptop
```

On Windows without bash, run the `python` lines in `run_all.sh` one by one from the repository folder.

Outputs are written to `outputs/`: `p3_results.pkl` (all estimates), `tipping.pkl`, `figures/` (PNG, PDF, SVG) and a
Word report with every table. Random seeds are fixed (`20261103`). Multiple-imputation estimates can differ from
the article in the fourth decimal place because the chained-equations sampler is not fully seeded; all reported
figures (to one or two decimals) are unchanged.

## Data access

The survey data are not public because they include household locations and community names. De-identified files can
be requested from the corresponding author; see `data/README.md`. Communities are reported as codes C01-C22.

## Citation and licence

Please cite the article and this repository (`CITATION.cff`). Code is released under the MIT licence. The study was
approved by [FILL: ethics committee and approval number].
