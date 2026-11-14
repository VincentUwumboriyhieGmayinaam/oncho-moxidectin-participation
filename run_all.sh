#!/usr/bin/env bash
# Reproduces all results, tables and figures for Paper 3 from the analysis file.
# Usage: bash run_all.sh            (expects data/paper3_uptake.csv and data/community_codes.json)
#        bash run_all.sh --build <survey.xlsx> <variable_audit.xlsx>   (rebuild the analysis file from the raw survey export first)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/figures
if [[ "${1:-}" == "--build" ]]; then python -I src/01_build_datasets.py "$2" "$3" data; fi
python -I src/02_analysis.py data/paper3_uptake.csv outputs
python -I src/03_tipping_point.py data/paper3_uptake.csv src/02_analysis.py outputs/tipping.pkl
python -I src/04_figures.py outputs/p3_results.pkl outputs/figures
python -I src/05_figure_s1.py outputs/tipping.pkl outputs/figures
python -I src/06_results_report.py outputs/p3_results.pkl outputs/figures "outputs/Paper 3 results - tables and figures.docx"
echo "Done. Results are in outputs/."
