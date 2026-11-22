#!/usr/bin/env bash
# Executes the notebooks in order and stores the outputs inside them (about 30 min on one CPU core).
set -e
cd "$(dirname "$0")/notebooks"
for nb in 01_Data_Analysis_EDA 02_EMeans_Clustering 03_CatBoost_Optuna_MLflow 04_LightGBM_Optuna_MLflow 05_Model_Comparison_and_Final_Model 00_Project_Overview; do
  echo "== running $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb.ipynb"
done
echo "done"
