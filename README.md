# SDSS17 Astronomical Object Classification

GALAXY / QSO / STAR classification on the SDSS17 dataset (`data/star_classification.csv`, the uploaded file; nothing else downloaded).

**Stack:** official Faiss **E-Means** (`faiss.contrib.e_means.NumpyEmeans`, in `faiss-cpu>=1.16.0`; not K-Means), **CatBoost**, **LightGBM**, **Optuna**, **MLflow**, **Streamlit**.

## Structure
| Path | Content |
|---|---|
| `notebooks/Overview.ipynb` | project map, design decisions, how to run |
| `notebooks/Analysis.ipynb` | inspection, cleaning, EDA, colour indices, stratified 70/15/15 split |
| `notebooks/EMeans.ipynb` | E-Means (k=3), Silhouette / Davies-Bouldin / ARI / NMI, clustering vs classification |
| `notebooks/CatBoost.ipynb` | CatBoost + Optuna (validation Macro-F1) + MLflow, with/without redshift |
| `notebooks/LightGBM.ipynb` | LightGBM + Optuna + MLflow, with/without redshift |
| `notebooks/ModelBenchmarks.ipynb` | test-set comparison, selection, bootstrap, explanation, final model, MLflow registry |
| `src/common.py` | shared helpers (constants, cleaning, features, split, metrics, plots, MLflow setup) |
| `app.py` | Streamlit app (Overview, EDA, E-Means, Model comparison, Predict single/batch, MLflow) |
| `artifacts/` | figures, tables, JSON results, saved models, `final_model.json` (written by the notebooks) |
| `mlflow.db`, `mlruns/` | MLflow tracking store |

## Run
```bash
pip install -r requirements.txt
./run_notebooks.sh              # executes notebooks 01 -> 05 in place (~35 min on one CPU core)
streamlit run app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Executed notebooks, artifacts, models and the MLflow store are already included, so the app runs immediately. Optuna budgets are `N_TRIALS` / `N_TRIALS_NOZ` at the top of notebooks 03 and 04.

## Method
* **Cleaning:** 1 row with the SDSS `-9999` sentinel removed. No missing values or duplicate rows. `obj_ID` repeats but the repeated rows differ in photometry and class (float64 cannot represent the IDs), so there is no cross-split leakage.
* **Features:** `u g r i z`, `u-g g-r r-i i-z`, `redshift` (ablation without it). Excluded: identifiers/metadata and `alpha`, `delta`.
* **Split:** stratified 70/15/15 (69,999 / 15,000 / 15,000), seed 42; E-Means scaler fit on train only.
* **Optuna:** TPE, objective = validation Macro-F1, independent studies (20 trials with redshift, 10 without), early stopping on validation.
* **Selection:** highest test Macro-F1 of the two tuned models, plus a paired bootstrap.

## Results (test set, with redshift)
| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| CatBoost | 0.9782 | 0.9749 | 0.9781 |
| **LightGBM (final)** | **0.9805** | **0.9773** | **0.9804** |

Macro-F1 difference +0.0024, bootstrap 95% CI [+0.0009, +0.0039]; the models disagree on 102 of 15,000 objects. Without redshift Macro-F1 is about 0.859 for both. E-Means (k=3, test): Silhouette 0.389, Davies-Bouldin 1.082, ARI 0.237, NMI 0.246.

## Caveats
Selecting between two finalists on the test set (as specified) makes the winner's score very slightly optimistic; Optuna budgets are modest; the "why LightGBM won" text is a plausible explanation, not a controlled experiment. Timings come from a 1-core sandbox.
