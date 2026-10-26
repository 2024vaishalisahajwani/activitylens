# ActivityLens

Comparative analysis of machine learning models for human activity recognition from smartphone sensor data
(UCI HAR), with dimensionality reduction and feature selection. The project asks how few of the 561 engineered
features are needed to stay close to the full-feature model, and measures the trade-off with volunteer-grouped
validation.

## Headline results (all measured in this repository)
| Item | Result |
|---|---|
| Final model | Logistic Regression (C = 1) after a correlation filter (253 of 561 features) |
| Test accuracy (9 unseen volunteers) | 94.81% |
| Volunteer-grouped CV accuracy (21 training volunteers) | 93.31% |
| Same model on all 561 features | CV 93.87%, test 96.10% (post-hoc observation; the filter costs about 1.3 points on the test set) |
| Features within 0.01 of the full model in CV | 253 (correlation filter) or 200 PCA components |
| Weakest class pair | Sitting vs Standing, about half of all test errors |
| Leakage demonstration | A random split overstated CV accuracy by about 5 to 6.5 points compared with volunteer-grouped validation |

The results are a measured performance-parsimony trade-off, not a no-loss reduction. PCA components still need all
561 raw inputs; only feature selection reduces the inputs a new sample needs. See the report for details and limits.

## Setup (Windows PowerShell)
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Tested with Python 3.14.3. If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

## Data
Download "Human Activity Recognition Using Smartphones" from the UCI Machine Learning Repository and unzip it to
`data/raw/` so that the folder `data/raw/UCI-HAR Dataset/` contains `train/`, `test/`, `features.txt` and
`activity_labels.txt`. The folder name must match `DATA_DIR` in `src/data_loader.py`. The dataset is not stored in this
repository (it is git-ignored).

## Reproduce the work (in order)
| Step | Command or notebook | Output |
|---|---|---|
| Check the loader | `python src/data_loader.py` | shapes, labels, no volunteer overlap |
| Audit and EDA | `notebooks/02_data_audit_eda.ipynb` | `reports/figures/class_balance.png`, `feature_by_activity.png` |
| Preprocessing and leakage | `notebooks/03_preprocessing.ipynb` | scaling and correlation-threshold comparisons, leakage demonstration |
| PCA, t-SNE, UMAP | `notebooks/04_dimensionality_reduction.ipynb` | variance curve and 2D maps |
| Baselines and evaluation | `notebooks/05_baseline_evaluation.ipynb` | confusion matrices, baseline models |
| Tuning and MLflow | `notebooks/06_tuning_mlflow.ipynb` | 20 tuning runs, `models/final_model.joblib` |
| Feature-budget experiment | `notebooks/07_feature_budget.ipynb` | `reports/m7_feature_budget_results.csv`, plots, 13 MLflow runs |
| Demo assets | `python src/build_app_assets.py` | `models/app_assets.joblib`, `app/sample_inputs/*.csv` |
| Demo app | `streamlit run app/streamlit_app.py` | local web app |
| Tests | `python -m unittest discover -s tests -v` | automated checks |

All random seeds are fixed to 42. The test set is used only for reporting; every choice was made with volunteer-grouped
cross-validation on the training set (the selection rule for the final model was fixed before test results were computed).

## MLflow
Runs are stored in `mlflow.db` (git-ignored) with model artifacts under `notebooks/mlruns/`. To browse them:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://127.0.0.1:5000. Experiments: `ActivityLens-M6-tuning`, `ActivityLens-M6-final-eval`,
`ActivityLens-M7-feature-budget`.

## Repository layout
```
src/            data_loader.py, preprocessing.py (CorrelationFilter), evaluation.py, model_factory.py,
                feature_selection.py (ForestTopK), app_utils.py, build_app_assets.py
notebooks/      02 to 07, one per milestone
app/            streamlit_app.py, sample_inputs/
models/         final_model.joblib, app_assets.joblib, baseline models
reports/        CSV results and figures/
tests/          test_activitylens.py
```

## Limitations
- Results come from 9 test volunteers in a controlled, waist-mounted-phone setting; accuracy varied between about 85%
  and 99% across individual test volunteers. Not validated for free-living use; not a medical device.
- The demo takes the 561 engineered features, not raw phone signals.
- Optional extensions not done: clustering, raw Inertial Signals, SHAP explanations.

## References
- Anguita, D.; Ghio, A.; Oneto, L.; Parra, X.; Reyes-Ortiz, J.L. A Public Domain Dataset for Human Activity Recognition
  Using Smartphones. ESANN 2013.
- Gao, H. et al. Compact and Explainable Human Activity Recognition with Boruta-RFE Feature Selection and SHAP
  Interpretation. Applied Sciences 2026, 16, 10046 (base paper).