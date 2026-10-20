"""Builds the full pipeline for each model family, so every notebook uses identical settings."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.preprocessing import CorrelationFilter

RANDOM_STATE = 42
CORR_THRESHOLD = 0.95   # decision from Milestone 3


def build_pipeline(family, scale=True, **params):
    """family: 'Logistic Regression', 'SVM (RBF)', 'Random Forest' or 'XGBoost'.
    params: the hyperparameters being tuned for that family."""
    steps = [("corr_filter", CorrelationFilter(CORR_THRESHOLD))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    if family == "Logistic Regression":
        model = LogisticRegression(max_iter=3000, **params)
    elif family == "SVM (RBF)":
        model = SVC(kernel="rbf", **params)
    elif family == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                       n_jobs=-1, **params)
    elif family == "XGBoost":
        model = XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", eval_metric="mlogloss",
                              random_state=RANDOM_STATE, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unknown model family: {family}")

    steps.append(("model", model))
    return Pipeline(steps)