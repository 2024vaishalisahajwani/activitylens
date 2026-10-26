"""Builds the small files the Streamlit demo needs.

Run once from the project root (needs the dataset and models/final_model.joblib):
    python src/build_app_assets.py
Everything is fitted on TRAINING data only; example rows come from the test set."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so that "src" can be imported when running this file directly

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data_loader import load_split, load_activity_labels
from src.preprocessing import CorrelationFilter   # noqa: F401  (needed so joblib can load the saved model)

RANDOM_STATE = 42
BACKGROUND_POINTS = 2000


def save_row_as_csv(row, path):
    np.savetxt(path, np.asarray(row).reshape(1, -1), delimiter=",", fmt="%.8g")


def main():
    X_train, y_train, _ = load_split("train")
    X_test, y_test, _ = load_split("test")
    labels = load_activity_labels()
    class_names = [labels[i] for i in sorted(labels)]            # codes 1..6 -> names, in order

    # 2D PCA map, fitted on scaled TRAINING data only (numpy arrays, so no feature-name warnings later)
    scaler = StandardScaler().fit(X_train.to_numpy())
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaler.transform(X_train.to_numpy()))
    Z = pca.transform(scaler.transform(X_train.to_numpy()))
    rng = np.random.RandomState(RANDOM_STATE)
    keep = rng.choice(len(Z), size=min(BACKGROUND_POINTS, len(Z)), replace=False)

    assets = {
        "scaler": scaler,
        "pca": pca,
        "bg_points": Z[keep],
        "bg_labels": np.array([class_names[c - 1] for c in y_train.to_numpy()[keep]]),
        "class_names": class_names,
        "n_features": X_train.shape[1],
        "explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(assets, ROOT / "models" / "app_assets.joblib")
    print("Saved models/app_assets.joblib | 2D PCA keeps",
          f"{assets['explained_variance_2d'] * 100:.1f}% of the variance")

    # Example input rows from the TEST set (first correctly classified window of each activity)
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    pred = model.predict(X_test.to_numpy())                       # the model uses labels 0..5
    true = (y_test - 1).to_numpy()
    out_dir = ROOT / "app" / "sample_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for code, name in enumerate(class_names):
        hits = np.where((true == code) & (pred == code))[0]
        if len(hits):
            save_row_as_csv(X_test.to_numpy()[hits[0]], out_dir / f"sample_{name.lower()}.csv")

    sit, stand = class_names.index("SITTING"), class_names.index("STANDING")
    wrong = np.where((true == sit) & (pred == stand))[0]          # a real mistake, for the honesty demo
    if len(wrong):
        save_row_as_csv(X_test.to_numpy()[wrong[0]], out_dir / "sample_sitting_misclassified_as_standing.csv")
    print("Saved example rows to", out_dir, "->", sorted(p.name for p in out_dir.glob("*.csv")))


if __name__ == "__main__":
    main()