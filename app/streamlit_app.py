"""ActivityLens demo: predict one of six activities from the 561 engineered UCI HAR features.

Run from the project root:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets joblib find src.preprocessing.CorrelationFilter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.preprocessing import CorrelationFilter   # noqa: F401  (needed to load the saved model)
from src.app_utils import N_FEATURES, parse_feature_text, uncertainty_notes

st.set_page_config(page_title="ActivityLens", layout="wide")

MODEL_PATH = ROOT / "models" / "final_model.joblib"
ASSETS_PATH = ROOT / "models" / "app_assets.joblib"
RESULTS_PATH = ROOT / "reports" / "m7_feature_budget_results.csv"
SAMPLE_DIR = ROOT / "app" / "sample_inputs"
BUDGETS = [10, 30, 100, 200, 561]


# ---------- loading (cached, so files are read once) ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_assets():
    return joblib.load(ASSETS_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


missing = [p.name for p in (MODEL_PATH, ASSETS_PATH, RESULTS_PATH) if not p.exists()]
if missing:
    st.error(f"Missing file(s): {', '.join(missing)}. Run `python src/build_app_assets.py` from the project "
             "root (it needs the dataset and models/final_model.joblib) and make sure "
             "reports/m7_feature_budget_results.csv exists.")
    st.stop()

model = load_model()
assets = load_assets()
results = load_results()
class_names = assets["class_names"]

st.title("ActivityLens")
st.caption("Recognize six activities (walking, walking upstairs, walking downstairs, sitting, standing, laying) "
           "from the 561 engineered smartphone-sensor features of the UCI HAR dataset.")

tab_predict, tab_budget, tab_about = st.tabs(["Predict", "Feature budget", "About and limits"])


# ---------- tab 1: prediction ----------
with tab_predict:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1. Choose an input row")
        source = st.radio("Input source", ["Bundled example", "Upload a CSV row"], horizontal=True)
        text = None
        if source == "Bundled example":
            examples = sorted(SAMPLE_DIR.glob("*.csv")) if SAMPLE_DIR.exists() else []
            if not examples:
                st.info("No bundled examples found. Run `python src/build_app_assets.py`.")
            else:
                choice = st.selectbox("Example (real windows from the test volunteers)",
                                      examples, format_func=lambda p: p.stem.replace("sample_", "").replace("_", " "))
                text = choice.read_text()
        else:
            upload = st.file_uploader(f"CSV or TXT with exactly {N_FEATURES} numbers per row", type=["csv", "txt"])
            if upload is not None:
                text = upload.getvalue().decode("utf-8", errors="replace")

        X = None
        if text is not None:
            try:
                X, messages = parse_feature_text(text)
                for msg in messages:
                    st.warning(msg)
            except ValueError as err:
                st.error(str(err))

        row_index = 0
        if X is not None and len(X) > 1:
            row_index = int(st.number_input("Row to inspect", min_value=1, max_value=len(X), value=1)) - 1

    if X is not None:
        proba_all = model.predict_proba(X)
        classes = [class_names[int(c)] for c in model.classes_]          # model classes are 0..5
        proba = proba_all[row_index]
        top = int(np.argmax(proba))

        with left:
            st.subheader("2. Prediction")
            c1, c2 = st.columns(2)
            c1.metric("Predicted activity", classes[top].replace("_", " ").title())
            c2.metric("Confidence", f"{proba[top]:.1%}")
            for note in uncertainty_notes(proba, classes):
                st.warning(note)

            fig, ax = plt.subplots(figsize=(6, 3.2))
            colors = ["tab:green" if i == top else "lightgray" for i in range(len(classes))]
            ax.barh([c.replace("_", " ").title() for c in classes], proba, color=colors)
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            for i, p in enumerate(proba):
                ax.text(min(p + 0.01, 0.9), i, f"{p:.1%}", va="center", fontsize=8)
            ax.set_xlabel("Predicted probability")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("3. Where the sample sits on the PCA map")
            point = assets["pca"].transform(assets["scaler"].transform(X[[row_index]]))[0]
            fig, ax = plt.subplots(figsize=(6.5, 5))
            palette = plt.get_cmap("tab10")
            for i, name in enumerate(assets["class_names"]):
                mask = assets["bg_labels"] == name
                ax.scatter(assets["bg_points"][mask, 0], assets["bg_points"][mask, 1], s=6, alpha=0.35,
                           color=palette(i), label=name.replace("_", " ").title())
            ax.scatter([point[0]], [point[1]], marker="*", s=420, color="black", edgecolor="yellow",
                       linewidth=1.2, zorder=5, label="_nolegend_")
            ax.annotate("Your sample", point, xytext=(10, 12), textcoords="offset points", fontsize=9, weight="bold")
            ax.set_xlabel("PCA component 1", labelpad=8)
            ax.set_ylabel("PCA component 2")
            ax.legend(fontsize=7, loc="best", markerscale=2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"2D PCA fitted on scaled training data (2000 training windows shown). Two components keep only "
                       f"{assets['explained_variance_2d']:.1%} of the variance, so overlap on this map does not mean the "
                       "classes cannot be separated; the classifier uses all kept features.")

        if len(X) > 1:
            st.subheader("All uploaded rows")
            summary = pd.DataFrame({
                "row": np.arange(1, len(X) + 1),
                "predicted": [classes[i].replace("_", " ").title() for i in proba_all.argmax(axis=1)],
                "confidence": proba_all.max(axis=1).round(3),
            })
            st.dataframe(summary, hide_index=True)


# ---------- tab 2: feature budget ----------
with tab_budget:
    st.subheader("How many features are enough?")
    st.write("Results of the feature-budget experiment (Logistic Regression). CV = volunteer-grouped 5-fold "
             "cross-validation on the 21 training volunteers; test = the 9 held-out test volunteers, evaluated once.")

    arm = st.radio("Reduction method", ["PCA components", "Top features by Random Forest importance"], horizontal=True)
    arm_key = "PCA" if arm.startswith("PCA") else "Importance"
    budget = st.select_slider("Feature budget", options=BUDGETS, value=561)

    full = results[(results.method == "Full") & (results.budget == 561)].iloc[0]
    arm_rows = pd.concat([results[results.method == arm_key], full.to_frame().T]).copy()
    arm_rows["budget"] = arm_rows["budget"].astype(int)
    for col in ["cv_accuracy", "cv_std", "test_accuracy", "test_macro_f1"]:
        arm_rows[col] = arm_rows[col].astype(float)
    arm_rows = arm_rows.sort_values("budget")
    chosen = arm_rows[arm_rows.budget == budget].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", f"{chosen.test_accuracy:.2%}",
              f"{(chosen.test_accuracy - full.test_accuracy) * 100:+.2f} points vs full 561")
    m2.metric("CV accuracy", f"{chosen.cv_accuracy:.2%}",
              f"{(chosen.cv_accuracy - full.cv_accuracy) * 100:+.2f} points vs full 561")
    m3.metric("Sitting/Standing errors", f"{int(chosen.sitting_standing_errors)}", f"of {int(chosen.total_errors)} errors")
    within = chosen.cv_accuracy >= float(full.cv_accuracy) - 0.01
    m4.metric("Within 0.01 of full (CV)", "Yes" if within else "No")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, color, label in [("PCA", "tab:blue", "PCA components"), ("Importance", "tab:orange", "Importance top-k")]:
        part = pd.concat([results[results.method == key], full.to_frame().T]).copy()
        part["budget"] = part["budget"].astype(int)
        part["test_accuracy"] = part["test_accuracy"].astype(float)
        part = part.sort_values("budget")
        ax.plot(part.budget, part.test_accuracy, marker="o", color=color, label=label)
    final_rows = results[results.method == "Corr. filter"]
    if len(final_rows):
        fr = final_rows.iloc[0]
        ax.scatter([253], [float(fr.test_accuracy)], marker="*", s=220, color="tab:green", zorder=5,
                   label="Final model (correlation filter, 253 features)")
    ax.axvline(budget, color="black", linestyle=":", label="Selected budget")
    ax.axhline(float(full.test_accuracy), color="gray", linestyle="--", label="Full 561 features")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS + [253])
    ax.set_xticklabels([str(b) for b in BUDGETS] + ["253"], fontsize=8)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Number of features or components (log scale)")
    ax.set_ylabel("Test accuracy (axis starts at 0.5)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if arm_key == "PCA" and budget < 561:
        st.info("PCA components are mixes of all 561 original features, so a PCA model still needs every raw input. "
                "Only feature selection reduces the number of inputs a new sample needs.")
    st.caption("The test set has only 9 volunteers, so differences of about a point are within noise. Cross-validation "
               "and test results can disagree by a point or two.")

    with st.expander("All experiment results"):
        show = results[["method", "budget", "cv_accuracy", "cv_std", "test_accuracy", "test_macro_f1",
                        "sitting_standing_errors", "total_errors"]].copy()
        st.dataframe(show.round(4), hide_index=True)


# ---------- tab 3: about ----------
with tab_about:
    st.subheader("About this demo")
    final_row = results[results.method == "Corr. filter"]
    if len(final_row):
        fr = final_row.iloc[0]
        st.write(f"**Model served:** Logistic Regression (C = 1) after a correlation filter that keeps 253 of the 561 "
                 f"features. Test accuracy {float(fr.test_accuracy):.2%} on 9 unseen volunteers; "
                 f"volunteer-grouped CV accuracy {float(fr.cv_accuracy):.2%} on the 21 training volunteers.")
    st.markdown(
        "- **Dataset:** UCI Human Activity Recognition Using Smartphones (Anguita et al., ESANN 2013): 30 volunteers, "
        "waist-mounted phone, 2.56 s windows with 50% overlap, 561 engineered features.\n"
        "- **Input:** one row of the 561 engineered features, not raw phone signals. This demo cannot compute those "
        "features from raw sensor readings.\n"
        "- **Weakest spot:** Sitting versus Standing makes up about half of the model's test errors.\n"
        "- **Generalization:** results come from 9 test volunteers in a controlled setting; accuracy varied between "
        "about 85% and 99% across individual volunteers in earlier evaluation. It is not validated for free-living use "
        "and is not a medical device.\n"
        "- **Reproduce:** see the project README and the Master Guide; every experiment run is logged in MLflow."
    )