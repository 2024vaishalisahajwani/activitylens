"""Helpers to evaluate a fitted classifier and draw its confusion matrix."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate(pipe, X, y, class_ids, class_names):
    """Predict with an already-fitted pipeline.
    Returns: accuracy, per-class table, confusion matrix, predictions."""
    y_pred = pipe.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=class_ids, zero_division=0)
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=class_names).round(3)
    cm = confusion_matrix(y, y_pred, labels=class_ids)   # rows = true, columns = predicted
    return accuracy, per_class, cm, y_pred


def plot_confusion_matrix(cm, class_names, title, path=None):
    """Left: raw counts. Right: each row as % of that true class (the diagonal is the recall)."""
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title}: counts")
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title}: % of each true class")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()