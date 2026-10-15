"""Loads the UCI HAR dataset into pandas objects with proper column names."""
from pathlib import Path
import pandas as pd

# Path is built from this file's location, so it works from any folder
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI-HAR Dataset"


def make_unique(names):
    """features.txt repeats some names; add a suffix so every column is unique."""
    seen, unique = {}, []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}__{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    return unique


def load_feature_names():
    df = pd.read_csv(DATA_DIR / "features.txt", sep=r"\s+", header=None,
                     names=["index", "name"])
    return make_unique(df["name"].tolist())


def load_activity_labels():
    df = pd.read_csv(DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None,
                     names=["id", "activity"])
    return dict(zip(df["id"], df["activity"]))


def load_split(split):
    """split must be 'train' or 'test'. Returns X (features), y (labels), subjects."""
    folder = DATA_DIR / split
    X = pd.read_csv(folder / f"X_{split}.txt", sep=r"\s+", header=None,
                    names=load_feature_names())
    y = pd.read_csv(folder / f"y_{split}.txt", header=None, names=["activity_id"])["activity_id"]
    subjects = pd.read_csv(folder / f"subject_{split}.txt", header=None,
                           names=["subject"])["subject"]
    return X, y, subjects


if __name__ == "__main__":
    X_train, y_train, s_train = load_split("train")
    X_test, y_test, s_test = load_split("test")
    labels = load_activity_labels()

    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("Activities:", labels)
    print("Train labels:", sorted(y_train.unique()), "| Test labels:", sorted(y_test.unique()))
    print("Train volunteers:", s_train.nunique(), "| Test volunteers:", s_test.nunique())
    print("Volunteer overlap (train vs test):", set(s_train) & set(s_test))
    print("Column names unique:", X_train.columns.is_unique)
    print("Missing values (train, test):", X_train.isna().sum().sum(), X_test.isna().sum().sum())