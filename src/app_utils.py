"""Helpers for the ActivityLens Streamlit demo: input parsing/validation and a confidence note."""
import re
import numpy as np

N_FEATURES = 561       # number of engineered features in the UCI HAR dataset
VALUE_LIMIT = 1.0      # the dataset authors normalized every feature to the range [-1, 1]
TOLERANCE = 1e-6
MAX_ROWS = 1000        # keeps the demo responsive


def _split(line):
    """Split one line on commas, spaces, tabs or semicolons."""
    return [tok for tok in re.split(r"[,\s;]+", line) if tok]


def _is_number(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_feature_text(text):
    """Turn the text of an uploaded CSV/TXT into a (n_rows, 561) float array.

    Accepts commas, spaces, tabs or semicolons, with or without a header line.
    Returns (array, messages). Raises ValueError with a clear message if the input is unusable."""
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        raise ValueError("The file is empty.")

    messages = []
    if not all(_is_number(tok) for tok in _split(lines[0])):
        lines = lines[1:]                                   # first line looks like a header
        messages.append("A header line was detected and skipped.")
        if not lines:
            raise ValueError("The file contains only a header line and no data rows.")

    if len(lines) > MAX_ROWS:
        lines = lines[:MAX_ROWS]
        messages.append(f"Only the first {MAX_ROWS} rows were used.")

    rows = []
    for i, line in enumerate(lines, start=1):
        tokens = _split(line)
        if len(tokens) != N_FEATURES:
            raise ValueError(f"Row {i} has {len(tokens)} values, but exactly {N_FEATURES} are required.")
        try:
            rows.append([float(tok) for tok in tokens])
        except ValueError:
            raise ValueError(f"Row {i} contains a value that is not a number.")

    data = np.array(rows, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("The data contains NaN or infinite values.")

    n_outside = int((np.abs(data) > VALUE_LIMIT + TOLERANCE).sum())
    if n_outside:
        messages.append(f"{n_outside} value(s) lie outside [-1, 1]. The UCI HAR features are normalized to this "
                        "range, so this input may not be a genuine UCI HAR feature row; predictions may be unreliable.")
    return data, messages


def uncertainty_notes(proba, class_names, low=0.60, close=0.20):
    """Short plain-language notes about how much to trust one prediction."""
    order = np.argsort(proba)[::-1]
    top, second = class_names[order[0]], class_names[order[1]]
    notes = []
    if proba[order[0]] < low:
        notes.append(f"Low confidence: the top class has only {proba[order[0]]:.0%} probability.")
    if {top, second} == {"SITTING", "STANDING"} and proba[order[1]] >= close:
        notes.append("Sitting and Standing are the most confused pair for this model "
                     "(about half of its test errors), so treat this prediction with care.")
    return notes