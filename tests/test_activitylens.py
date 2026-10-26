"""Automated checks for ActivityLens.

Run from the project root:
    python -m unittest discover -s tests -v

Tests that need the dataset or the saved model are skipped automatically if those files are missing.
The expected numbers come from the verified results in the Master Guide (Milestones 2, 6 and 7)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src import data_loader
from src.app_utils import N_FEATURES, parse_feature_text, uncertainty_notes
from src.preprocessing import CorrelationFilter

DATASET_OK = data_loader.DATA_DIR.exists()
MODEL_PATH = ROOT / "models" / "final_model.joblib"
RESULTS_PATH = ROOT / "reports" / "m7_feature_budget_results.csv"
SAMPLE_DIR = ROOT / "app" / "sample_inputs"
SAMPLE_CLASS = {"walking": 0, "walking_upstairs": 1, "walking_downstairs": 2,
                "sitting": 3, "standing": 4, "laying": 5}     # model labels are 0..5


class TestCorrelationFilter(unittest.TestCase):
    def test_drops_the_later_copy_of_a_duplicated_feature(self):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 5)
        X = np.hstack([X, 2 * X[:, [0]] + 1e-3 * rng.randn(300, 1)])     # column 5 mirrors column 0
        cf = CorrelationFilter(0.95).fit(X)
        self.assertTrue(cf.keep_mask_[0])
        self.assertFalse(cf.keep_mask_[5])
        self.assertEqual(cf.transform(X).shape, (300, 5))

    def test_transform_applies_the_mask_learned_in_fit(self):
        rng = np.random.RandomState(1)
        train = rng.randn(200, 4)
        train = np.hstack([train, train[:, [1]] * 3])
        cf = CorrelationFilter(0.95).fit(train)
        mask_before = cf.keep_mask_.copy()
        other = rng.randn(50, 5)                                           # different data, same columns
        out = cf.transform(other)
        self.assertEqual(out.shape[1], mask_before.sum())
        np.testing.assert_array_equal(cf.keep_mask_, mask_before)          # transform must not refit


class TestInputParser(unittest.TestCase):
    ROW = ",".join(["0.1"] * N_FEATURES)

    def test_valid_row_with_and_without_header(self):
        arr, msgs = parse_feature_text(self.ROW)
        self.assertEqual(arr.shape, (1, N_FEATURES))
        arr, msgs = parse_feature_text("name1,name2\n" + self.ROW + "\n" + self.ROW)
        self.assertEqual(arr.shape, (2, N_FEATURES))
        self.assertTrue(any("header" in m for m in msgs))

    def test_header_names_containing_commas_are_skipped(self):
        header = ",".join(["fBodyAcc-bandsEnergy()-1,8"] * N_FEATURES)
        arr, _ = parse_feature_text(header + "\n" + self.ROW)
        self.assertEqual(arr.shape, (1, N_FEATURES))

    def test_space_separated_input(self):
        arr, _ = parse_feature_text(" ".join(["0.2"] * N_FEATURES))
        self.assertEqual(arr.shape, (1, N_FEATURES))

    def test_bad_inputs_raise_clear_errors(self):
        bad_inputs = ["", ",".join(["0.1"] * 10), ",".join(["0.1"] * 560 + ["nan"]), "a,b,c",
                      self.ROW + "\n" + ",".join(["0.1"] * 559)]
        for text in bad_inputs:
            with self.assertRaises(ValueError):
                parse_feature_text(text)

    def test_out_of_range_values_warn_but_do_not_fail(self):
        _, msgs = parse_feature_text(",".join(["3.0"] + ["0.1"] * (N_FEATURES - 1)))
        self.assertTrue(any("outside" in m for m in msgs))


class TestUncertaintyNotes(unittest.TestCase):
    NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

    def test_sitting_standing_note(self):
        notes = uncertainty_notes(np.array([0, 0, 0, 0.5, 0.45, 0.05]), self.NAMES)
        self.assertEqual(len(notes), 2)

    def test_confident_prediction_has_no_notes(self):
        self.assertEqual(uncertainty_notes(np.array([0.97, 0.01, 0.01, 0.005, 0.003, 0.002]), self.NAMES), [])


@unittest.skipUnless(DATASET_OK, "UCI HAR dataset not found")
class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.s_train = data_loader.load_split("train")
        cls.X_test, cls.y_test, cls.s_test = data_loader.load_split("test")

    def test_shapes_and_labels(self):
        self.assertEqual(self.X_train.shape, (7352, 561))
        self.assertEqual(self.X_test.shape, (2947, 561))
        self.assertEqual(sorted(self.y_train.unique()), [1, 2, 3, 4, 5, 6])
        self.assertEqual(sorted(self.y_test.unique()), [1, 2, 3, 4, 5, 6])

    def test_volunteers_do_not_overlap_between_train_and_test(self):
        self.assertEqual(self.s_train.nunique(), 21)
        self.assertEqual(self.s_test.nunique(), 9)
        self.assertEqual(set(self.s_train) & set(self.s_test), set())

    def test_columns_unique_and_no_missing_values(self):
        self.assertTrue(self.X_train.columns.is_unique)
        self.assertEqual(int(self.X_train.isna().sum().sum()), 0)
        self.assertEqual(int(self.X_test.isna().sum().sum()), 0)

    def test_correlation_filter_reproduces_milestone_2_counts(self):
        for threshold, expected_kept in [(0.99, 439), (0.95, 253), (0.90, 202)]:
            cf = CorrelationFilter(threshold).fit(self.X_train)
            self.assertEqual(int(cf.keep_mask_.sum()), expected_kept, f"threshold {threshold}")


@unittest.skipUnless(MODEL_PATH.exists(), "models/final_model.joblib not found")
class TestSavedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(MODEL_PATH)

    def test_probabilities_are_valid(self):
        files = sorted(SAMPLE_DIR.glob("*.csv"))
        self.assertTrue(files, "no bundled example files found")
        for path in files:
            X, _ = parse_feature_text(path.read_text())
            proba = self.model.predict_proba(X)
            self.assertEqual(proba.shape, (1, 6))
            self.assertAlmostEqual(float(proba.sum()), 1.0, places=6)

    def test_bundled_examples_predict_as_labelled(self):
        for path in sorted(SAMPLE_DIR.glob("*.csv")):
            X, _ = parse_feature_text(path.read_text())
            predicted = int(self.model.predict(X)[0])
            name = path.stem.replace("sample_", "")
            if name == "sitting_misclassified_as_standing":
                self.assertEqual(predicted, SAMPLE_CLASS["standing"])      # a real, documented mistake
            else:
                self.assertEqual(predicted, SAMPLE_CLASS[name], name)

    @unittest.skipUnless(DATASET_OK, "UCI HAR dataset not found")
    def test_reproduces_the_reported_test_accuracy(self):
        X_test, y_test, _ = data_loader.load_split("test")
        accuracy = float((self.model.predict(X_test.to_numpy()) == (y_test - 1).to_numpy()).mean())
        self.assertAlmostEqual(accuracy, 0.9481, delta=0.0006)            # Milestone 6 / 7 value


@unittest.skipUnless(RESULTS_PATH.exists(), "feature-budget results CSV not found")
class TestResultsCsv(unittest.TestCase):
    def test_structure_and_key_values(self):
        df = pd.read_csv(RESULTS_PATH)
        self.assertEqual(len(df), 13)
        for col in ["method", "budget", "cv_accuracy", "test_accuracy", "sitting_standing_errors", "total_errors"]:
            self.assertIn(col, df.columns)
        self.assertFalse(df[["cv_accuracy", "test_accuracy"]].isna().any().any())
        full = df[(df.method == "Full") & (df.budget == 561)].iloc[0]
        final = df[(df.method == "Corr. filter") & (df.budget == 253)].iloc[0]
        self.assertAlmostEqual(float(full.test_accuracy), 0.9610, delta=0.0006)
        self.assertAlmostEqual(float(final.test_accuracy), 0.9481, delta=0.0006)


if __name__ == "__main__":
    unittest.main()