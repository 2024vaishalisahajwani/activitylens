"""Feature selection step for the feature-budget experiment."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class ForestTopK(BaseEstimator, TransformerMixin):
    """Keeps the k features with the highest Random Forest importance.
    The forest is fitted on the training data it receives, never on validation or test data."""

    def __init__(self, k=30, n_estimators=100, random_state=42):
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        forest = RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=3,
                                        random_state=self.random_state, n_jobs=-1)
        forest.fit(X, y)
        order = np.argsort(forest.feature_importances_)[::-1]   # most important first
        self.selected_ = order[:self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_]