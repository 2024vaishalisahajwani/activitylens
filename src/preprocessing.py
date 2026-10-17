"""Custom preprocessing steps for ActivityLens."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops a feature if it is highly correlated with any EARLIER feature.
    Fitted on training data only. threshold=1.0 keeps every feature."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X)
        corr = np.abs(np.corrcoef(X, rowvar=False))   # |correlation| of every feature pair
        corr = np.nan_to_num(corr)                     # constant columns give NaN, treat as 0
        upper = np.triu(corr, k=1)                     # keep each pair once
        self.drop_mask_ = (upper > self.threshold).any(axis=0)
        self.keep_mask_ = ~self.drop_mask_
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_mask_]