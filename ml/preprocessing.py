"""
GuardURL - Phase 2: Preprocessing Pipeline
Prepares feature representations, normalizes continuous scales,
and guarantees train/inference consistency without data leakage.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.feature_extraction import FEATURE_NAMES


class URLPreprocessor:
    """
    Standardized preprocessor for GuardURL feature vectors.
    Fits feature scalers on training splits only to prevent data leakage.
    Provides identical transformation for training batches and single real-time URLs.
    """

    def __init__(self, feature_names: list[str] = None):
        self.feature_names = feature_names or FEATURE_NAMES
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_means = {}

    def fit(self, X: pd.DataFrame) -> "URLPreprocessor":
        """Fits scaler on training feature DataFrame."""
        # Ensure column alignment
        X_aligned = X[self.feature_names].copy()
        # Impute any edge-case NaN/inf
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.scaler.fit(X_aligned)
        self.is_fitted = True
        self.feature_means = {col: float(X_aligned[col].mean()) for col in self.feature_names}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature DataFrame into scaled numpy array."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform.")
        X_aligned = X[self.feature_names].copy()
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled = self.scaler.transform(X_aligned)
        return np.clip(scaled, -10.0, 10.0)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits on training data and returns transformed array."""
        return self.fit(X).transform(X)

    def transform_single(self, features: dict) -> np.ndarray:
        """
        Transforms a single feature dictionary into a 1xN model-ready numpy array.
        Used at inference time for user-entered URLs.
        """
        row = [float(features.get(col, self.feature_means.get(col, 0.0))) for col in self.feature_names]
        df_single = pd.DataFrame([row], columns=self.feature_names)
        return self.transform(df_single)

    def save(self, filepath: str = "models/preprocessor.joblib"):
        """Saves fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Saved preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.joblib") -> "URLPreprocessor":
        """Loads fitted preprocessor from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor not found at {filepath}")
        return joblib.load(filepath)
