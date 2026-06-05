"""Ensemble combiner for the fraud-scoring model.

In production this averages an XGBoost and a CatBoost model (champion/challenger
managed in MLflow). CatBoost is optional, so when it is not installed the
ensemble gracefully reduces to the XGBoost model alone — the scoring contract is
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class FraudEnsemble:
    """Wraps one or more boosted models behind a single ``predict_proba``."""

    def __init__(self, model, feature_names: list[str], catboost_model=None):
        self.model = model
        self.catboost_model = catboost_model
        self.feature_names = list(feature_names)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for each row of ``X``."""

        X = X[self.feature_names]
        proba = self.model.predict_proba(X)[:, 1]
        if self.catboost_model is not None:
            cat_proba = self.catboost_model.predict_proba(X)[:, 1]
            proba = 0.5 * proba + 0.5 * cat_proba
        return proba

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path, feature_names: list[str]) -> "FraudEnsemble":
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(str(path))
        return cls(model=model, feature_names=feature_names)
