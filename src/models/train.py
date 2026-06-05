"""Trains the XGBoost (+ optional CatBoost) fraud-scoring ensemble.

The trainer demonstrates the headline result from the project design: graph
features lift AUC well above a tabular-only baseline. It trains two models —
one on transactional features only, one on transactional + graph features — and
reports both AUCs so the improvement is measurable on every run.

The fitted ensemble and feature metadata are persisted under
``src/models/artifacts/`` for the online scorer to load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import ModelConfig
from src.features.feature_defs import ALL_FEATURES, TRANSACTION_FEATURES
from src.models.ensemble import FraudEnsemble


@dataclass
class TrainResult:
    """Outcome of a training run, including the AUC uplift from graph features."""

    auc_tabular: float
    auc_graph: float
    n_train: int
    n_test: int
    fraud_rate: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    model_path: str = ""

    @property
    def auc_uplift(self) -> float:
        return round(self.auc_graph - self.auc_tabular, 4)


def _fit_xgb(X: pd.DataFrame, y: pd.Series, cfg: ModelConfig):
    from xgboost import XGBClassifier

    scale_pos_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_model(
    feature_matrix: pd.DataFrame, labels: pd.Series, cfg: ModelConfig
) -> TrainResult:
    """Train tabular-only and graph-enhanced models; persist the graph ensemble."""

    y = labels.astype(int).reset_index(drop=True)
    X_all = feature_matrix[ALL_FEATURES].reset_index(drop=True)
    X_tab = feature_matrix[TRANSACTION_FEATURES].reset_index(drop=True)

    X_all_tr, X_all_te, y_tr, y_te = train_test_split(
        X_all, y, test_size=0.25, random_state=42, stratify=y
    )
    X_tab_tr, X_tab_te = X_tab.loc[X_all_tr.index], X_tab.loc[X_all_te.index]

    logger.info("Training tabular-only baseline model...")
    model_tab = _fit_xgb(X_tab_tr, y_tr, cfg)
    auc_tab = roc_auc_score(y_te, model_tab.predict_proba(X_tab_te)[:, 1])

    logger.info("Training graph-enhanced model...")
    model_graph = _fit_xgb(X_all_tr, y_tr, cfg)
    auc_graph = roc_auc_score(y_te, model_graph.predict_proba(X_all_te)[:, 1])

    importance = dict(
        sorted(
            zip(ALL_FEATURES, model_graph.feature_importances_.astype(float)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )

    ensemble = FraudEnsemble(model=model_graph, feature_names=ALL_FEATURES)
    model_path = _persist(ensemble, importance, auc_tab, auc_graph, cfg.model_dir)

    result = TrainResult(
        auc_tabular=round(float(auc_tab), 4),
        auc_graph=round(float(auc_graph), 4),
        n_train=len(X_all_tr),
        n_test=len(X_all_te),
        fraud_rate=round(float(y.mean()), 5),
        feature_importance=importance,
        model_path=str(model_path),
    )
    logger.success(
        "AUC tabular={:.4f} | AUC graph={:.4f} | uplift=+{:.4f}",
        result.auc_tabular,
        result.auc_graph,
        result.auc_uplift,
    )
    return result


def _persist(
    ensemble: FraudEnsemble,
    importance: dict[str, float],
    auc_tab: float,
    auc_graph: float,
    model_dir: Path,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fraud_ensemble.json"
    ensemble.save(model_path)
    meta = {
        "feature_names": ensemble.feature_names,
        "auc_tabular": float(auc_tab),
        "auc_graph": float(auc_graph),
        "feature_importance": importance,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return model_path
