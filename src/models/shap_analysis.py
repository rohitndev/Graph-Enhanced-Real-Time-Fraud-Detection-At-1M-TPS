"""SHAP / feature-importance analysis for the fraud ensemble.

Quantifies how much the graph features contribute to each fraud decision. Uses
the optional ``shap`` package when available for per-transaction explanations,
and always falls back to the model's gain-based importance so it runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.features.feature_defs import ALL_FEATURES, GRAPH_FEATURES
from src.models.ensemble import FraudEnsemble


def graph_vs_tabular_importance(cfg: ModelConfig) -> dict:
    """Return the share of total importance attributable to graph features."""

    meta_path = Path(cfg.model_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Train the model first (run `python main.py`).")
    importance = json.loads(meta_path.read_text())["feature_importance"]

    graph_share = sum(importance.get(f, 0.0) for f in GRAPH_FEATURES)
    total = sum(importance.values()) or 1.0
    return {
        "graph_importance_share": round(graph_share / total, 4),
        "tabular_importance_share": round(1 - graph_share / total, 4),
        "top_graph_features": dict(
            sorted(
                ((f, importance.get(f, 0.0)) for f in GRAPH_FEATURES),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
        ),
    }


def shap_values(ensemble: FraudEnsemble, X: pd.DataFrame):  # pragma: no cover - optional
    """Per-transaction SHAP values (requires the optional ``shap`` package)."""

    import shap

    explainer = shap.TreeExplainer(ensemble.model)
    return explainer.shap_values(X[ALL_FEATURES])


if __name__ == "__main__":  # pragma: no cover - manual run
    print(json.dumps(graph_vs_tabular_importance(ModelConfig()), indent=2))
