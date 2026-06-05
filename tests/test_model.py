"""Model tests: graph features must improve AUC over a tabular-only baseline."""

from __future__ import annotations

import pandas as pd

from src.config import ModelConfig, StreamConfig
from src.features.feature_store import FeatureStore
from src.graph.graph_builder import build_transaction_graph
from src.graph.graph_features import compute_graph_features
from src.models.ensemble import FraudEnsemble
from src.models.score import ScoringService
from src.models.train import train_model
from src.streaming.transaction_stream import TransactionStream


def _train_on_small_stream(tmp_path):
    cfg_stream = StreamConfig()
    cfg_stream.total_transactions = 6_000
    cfg_stream.micro_batch_size = 6_000
    cfg_stream.fraud_ratio = 0.03
    cfg_stream.seed = 11

    stream = TransactionStream(cfg_stream)
    df = stream.build_dataframe()
    graph = build_transaction_graph(df)
    feats = compute_graph_features(graph, df)
    store = FeatureStore()
    fm = store.build_feature_matrix(df, feats)
    labels = df.set_index("tx_id").loc[fm["tx_id"], "is_fraud"].reset_index(drop=True)

    model_cfg = ModelConfig()
    model_cfg.model_dir = tmp_path
    result = train_model(fm, labels, model_cfg)
    return result, fm, df, model_cfg


def test_graph_features_improve_auc(tmp_path):
    result, *_ = _train_on_small_stream(tmp_path)
    assert result.auc_graph > result.auc_tabular
    assert result.auc_graph > 0.8


def test_scoring_produces_valid_decisions(tmp_path):
    result, fm, df, model_cfg = _train_on_small_stream(tmp_path)
    from src.features.feature_defs import ALL_FEATURES

    ensemble = FraudEnsemble.load(model_cfg.model_dir / "fraud_ensemble.json", ALL_FEATURES)
    scorer = ScoringService(ensemble, model_cfg)
    scored = scorer.score_batch(df, fm)

    assert len(scored) == len(df)
    for s in scored:
        assert 0.0 <= s.fraud_score <= 1.0
        assert s.decision in {"approve", "challenge", "decline"}
