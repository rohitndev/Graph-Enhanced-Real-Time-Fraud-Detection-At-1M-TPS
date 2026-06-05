"""Inline scoring service: turns feature vectors into fraud decisions.

Each transaction receives a fraud probability from the ensemble and a routing
decision — ``approve`` / ``challenge`` / ``decline`` — based on the configured
thresholds. High-risk transactions (score above the high-risk threshold) are
flagged for the LangChain fraud-investigation agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ModelConfig
from src.models.ensemble import FraudEnsemble


@dataclass
class ScoredTransaction:
    tx_id: str
    account_id: str
    device_id: str
    merchant_id: str
    amount: float
    fraud_score: float
    decision: str
    is_high_risk: bool
    is_fraud_label: int


class ScoringService:
    """Scores transactions and assigns decisions using config thresholds."""

    def __init__(self, ensemble: FraudEnsemble, cfg: ModelConfig):
        self.ensemble = ensemble
        self.cfg = cfg

    def _decide(self, score: float) -> str:
        if score >= self.cfg.decline_threshold:
            return "decline"
        if score >= self.cfg.challenge_threshold:
            return "challenge"
        return "approve"

    def score_batch(
        self, tx_df: pd.DataFrame, feature_matrix: pd.DataFrame
    ) -> list[ScoredTransaction]:
        """Score every transaction in the batch and return decisions."""

        scores = self.ensemble.predict_proba(feature_matrix)
        score_by_id = dict(zip(feature_matrix["tx_id"], scores))

        results: list[ScoredTransaction] = []
        for tx in tx_df.itertuples(index=False):
            score = float(score_by_id.get(tx.tx_id, 0.0))
            results.append(
                ScoredTransaction(
                    tx_id=tx.tx_id,
                    account_id=tx.account_id,
                    device_id=tx.device_id,
                    merchant_id=tx.merchant_id,
                    amount=float(tx.amount),
                    fraud_score=round(score, 4),
                    decision=self._decide(score),
                    is_high_risk=score >= self.cfg.high_risk_threshold,
                    is_fraud_label=int(getattr(tx, "is_fraud", 0)),
                )
            )
        return results
