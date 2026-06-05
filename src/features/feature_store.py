"""Feast-style online feature store with a Redis-backed fast path.

Materialises pre-computed graph features for online serving so that the model
sees identical features at training and inference time. The default backend is
an in-process dictionary (zero dependencies); when ``redis`` is installed and a
``REDIS_URL`` is configured, hot account/device features are cached in Redis for
sub-millisecond lookups exactly as described in the architecture.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_defs import (
    ALL_FEATURES,
    CHANNEL_CODES,
    TX_TYPE_CODES,
)


class FeatureStore:
    """Builds and serves the full feature vector for each transaction."""

    def __init__(self, use_redis: bool | None = None):
        self._online: dict[str, dict[str, float]] = {}
        self._redis = None
        if use_redis is None:
            use_redis = bool(os.getenv("REDIS_URL"))
        if use_redis:
            self._redis = self._connect_redis()

    @staticmethod
    def _connect_redis():
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(os.environ["REDIS_URL"])
            client.ping()
            logger.info("FeatureStore connected to Redis online cache.")
            return client
        except Exception as exc:  # pragma: no cover - optional path
            logger.warning("Redis unavailable ({}); using in-memory store.", exc)
            return None

    # ------------------------------------------------------------------
    # Transactional feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def transactional_features(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["amount"] = df["amount"].astype(float)
        out["amount_log"] = np.log1p(df["amount"].astype(float))
        out["tx_type_code"] = df["tx_type"].map(TX_TYPE_CODES).fillna(0).astype(int)
        out["channel_code"] = df["channel"].map(CHANNEL_CODES).fillna(0).astype(int)
        out["hour_of_day"] = pd.to_datetime(df["event_time"]).dt.hour
        out["tx_id"] = df["tx_id"].values
        return out

    # ------------------------------------------------------------------
    # Join transactional + graph features into the serving vector
    # ------------------------------------------------------------------
    def build_feature_matrix(
        self, tx_df: pd.DataFrame, graph_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a DataFrame with ``tx_id`` + the ordered ``ALL_FEATURES``."""

        tx_feats = self.transactional_features(tx_df)
        merged = tx_feats.merge(graph_features, on="tx_id", how="left")
        merged[ALL_FEATURES] = merged[ALL_FEATURES].fillna(0.0)
        return merged[["tx_id", *ALL_FEATURES]]

    # ------------------------------------------------------------------
    # Online materialization (write-behind to cache)
    # ------------------------------------------------------------------
    def materialize(self, feature_matrix: pd.DataFrame) -> None:
        """Persist the latest feature vectors for online serving."""

        for row in feature_matrix.itertuples(index=False):
            payload = {f: getattr(row, f) for f in ALL_FEATURES}
            self._online[row.tx_id] = payload
            if self._redis is not None:  # pragma: no cover - optional path
                self._redis.hset(f"feat:{row.tx_id}", mapping=payload)

    def get_online(self, tx_id: str) -> dict[str, Any] | None:
        if tx_id in self._online:
            return self._online[tx_id]
        if self._redis is not None:  # pragma: no cover - optional path
            cached = self._redis.hgetall(f"feat:{tx_id}")
            if cached:
                return {k.decode(): float(v) for k, v in cached.items()}
        return None
