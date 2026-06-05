"""Feature definitions shared between training and serving.

Centralising the feature list here is what eliminates **training-serving skew**:
the model trainer and the online scorer both import the exact same column order.
In a full deployment these definitions are registered with Feast as feature
views; the lightweight store in :mod:`src.features.feature_store` mirrors that
contract.
"""

from __future__ import annotations

from src.graph.graph_features import GRAPH_FEATURE_NAMES

# Raw transactional features derived directly from the event.
TRANSACTION_FEATURES: list[str] = [
    "amount",
    "amount_log",
    "tx_type_code",
    "channel_code",
    "hour_of_day",
]

# Graph features computed on the sliding-window transaction graph.
GRAPH_FEATURES: list[str] = list(GRAPH_FEATURE_NAMES)

# The full ordered feature vector consumed by the model (training == serving).
ALL_FEATURES: list[str] = TRANSACTION_FEATURES + GRAPH_FEATURES

TX_TYPE_CODES = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
CHANNEL_CODES = {"web": 0, "mobile": 1, "pos": 2, "atm": 3}
