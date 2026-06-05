"""Unit tests for Label-Propagation fraud-ring detection."""

from __future__ import annotations

import pandas as pd

from src.graph.graph_builder import build_transaction_graph
from src.graph.ring_detection import detect_fraud_rings


def _ring_window() -> pd.DataFrame:
    rows = []
    # A tight ring: 4 accounts funnel through 1 shared device + 1 merchant.
    for i, acc in enumerate(["A1", "A2", "A3", "A4"]):
        rows.append((f"R{i}", acc, "D_SHARED", "M_MULE", 200.0))
    # Scattered legit traffic (distinct devices, distinct merchants).
    for i in range(8):
        rows.append((f"L{i}", f"A1{i}", f"D1{i}", f"M1{i}", 25.0))
    df = pd.DataFrame(rows, columns=["tx_id", "account_id", "device_id", "merchant_id", "amount"])
    df["event_time"] = pd.to_datetime("2024-01-01")
    return df


def test_detects_ring_with_min_accounts():
    graph = build_transaction_graph(_ring_window())
    rings = detect_fraud_rings(graph, min_accounts=3)
    assert len(rings) >= 1
    top = rings[0]
    assert top.size >= 3
    assert top.shared_device_ratio > 0.0
    assert 0.0 <= top.risk_score <= 1.0


def test_no_rings_when_threshold_too_high():
    graph = build_transaction_graph(_ring_window())
    rings = detect_fraud_rings(graph, min_accounts=99)
    assert rings == []
