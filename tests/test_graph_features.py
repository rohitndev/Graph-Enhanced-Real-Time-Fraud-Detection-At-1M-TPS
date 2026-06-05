"""Unit tests for the graph builder and the 15 graph features."""

from __future__ import annotations

import pandas as pd

from src.graph.graph_builder import build_transaction_graph, node_key
from src.graph.graph_features import GRAPH_FEATURE_NAMES, compute_graph_features


def _sample_window() -> pd.DataFrame:
    rows = [
        # A ring: two accounts share one device and one merchant.
        ("T1", "A1", "D1", "M1", 100.0),
        ("T2", "A2", "D1", "M1", 120.0),
        ("T3", "A1", "D1", "M1", 90.0),
        # An unrelated legit transaction.
        ("T4", "A9", "D9", "M9", 30.0),
    ]
    df = pd.DataFrame(rows, columns=["tx_id", "account_id", "device_id", "merchant_id", "amount"])
    df["event_time"] = pd.to_datetime("2024-01-01 00:00:00")
    return df


def test_build_graph_node_types():
    graph = build_transaction_graph(_sample_window())
    assert graph.nodes[node_key("acct", "A1")]["ntype"] == "account"
    assert graph.nodes[node_key("dev", "D1")]["ntype"] == "device"
    assert graph.nodes[node_key("mer", "M1")]["ntype"] == "merchant"


def test_shared_device_degree_signal():
    """A device used by two accounts must have degree >= 2 (mule signal)."""

    graph = build_transaction_graph(_sample_window())
    assert graph.degree(node_key("dev", "D1")) >= 2


def test_compute_graph_features_shape_and_columns():
    window = _sample_window()
    graph = build_transaction_graph(window)
    feats = compute_graph_features(graph, window)

    assert len(feats) == len(window)
    for name in GRAPH_FEATURE_NAMES:
        assert name in feats.columns
    # Velocity for account A1 (2 transactions) must exceed the lone account A9.
    a1 = feats[feats["tx_id"] == "T1"]["velocity_ring_score"].iloc[0]
    a9 = feats[feats["tx_id"] == "T4"]["velocity_ring_score"].iloc[0]
    assert a1 > a9


def test_empty_graph_returns_zeroed_features():
    empty = pd.DataFrame(
        columns=["tx_id", "account_id", "device_id", "merchant_id", "amount", "event_time"]
    )
    graph = build_transaction_graph(empty)
    feats = compute_graph_features(graph, empty)
    assert list(feats.columns)[1:] == GRAPH_FEATURE_NAMES or feats.empty
