"""Computes the 15 real-time graph features per transaction.

These are the network signals that individual transaction features cannot
capture — the coordinated structure of a fraud ring. They are computed once per
sliding-window graph and then joined back onto each transaction by its account,
device and merchant nodes.

The 15 features (matching the project design):

  1.  account_degree                  - distinct devices+merchants for the account
  2.  account_pagerank               - account centrality in the window graph
  3.  account_clustering_coef        - how tightly the account's neighbours link
  4.  account_betweenness            - bridge score of the account node
  5.  device_degree                  - accounts sharing the device (mule signal)
  6.  device_betweenness_centrality  - device as a bridge across the graph
  7.  device_pagerank                - device centrality
  8.  device_shared_account_count    - distinct accounts on the device
  9.  merchant_degree                - distinct accounts hitting the merchant
  10. merchant_cluster_coefficient   - merchant neighbourhood density
  11. merchant_betweenness           - merchant as a bridge
  12. velocity_ring_score            - account transaction velocity in window
  13. component_size                 - size of the account's connected component
  14. unique_devices_per_account     - device fan-out for the account
  15. unique_merchants_per_account   - merchant fan-out for the account
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from .graph_builder import node_key

GRAPH_FEATURE_NAMES: list[str] = [
    "account_degree",
    "account_pagerank",
    "account_clustering_coef",
    "account_betweenness",
    "device_degree",
    "device_betweenness_centrality",
    "device_pagerank",
    "device_shared_account_count",
    "merchant_degree",
    "merchant_cluster_coefficient",
    "merchant_betweenness",
    "velocity_ring_score",
    "component_size",
    "unique_devices_per_account",
    "unique_merchants_per_account",
]


def _safe_betweenness(graph: nx.Graph) -> dict[str, float]:
    """Betweenness with sampling on large graphs to stay within latency budget."""

    n = graph.number_of_nodes()
    if n == 0:
        return {}
    # Sample pivots on large windows — keeps stream latency bounded.
    k = None if n <= 400 else min(200, n)
    try:
        return nx.betweenness_centrality(graph, k=k, weight=None, seed=7)
    except Exception:  # pragma: no cover - defensive
        return {node: 0.0 for node in graph.nodes}


def compute_graph_features(
    graph: nx.Graph, window_df: pd.DataFrame
) -> pd.DataFrame:
    """Return a per-transaction DataFrame of the 15 graph features.

    The output is aligned 1:1 with ``window_df`` rows (indexed by ``tx_id``).
    """

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            0.0, index=window_df["tx_id"], columns=GRAPH_FEATURE_NAMES
        ).reset_index()

    pagerank = nx.pagerank(graph, weight="weight", max_iter=100)
    betweenness = _safe_betweenness(graph)
    clustering = nx.clustering(graph)
    degree = dict(graph.degree())

    # Connected-component size per node.
    comp_size: dict[str, int] = {}
    for comp in nx.connected_components(graph):
        size = len(comp)
        for node in comp:
            comp_size[node] = size

    # Account transaction velocity in this window (ring burst signal).
    velocity = window_df.groupby("account_id").size().to_dict()

    rows = []
    for tx in window_df.itertuples(index=False):
        acct = node_key("acct", tx.account_id)
        dev = node_key("dev", tx.device_id)
        mer = node_key("mer", tx.merchant_id)

        rows.append(
            {
                "tx_id": tx.tx_id,
                "account_degree": degree.get(acct, 0),
                "account_pagerank": pagerank.get(acct, 0.0),
                "account_clustering_coef": clustering.get(acct, 0.0),
                "account_betweenness": betweenness.get(acct, 0.0),
                "device_degree": degree.get(dev, 0),
                "device_betweenness_centrality": betweenness.get(dev, 0.0),
                "device_pagerank": pagerank.get(dev, 0.0),
                "device_shared_account_count": degree.get(dev, 0),
                "merchant_degree": degree.get(mer, 0),
                "merchant_cluster_coefficient": clustering.get(mer, 0.0),
                "merchant_betweenness": betweenness.get(mer, 0.0),
                "velocity_ring_score": velocity.get(tx.account_id, 1),
                "component_size": comp_size.get(acct, 1),
                "unique_devices_per_account": _account_neighbour_count(graph, acct, "dev"),
                "unique_merchants_per_account": _account_neighbour_count(graph, acct, "mer"),
            }
        )

    return pd.DataFrame(rows)


def _account_neighbour_count(graph: nx.Graph, acct: str, prefix: str) -> int:
    if acct not in graph:
        return 0
    return sum(1 for nbr in graph.neighbors(acct) if nbr.startswith(f"{prefix}:"))
