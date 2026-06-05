"""Builds the heterogeneous transaction graph for a sliding window.

Nodes are typed: ``account``, ``device`` and ``merchant``. An edge connects an
account to the device and the merchant used in each transaction. In production
this is computed with Spark GraphFrames on each micro-batch; here we use
NetworkX so the project runs without a Spark cluster. The resulting graph is the
substrate for both the 15 graph features and Label-Propagation ring detection.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def node_key(node_type: str, node_id: str) -> str:
    """Namespaced node identifier, e.g. ``acct:A000123``."""

    return f"{node_type}:{node_id}"


def build_transaction_graph(window_df: pd.DataFrame) -> nx.Graph:
    """Construct an undirected, weighted transaction graph for one window.

    Parameters
    ----------
    window_df:
        Transactions inside the current sliding window. Must contain
        ``account_id``, ``device_id``, ``merchant_id`` and ``amount`` columns.
    """

    graph = nx.Graph()

    for row in window_df.itertuples(index=False):
        acct = node_key("acct", row.account_id)
        dev = node_key("dev", row.device_id)
        mer = node_key("mer", row.merchant_id)

        graph.add_node(acct, ntype="account")
        graph.add_node(dev, ntype="device")
        graph.add_node(mer, ntype="merchant")

        _bump_edge(graph, acct, dev, row.amount)
        _bump_edge(graph, acct, mer, row.amount)

    return graph


def _bump_edge(graph: nx.Graph, u: str, v: str, amount: float) -> None:
    """Increment edge weight / transaction count between two nodes."""

    if graph.has_edge(u, v):
        graph[u][v]["weight"] += 1
        graph[u][v]["amount"] += float(amount)
    else:
        graph.add_edge(u, v, weight=1, amount=float(amount))


def nodes_by_type(graph: nx.Graph, ntype: str) -> list[str]:
    return [n for n, d in graph.nodes(data=True) if d.get("ntype") == ntype]
