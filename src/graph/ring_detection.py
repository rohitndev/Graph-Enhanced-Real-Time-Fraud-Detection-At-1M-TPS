"""Fraud-ring detection via Label Propagation community detection.

Runs the Label Propagation Algorithm (LPA) on the hourly transaction-graph
snapshot — the same algorithm provided by Spark GraphX in production — to surface
communities of 3+ accounts that share devices and target the same merchants.
Each detected community becomes a candidate **fraud ring** scored by its
internal density and shared-device concentration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .graph_builder import node_key


@dataclass
class FraudRing:
    """A coordinated community detected in the transaction graph."""

    ring_id: int
    accounts: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    merchants: list[str] = field(default_factory=list)
    shared_device_ratio: float = 0.0
    density: float = 0.0
    size: int = 0

    @property
    def risk_score(self) -> float:
        """Heuristic ring risk in [0, 1] from density + device sharing."""

        return round(min(1.0, 0.5 * self.density + 0.5 * self.shared_device_ratio), 4)


def detect_fraud_rings(graph: nx.Graph, min_accounts: int = 3) -> list[FraudRing]:
    """Detect fraud-ring communities containing at least ``min_accounts`` accounts."""

    if graph.number_of_nodes() == 0:
        return []

    communities = nx.community.label_propagation_communities(graph)

    rings: list[FraudRing] = []
    ring_id = 0
    for community in communities:
        accounts = [n for n in community if n.startswith("acct:")]
        if len(accounts) < min_accounts:
            continue

        devices = [n for n in community if n.startswith("dev:")]
        merchants = [n for n in community if n.startswith("mer:")]
        sub = graph.subgraph(community)

        # Shared-device ratio: fewer devices than accounts => heavy sharing.
        shared_device_ratio = 0.0
        if accounts:
            shared_device_ratio = 1.0 - min(1.0, len(devices) / len(accounts))

        rings.append(
            FraudRing(
                ring_id=ring_id,
                accounts=accounts,
                devices=devices,
                merchants=merchants,
                shared_device_ratio=round(shared_device_ratio, 4),
                density=round(nx.density(sub), 4),
                size=len(accounts),
            )
        )
        ring_id += 1

    # Most suspicious rings first.
    rings.sort(key=lambda r: r.risk_score, reverse=True)
    return rings


def account_ring_lookup(rings: list[FraudRing]) -> dict[str, FraudRing]:
    """Map each account node to the ring it belongs to (for feature joins)."""

    lookup: dict[str, FraudRing] = {}
    for ring in rings:
        for acct in ring.accounts:
            lookup[acct] = ring
    return lookup
