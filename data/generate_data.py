"""Synthetic transaction generator with embedded coordinated fraud rings.

This mirrors the structure of the PaySim / IEEE-CIS datasets referenced in the
project design: each transaction links an *account*, a *device*, and a
*merchant*. Legitimate traffic spreads naturally across the graph, while a small
number of **fraud rings** share devices and hammer a handful of merchants in
short bursts — exactly the coordinated pattern that is only visible in the
transaction graph, not in any single transaction's features.

Run standalone to materialise a Parquet/CSV dataset::

    python -m data.generate_data --rows 50000 --out data/transactions.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
CHANNELS = ["web", "mobile", "pos", "atm"]


@dataclass
class GeneratorParams:
    total_transactions: int = 20_000
    n_accounts: int = 4_000
    n_devices: int = 2_500
    n_merchants: int = 800
    n_fraud_rings: int = 12
    fraud_ratio: float = 0.012
    window_minutes: int = 60
    seed: int = 42


class TransactionGenerator:
    """Generates a realistic transaction graph with planted fraud rings."""

    def __init__(self, params: GeneratorParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self._fraud_ring_accounts = self._build_fraud_rings()
        # Legitimate "hub" devices: public terminals / shared family devices
        # that many honest accounts use. These create benign high-degree nodes
        # so device sharing is a strong-but-imperfect fraud signal (the graph
        # model must combine several features, not memorise one).
        self._hub_devices = self.rng.choice(
            params.n_devices, size=max(5, params.n_devices // 80), replace=False
        )

    # ------------------------------------------------------------------
    # Fraud-ring construction
    # ------------------------------------------------------------------
    def _build_fraud_rings(self) -> dict[int, dict]:
        """Assign clusters of accounts that share a small pool of devices."""

        rings: dict[int, dict] = {}
        for ring_id in range(self.p.n_fraud_rings):
            ring_size = int(self.rng.integers(3, 9))  # 3+ accounts per the design
            accounts = self.rng.choice(self.p.n_accounts, size=ring_size, replace=False)
            # A ring shares a tiny device pool (device clustering signal):
            # many accounts funnel through 1-2 devices, which is the strongest
            # graph signal of a coordinated mule ring.
            shared_devices = self.rng.choice(
                self.p.n_devices, size=max(1, ring_size // 3), replace=False
            )
            # And targets a small set of mule merchants.
            target_merchants = self.rng.choice(self.p.n_merchants, size=3, replace=False)
            for acc in accounts:
                rings[int(acc)] = {
                    "ring_id": ring_id,
                    "devices": shared_devices,
                    "merchants": target_merchants,
                }
        return rings

    # ------------------------------------------------------------------
    # Single-transaction sampling
    # ------------------------------------------------------------------
    def _sample_legit(self, ts: float) -> dict:
        account = int(self.rng.integers(0, self.p.n_accounts))
        # ~15% of honest traffic flows through a shared hub device.
        if self.rng.random() < 0.15:
            device = int(self.rng.choice(self._hub_devices))
        else:
            device = int(self.rng.integers(0, self.p.n_devices))
        merchant = int(self.rng.integers(0, self.p.n_merchants))
        amount = float(np.round(self.rng.gamma(2.0, 45.0), 2))
        return {
            "account_id": f"A{account:06d}",
            "device_id": f"D{device:06d}",
            "merchant_id": f"M{merchant:05d}",
            "amount": amount,
            "tx_type": self.rng.choice(TX_TYPES, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
            "channel": self.rng.choice(CHANNELS, p=[0.4, 0.4, 0.15, 0.05]),
            "timestamp": ts,
            "is_fraud": 0,
            "ring_id": -1,
        }

    def _sample_fraud(self, ts: float) -> dict:
        account = int(self.rng.choice(list(self._fraud_ring_accounts.keys())))
        ring = self._fraud_ring_accounts[account]
        device = int(self.rng.choice(ring["devices"]))
        merchant = int(self.rng.choice(ring["merchants"]))
        # IMPORTANT: fraud is deliberately *camouflaged* on transactional
        # features — amounts, types and channels mirror legitimate traffic so
        # the fraud is almost invisible to a tabular-only model. The signal
        # lives entirely in the graph structure: the ring's shared devices and
        # repeated targeting of a small merchant pool. This is what lets the
        # graph-enhanced model out-perform the tabular baseline.
        amount = float(np.round(self.rng.gamma(2.0, 45.0) * self.rng.uniform(2.2, 3.6), 2))
        return {
            "account_id": f"A{account:06d}",
            "device_id": f"D{device:06d}",
            "merchant_id": f"M{merchant:05d}",
            "amount": amount,
            "tx_type": self.rng.choice(TX_TYPES, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
            "channel": self.rng.choice(CHANNELS, p=[0.4, 0.4, 0.15, 0.05]),
            "timestamp": ts,
            "is_fraud": 1,
            "ring_id": int(ring["ring_id"]),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_transactions(self) -> Iterator[dict]:
        """Yield transactions one at a time with monotonically rising timestamps."""

        base_ts = 1_700_000_000.0  # arbitrary epoch start
        # Spread events across several windows for realistic sliding-window graphs.
        seconds_span = self.p.window_minutes * 60 * 6
        step = seconds_span / max(1, self.p.total_transactions)
        for i in range(self.p.total_transactions):
            ts = base_ts + i * step + float(self.rng.uniform(0, step))
            if self.rng.random() < self.p.fraud_ratio:
                yield self._sample_fraud(ts)
            else:
                yield self._sample_legit(ts)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.iter_transactions())
        df.insert(0, "tx_id", [f"T{idx:09d}" for idx in range(len(df))])
        df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")
        return df


def generate_dataframe(params: GeneratorParams | None = None) -> pd.DataFrame:
    """Convenience helper returning a fully built transaction DataFrame."""

    return TransactionGenerator(params or GeneratorParams()).to_frame()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud transactions.")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--rings", type=int, default=12)
    parser.add_argument("--fraud-ratio", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/transactions.parquet")
    args = parser.parse_args()

    params = GeneratorParams(
        total_transactions=args.rows,
        n_fraud_rings=args.rings,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
    )
    df = generate_dataframe(params)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    fraud = int(df["is_fraud"].sum())
    print(f"Wrote {len(df):,} transactions ({fraud:,} fraud) -> {out}")


if __name__ == "__main__":
    _main()
