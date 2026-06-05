"""Micro-batch transaction source emulating PySpark Structured Streaming.

The production pipeline consumes a Kafka topic with PySpark Structured
Streaming using watermarks and checkpointing for exactly-once processing. To
keep the project runnable anywhere (no broker, no Spark cluster required) this
module replays a synthetic transaction stream in **micro-batches** with the same
semantics the streaming job relies on:

* a sliding **event-time window** (default 1 hour) used by the graph layer,
* a monotonic **watermark** that advances with each batch,
* **at-least-once** delivery with a de-duplication set that upgrades the
  effective guarantee to **exactly-once** scoring.

When ``KAFKA_BOOTSTRAP_SERVERS`` is configured and ``confluent-kafka`` is
installed, the real Kafka path in :mod:`src.streaming.kafka_connector` can be
used instead of the replay source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import pandas as pd
from loguru import logger

from data.generate_data import GeneratorParams, TransactionGenerator
from src.config import StreamConfig


@dataclass
class MicroBatch:
    """A single Structured-Streaming micro-batch."""

    batch_id: int
    transactions: pd.DataFrame  # fresh, de-duplicated events to score
    window_transactions: pd.DataFrame  # all events in the sliding graph window
    watermark: datetime
    window_start: datetime
    window_end: datetime

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.transactions)


class TransactionStream:
    """Replays transactions as watermarked micro-batches."""

    def __init__(self, cfg: StreamConfig):
        self.cfg = cfg
        self._seen_tx_ids: set[str] = set()  # exactly-once de-dup set

    def build_dataframe(self) -> pd.DataFrame:
        """Materialise the full ordered transaction log for this run."""

        params = GeneratorParams(
            total_transactions=self.cfg.total_transactions,
            n_accounts=self.cfg.n_accounts,
            n_devices=self.cfg.n_devices,
            n_merchants=self.cfg.n_merchants,
            n_fraud_rings=self.cfg.n_fraud_rings,
            fraud_ratio=self.cfg.fraud_ratio,
            window_minutes=self.cfg.window_minutes,
            seed=self.cfg.seed,
        )
        df = TransactionGenerator(params).to_frame()
        return df.sort_values("event_time").reset_index(drop=True)

    def iter_batches(self, df: pd.DataFrame | None = None) -> Iterator[MicroBatch]:
        """Yield watermarked micro-batches with exactly-once de-duplication."""

        if df is None:
            df = self.build_dataframe()
        window = timedelta(minutes=self.cfg.window_minutes)
        batch_size = self.cfg.micro_batch_size
        n_batches = (len(df) + batch_size - 1) // batch_size

        logger.info(
            "Streaming {n:,} transactions in {b} micro-batches "
            "(batch={bs}, window={w}min)",
            n=len(df),
            b=n_batches,
            bs=batch_size,
            w=self.cfg.window_minutes,
        )

        for batch_id in range(n_batches):
            chunk = df.iloc[batch_id * batch_size : (batch_id + 1) * batch_size]

            # Exactly-once: drop any transaction id already processed.
            fresh = chunk[~chunk["tx_id"].isin(self._seen_tx_ids)]
            self._seen_tx_ids.update(fresh["tx_id"].tolist())
            if fresh.empty:
                continue

            watermark = fresh["event_time"].max()
            window_end = watermark
            window_start = window_end - window
            window_view = df[
                (df["event_time"] > window_start) & (df["event_time"] <= window_end)
            ].reset_index(drop=True)

            yield MicroBatch(
                batch_id=batch_id,
                transactions=fresh.reset_index(drop=True),
                window_transactions=window_view,
                watermark=watermark,
                window_start=window_start,
                window_end=window_end,
            )
