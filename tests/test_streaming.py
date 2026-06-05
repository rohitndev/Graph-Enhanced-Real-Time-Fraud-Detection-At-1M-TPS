"""Integration tests for the watermarked micro-batch stream."""

from __future__ import annotations

from src.config import StreamConfig
from src.streaming.transaction_stream import TransactionStream


def _small_cfg() -> StreamConfig:
    cfg = StreamConfig()
    cfg.total_transactions = 1_000
    cfg.micro_batch_size = 250
    cfg.seed = 7
    return cfg


def test_stream_yields_all_transactions_exactly_once():
    stream = TransactionStream(_small_cfg())
    seen: set[str] = set()
    total = 0
    for batch in stream.iter_batches():
        total += len(batch)
        ids = set(batch.transactions["tx_id"])
        # Exactly-once: no transaction id appears in two batches.
        assert seen.isdisjoint(ids)
        seen.update(ids)
    assert total == 1_000
    assert len(seen) == 1_000


def test_watermark_is_monotonic():
    stream = TransactionStream(_small_cfg())
    last = None
    for batch in stream.iter_batches():
        if last is not None:
            assert batch.watermark >= last
        last = batch.watermark


def test_window_bounds_consistent():
    stream = TransactionStream(_small_cfg())
    for batch in stream.iter_batches():
        assert batch.window_start < batch.window_end
        # The sliding window is never empty and every event in it falls
        # within (window_start, window_end].
        assert len(batch.window_transactions) >= 1
        assert batch.window_transactions["event_time"].min() > batch.window_start
        assert batch.window_transactions["event_time"].max() <= batch.window_end
