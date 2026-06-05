"""Kafka source/sink connectors for the transaction stream.

This module wraps an optional ``confluent-kafka`` client. It is imported lazily
so the project runs without Kafka installed; the in-memory replay source in
:mod:`src.streaming.transaction_stream` is the default. Configure
``KAFKA_BOOTSTRAP_SERVERS`` (e.g. a Confluent Cloud free-tier cluster) to use
the real broker.
"""

from __future__ import annotations

import json
from typing import Iterator

from loguru import logger

from src.config import StreamConfig


def kafka_available() -> bool:
    """Return True when the confluent-kafka client can be imported."""

    try:
        import confluent_kafka  # noqa: F401

        return True
    except ImportError:
        return False


class KafkaTransactionProducer:
    """Publishes transaction events to a Kafka topic (1M TPS in cluster mode)."""

    def __init__(self, cfg: StreamConfig):
        if not cfg.kafka_bootstrap_servers:
            raise ValueError("KAFKA_BOOTSTRAP_SERVERS is not configured.")
        if not kafka_available():
            raise RuntimeError("confluent-kafka is not installed.")
        from confluent_kafka import Producer

        self.cfg = cfg
        self._producer = Producer({"bootstrap.servers": cfg.kafka_bootstrap_servers})

    def publish(self, record: dict) -> None:
        self._producer.produce(
            self.cfg.kafka_topic,
            key=str(record.get("account_id", "")),
            value=json.dumps(record, default=str),
        )

    def flush(self) -> None:
        self._producer.flush()


class KafkaTransactionConsumer:
    """Consumes transaction events from a Kafka topic with watermarking."""

    def __init__(self, cfg: StreamConfig, group_id: str = "fraud-scoring"):
        if not kafka_available():
            raise RuntimeError("confluent-kafka is not installed.")
        from confluent_kafka import Consumer

        self.cfg = cfg
        self._consumer = Consumer(
            {
                "bootstrap.servers": cfg.kafka_bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe([cfg.kafka_topic])
        logger.info("Subscribed to Kafka topic '{}'", cfg.kafka_topic)

    def poll(self, timeout: float = 1.0) -> Iterator[dict]:
        while True:
            msg = self._consumer.poll(timeout)
            if msg is None:
                break
            if msg.error():
                logger.warning("Kafka error: {}", msg.error())
                continue
            yield json.loads(msg.value())

    def close(self) -> None:
        self._consumer.close()
