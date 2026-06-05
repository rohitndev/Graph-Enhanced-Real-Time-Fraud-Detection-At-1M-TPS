"""Production PySpark Structured Streaming job (cluster mode).

This is the 1M-TPS path: it reads the Kafka transaction topic with watermarks
and checkpointing for exactly-once processing, and hands each micro-batch to the
graph + scoring layers via ``foreachBatch``. PySpark is an optional dependency,
so this module imports it lazily — the default local demo uses the pure-Python
micro-batch source in :mod:`src.streaming.transaction_stream`.

Submit on Databricks / a Spark cluster::

    spark-submit src/streaming/spark_streaming_job.py
"""

from __future__ import annotations

from loguru import logger

from src.config import load_config

# JSON schema of the Kafka transaction value.
TRANSACTION_SCHEMA = (
    "tx_id STRING, account_id STRING, device_id STRING, merchant_id STRING, "
    "amount DOUBLE, tx_type STRING, channel STRING, event_time TIMESTAMP"
)


def build_spark():  # pragma: no cover - requires a Spark runtime
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("graph-enhanced-fraud-detection")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/fraud-checkpoint")
        .getOrCreate()
    )


def run_stream():  # pragma: no cover - requires a Spark runtime
    from pyspark.sql.functions import col, from_json

    cfg = load_config()
    spark = build_spark()

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", cfg.stream.kafka_bootstrap_servers)
        .option("subscribe", cfg.stream.kafka_topic)
        .option("startingOffsets", "latest")
        .load()
    )

    transactions = (
        raw.select(from_json(col("value").cast("string"), TRANSACTION_SCHEMA).alias("tx"))
        .select("tx.*")
        # Watermark for late arrivals over the sliding graph window.
        .withWatermark("event_time", f"{cfg.stream.window_minutes} minutes")
    )

    def process_batch(batch_df, batch_id: int):
        """Score one micro-batch with the graph + model layers."""
        import pandas as pd

        from src.features.feature_store import FeatureStore
        from src.graph.graph_builder import build_transaction_graph
        from src.graph.graph_features import compute_graph_features
        from src.models.ensemble import FraudEnsemble
        from src.features.feature_defs import ALL_FEATURES
        from src.models.score import ScoringService

        pdf: pd.DataFrame = batch_df.toPandas()
        if pdf.empty:
            return
        graph = build_transaction_graph(pdf)
        feats = compute_graph_features(graph, pdf)
        fm = FeatureStore().build_feature_matrix(pdf, feats)
        ensemble = FraudEnsemble.load(
            cfg.model.model_dir / "fraud_ensemble.json", ALL_FEATURES
        )
        scored = ScoringService(ensemble, cfg.model).score_batch(pdf, fm)
        logger.info("batch {} scored {} transactions", batch_id, len(scored))

    query = (
        transactions.writeStream.foreachBatch(process_batch)
        .outputMode("update")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":  # pragma: no cover
    run_stream()
