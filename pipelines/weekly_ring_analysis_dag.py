"""Airflow DAG — weekly fraud-ring community analysis.

Runs Label-Propagation ring detection over the past week's transaction graph,
surfaces the highest-risk coordinated rings, and writes a report to the data
lake for the investigations team.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None
    PythonOperator = None


DEFAULT_ARGS = {"owner": "fraud-platform", "retries": 1, "retry_delay": timedelta(minutes=10)}


def ring_analysis_callable(**_):
    from src.config import load_config
    from src.graph.graph_builder import build_transaction_graph
    from src.graph.ring_detection import detect_fraud_rings
    from src.streaming.transaction_stream import TransactionStream

    cfg = load_config()
    df = TransactionStream(cfg.stream).build_dataframe()
    graph = build_transaction_graph(df)
    rings = detect_fraud_rings(graph)
    top = [
        {"ring_id": r.ring_id, "size": r.size, "risk_score": r.risk_score}
        for r in rings[:25]
    ]
    return {"rings_detected": len(rings), "top_rings": top}


if DAG is not None:
    with DAG(
        dag_id="weekly_fraud_ring_analysis",
        description="Weekly Label-Propagation fraud-ring detection report.",
        default_args=DEFAULT_ARGS,
        schedule="0 4 * * 1",  # 04:00 every Monday
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["fraud", "graph", "rings"],
    ) as dag:
        PythonOperator(task_id="detect_rings", python_callable=ring_analysis_callable)
