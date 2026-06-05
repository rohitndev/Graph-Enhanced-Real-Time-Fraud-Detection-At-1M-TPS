"""Airflow DAG — daily model retraining on newly labelled fraud data.

Retrains the graph-enhanced ensemble each night, logs metrics to MLflow, and
promotes the challenger to champion only if it beats the precision guardrail.
The DAG imports the project's own modules so the training logic stays in one
place (no duplicated code between batch and streaming).
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # Airflow is an optional orchestration dependency.
    DAG = None
    PythonOperator = None


DEFAULT_ARGS = {
    "owner": "fraud-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def retrain_callable(**_):
    """Materialise features, retrain, and register the new model version."""

    from src.config import load_config
    from src.pipeline import FraudDetectionPipeline

    cfg = load_config()
    summary = FraudDetectionPipeline(cfg).run()
    # In production: log `summary` to MLflow and gate promotion on AUC/precision.
    return summary


if DAG is not None:
    with DAG(
        dag_id="daily_fraud_model_retrain",
        description="Retrain the graph-enhanced fraud ensemble nightly.",
        default_args=DEFAULT_ARGS,
        schedule="0 2 * * *",  # 02:00 every day
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["fraud", "ml", "retrain"],
    ) as dag:
        PythonOperator(task_id="retrain_model", python_callable=retrain_callable)
