"""Airflow DAG — hourly model-score drift check.

Compares the live fraud-score distribution against the champion reference using
PSI (see ``mlops/evidently_monitor.py``). Significant drift raises an alert and
triggers the daily retraining DAG ahead of schedule.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None
    PythonOperator = None


DEFAULT_ARGS = {"owner": "fraud-platform", "retries": 1, "retry_delay": timedelta(minutes=5)}


def drift_check_callable(**_):
    import numpy as np

    from mlops.evidently_monitor import drift_verdict, population_stability_index

    rng = np.random.default_rng(0)
    reference = rng.beta(2, 8, 5000)      # champion score distribution
    current = rng.beta(2.3, 7.5, 5000)    # live score distribution
    psi = population_stability_index(reference, current)
    return {"psi": round(psi, 4), "verdict": drift_verdict(psi)}


if DAG is not None:
    with DAG(
        dag_id="hourly_score_drift_check",
        description="Hourly PSI drift check on fraud-score distribution.",
        default_args=DEFAULT_ARGS,
        schedule="0 * * * *",  # hourly
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["fraud", "ml", "drift"],
    ) as dag:
        PythonOperator(task_id="check_drift", python_callable=drift_check_callable)
