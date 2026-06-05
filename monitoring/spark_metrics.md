# Spark Structured Streaming Metrics

Custom metrics exported from the streaming job and scraped by Prometheus, then
visualised in `grafana_fraud_dashboard.json`.

| Metric                                   | Type    | Description                                  |
| ---------------------------------------- | ------- | -------------------------------------------- |
| `transactions_scored_total`              | counter | Total transactions scored (for TPS rate).    |
| `fraud_flagged_total{channel}`           | counter | Transactions flagged, labelled by channel.   |
| `fraud_detection_rate`                   | gauge   | Caught fraud / total fraud in window.        |
| `fraud_false_positive_rate`              | gauge   | False positives / flagged transactions.      |
| `fraud_rings_detected`                   | gauge   | Fraud-ring communities detected per window.  |
| `spark_streaming_watermark_lag_seconds`  | gauge   | Event-time lag behind wall-clock.            |
| `fraud_score_psi`                        | gauge   | PSI drift of the fraud-score distribution.   |

## Enabling Spark metrics

Configure the Spark streaming job with the Prometheus servlet sink:

```properties
# metrics.properties
*.sink.prometheusServlet.class=org.apache.spark.metrics.sink.PrometheusServlet
*.sink.prometheusServlet.path=/metrics/prometheus
```

The application-level fraud metrics (`fraud_detection_rate`,
`fraud_false_positive_rate`, etc.) are emitted from the pipeline's run summary
(`output/run_summary_*.json`) and pushed via a Prometheus pushgateway.
