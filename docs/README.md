# Documentation Images

The README references the following images. Generate each one (e.g. in ChatGPT)
and save it in this `docs/` folder using the **exact filename** below. All
diagrams use a clean, white-background, professional infographic style with the
correct tool logos/icons.

| # | Filename | Purpose |
| - | -------- | ------- |
| 1 | `project-overview.png` | Banner: end-to-end fraud-detection platform overview |
| 2 | `architecture.png` | High-level layered architecture (stream → graph → model → agent → cloud) |
| 3 | `data-flow.png` | Data flow from Kafka through scoring to outputs |
| 4 | `pipeline-stages.png` | The 4 stages: materialize → train → score → investigate |
| 5 | `streaming-watermark.png` | Micro-batches, sliding window, watermark, exactly-once |
| 6 | `transaction-graph.png` | Heterogeneous account/device/merchant graph with a fraud ring |
| 7 | `graph-features.png` | The 15 graph features grouped by node role |
| 8 | `ring-detection.png` | Label-Propagation fraud-ring community detection |
| 9 | `feature-store.png` | Feast-style store + training-serving skew elimination |
| 10 | `auc-comparison.png` | AUC: tabular-only vs graph-enhanced |
| 11 | `shap-importance.png` | Feature importance — graph features dominate |
| 12 | `scoring-decision.png` | Decision routing: approve / challenge / decline |
| 13 | `fraud-agent.png` | LangChain agent generating a fraud narrative |
| 14 | `aws-architecture.png` | AWS deployment: S3, Kinesis, MSK, IAM |
| 15 | `grafana-dashboard.png` | Real-time fraud monitoring dashboard |
| 16 | `run-summary.png` | Terminal run-summary output |

The exact generation prompts are provided alongside the project hand-off notes.
