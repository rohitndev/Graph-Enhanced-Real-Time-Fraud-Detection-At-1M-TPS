"""End-to-end fraud-detection pipeline orchestrator.

Wires every layer together into a single runnable flow:

    stream -> graph features + ring detection -> feature store ->
    model training -> inline scoring -> fraud agent -> AWS materialization

Run via ``python main.py`` (see the project README for options).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from src.agent.fraud_agent import FraudInvestigationAgent
from src.aws.aws_connector import AWSConnector
from src.config import PipelineConfig
from src.features.feature_store import FeatureStore
from src.graph.graph_builder import build_transaction_graph, node_key
from src.graph.graph_features import compute_graph_features
from src.graph.ring_detection import FraudRing, detect_fraud_rings
from src.models.score import ScoringService
from src.models.train import train_model
from src.streaming.transaction_stream import TransactionStream


class FraudDetectionPipeline:
    """Coordinates the full streaming fraud-detection workflow."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.feature_store = FeatureStore()
        self.aws = AWSConnector(cfg.aws)
        self.agent = FraudInvestigationAgent(cfg.agent)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        logger.info("=" * 64)
        logger.info("Graph-Enhanced Real-Time Fraud Detection — pipeline start")
        logger.info("=" * 64)

        stream = TransactionStream(self.cfg.stream)
        full_log = stream.build_dataframe()

        # 1) Materialise sliding-window graph features for every transaction.
        #    These exact vectors feed BOTH training and serving — this is how
        #    the Feast feature store eliminates training-serving skew.
        materialized = self._materialize_phase(stream, full_log)

        # 2) Train on the materialised feature vectors.
        logger.info("[2/4] Training the model on materialised features...")
        train_result = train_model(
            materialized["feature_matrix"], materialized["labels"], self.cfg.model
        )

        # 3) Score the stream with the same materialised vectors + run the agent.
        scored_df, rings_found, narratives = self._score_phase(materialized)

        # 4) Materialise outputs locally and (optionally) to AWS.
        summary = self._finalize(train_result, scored_df, rings_found, narratives)
        logger.success("Pipeline complete.")
        return summary

    # ------------------------------------------------------------------
    def _materialize_phase(self, stream: TransactionStream, full_log: pd.DataFrame):
        """One streaming pass: window graph -> features -> ring detection."""

        logger.info("[1/4] Streaming + materialising graph features...")
        feature_chunks: list[pd.DataFrame] = []
        tx_chunks: list[pd.DataFrame] = []
        graph_row_by_tx: dict[str, dict] = {}
        account_to_ring: dict[str, dict] = {}
        all_rings: list[dict] = []
        seen_ring_keys: set[tuple] = set()

        for batch in stream.iter_batches(full_log):
            graph = build_transaction_graph(batch.window_transactions)
            graph_features = compute_graph_features(graph, batch.transactions)
            feature_matrix = self.feature_store.build_feature_matrix(
                batch.transactions, graph_features
            )
            self.feature_store.materialize(feature_matrix)

            feature_chunks.append(feature_matrix)
            tx_chunks.append(batch.transactions)
            graph_row_by_tx.update(graph_features.set_index("tx_id").to_dict("index"))

            rings = detect_fraud_rings(graph)
            for ring in rings:
                ring_dict = {
                    "ring_id": ring.ring_id,
                    "size": ring.size,
                    "accounts": ring.accounts,
                    "merchants": ring.merchants,
                    "shared_device_ratio": ring.shared_device_ratio,
                    "density": ring.density,
                    "risk_score": ring.risk_score,
                }
                for acct in ring.accounts:
                    account_to_ring[acct] = ring_dict
                key = tuple(sorted(ring.accounts))
                if key not in seen_ring_keys:
                    seen_ring_keys.add(key)
                    all_rings.append(ring_dict)

            logger.info(
                "batch {:>3} | window {} -> {} | events {:>4} | rings {:>2}",
                batch.batch_id,
                batch.window_start.strftime("%H:%M"),
                batch.window_end.strftime("%H:%M"),
                len(batch.transactions),
                len(rings),
            )

        feature_matrix = pd.concat(feature_chunks, ignore_index=True)
        tx_meta = pd.concat(tx_chunks, ignore_index=True)
        labels = tx_meta.set_index("tx_id").loc[feature_matrix["tx_id"], "is_fraud"]
        labels = labels.reset_index(drop=True)

        return {
            "feature_matrix": feature_matrix,
            "labels": labels,
            "tx_meta": tx_meta,
            "graph_row_by_tx": graph_row_by_tx,
            "account_to_ring": account_to_ring,
            "all_rings": all_rings,
        }

    # ------------------------------------------------------------------
    def _score_phase(self, materialized: dict):
        from src.features.feature_defs import ALL_FEATURES
        from src.models.ensemble import FraudEnsemble

        ensemble = FraudEnsemble.load(
            self.cfg.model.model_dir / "fraud_ensemble.json", ALL_FEATURES
        )
        scorer = ScoringService(ensemble, self.cfg.model)

        logger.info("[3/4] Scoring transactions + running the fraud agent...")
        scored = scorer.score_batch(materialized["tx_meta"], materialized["feature_matrix"])
        scored_df = pd.DataFrame(asdict(s) for s in scored)

        narratives: list[dict] = []
        graph_row_by_tx = materialized["graph_row_by_tx"]
        account_to_ring = materialized["account_to_ring"]

        high_risk = sorted(
            (s for s in scored if s.is_high_risk), key=lambda s: s.fraud_score, reverse=True
        )
        for s in high_risk:
            if len(narratives) >= self.cfg.agent.max_cases:
                break
            ring_dict = account_to_ring.get(node_key("acct", s.account_id))
            ring = _ring_from_dict(ring_dict)
            graph_row = graph_row_by_tx.get(s.tx_id, {})
            narrative = self.agent.investigate(s, graph_row, ring)
            narratives.append(asdict(narrative))
            if self.aws.enabled:
                self.aws.publish_alert(
                    {
                        "tx_id": s.tx_id,
                        "account_id": s.account_id,
                        "fraud_score": s.fraud_score,
                        "decision": s.decision,
                    }
                )

        return scored_df, materialized["all_rings"], narratives

    # ------------------------------------------------------------------
    def _finalize(self, train_result, scored_df, rings_found, narratives) -> dict:
        logger.info("[4/4] Writing outputs...")
        out = self.cfg.output_dir
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        scores_path = out / f"scored_transactions_{ts}.parquet"
        scored_df.to_parquet(scores_path, index=False)
        (out / f"fraud_rings_{ts}.json").write_text(json.dumps(rings_found, indent=2))
        (out / f"fraud_narratives_{ts}.json").write_text(
            json.dumps(narratives, indent=2, default=str)
        )

        # Detection metrics on the labelled stream.
        metrics = self._metrics(scored_df, train_result)
        (out / f"run_summary_{ts}.json").write_text(json.dumps(metrics, indent=2, default=str))

        # Optional AWS materialisation.
        if self.aws.enabled:
            self.aws.upload_dataframe(scored_df, f"scores/dt={ts}/scored_transactions.parquet")
            self.aws.upload_json(rings_found, f"rings/dt={ts}/fraud_rings.json")
            self.aws.upload_json(narratives, f"narratives/dt={ts}/fraud_narratives.json")
            self.aws.upload_file(
                self.cfg.model.model_dir / "fraud_ensemble.json",
                f"models/dt={ts}/fraud_ensemble.json",
            )

        self._print_report(metrics, narratives)
        return metrics

    def _metrics(self, scored_df: pd.DataFrame, train_result) -> dict:
        total = len(scored_df)
        flagged = scored_df[scored_df["decision"] != "approve"]
        true_fraud = scored_df[scored_df["is_fraud_label"] == 1]
        caught = flagged[flagged["is_fraud_label"] == 1]

        detection_rate = round(len(caught) / max(1, len(true_fraud)), 4)
        false_positive_rate = round(
            len(flagged[flagged["is_fraud_label"] == 0]) / max(1, len(flagged)), 4
        )
        return {
            "transactions_scored": int(total),
            "true_fraud": int(len(true_fraud)),
            "flagged": int(len(flagged)),
            "fraud_caught": int(len(caught)),
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "auc_tabular": train_result.auc_tabular,
            "auc_graph": train_result.auc_graph,
            "auc_uplift": train_result.auc_uplift,
            "top_features": dict(list(train_result.feature_importance.items())[:10]),
            "model_path": train_result.model_path,
            "aws_enabled": self.aws.enabled,
        }

    @staticmethod
    def _print_report(metrics: dict, narratives: list[dict]) -> None:
        print("\n" + "=" * 64)
        print(" RUN SUMMARY")
        print("=" * 64)
        print(f" Transactions scored      : {metrics['transactions_scored']:,}")
        print(f" True fraud in stream      : {metrics['true_fraud']:,}")
        print(f" Flagged (challenge/decline): {metrics['flagged']:,}")
        print(f" Fraud caught              : {metrics['fraud_caught']:,}")
        print(f" Detection rate            : {metrics['detection_rate']:.1%}")
        print(f" False-positive rate       : {metrics['false_positive_rate']:.1%}")
        print("-" * 64)
        print(f" AUC (tabular only)        : {metrics['auc_tabular']:.4f}")
        print(f" AUC (graph-enhanced)      : {metrics['auc_graph']:.4f}")
        print(f" AUC uplift from graph     : +{metrics['auc_uplift']:.4f}")
        print("-" * 64)
        print(f" Fraud narratives generated: {len(narratives)}")
        if narratives:
            print("\n Example narrative:\n")
            print(narratives[0]["narrative"])
        print("=" * 64 + "\n")


def _ring_from_dict(ring_dict: dict | None) -> FraudRing | None:
    """Rehydrate a :class:`FraudRing` from its serialised form for the agent."""

    if ring_dict is None:
        return None
    return FraudRing(
        ring_id=ring_dict["ring_id"],
        accounts=ring_dict["accounts"],
        merchants=ring_dict["merchants"],
        shared_device_ratio=ring_dict["shared_device_ratio"],
        density=ring_dict["density"],
        size=ring_dict["size"],
    )
