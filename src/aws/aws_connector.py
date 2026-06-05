"""AWS cloud connectivity for the fraud-detection pipeline.

The pipeline is cloud-ready: with ``AWS_ENABLED=true`` it materialises scored
transactions, detected fraud rings and model artifacts to an **S3** data lake and
can publish high-risk alerts to a **Kinesis** stream. Every method degrades to a
safe no-op when AWS is disabled or credentials are absent, so the pipeline always
runs locally first.

Required environment to enable:
    AWS_ENABLED=true
    AWS_REGION=us-east-1
    AWS_S3_BUCKET=your-bucket
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   (or an AWS_PROFILE / IAM role)
    AWS_KINESIS_STREAM=fraud-alerts             (optional)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import AWSConfig


class AWSConnector:
    """Thin boto3 wrapper for S3 + Kinesis, safe to use when AWS is disabled."""

    def __init__(self, cfg: AWSConfig):
        self.cfg = cfg
        self._s3 = None
        self._kinesis = None
        if cfg.enabled:
            self._init_clients()
        else:
            logger.info("AWS disabled — cloud writes will be skipped (local-only run).")

    # ------------------------------------------------------------------
    def _session(self):
        import boto3

        if self.cfg.profile:
            return boto3.Session(profile_name=self.cfg.profile, region_name=self.cfg.region)
        return boto3.Session(region_name=self.cfg.region)

    def _init_clients(self) -> None:
        try:
            session = self._session()
            self._s3 = session.client("s3")
            if self.cfg.kinesis_stream:
                self._kinesis = session.client("kinesis")
            logger.success("AWS clients initialised (region={}).", self.cfg.region)
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Could not initialise AWS clients ({}); disabling.", exc)
            self.cfg.enabled = False

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and self._s3 is not None

    def _key(self, *parts: str) -> str:
        return "/".join([self.cfg.s3_prefix.strip("/"), *parts])

    # ------------------------------------------------------------------
    # S3 writes
    # ------------------------------------------------------------------
    def upload_dataframe(self, df: pd.DataFrame, key: str) -> str | None:
        """Upload a DataFrame to S3 as Parquet. Returns the s3:// URI or None."""

        if not self.enabled:
            return None
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        full_key = self._key(key)
        self._s3.put_object(Bucket=self.cfg.s3_bucket, Key=full_key, Body=buffer.getvalue())
        uri = f"s3://{self.cfg.s3_bucket}/{full_key}"
        logger.info("Uploaded {} rows -> {}", len(df), uri)
        return uri

    def upload_json(self, obj: dict | list, key: str) -> str | None:
        if not self.enabled:
            return None
        full_key = self._key(key)
        self._s3.put_object(
            Bucket=self.cfg.s3_bucket,
            Key=full_key,
            Body=json.dumps(obj, indent=2, default=str).encode(),
        )
        uri = f"s3://{self.cfg.s3_bucket}/{full_key}"
        logger.info("Uploaded JSON -> {}", uri)
        return uri

    def upload_file(self, path: str | Path, key: str) -> str | None:
        if not self.enabled:
            return None
        full_key = self._key(key)
        self._s3.upload_file(str(path), self.cfg.s3_bucket, full_key)
        return f"s3://{self.cfg.s3_bucket}/{full_key}"

    # ------------------------------------------------------------------
    # Kinesis high-risk alerts
    # ------------------------------------------------------------------
    def publish_alert(self, alert: dict) -> bool:
        """Publish a high-risk fraud alert to Kinesis. Returns success flag."""

        if not self.enabled or self._kinesis is None:
            return False
        try:  # pragma: no cover - depends on env
            self._kinesis.put_record(
                StreamName=self.cfg.kinesis_stream,
                Data=json.dumps(alert, default=str).encode(),
                PartitionKey=str(alert.get("account_id", "unknown")),
            )
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("Kinesis publish failed: {}", exc)
            return False
