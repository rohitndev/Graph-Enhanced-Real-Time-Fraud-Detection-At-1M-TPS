"""Entrypoint for the Graph-Enhanced Real-Time Fraud Detection pipeline.

Usage examples
--------------
Run the full pipeline on synthetic data::

    python main.py

Override stream size / fraud ratio via environment or flags::

    python main.py --transactions 40000 --batch 4000

Enable AWS materialisation (requires AWS_* env vars)::

    AWS_ENABLED=true AWS_S3_BUCKET=my-bucket python main.py
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from src.config import load_config
from src.pipeline import FraudDetectionPipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph-Enhanced Real-Time Fraud Detection at 1M TPS"
    )
    parser.add_argument("--transactions", type=int, help="Total transactions to stream.")
    parser.add_argument("--batch", type=int, help="Micro-batch size.")
    parser.add_argument("--fraud-ratio", type=float, help="Fraction of fraudulent transactions.")
    parser.add_argument("--rings", type=int, help="Number of planted fraud rings.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument(
        "--log-level", default="INFO", help="Loguru level (DEBUG/INFO/WARNING)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, format="<level>{message}</level>")

    cfg = load_config()
    if args.transactions is not None:
        cfg.stream.total_transactions = args.transactions
    if args.batch is not None:
        cfg.stream.micro_batch_size = args.batch
    if args.fraud_ratio is not None:
        cfg.stream.fraud_ratio = args.fraud_ratio
    if args.rings is not None:
        cfg.stream.n_fraud_rings = args.rings
    if args.seed is not None:
        cfg.stream.seed = args.seed

    pipeline = FraudDetectionPipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
