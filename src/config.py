"""Central configuration for the fraud-detection pipeline.

Values are read from environment variables (optionally loaded from a local
``.env`` file) with sensible defaults, so the pipeline runs out-of-the-box
without any cloud credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if present (non-fatal when missing).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class StreamConfig:
    """Transaction stream settings (simulated Kafka source)."""

    total_transactions: int = field(default_factory=lambda: _env_int("STREAM_TOTAL", 20_000))
    micro_batch_size: int = field(default_factory=lambda: _env_int("STREAM_BATCH", 2_000))
    fraud_ratio: float = field(default_factory=lambda: _env_float("STREAM_FRAUD_RATIO", 0.012))
    n_accounts: int = field(default_factory=lambda: _env_int("STREAM_ACCOUNTS", 4_000))
    n_devices: int = field(default_factory=lambda: _env_int("STREAM_DEVICES", 2_500))
    n_merchants: int = field(default_factory=lambda: _env_int("STREAM_MERCHANTS", 800))
    n_fraud_rings: int = field(default_factory=lambda: _env_int("STREAM_RINGS", 12))
    window_minutes: int = field(default_factory=lambda: _env_int("GRAPH_WINDOW_MINUTES", 60))
    seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))

    # Kafka connection (only used when a real broker is configured).
    kafka_bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    )
    kafka_topic: str = field(default_factory=lambda: os.getenv("KAFKA_TOPIC", "transactions"))


@dataclass
class ModelConfig:
    """Gradient-boosted scoring model settings."""

    high_risk_threshold: float = field(
        default_factory=lambda: _env_float("MODEL_HIGH_RISK_THRESHOLD", 0.85)
    )
    decline_threshold: float = field(
        default_factory=lambda: _env_float("MODEL_DECLINE_THRESHOLD", 0.85)
    )
    challenge_threshold: float = field(
        default_factory=lambda: _env_float("MODEL_CHALLENGE_THRESHOLD", 0.45)
    )
    n_estimators: int = field(default_factory=lambda: _env_int("MODEL_N_ESTIMATORS", 300))
    max_depth: int = field(default_factory=lambda: _env_int("MODEL_MAX_DEPTH", 6))
    learning_rate: float = field(default_factory=lambda: _env_float("MODEL_LR", 0.1))
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "src" / "models" / "artifacts")


@dataclass
class AWSConfig:
    """AWS cloud connectivity settings.

    When ``enabled`` is False (the default) every cloud call is a no-op, so the
    pipeline runs locally without credentials. Set ``AWS_ENABLED=true`` and the
    relevant variables to materialise scores/artifacts to S3 and stream through
    Kinesis.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("AWS_ENABLED", False))
    region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    s3_bucket: str = field(default_factory=lambda: os.getenv("AWS_S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: os.getenv("AWS_S3_PREFIX", "fraud-detection"))
    kinesis_stream: str = field(default_factory=lambda: os.getenv("AWS_KINESIS_STREAM", ""))
    profile: str = field(default_factory=lambda: os.getenv("AWS_PROFILE", ""))


@dataclass
class AgentConfig:
    """LangChain fraud-investigation agent settings."""

    enabled: bool = field(default_factory=lambda: _env_bool("AGENT_ENABLED", True))
    provider: str = field(default_factory=lambda: os.getenv("AGENT_PROVIDER", "template"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "mixtral-8x7b-32768"))
    max_cases: int = field(default_factory=lambda: _env_int("AGENT_MAX_CASES", 10))


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating every sub-config."""

    stream: StreamConfig = field(default_factory=StreamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> PipelineConfig:
    """Build a :class:`PipelineConfig` from the current environment."""

    cfg = PipelineConfig()
    cfg.ensure_dirs()
    return cfg
