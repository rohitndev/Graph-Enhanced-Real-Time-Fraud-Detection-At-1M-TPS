"""Model layer: gradient-boosted trainers, ensemble combiner, scoring service."""

from .train import train_model, TrainResult
from .ensemble import FraudEnsemble
from .score import ScoringService, ScoredTransaction

__all__ = [
    "train_model",
    "TrainResult",
    "FraudEnsemble",
    "ScoringService",
    "ScoredTransaction",
]
