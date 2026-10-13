"""Configurable scoring weights for memory recall."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ScoringConfig:
    """Configurable scoring weights for memory recall."""

    weight_similarity: float = 0.70
    weight_importance: float = 0.15
    weight_recency: float = 0.10
    weight_reinforcement: float = 0.05
    recency_half_life_days: float = 7.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
