from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.state import MarketState


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str = "ok"


class PredicateRuleEngine:
    """Economic constraints and transition-level truth filter."""

    def __init__(self, max_capacity: float = 10_000.0) -> None:
        self._max_capacity = max_capacity

    def validate_transition(self, current: MarketState, nxt: MarketState, price_action: float) -> ValidationResult:
        if nxt.price < 0 or nxt.supply < 0 or nxt.demand < 0:
            return ValidationResult(False, "negative value")
        if nxt.supply > self._max_capacity:
            return ValidationResult(False, "capacity exceeded")
        if price_action > 0 and nxt.demand > current.demand:
            return ValidationResult(False, "law of demand violated")
        if price_action < 0 and nxt.supply > current.supply:
            return ValidationResult(False, "law of supply violated")
        return ValidationResult(True)


class VectorStoreFacade:
    """Primary/fallback vector DB abstraction (Pinecone/FAISS)."""

    def __init__(self) -> None:
        self.primary_name = "pinecone"
        self.fallback_name = "faiss"
        self.primary_available = False

    def upsert(self, key: str, vector: list[float]) -> str:
        if self.primary_available:
            return f"{self.primary_name}:{key}"
        return f"{self.fallback_name}:{key}"
