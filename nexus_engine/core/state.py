from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256


class MarketRegime(str, Enum):
    BULL = "Bull"
    BEAR = "Bear"
    STAGNANT = "Stagnant"


@dataclass(frozen=True)
class MarketState:
    """Immutable canonical state for search and planning."""

    price: float
    supply: float
    demand: float
    timestamp: int
    regime: MarketRegime

    def __post_init__(self) -> None:
        if min(self.price, self.supply, self.demand, self.timestamp) < 0:
            raise ValueError("MarketState fields must be non-negative")

    def __lt__(self, other: object) -> bool:
        """Deterministic tie-break ordering for heap safety."""
        if not isinstance(other, MarketState):
            return NotImplemented
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        if self.price != other.price:
            return self.price < other.price
        return self.bin_hash() < other.bin_hash()

    def bin_hash(self, near_equilibrium_threshold: float = 10.0) -> str:
        mismatch = abs(self.supply - self.demand)
        price_precision = 2 if mismatch <= near_equilibrium_threshold else 1
        binned = (
            round(self.price, price_precision),
            round(self.supply, 1),
            round(self.demand, 1),
            self.timestamp,
            self.regime.value,
        )
        return sha256(str(binned).encode("utf-8")).hexdigest()
