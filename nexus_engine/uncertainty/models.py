from __future__ import annotations

from dataclasses import dataclass, field

from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.data.market_data import MarketDataRecord


@dataclass
class MarkovTransitionModel:
    demand_drift: float = 1.0
    supply_drift: float = 0.8

    @property
    def demand_sensitivity(self) -> float:
        return 2.0 + self.demand_drift

    @property
    def supply_sensitivity(self) -> float:
        return 1.2 + self.supply_drift

    def fit(self, records: list[MarketDataRecord]) -> None:
        demand_slopes: list[float] = []
        supply_slopes: list[float] = []
        ordered = sorted(records, key=lambda item: item.timestamp)
        for previous, current in zip(ordered, ordered[1:]):
            delta_price = current.price - previous.price
            if abs(delta_price) <= 1e-9:
                continue
            demand_slopes.append(abs((current.demand - previous.demand) / delta_price))
            supply_slopes.append(abs((current.supply - previous.supply) / delta_price))
        if demand_slopes:
            self.demand_drift = max(0.0, (sum(demand_slopes) / len(demand_slopes)) - 2.0)
        if supply_slopes:
            self.supply_drift = max(0.0, (sum(supply_slopes) / len(supply_slopes)) - 1.2)

    def evolve(self, state: MarketState, price_action: float) -> tuple[float, float]:
        next_demand = max(0.0, state.demand - self.demand_sensitivity * price_action)
        next_supply = max(0.0, state.supply + self.supply_sensitivity * price_action)
        return next_supply, next_demand

    def equilibrium_action(self, state: MarketState) -> float:
        """Continuous one-step action that closes mismatch in linear dynamics."""
        denom = self.demand_sensitivity + self.supply_sensitivity
        if denom == 0:
            return 0.0
        return (state.demand - state.supply) / denom


@dataclass
class HiddenMarkovModel:
    default_regime: MarketRegime = MarketRegime.STAGNANT
    transition_counts: dict[MarketRegime, dict[MarketRegime, int]] = field(default_factory=dict)

    def fit(self, records: list[MarketDataRecord]) -> None:
        ordered = sorted(records, key=lambda item: item.timestamp)
        for previous, current in zip(ordered, ordered[1:]):
            bucket = self.transition_counts.setdefault(previous.regime, {})
            bucket[current.regime] = bucket.get(current.regime, 0) + 1
        if ordered:
            self.default_regime = ordered[-1].regime

    def infer_regime(self, latest_mismatch: float) -> MarketRegime:
        if latest_mismatch < 5:
            return MarketRegime.STAGNANT
        if latest_mismatch > 40:
            return MarketRegime.BEAR
        return MarketRegime.BULL

    def most_likely_next(self, current: MarketRegime | None = None) -> MarketRegime:
        source = current or self.default_regime
        counts = self.transition_counts.get(source, {})
        if not counts:
            return self.default_regime
        return max(counts, key=counts.get)

    def risk_proxy_lower_bound(self) -> float:
        return 0.1
