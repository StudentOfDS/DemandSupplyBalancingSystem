from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.state import MarketRegime, MarketState


@dataclass
class MarkovTransitionModel:
    demand_drift: float = 1.0
    supply_drift: float = 0.8

    def evolve(self, state: MarketState, price_action: float) -> tuple[float, float]:
        next_demand = max(0.0, state.demand - (2.0 + self.demand_drift) * price_action)
        next_supply = max(0.0, state.supply + (1.2 + self.supply_drift) * price_action)
        return next_supply, next_demand


@dataclass
class HiddenMarkovModel:
    default_regime: MarketRegime = MarketRegime.STAGNANT

    def infer_regime(self, latest_mismatch: float) -> MarketRegime:
        if latest_mismatch < 5:
            return MarketRegime.STAGNANT
        if latest_mismatch > 40:
            return MarketRegime.BEAR
        return MarketRegime.BULL

    def risk_proxy_lower_bound(self) -> float:
        return 0.1
