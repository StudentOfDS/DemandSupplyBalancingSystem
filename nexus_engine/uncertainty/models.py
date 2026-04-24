from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.state import MarketRegime, MarketState


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

    def infer_regime(self, latest_mismatch: float) -> MarketRegime:
        if latest_mismatch < 5:
            return MarketRegime.STAGNANT
        if latest_mismatch > 40:
            return MarketRegime.BEAR
        return MarketRegime.BULL

    def risk_proxy_lower_bound(self) -> float:
        return 0.1
