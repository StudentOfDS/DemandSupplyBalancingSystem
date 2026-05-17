from __future__ import annotations

import math
from dataclasses import dataclass, field

from nexus_engine.core.state import MarketRegime
from nexus_engine.data.features import FeatureVector
from nexus_engine.data.market_data import MarketDataRecord


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return numerator / denominator


@dataclass
class ElasticityModel:
    """Interpretable demand elasticity model with lightweight training support."""

    base_elasticity: float = 1.0
    trained_samples: int = 0

    def fit(self, records: list[MarketDataRecord]) -> None:
        elasticities: list[float] = []
        ordered = sorted(records, key=lambda item: item.timestamp)
        for previous, current in zip(ordered, ordered[1:]):
            pct_demand = _safe_div(current.demand - previous.demand, previous.demand)
            pct_price = _safe_div(current.price - previous.price, previous.price)
            if abs(pct_price) > 1e-9:
                elasticities.append(abs(_safe_div(pct_demand, pct_price)))
        if elasticities:
            self.base_elasticity = sum(elasticities) / len(elasticities)
            self.trained_samples = len(elasticities)

    def predict_adjustment(self, features: FeatureVector) -> float:
        return self.base_elasticity * (0.6 + 0.4 * features.traffic)


@dataclass
class SupplyResponseModel:
    """Learns average supply response to price changes from observed market data."""

    response_coefficient: float = 1.0
    trained_samples: int = 0

    def fit(self, records: list[MarketDataRecord]) -> None:
        responses: list[float] = []
        ordered = sorted(records, key=lambda item: item.timestamp)
        for previous, current in zip(ordered, ordered[1:]):
            delta_price = current.price - previous.price
            delta_supply = current.supply - previous.supply
            if abs(delta_price) > 1e-9:
                responses.append(_safe_div(delta_supply, delta_price))
        if responses:
            self.response_coefficient = sum(responses) / len(responses)
            self.trained_samples = len(responses)

    def predict_supply_delta(self, price_delta: float) -> float:
        return self.response_coefficient * price_delta


@dataclass
class ResidualNeuralModel:
    """Tiny deterministic residual learner trained with gradient descent."""

    weights: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    bias: float = 0.0
    trained_epochs: int = 0

    def fit(self, samples: list[tuple[FeatureVector, MarketRegime, float]], epochs: int = 100, lr: float = 0.01) -> None:
        if not samples:
            return
        for _ in range(epochs):
            for features, regime, target in samples:
                x = self._encode(features, regime)
                prediction = self._forward(x)
                error = prediction - target
                self.weights = [weight - lr * error * value for weight, value in zip(self.weights, x)]
                self.bias -= lr * error
            self.trained_epochs += 1

    def residual(self, features: FeatureVector, regime: MarketRegime) -> float:
        if self.trained_epochs > 0:
            return self._forward(self._encode(features, regime))
        regime_bias = {
            MarketRegime.BULL: 0.04,
            MarketRegime.STAGNANT: 0.01,
            MarketRegime.BEAR: -0.03,
        }[regime]
        nonlinear = 0.02 * (features.inventory_pressure**2) - 0.015 * features.competitor_price_gap
        return regime_bias + nonlinear

    def _forward(self, encoded: list[float]) -> float:
        raw = sum(weight * value for weight, value in zip(self.weights, encoded)) + self.bias
        return math.tanh(raw)

    def _encode(self, features: FeatureVector, regime: MarketRegime) -> list[float]:
        regime_value = {
            MarketRegime.BULL: 1.0,
            MarketRegime.STAGNANT: 0.0,
            MarketRegime.BEAR: -1.0,
        }[regime]
        return [
            features.seasonality,
            features.traffic,
            features.competitor_price_gap,
            features.inventory_pressure,
            features.production_cost_index,
            regime_value,
        ]
