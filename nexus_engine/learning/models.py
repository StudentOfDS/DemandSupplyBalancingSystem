from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.state import MarketRegime
from nexus_engine.data.features import FeatureVector


@dataclass
class ElasticityModel:
    """Interpretable baseline model."""

    base_elasticity: float = 1.0

    def predict_adjustment(self, features: FeatureVector) -> float:
        return self.base_elasticity * (0.6 + 0.4 * features.traffic)


@dataclass
class ResidualNeuralModel:
    """NN-style residual approximator (lightweight deterministic proxy)."""

    def residual(self, features: FeatureVector, regime: MarketRegime) -> float:
        regime_bias = {
            MarketRegime.BULL: 0.04,
            MarketRegime.STAGNANT: 0.01,
            MarketRegime.BEAR: -0.03,
        }[regime]
        nonlinear = 0.02 * (features.inventory_pressure**2) - 0.015 * features.competitor_price_gap
        return regime_bias + nonlinear
