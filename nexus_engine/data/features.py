from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureVector:
    seasonality: float
    traffic: float
    competitor_price_gap: float
    inventory_pressure: float
    production_cost_index: float
    macro_index: float


class FeatureIngestionService:
    def ingest(self, timestamp: int) -> FeatureVector:
        return FeatureVector(
            seasonality=(timestamp % 12) / 12.0,
            traffic=1.0 + ((timestamp % 7) / 20.0),
            competitor_price_gap=0.1,
            inventory_pressure=0.5,
            production_cost_index=1.0,
            macro_index=1.0,
        )


class FeatureExtrapolator:
    def extrapolate(self, base: FeatureVector, delta_steps: int) -> FeatureVector:
        decay = max(0.85, 1.0 - 0.01 * delta_steps)
        return FeatureVector(
            seasonality=min(1.0, base.seasonality + (delta_steps * 0.02)),
            traffic=base.traffic * decay,
            competitor_price_gap=base.competitor_price_gap,
            inventory_pressure=min(1.0, base.inventory_pressure + 0.02 * delta_steps),
            production_cost_index=base.production_cost_index * (1.0 + 0.003 * delta_steps),
            macro_index=base.macro_index,
        )
