from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.data.features import FeatureVector


REQUIRED_COLUMNS = {
    "timestamp",
    "price",
    "supply",
    "demand",
    "inventory",
    "competitor_price",
    "seasonality",
    "traffic",
    "cost",
}


@dataclass(frozen=True)
class MarketDataRecord:
    timestamp: int
    price: float
    supply: float
    demand: float
    inventory: float
    competitor_price: float
    seasonality: float
    traffic: float
    cost: float
    regime: MarketRegime = MarketRegime.STAGNANT

    def to_state(self) -> MarketState:
        return MarketState(
            price=self.price,
            supply=self.supply,
            demand=self.demand,
            timestamp=self.timestamp,
            regime=self.regime,
        )

    def to_features(self) -> FeatureVector:
        return FeatureVector(
            seasonality=self.seasonality,
            traffic=self.traffic,
            competitor_price_gap=self.price - self.competitor_price,
            inventory_pressure=self.inventory / max(self.demand, 1.0),
            production_cost_index=self.cost / max(self.price, 1.0),
            macro_index=1.0,
        )


class MarketDataIngestionError(ValueError):
    pass


class MarketDataIngestionService:
    """Loads real market observations from CSV files or JSON API endpoints."""

    def from_csv(self, path: str | Path) -> list[MarketDataRecord]:
        csv_path = Path(path)
        if not csv_path.exists():
            raise MarketDataIngestionError(f"CSV file does not exist: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise MarketDataIngestionError("CSV file has no header row")
            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise MarketDataIngestionError(f"CSV missing required columns: {sorted(missing)}")
            return [self._record_from_mapping(row) for row in reader]

    def from_api(self, url: str, timeout_seconds: float = 10.0) -> list[MarketDataRecord]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise MarketDataIngestionError("API response must be a JSON array of records")
        return [self._record_from_mapping(item) for item in payload]

    def latest_state_and_features(self, records: list[MarketDataRecord]) -> tuple[MarketState, FeatureVector]:
        if not records:
            raise MarketDataIngestionError("No market records supplied")
        latest = max(records, key=lambda item: item.timestamp)
        return latest.to_state(), latest.to_features()

    def _record_from_mapping(self, row: dict[str, object]) -> MarketDataRecord:
        try:
            regime_value = str(row.get("regime", MarketRegime.STAGNANT.value))
            regime = MarketRegime(regime_value)
            return MarketDataRecord(
                timestamp=int(row["timestamp"]),
                price=float(row["price"]),
                supply=float(row["supply"]),
                demand=float(row["demand"]),
                inventory=float(row["inventory"]),
                competitor_price=float(row["competitor_price"]),
                seasonality=float(row["seasonality"]),
                traffic=float(row["traffic"]),
                cost=float(row["cost"]),
                regime=regime,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataIngestionError(f"Invalid market data row: {row}") from exc
