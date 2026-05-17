from __future__ import annotations

import argparse

from nexus_engine.core.dto import PlannerRequestDTO
from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.data.features import FeatureVector
from nexus_engine.data.market_data import MarketDataIngestionService
from nexus_engine.orchestration.engine import NexusEngine
from nexus_engine.storage.sqlite import SQLiteRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Nexus Engine planner.")
    parser.add_argument("--csv", help="Path to market-data CSV file for real input/training.")
    parser.add_argument("--sqlite", help="SQLite database path for persisted runs.")
    parser.add_argument("--horizon", type=int, default=15, help="Planning horizon/depth.")
    parser.add_argument("--max-nodes", type=int, default=4000, help="Maximum search nodes.")
    parser.add_argument("--price", type=float, default=100.0)
    parser.add_argument("--supply", type=float, default=50.0)
    parser.add_argument("--demand", type=float, default=150.0)
    parser.add_argument("--timestamp", type=int, default=0)
    parser.add_argument("--regime", choices=[item.value for item in MarketRegime], default=MarketRegime.STAGNANT.value)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = None
    latest_features: FeatureVector | None = None
    initial_state = MarketState(
        price=args.price,
        supply=args.supply,
        demand=args.demand,
        timestamp=args.timestamp,
        regime=MarketRegime(args.regime),
    )

    if args.csv:
        ingestion = MarketDataIngestionService()
        records = ingestion.from_csv(args.csv)
        initial_state, latest_features = ingestion.latest_state_and_features(records)

    storage = SQLiteRepository(args.sqlite) if args.sqlite else None
    engine = NexusEngine(records=records, storage=storage, latest_features=latest_features)
    result = engine.run(PlannerRequestDTO(initial_state=initial_state, horizon=args.horizon, max_nodes=args.max_nodes))

    print(result.explanation)
    if result.best_trajectory:
        final = result.best_trajectory.steps[-1]
        print(f"Final price: {final.price:.2f}")
        print(f"Final supply: {final.supply:.2f}")
        print(f"Final demand: {final.demand:.2f}")
    print(f"Algorithm: {result.selected_algorithm}")
    print(f"Explored nodes: {result.explored_nodes}")


if __name__ == "__main__":
    main()
