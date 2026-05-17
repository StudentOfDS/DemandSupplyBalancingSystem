from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import asdict

from nexus_engine.core.dto import PlannerRequestDTO, PlannerResponseDTO
from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.data.features import FeatureVector
from nexus_engine.data.market_data import MarketDataIngestionService, MarketDataRecord
from nexus_engine.orchestration.engine import NexusEngine
from nexus_engine.storage.sqlite import SQLiteRepository


def build_engine_from_env() -> NexusEngine:
    records: list[MarketDataRecord] | None = None
    latest_features: FeatureVector | None = None
    csv_path = os.getenv("NEXUS_MARKET_CSV")
    if csv_path:
        ingestion = MarketDataIngestionService()
        records = ingestion.from_csv(csv_path)
        _, latest_features = ingestion.latest_state_and_features(records)

    sqlite_path = os.getenv("NEXUS_SQLITE_PATH")
    storage = SQLiteRepository(sqlite_path) if sqlite_path else None
    return NexusEngine(records=records, storage=storage, latest_features=latest_features)


engine = build_engine_from_env()


def configure_engine(new_engine: NexusEngine) -> None:
    global engine
    engine = new_engine


def _response_to_dict(response: PlannerResponseDTO) -> dict[str, object]:
    payload = asdict(response)
    if response.best_trajectory is not None:
        for step in payload["best_trajectory"]["steps"]:
            step["regime"] = step["regime"].value
    return payload


def _request_from_dict(payload: dict[str, object]) -> PlannerRequestDTO:
    initial = payload.get("initial_state")
    if not isinstance(initial, dict):
        raise ValueError("initial_state must be an object")
    state = MarketState(
        price=float(initial["price"]),
        supply=float(initial["supply"]),
        demand=float(initial["demand"]),
        timestamp=int(initial["timestamp"]),
        regime=MarketRegime(str(initial.get("regime", MarketRegime.STAGNANT.value))),
    )
    return PlannerRequestDTO(
        initial_state=state,
        horizon=int(payload.get("horizon", 12)),
        max_nodes=int(payload.get("max_nodes", 3000)),
    )


def plan(request: PlannerRequestDTO) -> PlannerResponseDTO:
    return engine.run(request)


def health() -> dict[str, str]:
    return {"status": "ok", "service": "nexus-engine"}


def model_status() -> dict[str, object]:
    return engine.status()


if importlib.util.find_spec("fastapi") is not None:
    fastapi = importlib.import_module("fastapi")
    FastAPI = fastapi.FastAPI
    HTTPException = fastapi.HTTPException
    app = FastAPI(title="Nexus Engine", version="0.2.0")

    @app.get("/health")
    def health_endpoint() -> dict[str, str]:
        return health()

    @app.get("/status")
    def status_endpoint() -> dict[str, object]:
        return model_status()

    @app.post("/plan")
    def plan_endpoint(payload: dict[str, object]) -> dict[str, object]:
        try:
            request = _request_from_dict(payload)
            response = plan(request)
            return _response_to_dict(response)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
else:
    class LocalApp:
        title = "Nexus Engine"

        def get(self, _path: str):
            def decorator(func):
                return func

            return decorator

        def post(self, _path: str):
            def decorator(func):
                return func

            return decorator

    app = LocalApp()
