from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path

from nexus_engine.core.dto import TrajectoryDTO
from nexus_engine.core.state import MarketRegime, MarketState


class SQLiteRepository:
    """Persistent storage for states, scenarios, trajectories, model outputs, and experiment logs."""

    def __init__(self, database_path: str | Path = "nexus_engine.db") -> None:
        self.database_path = Path(database_path)
        self._ensure_schema()

    def save_market_state(self, state: MarketState) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO market_states (price, supply, demand, timestamp, regime, state_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (state.price, state.supply, state.demand, state.timestamp, state.regime.value, state.bin_hash()),
            )
            return int(cursor.lastrowid)

    def save_user_scenario(self, name: str, request_payload: dict[str, object]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO user_scenarios (name, request_json) VALUES (?, ?)",
                (name, json.dumps(request_payload, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def save_trajectory(self, trajectory: TrajectoryDTO, scenario_id: int | None = None) -> int:
        payload = {
            "is_equilibrium_reached": trajectory.is_equilibrium_reached,
            "steps": [self._state_to_dict(step) for step in trajectory.steps],
        }
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO trajectories (scenario_id, trajectory_json) VALUES (?, ?)",
                (scenario_id, json.dumps(payload, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def save_model_output(self, model_name: str, output: dict[str, object]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO model_outputs (model_name, output_json) VALUES (?, ?)",
                (model_name, json.dumps(output, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def log_experiment(self, name: str, metrics: dict[str, float]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO experiment_logs (name, metrics_json) VALUES (?, ?)",
                (name, json.dumps(metrics, sort_keys=True)),
            )
            return int(cursor.lastrowid)

    def count_rows(self, table_name: str) -> int:
        allowed = {"market_states", "user_scenarios", "trajectories", "model_outputs", "experiment_logs"}
        if table_name not in allowed:
            raise ValueError(f"Unsupported table for count: {table_name}")
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
        return int(row["total"])

    def latest_market_state(self) -> MarketState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT price, supply, demand, timestamp, regime
                FROM market_states
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return MarketState(
            price=float(row["price"]),
            supply=float(row["supply"]),
            demand=float(row["demand"]),
            timestamp=int(row["timestamp"]),
            regime=MarketRegime(str(row["regime"])),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    price REAL NOT NULL,
                    supply REAL NOT NULL,
                    demand REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    regime TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS user_scenarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_id INTEGER,
                    trajectory_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scenario_id) REFERENCES user_scenarios(id)
                );
                CREATE TABLE IF NOT EXISTS model_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS experiment_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _state_to_dict(self, state: MarketState) -> dict[str, object]:
        payload = asdict(state) if is_dataclass(state) else dict(state)
        payload["regime"] = state.regime.value
        return payload
