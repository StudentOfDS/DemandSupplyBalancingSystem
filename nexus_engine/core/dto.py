from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.state import MarketState


@dataclass(frozen=True)
class TrajectoryDTO:
    steps: tuple[MarketState, ...]
    is_equilibrium_reached: bool


@dataclass(frozen=True)
class PlannerRequestDTO:
    initial_state: MarketState
    horizon: int = 12
    max_nodes: int = 3000


@dataclass(frozen=True)
class PlannerResponseDTO:
    best_trajectory: TrajectoryDTO | None
    explored_nodes: int
    selected_algorithm: str
    explanation: str
