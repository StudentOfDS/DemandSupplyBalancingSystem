from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.state import MarketState


@dataclass(frozen=True)
class TrajectoryDTO:
    steps: tuple[MarketState, ...]
    is_equilibrium_reached: bool

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("trajectory must contain at least one state")


@dataclass(frozen=True)
class PlannerRequestDTO:
    initial_state: MarketState
    horizon: int = 12
    max_nodes: int = 3000

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")


@dataclass(frozen=True)
class PlannerResponseDTO:
    best_trajectory: TrajectoryDTO | None
    explored_nodes: int
    selected_algorithm: str
    explanation: str

    def __post_init__(self) -> None:
        if self.explored_nodes < 0:
            raise ValueError("explored_nodes must be non-negative")
        if not self.selected_algorithm:
            raise ValueError("selected_algorithm is required")
