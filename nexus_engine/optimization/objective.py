from __future__ import annotations

from dataclasses import dataclass

from nexus_engine.core.dto import TrajectoryDTO


@dataclass
class ObjectiveEvaluator:
    alpha: float = 1.0
    beta: float = 0.4
    gamma: float = 1.5

    regime_risk: dict[str, float] = None

    def __post_init__(self) -> None:
        if self.regime_risk is None:
            self.regime_risk = {"Bull": 0.2, "Stagnant": 0.5, "Bear": 1.4}

    def evaluate_trajectory(self, trajectory: TrajectoryDTO) -> float:
        total = 0.0
        steps = trajectory.steps
        for idx, state in enumerate(steps):
            mismatch = min(abs(state.supply - state.demand) / 1000.0, 1.0)
            volatility = 0.0
            if idx > 0:
                volatility = min(abs(state.price - steps[idx - 1].price) / 50.0, 1.0)
            risk = min(self.regime_risk.get(state.regime.value, 1.0) / 2.0, 1.0)
            total += self.alpha * mismatch + self.beta * volatility + self.gamma * risk
        return total


class TrajectoryOptimizer:
    def select_best(self, candidates: list[TrajectoryDTO], objective: ObjectiveEvaluator) -> TrajectoryDTO | None:
        if not candidates:
            return None
        return min(candidates, key=objective.evaluate_trajectory)
