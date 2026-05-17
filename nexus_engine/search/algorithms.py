from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from nexus_engine.core.dto import TrajectoryDTO
from nexus_engine.core.state import MarketState
from nexus_engine.optimization.objective import ObjectiveEvaluator
from nexus_engine.uncertainty.models import HiddenMarkovModel, MarkovTransitionModel


Validator = Callable[[MarketState, MarketState, float], bool]


@dataclass
class SearchLimits:
    max_depth: int = 12
    max_nodes: int = 3000
    timeout_seconds: float = 1.0


class SearchLayer:
    def __init__(
        self,
        validate_transition: Validator,
        objective: ObjectiveEvaluator,
        markov: MarkovTransitionModel,
        hmm: HiddenMarkovModel,
    ) -> None:
        self._validate = validate_transition
        self._objective = objective
        self._markov = markov
        self._hmm = hmm

    def generate_paths(self, initial_state: MarketState, limits: SearchLimits) -> tuple[list[TrajectoryDTO], str, int]:
        try:
            return self._astar(initial_state, limits)
        except RuntimeError:
            return self._gbfs(initial_state, limits)

    def _astar(self, initial_state: MarketState, limits: SearchLimits) -> tuple[list[TrajectoryDTO], str, int]:
        start = time.monotonic()
        frontier: list[tuple[float, int, MarketState, tuple[MarketState, ...]]] = []
        heapq.heappush(frontier, (0.0, 0, initial_state, (initial_state,)))
        explored: set[str] = set()
        results: list[TrajectoryDTO] = []
        node_count = 0

        while frontier and len(results) < 5:
            if time.monotonic() - start > limits.timeout_seconds or node_count > limits.max_nodes:
                raise RuntimeError("A* resource limits exceeded")

            _, _, state, path = heapq.heappop(frontier)
            state_hash = state.bin_hash()
            if state_hash in explored:
                continue
            explored.add(state_hash)

            if abs(state.supply - state.demand) < 1.0:
                results.append(TrajectoryDTO(steps=path, is_equilibrium_reached=True))
                continue
            if len(path) >= limits.max_depth:
                continue

            for action in self._actions_for_state(state):
                next_state = self._simulate(state, action)
                node_count += 1
                if not self._validate(state, next_state, action):
                    continue
                g_cost = self._objective.evaluate_trajectory(
                    TrajectoryDTO(steps=path + (next_state,), is_equilibrium_reached=False)
                )
                h_cost = self._heuristic(next_state)
                heapq.heappush(frontier, (g_cost + h_cost, node_count, next_state, path + (next_state,)))

        return results, "astar", node_count

    def _gbfs(self, initial_state: MarketState, limits: SearchLimits) -> tuple[list[TrajectoryDTO], str, int]:
        frontier: list[tuple[float, int, MarketState, tuple[MarketState, ...]]] = []
        heapq.heappush(frontier, (self._heuristic(initial_state), 0, initial_state, (initial_state,)))
        explored: set[str] = set()
        results: list[TrajectoryDTO] = []
        node_count = 0

        while frontier and len(results) < 3 and node_count <= limits.max_nodes:
            _, _, state, path = heapq.heappop(frontier)
            state_hash = state.bin_hash()
            if state_hash in explored:
                continue
            explored.add(state_hash)

            if abs(state.supply - state.demand) < 1.0:
                results.append(TrajectoryDTO(steps=path, is_equilibrium_reached=True))
                continue
            if len(path) >= limits.max_depth:
                continue

            for action in self._actions_for_state(state):
                next_state = self._simulate(state, action)
                node_count += 1
                if not self._validate(state, next_state, action):
                    continue
                heapq.heappush(frontier, (self._heuristic(next_state), node_count, next_state, path + (next_state,)))

        return results, "gbfs_fallback", node_count

    def bfs(self, initial_state: MarketState, max_depth: int = 6) -> list[TrajectoryDTO]:
        queue = deque([(initial_state, (initial_state,))])
        results: list[TrajectoryDTO] = []
        while queue:
            state, path = queue.popleft()
            if abs(state.supply - state.demand) < 1.0:
                results.append(TrajectoryDTO(steps=path, is_equilibrium_reached=True))
                continue
            if len(path) >= max_depth:
                continue
            for action in self._actions_for_state(state):
                nxt = self._simulate(state, action)
                if self._validate(state, nxt, action):
                    queue.append((nxt, path + (nxt,)))
        return results

    def _actions_for_state(self, state: MarketState) -> tuple[float, ...]:
        mismatch = abs(state.supply - state.demand)
        if mismatch > 25:
            base_actions = (5.0, 1.0, -1.0, -5.0)
        elif mismatch > 8:
            base_actions = (2.0, 1.0, -1.0, -2.0)
        else:
            base_actions = (1.0, 0.5, 0.25, -0.25, -0.5, -1.0)

        if mismatch <= 8:
            exact = self._markov.equilibrium_action(state)
            exact = max(min(exact, 5.0), -5.0)
            if abs(exact) > 0.05:
                return tuple(dict.fromkeys(base_actions + (round(exact, 3),)))
        return base_actions

    def _simulate(self, current: MarketState, action: float) -> MarketState:
        next_supply, next_demand = self._markov.evolve(current, action)
        regime = self._hmm.infer_regime(abs(next_supply - next_demand))
        return MarketState(
            price=max(0.0, current.price + action),
            supply=next_supply,
            demand=next_demand,
            timestamp=current.timestamp + 1,
            regime=regime,
        )

    def _heuristic(self, state: MarketState) -> float:
        mismatch_lb = min(abs(state.supply - state.demand) / 1000.0, 1.0)
        min_action = 1.0
        volatility_lb = self._objective.beta * (min_action / 50.0)
        return self._objective.alpha * mismatch_lb + volatility_lb + self._hmm.risk_proxy_lower_bound()
