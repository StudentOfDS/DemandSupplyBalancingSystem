from __future__ import annotations

from dataclasses import asdict

from nexus_engine.core.dto import PlannerRequestDTO, PlannerResponseDTO
from nexus_engine.core.state import MarketState
from nexus_engine.data.features import FeatureExtrapolator, FeatureIngestionService, FeatureVector
from nexus_engine.data.market_data import MarketDataRecord
from nexus_engine.knowledge.rules import PredicateRuleEngine
from nexus_engine.language.interface import ConstrainedLanguageLayer
from nexus_engine.learning.models import ElasticityModel, ResidualNeuralModel, SupplyResponseModel
from nexus_engine.optimization.objective import ObjectiveEvaluator, TrajectoryOptimizer
from nexus_engine.search.algorithms import SearchLayer, SearchLimits
from nexus_engine.storage.sqlite import SQLiteRepository
from nexus_engine.uncertainty.models import HiddenMarkovModel, MarkovTransitionModel


class NexusEngine:
    def __init__(
        self,
        records: list[MarketDataRecord] | None = None,
        storage: SQLiteRepository | None = None,
        latest_features: FeatureVector | None = None,
    ) -> None:
        self._features = FeatureIngestionService(latest_features)
        self._feature_extrapolator = FeatureExtrapolator()
        self._rules = PredicateRuleEngine()
        self._baseline_model = ElasticityModel()
        self._supply_model = SupplyResponseModel()
        self._residual_model = ResidualNeuralModel()
        self._objective = ObjectiveEvaluator()
        self._optimizer = TrajectoryOptimizer()
        self._hmm = HiddenMarkovModel()
        self._markov = MarkovTransitionModel()
        self._storage = storage
        self._search = SearchLayer(self._validate, self._objective, self._markov, self._hmm)
        self._language = ConstrainedLanguageLayer()
        if records:
            self.train(records)

    def train(self, records: list[MarketDataRecord]) -> None:
        self._baseline_model.fit(records)
        self._supply_model.fit(records)
        self._markov.fit(records)
        self._hmm.fit(records)
        residual_samples = [
            (record.to_features(), record.regime, (record.demand - record.supply) / max(record.demand, 1.0))
            for record in records
        ]
        self._residual_model.fit(residual_samples, epochs=20, lr=0.005)
        if self._storage is not None:
            self._storage.save_model_output(
                "training_summary",
                {
                    "records": len(records),
                    "elasticity_samples": self._baseline_model.trained_samples,
                    "supply_samples": self._supply_model.trained_samples,
                    "residual_epochs": self._residual_model.trained_epochs,
                },
            )

    def run(self, request: PlannerRequestDTO) -> PlannerResponseDTO:
        self._validate_request(request)
        scenario_id = None
        if self._storage is not None:
            self._storage.save_market_state(request.initial_state)
            scenario_id = self._storage.save_user_scenario("planner_request", self._request_to_dict(request))

        base_features = self._features.ingest(request.initial_state.timestamp)
        projected_features = self._feature_extrapolator.extrapolate(base_features, request.horizon)

        baseline_signal = self._baseline_model.predict_adjustment(projected_features)
        residual_signal = self._residual_model.residual(projected_features, request.initial_state.regime)
        supply_signal = self._supply_model.predict_supply_delta(1.0)
        if self._storage is not None:
            self._storage.save_model_output(
                "planner_signals",
                {
                    "baseline_signal": baseline_signal,
                    "residual_signal": residual_signal,
                    "supply_signal": supply_signal,
                },
            )

        candidates, algorithm, explored = self._search.generate_paths(
            request.initial_state,
            SearchLimits(max_depth=request.horizon, max_nodes=request.max_nodes),
        )

        best = self._optimizer.select_best(candidates, self._objective)
        if best is not None and self._storage is not None:
            self._storage.save_trajectory(best, scenario_id=scenario_id)
            self._storage.log_experiment("planner_run", {"explored_nodes": float(explored)})

        explanation = self._language.explain(
            algorithm=algorithm,
            reached=bool(best and best.is_equilibrium_reached),
            steps=(len(best.steps) - 1) if best else 0,
        )

        return PlannerResponseDTO(
            best_trajectory=best,
            explored_nodes=explored,
            selected_algorithm=algorithm,
            explanation=explanation,
        )

    def status(self) -> dict[str, object]:
        return {
            "service": "nexus-engine",
            "trained": self._baseline_model.trained_samples > 0 or self._residual_model.trained_epochs > 0,
            "elasticity_samples": self._baseline_model.trained_samples,
            "supply_samples": self._supply_model.trained_samples,
            "residual_epochs": self._residual_model.trained_epochs,
            "default_regime": self._hmm.default_regime.value,
            "storage_enabled": self._storage is not None,
        }

    def _validate(self, current: MarketState, nxt: MarketState, action: float) -> bool:
        return self._rules.validate_transition(current, nxt, action).is_valid

    def _validate_request(self, request: PlannerRequestDTO) -> None:
        if request.horizon <= 0:
            raise ValueError("horizon must be positive")
        if request.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

    def _request_to_dict(self, request: PlannerRequestDTO) -> dict[str, object]:
        payload = asdict(request)
        payload["initial_state"]["regime"] = request.initial_state.regime.value
        return payload
