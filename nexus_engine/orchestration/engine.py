from __future__ import annotations

from nexus_engine.core.dto import PlannerRequestDTO, PlannerResponseDTO
from nexus_engine.core.state import MarketState
from nexus_engine.data.features import FeatureExtrapolator, FeatureIngestionService
from nexus_engine.knowledge.rules import PredicateRuleEngine
from nexus_engine.language.interface import ConstrainedLanguageLayer
from nexus_engine.learning.models import ElasticityModel, ResidualNeuralModel
from nexus_engine.optimization.objective import ObjectiveEvaluator, TrajectoryOptimizer
from nexus_engine.search.algorithms import SearchLayer, SearchLimits
from nexus_engine.uncertainty.models import HiddenMarkovModel, MarkovTransitionModel


class NexusEngine:
    def __init__(self) -> None:
        self._features = FeatureIngestionService()
        self._feature_extrapolator = FeatureExtrapolator()
        self._rules = PredicateRuleEngine()
        self._baseline_model = ElasticityModel()
        self._residual_model = ResidualNeuralModel()
        self._objective = ObjectiveEvaluator()
        self._optimizer = TrajectoryOptimizer()
        self._hmm = HiddenMarkovModel()
        self._markov = MarkovTransitionModel()
        self._search = SearchLayer(self._validate, self._objective, self._markov, self._hmm)
        self._language = ConstrainedLanguageLayer()

    def run(self, request: PlannerRequestDTO) -> PlannerResponseDTO:
        base_features = self._features.ingest(request.initial_state.timestamp)
        projected_features = self._feature_extrapolator.extrapolate(base_features, request.horizon)

        # Baseline + residual modeling (used as policy signal / diagnostics)
        _ = self._baseline_model.predict_adjustment(projected_features) + self._residual_model.residual(
            projected_features,
            request.initial_state.regime,
        )

        candidates, algorithm, explored = self._search.generate_paths(
            request.initial_state,
            SearchLimits(max_depth=request.horizon, max_nodes=request.max_nodes),
        )

        best = self._optimizer.select_best(candidates, self._objective)
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

    def _validate(self, current: MarketState, nxt: MarketState, action: float) -> bool:
        return self._rules.validate_transition(current, nxt, action).is_valid
