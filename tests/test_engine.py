from nexus_engine.core.dto import PlannerRequestDTO
from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.optimization.objective import ObjectiveEvaluator
from nexus_engine.orchestration.engine import NexusEngine
from nexus_engine.search.algorithms import SearchLayer
from nexus_engine.uncertainty.models import HiddenMarkovModel, MarkovTransitionModel


def test_engine_runs_and_returns_response() -> None:
    engine = NexusEngine()
    request = PlannerRequestDTO(
        initial_state=MarketState(
            price=100.0,
            supply=60.0,
            demand=120.0,
            timestamp=0,
            regime=MarketRegime.STAGNANT,
        ),
        horizon=10,
        max_nodes=2000,
    )

    response = engine.run(request)

    assert response.selected_algorithm in {"astar", "gbfs_fallback"}
    assert response.explored_nodes >= 0
    assert isinstance(response.explanation, str)


def test_heuristic_includes_safe_volatility_lower_bound() -> None:
    search = SearchLayer(
        validate_transition=lambda _c, _n, _a: True,
        objective=ObjectiveEvaluator(beta=0.7),
        markov=MarkovTransitionModel(),
        hmm=HiddenMarkovModel(),
    )
    state = MarketState(price=100.0, supply=100.0, demand=110.0, timestamp=0, regime=MarketRegime.STAGNANT)

    h = search._heuristic(state)

    assert h >= 0.7 * (1.0 / 50.0) + 0.1


def test_action_space_adds_continuous_equilibrium_candidate() -> None:
    search = SearchLayer(
        validate_transition=lambda _c, _n, _a: True,
        objective=ObjectiveEvaluator(),
        markov=MarkovTransitionModel(),
        hmm=HiddenMarkovModel(),
    )
    near_eq_state = MarketState(
        price=102.0,
        supply=100.0,
        demand=102.35,
        timestamp=3,
        regime=MarketRegime.STAGNANT,
    )

    actions = search._actions_for_state(near_eq_state)

    assert any(abs(a - 0.47) < 0.02 for a in actions)


def test_market_state_lt_is_deterministic() -> None:
    a = MarketState(price=10, supply=5, demand=6, timestamp=1, regime=MarketRegime.BULL)
    b = MarketState(price=11, supply=5, demand=6, timestamp=2, regime=MarketRegime.BULL)

    assert a < b
