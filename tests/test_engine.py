import json
from pathlib import Path

import pytest

from nexus_engine.api.app import _request_from_dict, configure_engine, health, model_status, plan
from nexus_engine.core.dto import PlannerRequestDTO, TrajectoryDTO
from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.data.market_data import MarketDataIngestionError, MarketDataIngestionService
from nexus_engine.language.interface import ConstrainedLanguageLayer, HuggingFaceLLMClient
from nexus_engine.learning.models import ElasticityModel, ResidualNeuralModel, SupplyResponseModel
from nexus_engine.optimization.objective import ObjectiveEvaluator
from nexus_engine.orchestration.engine import NexusEngine
from nexus_engine.search.algorithms import SearchLayer
from nexus_engine.storage.sqlite import SQLiteRepository
from nexus_engine.tasks import run_plan_sync
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


def test_csv_ingestion_loads_real_market_features() -> None:
    records = MarketDataIngestionService().from_csv("examples/market_data.csv")
    state, features = MarketDataIngestionService().latest_state_and_features(records)

    assert len(records) == 4
    assert state.price == 105.0
    assert features.competitor_price_gap == 1.0


def test_csv_ingestion_rejects_missing_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("timestamp,price\n1,10\n", encoding="utf-8")

    with pytest.raises(MarketDataIngestionError):
        MarketDataIngestionService().from_csv(bad_csv)


def test_learning_and_uncertainty_models_train_from_records() -> None:
    records = MarketDataIngestionService().from_csv("examples/market_data.csv")
    elasticity = ElasticityModel()
    supply = SupplyResponseModel()
    markov = MarkovTransitionModel()
    hmm = HiddenMarkovModel()
    residual = ResidualNeuralModel()

    elasticity.fit(records)
    supply.fit(records)
    markov.fit(records)
    hmm.fit(records)
    residual.fit([(record.to_features(), record.regime, 0.1) for record in records], epochs=2)

    assert elasticity.trained_samples > 0
    assert supply.trained_samples > 0
    assert markov.demand_sensitivity > 0
    assert hmm.most_likely_next(MarketRegime.BEAR) in set(MarketRegime)
    assert residual.trained_epochs == 2


def test_sqlite_repository_persists_core_outputs(tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "nexus.db")
    state = MarketState(price=10, supply=5, demand=8, timestamp=1, regime=MarketRegime.STAGNANT)
    trajectory = TrajectoryDTO(steps=(state,), is_equilibrium_reached=False)

    state_id = repo.save_market_state(state)
    scenario_id = repo.save_user_scenario("test", {"horizon": 1})
    trajectory_id = repo.save_trajectory(trajectory, scenario_id=scenario_id)
    output_id = repo.save_model_output("model", {"value": 1})
    log_id = repo.log_experiment("exp", {"nodes": 3.0})

    assert min(state_id, scenario_id, trajectory_id, output_id, log_id) > 0
    assert repo.latest_market_state() == state


def test_language_layer_validates_json_and_uses_fallback_client() -> None:
    layer = ConstrainedLanguageLayer(HuggingFaceLLMClient(settings=None))

    parsed = layer.parse_user_prompt(json.dumps({"horizon": 3, "max_nodes": 9}))
    fallback = layer.parse_natural_language("plan with defaults")

    assert parsed.horizon == 3
    assert fallback.max_nodes == 3000
    with pytest.raises(ValueError):
        layer.parse_user_prompt("not-json")


def test_api_helpers_validate_and_execute_plan() -> None:
    payload = {
        "initial_state": {"price": 100, "supply": 80, "demand": 100, "timestamp": 0, "regime": "Stagnant"},
        "horizon": 6,
        "max_nodes": 500,
    }

    request = _request_from_dict(payload)
    response = plan(request)

    assert health()["status"] == "ok"
    assert model_status()["service"] == "nexus-engine"
    assert response.explored_nodes >= 0


def test_api_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        _request_from_dict({"horizon": 1})
    with pytest.raises(ValueError):
        PlannerRequestDTO(
            initial_state=MarketState(price=1, supply=1, demand=1, timestamp=0, regime=MarketRegime.STAGNANT),
            horizon=0,
        )


def test_celery_task_sync_fallback_runs_plan() -> None:
    response = run_plan_sync(
        {
            "initial_state": {"price": 100, "supply": 90, "demand": 110, "timestamp": 0, "regime": "Stagnant"},
            "horizon": 6,
            "max_nodes": 500,
        }
    )

    assert response["selected_algorithm"] in {"astar", "gbfs_fallback"}


def test_search_performance_regression_under_node_limit() -> None:
    engine = NexusEngine()
    response = engine.run(
        PlannerRequestDTO(
            initial_state=MarketState(price=100, supply=50, demand=150, timestamp=0, regime=MarketRegime.STAGNANT),
            horizon=15,
            max_nodes=4000,
        )
    )

    assert response.explored_nodes <= 4000


def test_end_to_end_csv_training_search_optimization_persistence_api_response(tmp_path: Path) -> None:
    records = MarketDataIngestionService().from_csv("examples/market_data.csv")
    initial_state, latest_features = MarketDataIngestionService().latest_state_and_features(records)
    repo = SQLiteRepository(tmp_path / "e2e.db")
    configure_engine(NexusEngine(records=records, storage=repo, latest_features=latest_features))

    response = plan(PlannerRequestDTO(initial_state=initial_state, horizon=15, max_nodes=4000))

    assert response.best_trajectory is not None
    assert response.explored_nodes > 0
    assert model_status()["trained"] is True
    assert model_status()["storage_enabled"] is True
    assert repo.count_rows("market_states") >= 1
    assert repo.count_rows("user_scenarios") >= 1
    assert repo.count_rows("trajectories") >= 1
    assert repo.count_rows("model_outputs") >= 2
    assert repo.count_rows("experiment_logs") >= 1
    configure_engine(NexusEngine())
