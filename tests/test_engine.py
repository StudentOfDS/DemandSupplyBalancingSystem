from nexus_engine.core.dto import PlannerRequestDTO
from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.orchestration.engine import NexusEngine


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
