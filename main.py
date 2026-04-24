from nexus_engine.core.dto import PlannerRequestDTO
from nexus_engine.core.state import MarketRegime, MarketState
from nexus_engine.orchestration.engine import NexusEngine

if __name__ == "__main__":
    engine = NexusEngine()
    req = PlannerRequestDTO(
        initial_state=MarketState(
            price=100.0,
            supply=50.0,
            demand=150.0,
            timestamp=0,
            regime=MarketRegime.STAGNANT,
        ),
        horizon=15,
        max_nodes=4000,
    )
    result = engine.run(req)
    print(result.explanation)
    if result.best_trajectory:
        print("final price", result.best_trajectory.steps[-1].price)
