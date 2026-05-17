# DemandSupplyBalancingSystem (Nexus Engine)

A modular, production-style demand-supply balancing engine that models market equilibrium as a **trajectory planning** problem.

## Implemented Architecture

- **Core state and DTO layer**
  - Immutable `MarketState` and immutable transfer objects (`TrajectoryDTO`, request/response DTOs).
  - Adaptive state binning + hash for search deduplication.
- **Knowledge/rule layer**
  - Predicate-style transition validator enforcing non-negativity, capacity bounds, law of demand, and law of supply.
  - Vector-store facade with Pinecone primary + FAISS fallback semantics.
- **Data/feature layer**
  - Deterministic feature ingestion and future-time extrapolation to prevent feature leakage.
- **Uncertainty layer**
  - Markov transition model for observable evolution.
  - HMM-style regime inference (`Bull`, `Bear`, `Stagnant`) plus conservative risk lower bound.
- **Learning layer**
  - Baseline interpretable elasticity model + residual neural approximation.
- **Search layer**
  - A* with admissible heuristic aligned with normalized objective terms.
  - Resource guards (`max_depth`, `max_nodes`, `timeout`) and automatic GBFS fallback.
  - Additional BFS helper for diagnostics.
- **Optimization layer**
  - Multi-objective trajectory scoring over mismatch, volatility, and regime risk.
  - Best-candidate selector.
- **Language layer**
  - JSON-constrained LLM interface for config parsing + human-readable explanations.
- **Orchestration layer**
  - `NexusEngine` pipeline wiring all modules with strict interfaces.
- **API layer**
  - FastAPI endpoint `POST /plan`.

## Run

```bash
python main.py
```

## API

```bash
uvicorn nexus_engine.api.app:app --reload
```
