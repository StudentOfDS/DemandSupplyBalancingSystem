# DemandSupplyBalancingSystem (Nexus Engine)

A modular demand-supply balancing engine that models market equilibrium as a **trajectory planning** problem. The repository now includes the algorithmic MVP plus production-enablement pieces for real input files, training hooks, persistence, API/CLI usage, optional Hugging Face integration, Docker deployment, and expanded tests.

## Implemented Architecture

- **Core state and DTO layer**
  - Immutable `MarketState` and immutable transfer objects (`TrajectoryDTO`, request/response DTOs).
  - Adaptive state binning + hash for search deduplication.
  - Request/response validation and deterministic heap tie-break ordering.
- **Real data input**
  - CSV ingestion for timestamp, price, supply, demand, inventory, competitor price, seasonality, traffic, cost, and optional regime.
  - JSON API ingestion for the same market record schema.
  - CLI support for `--csv examples/market_data.csv`.
- **Knowledge/rule layer**
  - Predicate-style transition validator enforcing non-negativity, capacity bounds, law of demand, and law of supply.
  - Vector-store facade with Pinecone primary + FAISS fallback semantics.
- **Data/feature layer**
  - Deterministic feature ingestion and future-time extrapolation to prevent feature leakage.
- **Uncertainty layer**
  - Markov transition model for observable evolution.
  - Trainable HMM-style regime transition counts and regime inference (`Bull`, `Bear`, `Stagnant`).
- **Learning layer**
  - Trainable demand elasticity model.
  - Trainable supply response model.
  - Tiny deterministic residual neural learner for grey-box residuals.
- **Search layer**
  - A* with admissible heuristic aligned with normalized objective terms.
  - Resource guards (`max_depth`, `max_nodes`, `timeout`) and automatic GBFS fallback.
  - Adaptive action space plus continuous near-equilibrium action candidate.
  - Additional BFS helper for diagnostics.
- **Optimization layer**
  - Multi-objective trajectory scoring over mismatch, volatility, and regime risk.
  - Best-candidate selector.
- **Language layer**
  - JSON-constrained LLM interface for config parsing and explanations.
  - Optional Hugging Face inference endpoint integration via environment variables, retries, JSON extraction, and local fallback.
- **Storage layer**
  - SQLite persistence for historical market states, user scenarios, generated trajectories, model outputs, and experiment logs.
- **API layer**
  - `POST /plan`, `GET /health`, and `GET /status`.
  - Request parsing, validation errors, response serialization, and model status reporting.
- **Async/deployment layer**
  - Optional Celery task wrapper backed by Redis when Celery is installed.
  - `Dockerfile`, `docker-compose.yml`, `.env.example`, and `requirements.txt`.

## Run the CLI

Synthetic/default state:

```bash
python main.py
```

Real CSV input + SQLite persistence:

```bash
python main.py --csv examples/market_data.csv --sqlite nexus_engine.db --horizon 15 --max-nodes 4000
```

## API

The API bootstraps from `NEXUS_MARKET_CSV` and `NEXUS_SQLITE_PATH` when set, so container or shell deployments can start with trained records and persistence enabled.

```bash
export NEXUS_MARKET_CSV=examples/market_data.csv
export NEXUS_SQLITE_PATH=nexus_engine.db
uvicorn nexus_engine.api.app:app --reload
```

Example request:

```bash
curl -X POST http://localhost:8000/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "initial_state": {
      "price": 100,
      "supply": 50,
      "demand": 150,
      "timestamp": 0,
      "regime": "Stagnant"
    },
    "horizon": 12,
    "max_nodes": 3000
  }'
```

Health and model status:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## Hugging Face configuration

Set these variables to enable real Hugging Face/local inference calls in the language layer:

```bash
export HF_INFERENCE_ENDPOINT="https://api-inference.huggingface.co/models/<model>"
export HF_API_TOKEN="..."
export HF_TIMEOUT_SECONDS=15
export HF_RETRIES=2
```

If no endpoint is configured, the language layer uses a deterministic local fallback JSON config.

## Docker

```bash
docker compose up --build
```

This starts the API, Redis, and a Celery worker. The API listens on `http://localhost:8000`.

## Tests

The suite includes an end-to-end regression proving CSV input → model training → search → optimization → SQLite persistence → API response.

```bash
python -m pytest -q
```
