from __future__ import annotations

from fastapi import FastAPI

from nexus_engine.core.dto import PlannerRequestDTO, PlannerResponseDTO
from nexus_engine.orchestration.engine import NexusEngine

app = FastAPI(title="Nexus Engine")
engine = NexusEngine()


@app.post("/plan", response_model=PlannerResponseDTO)
def plan(request: PlannerRequestDTO) -> PlannerResponseDTO:
    return engine.run(request)
