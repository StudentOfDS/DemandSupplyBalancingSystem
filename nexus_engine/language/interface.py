from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfigDTO:
    horizon: int = 12
    max_nodes: int = 3000


class ConstrainedLanguageLayer:
    """LLM interface constrained to valid JSON payloads only."""

    def parse_user_prompt(self, llm_raw_output: str) -> LLMConfigDTO:
        try:
            payload = json.loads(llm_raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON output from LLM") from exc

        if not isinstance(payload, dict):
            raise ValueError("Invalid schema output from LLM")

        horizon = int(payload.get("horizon", 12))
        max_nodes = int(payload.get("max_nodes", 3000))
        if horizon <= 0 or max_nodes <= 0:
            raise ValueError("horizon and max_nodes must be positive")
        return LLMConfigDTO(horizon=horizon, max_nodes=max_nodes)

    def explain(self, algorithm: str, reached: bool, steps: int) -> str:
        if reached:
            return f"Selected {algorithm}. Equilibrium reached in {steps} step(s)."
        return f"Selected {algorithm}. No equilibrium trajectory found within configured limits."
