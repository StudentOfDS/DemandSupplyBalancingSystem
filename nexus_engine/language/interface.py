from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LLMConfigDTO:
    horizon: int = 12
    max_nodes: int = 3000


@dataclass(frozen=True)
class HuggingFaceSettings:
    endpoint_url: str
    api_token: str | None = None
    timeout_seconds: float = 15.0
    retries: int = 2

    @classmethod
    def from_env(cls) -> HuggingFaceSettings | None:
        endpoint = os.getenv("HF_INFERENCE_ENDPOINT")
        if not endpoint:
            return None
        return cls(
            endpoint_url=endpoint,
            api_token=os.getenv("HF_API_TOKEN"),
            timeout_seconds=float(os.getenv("HF_TIMEOUT_SECONDS", "15")),
            retries=int(os.getenv("HF_RETRIES", "2")),
        )


class HuggingFaceLLMClient:
    """Minimal Hugging Face inference client with JSON-only response contract."""

    def __init__(self, settings: HuggingFaceSettings | None = None) -> None:
        self._settings = settings or HuggingFaceSettings.from_env()

    def generate_json(self, prompt: str, fallback: dict[str, object] | None = None) -> dict[str, object]:
        if self._settings is None:
            return fallback or {"horizon": 12, "max_nodes": 3000}

        payload = json.dumps({"inputs": prompt}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._settings.api_token:
            headers["Authorization"] = f"Bearer {self._settings.api_token}"

        last_error: Exception | None = None
        for _ in range(self._settings.retries + 1):
            request = Request(self._settings.endpoint_url, data=payload, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self._settings.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                return self._extract_json(raw)
            except (TimeoutError, URLError, ValueError) as exc:
                last_error = exc
        if fallback is not None:
            return fallback
        raise RuntimeError("Hugging Face inference failed") from last_error

    def _extract_json(self, raw: str) -> dict[str, object]:
        decoded = json.loads(raw)
        if isinstance(decoded, dict) and "generated_text" in decoded:
            return self._loads_json_object(str(decoded["generated_text"]))
        if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict) and "generated_text" in decoded[0]:
            return self._loads_json_object(str(decoded[0]["generated_text"]))
        if isinstance(decoded, dict):
            return decoded
        raise ValueError("Hugging Face response did not contain a JSON object")

    def _loads_json_object(self, text: str) -> dict[str, object]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Generated text did not contain JSON")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Generated JSON was not an object")
        return parsed


class ConstrainedLanguageLayer:
    """LLM interface constrained to valid JSON payloads only."""

    def __init__(self, client: HuggingFaceLLMClient | None = None) -> None:
        self._client = client or HuggingFaceLLMClient()

    def parse_natural_language(self, user_prompt: str) -> LLMConfigDTO:
        prompt = (
            "Return only JSON matching this schema: "
            '{"horizon": positive_integer, "max_nodes": positive_integer}. '
            f"User request: {user_prompt}"
        )
        payload = self._client.generate_json(prompt, fallback={"horizon": 12, "max_nodes": 3000})
        return self._validate_payload(payload)

    def parse_user_prompt(self, llm_raw_output: str) -> LLMConfigDTO:
        try:
            payload = json.loads(llm_raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON output from LLM") from exc
        return self._validate_payload(payload)

    def explain(self, algorithm: str, reached: bool, steps: int) -> str:
        if reached:
            return f"Selected {algorithm}. Equilibrium reached in {steps} step(s)."
        return f"Selected {algorithm}. No equilibrium trajectory found within configured limits."

    def _validate_payload(self, payload: object) -> LLMConfigDTO:
        if not isinstance(payload, dict):
            raise ValueError("Invalid schema output from LLM")
        horizon = int(payload.get("horizon", 12))
        max_nodes = int(payload.get("max_nodes", 3000))
        if horizon <= 0 or max_nodes <= 0:
            raise ValueError("horizon and max_nodes must be positive")
        return LLMConfigDTO(horizon=horizon, max_nodes=max_nodes)
