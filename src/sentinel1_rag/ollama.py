"""Minimal client for Ollama's REST API, without an SDK, so every request is visible."""

from typing import Any

import httpx

from sentinel1_rag.config import OLLAMA_URL


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self._base_url = base_url
        self._http = httpx.Client(base_url=base_url, timeout=300)

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """One vector per text. A text longer than the model's context fails instead of being cut."""
        data = self._call("POST", "/api/embed", {"model": model, "input": texts, "truncate": False})
        vectors = data["embeddings"]
        if len(vectors) != len(texts):
            raise OllamaError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def digest(self, model: str) -> str:
        """Digest of a pulled model; it changes if the model is pulled again with new weights."""
        name = model if ":" in model else f"{model}:latest"
        for entry in self._call("GET", "/api/tags")["models"]:
            if entry["name"] == name:
                return entry["digest"]
        raise OllamaError(f"model {name} is not pulled; run `ollama pull {model}`")

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict[str, Any]:
        try:
            response = self._http.request(method, path, json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self._base_url}; is the service running? "
                "(systemctl status ollama)"
            ) from exc
        if response.is_error:
            raise OllamaError(f"{method} {path} returned {response.status_code}: {response.text}")
        return response.json()
