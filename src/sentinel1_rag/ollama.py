"""Minimal client for Ollama's REST API, without an SDK, so every request is visible."""

from typing import Any

import httpx

from sentinel1_rag.config import OLLAMA_URL


# Sent with every request. When Ollama sees little free RAM in WSL it turns mmap off, and llama.cpp
# then copies the whole model file into RAM: Gemma's runner grew from 3.3 to 6.0 GB, and with
# bge-m3 loaded next to it WSL ran out of memory and froze (D-030). With mmap, only the weights
# that run on the CPU stay in RAM. Every request must send it: a request whose load options differ
# from the loaded model's makes Ollama load the model again.
LOAD_OPTIONS = {"use_mmap": True}
# Ollama refuses more ("top_logprobs must be between 0 and 20").
TOP_LOGPROBS = 20


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL) -> None:
        self._base_url = base_url
        self._http = httpx.Client(base_url=base_url, timeout=300)

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        """One vector per text. A text longer than the model's context fails instead of being cut."""
        payload = {"model": model, "input": texts, "truncate": False, "options": LOAD_OPTIONS}
        data = self._call("POST", "/api/embed", payload)
        vectors = data["embeddings"]
        if len(vectors) != len(texts):
            raise OllamaError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """One complete answer, not streamed, with thinking off unless asked for.

        Returns Ollama's whole response: the message plus token counts and timings. With
        `tools`, the message may hold tool calls instead of an answer (`message.tool_calls`).
        Ollama does not run them: the caller does and sends back the results (agent.py).
        """
        payload = {
            "model": model,
            "messages": messages,
            "options": LOAD_OPTIONS | options,
            "stream": False,
            "think": think,
        }
        if tools:
            payload["tools"] = tools
        return self._call("POST", "/api/chat", payload)

    def next_token_logprobs(self, model: str, prompt: str, options: dict[str, Any]) -> dict[str, float]:
        """Log-probabilities of the likeliest tokens to follow a prompt, as {token: logprob}.

        The prompt is sent raw, without the model's chat template, so it must already contain
        any special tokens. Ollama returns at most 20 tokens (TOP_LOGPROBS).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "options": LOAD_OPTIONS | options | {"num_predict": 1},
        }
        data = self._call("POST", "/api/generate", payload)
        return {entry["token"]: entry["logprob"] for entry in data["logprobs"][0]["top_logprobs"]}

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
