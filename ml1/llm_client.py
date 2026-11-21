"""Client Ollama (doar biblioteca standard — fără dependențe pe ținta embedded).

    from ml1.llm_client import OllamaClient
    llm = OllamaClient()          # qwen2.5:1.5b
    llm.warmup()                  # încarcă modelul + cache-ul prefixului
    decide(ctx, llm)              # din ml1.validate

Modelul se schimbă cu variabila de mediu VIGIL_MODEL (ex. llama3.2:1b).
"""

from __future__ import annotations

import json
import os
import urllib.request

from .actions import LLM_SCHEMA
from .prompt import build_messages

DEFAULT_MODEL = os.environ.get("VIGIL_MODEL", "qwen2.5:1.5b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

OPTIONS = {
    "temperature": 0,     # determinist
    "seed": 42,
    "num_predict": 48,    # răspunsul compact are ~25 tokeni
    "num_ctx": 2048,      # promptul are ~1000 tokeni MĂSURAȚI (prompt_eval_count); 1024 ar tăia începutul (regulile)
    "top_k": 1,
}


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: float = 2.5, keep_alive: str = "60m", options: dict | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive  # modelul rămâne în memorie: fără încărcare la fiecare apel
        self.options = {**OPTIONS, **(options or {})}

    def _post(self, path: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.host + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def payload(self, ctx: dict) -> dict:
        body = {
            "model": self.model,
            "messages": build_messages(ctx),
            "stream": False,
            "format": LLM_SCHEMA,     # structured output: modelul nu poate ieși din meniu
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        if self.model.startswith("qwen3"):
            body["think"] = False     # fără raționament ascuns: latență mică
        return body

    def __call__(self, ctx: dict) -> dict:
        """Întoarce răspunsul brut Ollama; `validate.decide` îl parsează și validează."""
        return self._post("/api/chat", self.payload(ctx), self.timeout)

    def warmup(self) -> float:
        """Primul apel încarcă modelul și pune SYSTEM + FEW_SHOT în cache. Întoarce durata (s)."""
        r = self._post("/api/chat", self.payload({"fusion": {"a": 1.0}}), timeout=120)
        return r.get("total_duration", 0) / 1e9

    def available(self) -> bool:
        try:
            tags = self._post_get("/api/tags")
        except OSError:
            return False
        return any(m.get("name", "").startswith(self.model) for m in tags.get("models", []))

    def _post_get(self, path: str) -> dict:
        with urllib.request.urlopen(self.host + path, timeout=2) as r:
            return json.loads(r.read().decode())
