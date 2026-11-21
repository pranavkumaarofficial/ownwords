"""Model access.

Two backends behind one call. Anthropic for quality, Ollama for a local model
that costs nothing per rewrite -- the pipeline does not care which, so the choice
can be made per task (a mail to a professor is worth an API call; reformatting
your own notes is not).

urllib rather than the vendor SDKs: this keeps the package installable with no
third-party packages at all, which also means no lockfile to audit.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


class LLMError(RuntimeError):
    pass


@dataclass
class LLM:
    backend: str = "ollama"             # "ollama" | "anthropic"
    model: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 4096
    ollama_host: str = "http://localhost:11434"
    timeout: int = 180

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = (
                DEFAULT_ANTHROPIC_MODEL if self.backend == "anthropic"
                else DEFAULT_OLLAMA_MODEL
            )

    def complete(self, system: str, user: str) -> str:
        if self.backend == "anthropic":
            return self._anthropic(system, user)
        if self.backend == "ollama":
            return self._ollama(system, user)
        raise LLMError("unknown backend %r" % self.backend)

    # --- backends ----------------------------------------------------------
    def _anthropic(self, system: str, user: str) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._post(
            ANTHROPIC_URL, body,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        try:
            return "".join(
                b["text"] for b in data["content"] if b.get("type") == "text"
            ).strip()
        except (KeyError, TypeError) as e:
            raise LLMError("unexpected response shape: %s" % data) from e

    def _ollama(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = self._post(self.ollama_host.rstrip("/") + "/api/chat", body, {})
        return (data.get("message") or {}).get("content", "").strip()

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMError("%s %s: %s" % (url, e.code, e.read().decode("utf-8", "replace")[:400])) from e
        except urllib.error.URLError as e:
            raise LLMError("cannot reach %s: %s" % (url, e.reason)) from e
