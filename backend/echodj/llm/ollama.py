"""
Ollama LLM Provider
====================
Local Ollama implementation of the LLMProvider protocol.
Calls the Ollama REST API running on localhost.

References:
    - Spec §4.1 (LLM Provider Abstraction)
    - Config: ECHODJ_LLM_PROVIDER=ollama, ECHODJ_LLM_MODEL, OLLAMA_BASE_URL
"""

from __future__ import annotations

import logging

import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class OllamaProvider:
    """Local Ollama LLM provider."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=_TIMEOUT,
        )
        self._model = settings.echodj_llm_model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text via the Ollama /api/generate endpoint.

        Uses non-streaming mode for simplicity.
        Returns empty string on any failure so callers can apply fallbacks.
        """
        payload = {
            "model": self._model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": 0.7},
        }

        try:
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except httpx.ConnectError:
            logger.error(
                "Ollama not running at %s. Start with: ollama serve",
                settings.ollama_base_url,
            )
            return ""
        except Exception:
            logger.exception("Ollama generate() failed")
            return ""

    async def close(self) -> None:
        await self._client.aclose()
