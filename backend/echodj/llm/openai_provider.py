"""
OpenAI-Compatible LLM Provider
================================
OpenAI (and compatible API) implementation of the LLMProvider protocol.

References:
    - Spec §4.1 (LLM Provider Abstraction)
    - Config: ECHODJ_LLM_PROVIDER=openai, ECHODJ_LLM_MODEL, OPENAI_API_KEY
"""

from __future__ import annotations

import logging

import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_OPENAI_BASE = "https://api.openai.com/v1"


class OpenAIProvider:
    """OpenAI chat completion provider."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or switch to a different provider."
            )
        self._client = httpx.AsyncClient(
            base_url=_OPENAI_BASE,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=_TIMEOUT,
        )
        self._model = settings.echodj_llm_model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text via the OpenAI chat completions endpoint."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }

        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenAI API error %s: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return ""
        except Exception:
            logger.exception("OpenAI generate() failed")
            return ""

    async def close(self) -> None:
        await self._client.aclose()
