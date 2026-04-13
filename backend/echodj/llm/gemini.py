"""
Gemini LLM Provider
====================
Google Gemini implementation of the LLMProvider protocol.
Uses the Gemini REST API directly via httpx (no SDK dependency).

References:
    - Spec §4.1 (LLM Provider Abstraction)
    - Config: ECHODJ_LLM_PROVIDER=gemini, ECHODJ_LLM_MODEL, GEMINI_API_KEY
"""

from __future__ import annotations

import logging

import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class GeminiProvider:
    """Google Gemini LLM provider via REST API."""

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or switch to a different provider."
            )
        self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        self._model = settings.echodj_llm_model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text via the Gemini generateContent endpoint.

        Combines system_prompt and user_prompt into Gemini's content format.
        Returns empty string on any failure so callers can apply fallbacks.
        """
        url = f"{_GEMINI_BASE}/{self._model}:generateContent"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 512,
            },
        }

        try:
            response = await self._client.post(
                url,
                json=payload,
                params={"key": settings.gemini_api_key},
            )
            response.raise_for_status()
            data = response.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return text.strip()

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Gemini API error %s: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return ""
        except Exception:
            logger.exception("Gemini generate() failed")
            return ""

    async def close(self) -> None:
        await self._client.aclose()
