"""
LLM Provider Protocol
======================
A plug-and-play abstraction for all LLM calls in EchoDJ.
Every agent calls a single ``generate()`` method — the implementation
is swappable via the ``ECHODJ_LLM_PROVIDER`` environment variable.

References:
    - Spec §4.1 (LLM Provider Abstraction)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for all LLM providers used in EchoDJ.

    All agents (Curator, Scriptwriter, Memory Manager, Observer intent
    classifier) call this single interface.  The implementation is
    swapped transparently via the ``ECHODJ_LLM_PROVIDER`` env var.

    Usage::

        provider = get_provider()
        text = await provider.generate(
            system_prompt="You are a DJ...",
            user_prompt="Write a liner for...",
        )
    """

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text response from the LLM.

        Args:
            system_prompt: The system-level context / persona.
            user_prompt: The specific request for this call.

        Returns:
            Generated text. Never raises — returns empty string on failure
            so callers can apply their own fallback logic.
        """
        ...


def get_provider() -> LLMProvider:
    """Factory: return the configured LLM provider instance.

    Reads ``ECHODJ_LLM_PROVIDER`` from settings and returns the
    appropriate implementation.  Providers are imported lazily so
    unused providers don't impose startup overhead.

    Raises:
        ValueError: If the configured provider name is unknown.
    """
    from echodj.config import settings

    provider_name = settings.echodj_llm_provider.lower()

    if provider_name == "gemini":
        from echodj.llm.gemini import GeminiProvider
        return GeminiProvider()

    if provider_name == "ollama":
        from echodj.llm.ollama import OllamaProvider
        return OllamaProvider()

    raise ValueError(
        f"Unknown LLM provider: {provider_name!r}. "
        "Set ECHODJ_LLM_PROVIDER to 'gemini' or 'ollama'."
    )
