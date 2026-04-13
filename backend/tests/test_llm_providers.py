"""
Tests for the LLM Provider abstraction.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from echodj.llm.provider import LLMProvider, get_provider


class TestLLMProviderProtocol:
    """Tests for the LLMProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """LLMProvider should be runtime_checkable."""
        class FakeProvider:
            async def generate(self, system_prompt: str, user_prompt: str) -> str:
                return "test"

        provider = FakeProvider()
        assert isinstance(provider, LLMProvider)

    def test_missing_generate_fails_check(self) -> None:
        """An object without generate() should not satisfy the protocol."""
        class NotAProvider:
            pass

        obj = NotAProvider()
        assert not isinstance(obj, LLMProvider)

    def test_get_provider_raises_on_unknown(self) -> None:
        """get_provider() should raise ValueError for unknown provider names."""
        # settings is imported inside get_provider() from echodj.config
        with patch("echodj.config.settings") as mock_settings:
            mock_settings.echodj_llm_provider = "invalid_provider"
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_provider()


class TestProviderConformance:
    """Ensure all provider classes satisfy the LLMProvider protocol."""

    @pytest.mark.asyncio
    async def test_gemini_provider_protocol(self) -> None:
        """GeminiProvider should implement LLMProvider protocol."""
        with patch("echodj.llm.gemini.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.echodj_llm_model = "gemini-2.0-flash"
            from echodj.llm.gemini import GeminiProvider
            provider = GeminiProvider()
            assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_ollama_provider_protocol(self) -> None:
        """OllamaProvider should implement LLMProvider protocol."""
        with patch("echodj.llm.ollama.settings") as mock_settings:
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.echodj_llm_model = "llama3"
            from echodj.llm.ollama import OllamaProvider
            provider = OllamaProvider()
            assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_openai_provider_protocol(self) -> None:
        """OpenAIProvider should implement LLMProvider protocol."""
        with patch("echodj.llm.openai_provider.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            mock_settings.echodj_llm_model = "gpt-4o"
            from echodj.llm.openai_provider import OpenAIProvider
            provider = OpenAIProvider()
            assert isinstance(provider, LLMProvider)

    @pytest.mark.asyncio
    async def test_gemini_returns_empty_on_error(self) -> None:
        """Gemini provider should return empty string on HTTP error (not raise)."""
        import httpx

        with patch("echodj.llm.gemini.settings") as mock_settings:
            mock_settings.gemini_api_key = "test-key"
            mock_settings.echodj_llm_model = "gemini-2.0-flash"
            from echodj.llm.gemini import GeminiProvider
            provider = GeminiProvider()

            with patch.object(
                provider._client,
                "post",
                side_effect=httpx.TimeoutException("timeout"),
            ):
                result = await provider.generate("system", "user")
                assert result == ""
