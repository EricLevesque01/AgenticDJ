"""
Tests for the TTS Service (edge-tts)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from echodj.config import settings
from echodj.services.tts import synthesize, TTSError


class TestTTSSynthesize:
    """Tests for the synthesize() function."""

    def test_empty_text_raises(self) -> None:
        """Empty text should raise TTSError without calling edge-tts."""
        with pytest.raises(TTSError, match="empty text"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(synthesize(""))

    @pytest.mark.asyncio
    async def test_empty_text_raises_async(self) -> None:
        """Async: empty text should raise TTSError."""
        with pytest.raises(TTSError, match="empty text"):
            await synthesize("")

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self) -> None:
        """Whitespace-only text should raise TTSError."""
        with pytest.raises(TTSError, match="empty text"):
            await synthesize("   ")

    @pytest.mark.asyncio
    async def test_timeout_raises_tts_error(self, monkeypatch) -> None:
        """asyncio.TimeoutError should be wrapped as TTSError."""
        import asyncio
        monkeypatch.setattr(settings, "echodj_tts_provider", "edge")

        async def slow_stream(*args, **kwargs):
            await asyncio.sleep(100)  # Will be cancelled by timeout

        with patch("echodj.services.tts._stream_audio", side_effect=asyncio.TimeoutError):
            with pytest.raises(TTSError, match="timed out"):
                await synthesize("Hello world")

    @pytest.mark.asyncio
    async def test_successful_synthesis_returns_tuple(self, monkeypatch) -> None:
        """Successful synthesis should return (bytes, duration_ms) using edge."""
        monkeypatch.setattr(settings, "echodj_tts_provider", "edge")
        fake_audio = b"\xff\xfb" + b"\x00" * 1000  # 1002 bytes of fake MP3

        with patch("echodj.services.tts._stream_audio", AsyncMock(return_value=fake_audio)):
            audio, duration_ms = await synthesize("Hello, welcome to EchoDJ.")
            assert isinstance(audio, bytes)
            assert len(audio) == 1002
            assert duration_ms > 0

    @pytest.mark.asyncio
    async def test_empty_audio_buffer_raises(self, monkeypatch) -> None:
        """Empty audio returned by edge-tts should raise TTSError."""
        monkeypatch.setattr(settings, "echodj_tts_provider", "edge")
        with patch("echodj.services.tts._stream_audio", AsyncMock(return_value=b"")):
            with pytest.raises(TTSError, match="empty audio buffer"):
                await synthesize("Some text")

    def test_duration_estimation(self) -> None:
        """Duration estimation: length_bytes * 8 / 32kbps should be correct."""
        # 32000 bytes = 32kB → 32000*8/32 = 8000ms = 8 seconds
        length_bytes = 32000
        expected_ms = (length_bytes * 8) // 32
        assert expected_ms == 8000

    @pytest.mark.asyncio
    async def test_successful_synthesis_gemini(self, monkeypatch) -> None:
        """Gemini backend should successfully return a WAV buffer and duration."""
        monkeypatch.setattr(settings, "echodj_tts_provider", "gemini")
        fake_wav = b"RIFF" + b"\x00" * 40 # Mock WAV
        
        with patch("echodj.services.tts._synthesize_gemini", AsyncMock(return_value=(fake_wav, 1000))):
            audio, duration_ms = await synthesize("Testing Gemini TTS.")
            assert isinstance(audio, bytes)
            assert len(audio) == 44
            assert duration_ms == 1000

    @pytest.mark.asyncio
    async def test_gemini_timeout(self, monkeypatch) -> None:
        """Gemini timeout should raise TTSError."""
        import asyncio
        monkeypatch.setattr(settings, "echodj_tts_provider", "gemini")
        
        with patch("echodj.services.tts._synthesize_gemini", side_effect=asyncio.TimeoutError):
            with pytest.raises(TTSError, match="Gemini TTS timed out"):
                await synthesize("Timeout text")
