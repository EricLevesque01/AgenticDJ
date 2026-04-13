"""
edge-tts Text-to-Speech Wrapper
================================
Converts DJ liner scripts into spoken MP3 audio.

References:
    - Spec §5.6 (Vocalizer Node)
    - Spec §6 (API Strategy — 3s timeout)

Latency target: < 1.5s for 15–20s of audio (Spec §5.6).
"""

from __future__ import annotations

import asyncio
import io
import logging

import edge_tts

from echodj.config import settings

logger = logging.getLogger(__name__)

# Duration estimation: edge-tts outputs MP3 at ~32kbps for speech (Spec §5.6)
_MP3_BITRATE_KBPS = 32
# Hard timeout for TTS synthesis (Spec §5.6: skip DJ break if > 3s)
_TTS_TIMEOUT_S = 3.0


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails."""


async def synthesize(text: str, voice: str | None = None) -> tuple[bytes, int]:
    """Convert text to spoken audio using edge-tts.

    Spec §5.6 implementation:
        - Voice: en-US-GuyNeural (configurable via ECHODJ_TTS_VOICE env var)
        - Timeout: 3s hard cap
        - On failure: raise TTSError (Broadcast node skips DJ break)

    Args:
        text: The DJ liner script to synthesize (40–55 words normally).
        voice: Optional voice override. Defaults to settings.echodj_tts_voice.

    Returns:
        Tuple of (mp3_audio_bytes, duration_ms).

    Raises:
        TTSError: If synthesis fails or times out.
    """
    if not text.strip():
        raise TTSError("Cannot synthesize empty text")

    selected_voice = voice or settings.echodj_tts_voice

    try:
        audio_bytes = await asyncio.wait_for(
            _stream_audio(text, selected_voice),
            timeout=_TTS_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        raise TTSError(
            f"TTS synthesis timed out after {_TTS_TIMEOUT_S}s for voice={selected_voice}"
        )
    except Exception as exc:
        raise TTSError(f"TTS synthesis failed: {exc}") from exc

    if not audio_bytes:
        raise TTSError("TTS returned empty audio buffer")

    # Duration estimation from Spec §5.6:
    # length_bytes * 8 bits/byte / 32 kbps = seconds * 1000 = ms
    duration_ms = (len(audio_bytes) * 8) // _MP3_BITRATE_KBPS

    logger.info(
        "TTS synthesized: %d bytes, ~%dms, voice=%s",
        len(audio_bytes),
        duration_ms,
        selected_voice,
    )
    return audio_bytes, duration_ms


async def _stream_audio(text: str, voice: str) -> bytes:
    """Stream audio chunks from edge-tts and concatenate into a single buffer."""
    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()


async def list_voices() -> list[dict]:
    """List available edge-tts voices. Useful for configuration."""
    return await edge_tts.list_voices()
