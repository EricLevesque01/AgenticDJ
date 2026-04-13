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
import base64
import io
import logging
import struct

import edge_tts
import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)

# Duration estimation for edge-tts (MP3 ~32kbps)
_MP3_BITRATE_KBPS = 32
# Timeout for TTS synthesis
_TTS_TIMEOUT_S = 7.0


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

    provider = settings.echodj_tts_provider

    if provider == "gemini":
        selected_voice = voice or settings.echodj_gemini_voice
        try:
            return await asyncio.wait_for(
                _synthesize_gemini(text, selected_voice),
                timeout=_TTS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise TTSError(f"Gemini TTS timed out after {_TTS_TIMEOUT_S}s")
    else:
        # Fall back to Edge TTS
        selected_voice = voice or settings.echodj_tts_voice
        try:
            audio_bytes = await asyncio.wait_for(
                _stream_audio(text, selected_voice),
                timeout=_TTS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise TTSError(f"Edge TTS timed out after {_TTS_TIMEOUT_S}s")
        except Exception as exc:
            raise TTSError(f"Edge TTS failure: {exc}") from exc

        if not audio_bytes:
            raise TTSError("Edge TTS returned empty audio buffer")

        duration_ms = (len(audio_bytes) * 8) // _MP3_BITRATE_KBPS
        logger.info(
            "Edge TTS synthesized: %d bytes, ~%dms, voice=%s",
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


def _build_wav_header(pcm_data: bytes, sample_rate: int = 24000, num_channels: int = 1, bit_depth: int = 16) -> bytes:
    """Builds a WAV container header for raw PCM audio data."""
    byte_rate = sample_rate * num_channels * (bit_depth // 8)
    block_align = num_channels * (bit_depth // 8)
    data_size = len(pcm_data)
    chunk_size = 36 + data_size
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', chunk_size, b'WAVE',
        b'fmt ', 16, 1, num_channels, sample_rate, byte_rate, block_align, bit_depth,
        b'data', data_size
    )
    return header + pcm_data


async def _synthesize_gemini(text: str, voice: str) -> tuple[bytes, int]:
    """Generates speech using the Gemini 2.0 multimodal audio API."""
    api_key = settings.gemini_api_key
    if not api_key:
        raise TTSError("GEMINI_API_KEY is not set for Gemini TTS.")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    # Minimal system instruction so the model just acts as a pure dictation TTS component
    # without trying to converse.
    payload = {
        "system_instruction": {
            "parts": [{"text": "You are a pure text-to-speech engine. Recite the exact user text word-for-word, beautifully, with the cadence of a radio DJ. Do not add any conversational filler. Wait half a second before starting."}]
        },
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice
                    }
                }
            }
        }
    }

    # Gemini Audio PCM format: 24kHz, 16-bit => 48000 bytes/sec
    pcm_bytes_per_second = 24000 * 2

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, params={"key": api_key})
            resp.raise_for_status()
            data = resp.json()

            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])

            audio_base64 = None
            for part in parts:
                if "inlineData" in part and part["inlineData"].get("mimeType", "").startswith("audio/pcm"):
                    audio_base64 = part["inlineData"]["data"]
                    break

            if not audio_base64:
                raise TTSError(f"Gemini API did not return native audio. Response limits hit?")

            pcm_data = base64.b64decode(audio_base64)
            wav_data = _build_wav_header(pcm_data)
            duration_ms = (len(pcm_data) * 1000) // pcm_bytes_per_second

            logger.info("Gemini TTS synthesized: %d bytes (PCM), ~%dms, voice=%s", 
                        len(pcm_data), duration_ms, voice)
            return wav_data, duration_ms

        except httpx.HTTPStatusError as exc:
            raise TTSError(f"Gemini TTS HTTP {exc.response.status_code}: {exc.response.text[:100]}") from exc
        except Exception as exc:
            raise TTSError(f"Gemini TTS internal failure: {exc}") from exc


async def list_voices() -> list[dict]:
    """List available edge-tts voices. Useful for configuration."""
    return await edge_tts.list_voices()
