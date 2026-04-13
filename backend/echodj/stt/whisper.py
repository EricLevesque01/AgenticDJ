"""
Faster-Whisper Speech-to-Text Wrapper
======================================
Transcribes PTT audio from the browser using the Faster-Whisper library.

References:
    - Spec §5.1 (Observer — PTT pipeline)
    - Spec §1.2 (GPU: RTX 3090, large-v3, int8 quantization)

Latency target: < 1s transcription on RTX 3090 (Spec §5.1).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from echodj.config import settings

if TYPE_CHECKING:
    from faster_whisper import WhisperModel as _WhisperModel

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Lazy-loading Faster-Whisper transcriber.

    The model is loaded on first use to avoid blocking application startup.
    Model lives in memory for the duration of the process — one load per
    server instance.

    Spec §5.1:
        - Model: large-v3
        - Device: cuda (RTX 3090)
        - Compute type: int8
        - Beam size: 2 (latency-optimized)
    """

    def __init__(self) -> None:
        self._model: _WhisperModel | None = None

    def _load_model(self) -> _WhisperModel:
        """Load the Whisper model.  Called once on first transcription."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        logger.info(
            "Loading Faster-Whisper model=%s device=%s compute_type=%s",
            settings.echodj_whisper_model,
            settings.echodj_whisper_device,
            settings.echodj_whisper_compute_type,
        )
        self._model = WhisperModel(
            settings.echodj_whisper_model,
            device=settings.echodj_whisper_device,
            compute_type=settings.echodj_whisper_compute_type,
        )
        logger.info("Faster-Whisper model loaded")
        return self._model

    def transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe raw PCM audio bytes to text.

        Spec §5.1:
            - Input: PCM 16kHz, 16-bit, mono (from browser WebSocket)
            - Beam size: 2 (latency optimized)
            - Returns None on empty transcription or error

        Args:
            audio_bytes: Raw PCM audio data as bytes.

        Returns:
            Transcribed text, or None if transcription fails or is empty.
        """
        if not audio_bytes:
            logger.warning("transcribe() called with empty audio bytes")
            return None

        try:
            import io
            import numpy as np

            # Convert raw PCM bytes → numpy float32 array
            # PCM 16kHz 16-bit mono → normalize to [-1.0, 1.0]
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio_np = audio_np / 32768.0  # Normalize int16 range

            model = self._load_model()
            segments, _info = model.transcribe(
                audio_np,
                beam_size=settings.echodj_whisper_beam_size,
                language="en",
                vad_filter=True,      # Skip silence automatically
                vad_parameters={
                    "min_silence_duration_ms": 300,
                },
            )

            # Collect all segments into a single string
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if not text:
                logger.info("Whisper returned empty transcription")
                return None

            logger.info("Transcribed: %r", text)
            return text

        except Exception:
            logger.exception("Whisper transcription failed")
            return None


# Module-level singleton — loaded once per process
transcriber = WhisperTranscriber()
