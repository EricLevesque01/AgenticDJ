"""
Vocalizer Node
===============
Converts the Scriptwriter's liner text into spoken MP3 audio using edge-tts.

References:
    - Spec §5.6 (Vocalizer Node)
    - Latency target: < 1.5s for 15–20s of audio
    - Error handling: if edge-tts fails, audio_buffer = None
                     (Broadcast skips the DJ break)
"""

from __future__ import annotations

import logging

from echodj.services.tts import TTSError, synthesize
from echodj.state import DJState

logger = logging.getLogger(__name__)


async def vocalizer_node(state: DJState) -> dict:
    """LangGraph node: synthesize script text into audio.

    Input reads (Spec §3.3):
        - script_text

    Output writes (Spec §3.3):
        - audio_buffer (bytes | None)
        - audio_duration_ms (int)
    """
    script_text = state.get("script_text", "")

    if not script_text:
        logger.warning("Vocalizer: empty script text, skipping synthesis")
        return {"audio_buffer": None, "audio_duration_ms": 0}

    try:
        audio_bytes, duration_ms = await synthesize(script_text)
        logger.info(
            "Vocalizer: synthesized %d bytes (~%dms)",
            len(audio_bytes), duration_ms,
        )
        return {
            "audio_buffer": audio_bytes,
            "audio_duration_ms": duration_ms,
        }

    except TTSError as exc:
        # Spec §5.6: edge-tts unavailable → audio_buffer = None
        # Broadcast node will skip the DJ break entirely
        logger.error("Vocalizer: TTS failed: %s", exc)
        return {"audio_buffer": None, "audio_duration_ms": 0}
