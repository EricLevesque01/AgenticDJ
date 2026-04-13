"""
Broadcast Node
===============
Sends ducking signals and DJ audio to the frontend via WebSocket.

References:
    - Spec §5.7 (Broadcast Node)
    - Sequence: duck_start → wait 300ms → stream audio → wait → duck_end
    - Spec §7.3: Message types (duck_start, duck_end, skip_to_next)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from echodj.state import DJState

logger = logging.getLogger(__name__)


# Type alias for the WebSocket send function passed in from the server
SendJsonFn = Callable[[dict[str, Any]], Awaitable[None]]
SendBytesFn = Callable[[bytes], Awaitable[None]]


async def broadcast_node(
    state: DJState,
    send_json: SendJsonFn,
    send_bytes: SendBytesFn,
) -> dict:
    """LangGraph node: send ducking signals and stream DJ audio.

    Input reads (Spec §3.3):
        - audio_buffer, audio_duration_ms, queue_action

    Output writes (Spec §3.3):
        - ducking_active

    Sequence (Spec §5.7):
        1. Send duck_start (fade_ms=300)
        2. Wait 300ms for frontend fade
        3. Stream audio_buffer as binary WebSocket frame
        4. Wait audio_duration_ms
        5. Send duck_end (fade_ms=500)
        6. If queue_action == "play_next": send skip_to_next
    """
    audio_buffer = state.get("audio_buffer")
    audio_duration_ms = state.get("audio_duration_ms", 0)
    queue_action = state.get("queue_action", "continue")

    # No audio — skip DJ break entirely (Spec §5.7)
    if not audio_buffer:
        logger.info("Broadcast: no audio buffer, skipping DJ break")
        if queue_action == "play_next":
            await _safe_send(send_json, {"type": "skip_to_next"})
        return {"ducking_active": False}

    logger.info(
        "Broadcast: starting DJ break (%dms audio, action=%s)",
        audio_duration_ms, queue_action,
    )

    # Step 1: Signal frontend to duck Spotify (Spec §5.7)
    await _safe_send(send_json, {"type": "duck_start", "fade_ms": 300})

    # Step 2: Wait for fade to complete
    await asyncio.sleep(0.300)

    # Step 3: Stream DJ audio
    try:
        await send_bytes(audio_buffer)
    except Exception:
        logger.exception("Broadcast: failed to send audio bytes")
        await _safe_send(send_json, {"type": "duck_end", "fade_ms": 100})
        return {"ducking_active": False}

    # Step 4: Wait for audio to finish playing
    await asyncio.sleep(audio_duration_ms / 1000.0)

    # Step 5: Restore Spotify volume
    await _safe_send(send_json, {"type": "duck_end", "fade_ms": 500})

    # Step 6: Advance to next queued track
    if queue_action in ("play_next", "interrupt"):
        await _safe_send(send_json, {"type": "skip_to_next"})

    logger.info("Broadcast: DJ break complete")
    return {"ducking_active": False}


async def _safe_send(send_fn: SendJsonFn, message: dict[str, Any]) -> None:
    """Send a JSON message, swallowing errors (Spec §5.7 error handling)."""
    try:
        await send_fn(message)
    except Exception:
        logger.warning("Broadcast: send_json failed for %s", message.get("type"))
