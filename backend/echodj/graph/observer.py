"""
Observer Node
=============
Monitors Spotify playback state and captures user voice input.
Entry point for the LangGraph loop — event-driven.

References:
    - Spec §5.1 (Observer Node)
    - Builds internal play log (NOT using Spotify recently-played API)
    - Pre-computation trigger at 75% track progress
    - PTT pipeline: buffer → Faster-Whisper → intent classifier
"""

from __future__ import annotations

import logging
from typing import Any

from echodj.models import SpotifyTrack, UserIntent
from echodj.state import DJState

logger = logging.getLogger(__name__)

# Spec §5.1: Pre-computation trigger threshold
_PRE_COMPUTE_THRESHOLD = 0.75
# Spec §5.1: Rolling history max length
_MAX_HISTORY = 20


def process_playback_state(
    state: DJState,
    playback_data: dict[str, Any],
) -> dict:
    """Process a playback_state WebSocket message and update DJState.

    Spec §5.1:
        - Extracts current_track, playback_progress from SDK state
        - Sets track_ending_soon = True when progress > 0.75
        - Maintains previous_tracks rolling list (max 20)

    Args:
        state: Current DJState.
        playback_data: Parsed playback_state message data from WebSocket.

    Returns:
        Partial state update dict.
    """
    # Track change threshold for skip detection
    # Spec §5.1 + skip tracking: progress < 0.3 at track change = skipped
    _SKIP_DETECTION_THRESHOLD = 0.30

    track_uri = playback_data.get("track_uri", "")
    position_ms = playback_data.get("position_ms", 0)
    duration_ms = playback_data.get("duration_ms", 1)
    is_playing = playback_data.get("is_playing", False)
    track_name = playback_data.get("track_name", "")
    artist_name = playback_data.get("artist_name", "")
    album_art_url = playback_data.get("album_art_url")

    if not track_uri or not track_name:
        return {}

    # Calculate progress
    progress = position_ms / max(duration_ms, 1)
    track_ending_soon = progress > _PRE_COMPUTE_THRESHOLD

    # Build SpotifyTrack from playback state
    try:
        new_track = SpotifyTrack(
            spotify_uri=track_uri,
            track_name=track_name,
            artist_name=artist_name,
            album_name="",  # Not available from playback state
            duration_ms=duration_ms,
            album_art_url=album_art_url,
        )
    except ValueError as exc:
        logger.warning("Observer: invalid track data: %s", exc)
        return {}

    # Detect track change → add previous track to history
    current = state.get("current_track")
    updates: dict = {}
    skip_detected = False

    if current and current.spotify_uri != new_track.spotify_uri:
        # Track changed — add old track to history
        history = list(state.get("previous_tracks", []))
        history.append(current)
        # Trim to max length (keep most recent)
        if len(history) > _MAX_HISTORY:
            history = history[-_MAX_HISTORY:]
        updates["previous_tracks"] = [current]  # add reducer appends this

        # ── Skip Detection ──────────────────────────────────────────────
        # If the old track changed before 30% progress, it was a skip.
        old_progress = state.get("playback_progress", 1.0)
        if old_progress < _SKIP_DETECTION_THRESHOLD:
            skip_detected = True
            updates["skipped_tracks"] = [current.spotify_uri]  # add reducer appends
            logger.info(
                "Observer: SKIP detected for %r (progress=%.2f < %.2f)",
                current.track_name, old_progress, _SKIP_DETECTION_THRESHOLD,
            )
        else:
            skip_detected = False

        logger.info(
            "Observer: track changed %r → %r (skip=%s)",
            current.track_name, new_track.track_name, skip_detected,
        )

    updates.update({
        "current_track": new_track,
        "playback_progress": progress,
        "track_ending_soon": track_ending_soon,
        "skip_detected": skip_detected,
    })

    if track_ending_soon and not state.get("track_ending_soon"):
        logger.info(
            "Observer: track ending soon (progress=%.2f), triggering pre-computation",
            progress,
        )

    return updates


def process_ptt_result(
    user_utterance: str | None,
    user_intent: UserIntent | None,
) -> dict:
    """Update state after PTT transcription completes.

    Args:
        user_utterance: Whisper transcription result (None if failed).
        user_intent: Pre-classified intent (None if classification failed).

    Returns:
        Partial state update.
    """
    return {
        "user_utterance": user_utterance,
        "user_intent": user_intent,
    }


def clear_ptt_state() -> dict:
    """Reset PTT fields after the agent loop handles the interrupt."""
    return {
        "user_utterance": None,
        "user_intent": None,
    }
