"""
Memory Manager Node
====================
Periodically condenses session data into a persistent Listener Profile
stored in LangGraph's SqliteStore.

References:
    - Spec §5.8 (Memory Manager Node)
    - Trigger: every 10 tracks or at session end
    - Latency target: < 2s (not on critical playback path)
    - Cold start: create profile from Spotify top tracks if none exists
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from echodj.llm.provider import LLMProvider
from echodj.models import ListenerProfile
from echodj.services.segment_cache import SegmentBuilder, SegmentCache
from echodj.services.spotify import SpotifyClient
from echodj.state import DJState

logger = logging.getLogger(__name__)

# Spec §5.8: Memory Manager trigger threshold
_MEMORY_TRIGGER_THRESHOLD = 10

# Spec §5.8: Memory Manager prompt — verbatim from spec
_MEMORY_SYSTEM_PROMPT = """\
You are a memory manager for a music DJ application. Given the user's existing
listener profile and their latest session data, output an UPDATED JSON profile.

Rules:
1. Adjust genre_affinity scores: increase for genres played fully, decrease for skipped.
2. Add artists to favorites if the user gave positive feedback or listened fully (>80%).
3. Add artists to artist_dislikes if skipped multiple times or given negative feedback.
4. Update recent_mood_trajectory based on session vibe evolution.
5. Update skip_patterns: describe patterns you observe (e.g. "skips tracks over 6 min").
6. Update skip_patterns_uris: a list of track URIs the user has skipped — use these
   to avoid recommending those specific tracks again. Append new URIs from skipped_track_uris.
7. Increment total_sessions.
8. Merge new trivia into trivia_discussed.
9. Keep discovery_openness between 0.0 (only familiar) and 1.0 (very adventurous).
   Decrease if many skips occurred this session; increase if user requested discovery.
10. Output ONLY valid JSON, no explanation."""


async def memory_manager_node(
    state: DJState,
    llm: LLMProvider,
    store: Any,  # LangGraph SqliteStore
    spotify: SpotifyClient,
    segment_cache: SegmentCache | None = None,
) -> dict:
    """LangGraph node: update the Listener Profile in SqliteStore.

    Also:
        - Records skipped tracks as negative signals in the profile
        - Triggers SegmentBuilder to refresh pre-curated playlists

    Input reads (Spec §3.3):
        - previous_tracks, session_vibe, discussed_trivia
        - skipped_tracks (skip tracking: negative signals)
        - user_id, tracks_since_last_memory_update

    Output writes (Spec §3.3):
        - tracks_since_last_memory_update reset to 0
        - (Side effect: SqliteStore updated, segments refreshed)
    """
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "unknown")
    previous_tracks = state.get("previous_tracks", [])
    session_vibe = state.get("session_vibe", "moderate")
    discussed_trivia = state.get("discussed_trivia", [])
    skipped_tracks = list(state.get("skipped_tracks", []))  # Skip tracking

    # ── Read existing profile from Store ─────────────────────────────────
    existing_profile = _read_profile(store, user_id)

    # ── Cold start if no profile exists ──────────────────────────────────
    if existing_profile is None:
        logger.info("Memory Manager: cold start for user=%s", user_id)
        existing_profile = await _create_initial_profile(spotify, user_id)

    # ── Build session summary for LLM ────────────────────────────────────
    session_data = _build_session_summary(
        previous_tracks, session_vibe, discussed_trivia, session_id, skipped_tracks
    )

    # ── LLM: update profile ───────────────────────────────────────────────
    updated_profile = await _update_profile_with_llm(
        llm, existing_profile, session_data
    )

    # ── Write updated profile to Store ───────────────────────────────────
    _write_profile(store, user_id, updated_profile)
    _write_session_summary(store, user_id, session_id, session_data)

    # ── Flush discussed_trivia to Store (GAP 8: cross-session dedup) ─────
    _write_discussed_trivia(store, user_id, discussed_trivia)

    # ── Refresh pre-curated segments in background ────────────────────
    if segment_cache:
        try:
            import asyncio
            builder = SegmentBuilder(segment_cache)
            # Run as background task — not on critical playback path
            asyncio.create_task(
                builder.build_from_profile(user_id, updated_profile, spotify)
            )
            logger.info("Memory Manager: segment refresh queued for user=%s", user_id)
        except Exception:
            logger.warning("Memory Manager: segment refresh failed", exc_info=True)

    logger.info(
        "Memory Manager: profile updated for user=%s (sessions=%d)",
        user_id,
        updated_profile.get("total_sessions", 0),
    )

    return {"tracks_since_last_memory_update": 0}


def should_run(state: DJState) -> bool:
    """Return True if the Memory Manager should run now.

    Spec §5.8: Trigger when tracks_since_last_memory_update >= 10.
    """
    return state.get("tracks_since_last_memory_update", 0) >= _MEMORY_TRIGGER_THRESHOLD


def _read_profile(store: Any, user_id: str) -> ListenerProfile | None:
    """Read Listener Profile from LangGraph SqliteStore.

    Spec §10.2: namespace ("users", user_id, "listener_profile")
    """
    try:
        result = store.get(("users", user_id, "listener_profile"), "profile")
        if result:
            return result.value if hasattr(result, "value") else result
    except Exception:
        logger.warning("Memory Manager: failed to read profile", exc_info=True)
    return None


def _write_profile(store: Any, user_id: str, profile: ListenerProfile) -> None:
    """Write Listener Profile to LangGraph SqliteStore."""
    try:
        store.put(("users", user_id, "listener_profile"), "profile", profile)
    except Exception:
        logger.error("Memory Manager: failed to write profile", exc_info=True)


def _write_session_summary(
    store: Any, user_id: str, session_id: str, summary: dict
) -> None:
    """Write session summary to SqliteStore (append-only log)."""
    try:
        store.put(("users", user_id, "session_summaries"), session_id, summary)
    except Exception:
        logger.warning("Memory Manager: failed to write session summary")


def load_discussed_trivia(store: Any, user_id: str) -> list[str]:
    """Load cross-session discussed trivia from SqliteStore.

    Spec §10.2: namespace ("users", user_id, "trivia_discussed")
    Called at session start to seed DJState.discussed_trivia.
    """
    try:
        result = store.get(("users", user_id, "trivia_discussed"), "all")
        if result:
            val = result.value if hasattr(result, "value") else result
            if isinstance(val, list):
                return val
    except Exception:
        logger.warning("Memory Manager: failed to load discussed trivia")
    return []


def _write_discussed_trivia(
    store: Any, user_id: str, trivia: list[str]
) -> None:
    """Flush discussed_trivia to SqliteStore for cross-session persistence."""
    try:
        store.put(("users", user_id, "trivia_discussed"), "all", trivia)
    except Exception:
        logger.warning("Memory Manager: failed to write discussed trivia")


def _build_session_summary(
    tracks: list,
    vibe: str,
    trivia_discussed: list[str],
    session_id: str,
    skipped_uris: list[str] | None = None,
) -> dict:
    """Build a JSON-serializable session summary for the LLM.

    Includes skip tracking data as a negative signal.
    """
    artists = list({t.artist_name for t in tracks})
    genres: list[str] = []
    for t in tracks:
        genres.extend(t.genres)

    # Count genre occurrences for dominant genres
    genre_counts: dict[str, int] = {}
    for g in genres:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    dominant = sorted(genre_counts, key=lambda x: genre_counts[x], reverse=True)[:3]

    return {
        "session_id": session_id,
        "tracks_played": len(tracks),
        "artists_played": artists[:10],  # Cap for token budget
        "dominant_genres": dominant,
        "session_vibe": vibe,
        "trivia_discussed": trivia_discussed[-10:],  # Most recent
        "skipped_track_uris": (skipped_uris or [])[-20:],  # Skip signal
        "skip_count": len(skipped_uris or []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def _update_profile_with_llm(
    llm: LLMProvider,
    existing: ListenerProfile,
    session_data: dict,
) -> ListenerProfile:
    """Use LLM to merge session data into the existing profile."""
    user_prompt = (
        f"EXISTING PROFILE:\n{json.dumps(existing, indent=2)}\n\n"
        f"SESSION DATA:\n{json.dumps(session_data, indent=2)}\n\n"
        "Output the UPDATED JSON profile:"
    )

    response = await llm.generate(
        system_prompt=_MEMORY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # Parse LLM JSON response — fall back to existing profile on parse failure
    try:
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        updated = json.loads(cleaned)
        
        # Merge valid keys into existing profile to prevent data loss
        merged = existing.copy()
        valid_keys = set(ListenerProfile.__annotations__.keys())
        for k, v in updated.items():
            if k in valid_keys:
                merged[k] = v  # type: ignore[literal-required]
                
        # Always refresh timestamp
        from datetime import datetime, timezone
        merged["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        return merged
    except Exception:
        logger.warning("Memory Manager: failed to parse LLM profile update, keeping existing")
        return existing


async def _create_initial_profile(
    spotify: SpotifyClient, user_id: str
) -> ListenerProfile:
    """Create cold-start profile from Spotify top tracks and artists.

    Spec §10.4: Cold Start Sequence.
    """
    try:
        top_artists = await spotify.get_top_artists(limit=20)
    except Exception:
        logger.warning("Memory Manager: cold start Spotify fetch failed")
        top_artists = []

    # Build genre affinity from top artists
    genre_counts: dict[str, int] = {}
    for artist in top_artists:
        for genre in artist.get("genres", []):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    total_genre_signals = sum(genre_counts.values()) or 1
    genre_affinity = {
        g: round(c / total_genre_signals, 3)
        for g, c in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    }

    artist_favorites = [a["name"] for a in top_artists[:5]]

    profile: ListenerProfile = {
        "user_id": user_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "genre_affinity": genre_affinity,
        "artist_favorites": artist_favorites,
        "artist_dislikes": [],
        "vibe_preference": "moderate",
        "discovery_openness": 0.5,  # Spec §5.8: "moderate default"
        "avg_session_length_tracks": 0,
        "total_sessions": 0,
        "recent_mood_trajectory": "",
        "skip_patterns": "",
        "trivia_discussed": [],
    }
    return profile
