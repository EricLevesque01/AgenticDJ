"""
Segment Cache — Pre-curated DJ Segment Playlists
=================================================
Builds and stores themed playlists with script templates ahead of time,
based on the user's listening history. Used by the Curator as a low-latency
fallback when Graph RAG enrichment is still running, eliminating perceived
latency between tracks.

Segment Types:
    - Throwback:      Era-specific nostalgia segments ("90s deep cuts")
    - Genre deep-dive: Single-genre focused runs ("jazz hour")
    - Artist spotlight: Rotating around a favorite artist's catalog
    - Discovery bridge: Connects known favorites to new artists

Each segment entry stores:
    - A pre-selected list of Spotify track URIs (ordered)
    - A pre-written script template (filled at runtime with specifics)
    - An expiry timestamp (segments rebuilt every 24 hours)

The Curator checks the cache before running Graph RAG — if a valid segment
is queued, it uses the pre-built track + script instead of real-time LLM calls.
This brings perceived latency to near-zero for the common case.

References:
    - Spec §5.4 (Curator: pre-computation and latency)
    - Spec §10 (Memory Architecture)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Segments expire after 24 hours — rebuilt in background
_SEGMENT_TTL_HOURS = 24
# Max tracks stored per segment
_MAX_SEGMENT_TRACKS = 15

_DEFAULT_DB = Path(__file__).resolve().parent.parent.parent.parent / "data" / "echodj_music_kg.db"


# ── Script templates per segment type ──────────────────────────────────────

_SCRIPT_TEMPLATES: dict[str, list[str]] = {
    "throwback": [
        "Alright, we're stepping back in time for a minute. "
        "This one's a {era} gem — pure {genre}, a sound that defined a generation. "
        "Here's {track} by {artist}.",

        "Throwback mode. {artist} doing what they did best back in the {era}. "
        "This is {track} — if you know, you know.",

        "Let's rewind. {artist}, {era}-era {genre}. "
        "This is {track}. Turn it up.",
    ],
    "genre_deep_dive": [
        "We're going deep on {genre} for a little while. "
        "This next one is a cornerstone — {artist}, {track}.",

        "Staying in {genre} territory. "
        "Here's {artist} with {track}.",

        "Full {genre} immersion. {track} by {artist} — absolute essential.",
    ],
    "artist_spotlight": [
        "Let's spend some time with {artist}. "
        "This is {track} — one of their finest.",

        "{artist} spotlight. {track}. Listen to how they do this.",

        "More {artist}. {track}. "
        "If you haven't gone deep on their catalog yet, now's the time.",
    ],
    "discovery_bridge": [
        "Here's something you might not have heard before. "
        "{artist} — if you're into {listener_genre}, this is going to hit. "
        "It's called {track}.",

        "Discovery time. {artist} sounds like {listener_genre} but goes somewhere new. "
        "Check out {track}.",

        "New territory. {artist} bridges {listener_genre} and something a little different. "
        "This is {track}.",
    ],
}


class SegmentCache:
    """Pre-curated DJ segment playlist store.

    Stores themed playlists (throwbacks, deep-dives, spotlights) built from
    the user's listening history. The Curator checks this cache first for a
    near-zero latency path that bypasses real-time Graph RAG enrichment.

    Shares the `echodj_music_kg.db` database with MusicKnowledgeGraph —
    segments are part of the Source Profile Layer.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or _DEFAULT_DB)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            -- Pre-curated segment playlists
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                segment_type TEXT NOT NULL,
                label TEXT NOT NULL,
                track_uris TEXT NOT NULL,        -- JSON list of Spotify URIs (ordered)
                track_metadata TEXT NOT NULL,    -- JSON list of {track_name, artist_name}
                script_templates TEXT NOT NULL,  -- JSON list of template strings
                context TEXT DEFAULT '{}',       -- genre, era, artist context for templates
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                used_count INTEGER DEFAULT 0,
                UNIQUE(user_id, label)
            );

            CREATE INDEX IF NOT EXISTS idx_segments_user ON segments(user_id, segment_type);
        """)
        self._conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────

    def store_segment(
        self,
        user_id: str,
        segment_type: str,
        label: str,
        tracks: list[dict[str, str]],  # [{"uri": ..., "track_name": ..., "artist_name": ...}]
        context: dict[str, str] | None = None,
        ttl_hours: int = _SEGMENT_TTL_HOURS,
    ) -> None:
        """Store a pre-curated segment playlist.

        Args:
            user_id: User identifier
            segment_type: "throwback", "genre_deep_dive", "artist_spotlight", "discovery_bridge"
            label: Human-readable segment name e.g. "90s Hip-Hop Deep Cut"
            tracks: Ordered list of {uri, track_name, artist_name}
            context: Template fill vars — genre, era, artist, listener_genre
            ttl_hours: How long until this segment expires (default 24h)
        """
        uris = [t["uri"] for t in tracks]
        metadata = [{"track_name": t["track_name"], "artist_name": t["artist_name"]} for t in tracks]
        templates = _SCRIPT_TEMPLATES.get(segment_type, _SCRIPT_TEMPLATES["genre_deep_dive"])
        expires = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()

        self._conn.execute(
            """INSERT INTO segments
               (user_id, segment_type, label, track_uris, track_metadata, script_templates, context, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, label) DO UPDATE SET
                 track_uris = excluded.track_uris,
                 track_metadata = excluded.track_metadata,
                 script_templates = excluded.script_templates,
                 context = excluded.context,
                 expires_at = excluded.expires_at,
                 created_at = CURRENT_TIMESTAMP""",
            (user_id, segment_type, label,
             json.dumps(uris), json.dumps(metadata),
             json.dumps(templates), json.dumps(context or {}), expires),
        )
        self._conn.commit()
        logger.info("SegmentCache: stored segment label=%r type=%s tracks=%d", label, segment_type, len(tracks))

    # ── Read ──────────────────────────────────────────────────────────────

    def get_next_segment_track(
        self,
        user_id: str,
        active_segment_label: str | None,
        recent_uris: set[str],
    ) -> dict[str, Any] | None:
        """Get the next track from an active segment playlist.

        Returns the first unused track from the active segment that isn't in
        recent playback history. Returns None if segment is exhausted or expired.

        Returns:
            Dict with keys: uri, track_name, artist_name, script, segment_label
            or None if no valid segment track available.
        """
        if not active_segment_label:
            return None

        row = self._conn.execute(
            """SELECT * FROM segments
               WHERE user_id = ? AND label = ? AND expires_at > ?""",
            (user_id, active_segment_label, datetime.now(timezone.utc).isoformat()),
        ).fetchone()

        if not row:
            return None

        uris = json.loads(row["track_uris"])
        metadata = json.loads(row["track_metadata"])
        templates = json.loads(row["script_templates"])
        context = json.loads(row["context"])

        # Find first track not in recent playback
        for uri, meta in zip(uris, metadata):
            if uri not in recent_uris:
                # Pick a template (rotate through them)
                template_idx = row["used_count"] % len(templates)
                script_template = templates[template_idx]

                # Fill template with context
                script = _fill_template(script_template, meta, context)

                # Increment used count
                self._conn.execute(
                    "UPDATE segments SET used_count = used_count + 1 WHERE id = ?",
                    (row["id"],),
                )
                self._conn.commit()

                return {
                    "uri": uri,
                    "track_name": meta["track_name"],
                    "artist_name": meta["artist_name"],
                    "script": script,
                    "segment_label": row["label"],
                    "segment_type": row["segment_type"],
                }

        logger.info("SegmentCache: segment %r exhausted", active_segment_label)
        return None

    def get_available_segments(
        self, user_id: str, segment_type: str | None = None
    ) -> list[dict[str, Any]]:
        """List all valid (non-expired) segments for a user.

        Args:
            user_id: User identifier
            segment_type: Optional filter by type

        Returns:
            List of segment summary dicts
        """
        now = datetime.now(timezone.utc).isoformat()
        if segment_type:
            rows = self._conn.execute(
                "SELECT id, segment_type, label, used_count, expires_at FROM segments "
                "WHERE user_id = ? AND segment_type = ? AND expires_at > ? ORDER BY created_at DESC",
                (user_id, segment_type, now),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, segment_type, label, used_count, expires_at FROM segments "
                "WHERE user_id = ? AND expires_at > ? ORDER BY created_at DESC",
                (user_id, now),
            ).fetchall()

        return [
            {
                "label": r["label"],
                "segment_type": r["segment_type"],
                "used_count": r["used_count"],
                "expires_at": r["expires_at"],
            }
            for r in rows
        ]

    def has_valid_segment(self, user_id: str, label: str) -> bool:
        """Check if a specific segment label exists and is valid."""
        row = self._conn.execute(
            "SELECT id FROM segments WHERE user_id = ? AND label = ? AND expires_at > ?",
            (user_id, label, datetime.now(timezone.utc).isoformat()),
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self._conn.close()


# ── Segment Builder ───────────────────────────────────────────────────────

class SegmentBuilder:
    """Builds pre-curated segment playlists from a user's listening history.

    Called by the Memory Manager after each profile update. Runs asynchronously
    in the background — does not block the main playback pipeline.

    Segment types built:
        - throwback_<era>: Top tracks from a decade the user listened to heavily
        - genre_dive_<genre>: Deep cut tracks from a top genre
        - artist_spotlight_<name>: Top tracks from a user's favorite artist
    """

    def __init__(self, cache: SegmentCache) -> None:
        self._cache = cache

    async def build_from_profile(
        self,
        user_id: str,
        profile: dict,
        spotify_client: Any,
    ) -> int:
        """Build segments from a listener profile. Returns count of segments built.

        Called after Memory Manager updates the profile. This method builds
        segments in the background — failures are logged but not raised.
        """
        built = 0

        genre_affinity: dict[str, float] = profile.get("genre_affinity", {})
        artist_favorites: list[str] = profile.get("artist_favorites", [])
        skip_history: list[str] = profile.get("skip_patterns_uris", [])  # URIs to avoid

        # Build genre deep-dive segments for top 3 genres
        top_genres = sorted(genre_affinity.items(), key=lambda x: x[1], reverse=True)[:3]
        for genre, _ in top_genres:
            try:
                tracks = await _fetch_genre_tracks(spotify_client, genre, limit=12)
                # Filter out previously skipped tracks
                tracks = [t for t in tracks if t["uri"] not in skip_history]
                if tracks:
                    self._cache.store_segment(
                        user_id=user_id,
                        segment_type="genre_deep_dive",
                        label=f"genre_dive_{genre.replace(' ', '_')}",
                        tracks=tracks,
                        context={"genre": genre},
                    )
                    built += 1
            except Exception:
                logger.warning("SegmentBuilder: failed to build genre segment for %r", genre)

        # Build artist spotlight for top 2 favorites
        for artist in artist_favorites[:2]:
            try:
                tracks = await _fetch_artist_top_tracks(spotify_client, artist, limit=8)
                tracks = [t for t in tracks if t["uri"] not in skip_history]
                if tracks:
                    self._cache.store_segment(
                        user_id=user_id,
                        segment_type="artist_spotlight",
                        label=f"artist_spotlight_{artist.replace(' ', '_').lower()}",
                        tracks=tracks,
                        context={"artist": artist},
                    )
                    built += 1
            except Exception:
                logger.warning("SegmentBuilder: failed to build artist segment for %r", artist)

        # Build throwback segment if era-specific listening patterns exist
        throwback_eras = _detect_throwback_eras(profile)
        for era, genre in throwback_eras[:1]:  # One throwback segment at a time
            try:
                tracks = await _fetch_era_tracks(spotify_client, genre, era, limit=10)
                tracks = [t for t in tracks if t["uri"] not in skip_history]
                if tracks:
                    self._cache.store_segment(
                        user_id=user_id,
                        segment_type="throwback",
                        label=f"throwback_{era}_{genre.replace(' ', '_')}",
                        tracks=tracks,
                        context={"era": era, "genre": genre},
                    )
                    built += 1
            except Exception:
                logger.warning("SegmentBuilder: failed to build throwback segment for %r %r", era, genre)

        logger.info("SegmentBuilder: built %d segments for user=%s", built, user_id)
        return built


# ── Helpers ──────────────────────────────────────────────────────────────

def _fill_template(template: str, meta: dict[str, str], context: dict[str, str]) -> str:
    """Fill a script template with track metadata and segment context."""
    fill = {
        "track": meta.get("track_name", "this track"),
        "artist": meta.get("artist_name", "the artist"),
        **context,
    }
    try:
        return template.format(**fill)
    except KeyError:
        # If template has unfilled placeholders, return generically
        return f"Here's {meta.get('track_name', 'this track')} by {meta.get('artist_name', 'the artist')}."


async def _fetch_genre_tracks(
    spotify: Any, genre: str, limit: int = 12
) -> list[dict[str, str]]:
    """Fetch tracks for a genre via Spotify search."""
    results = await spotify.search_tracks_by_genre(genre, limit=limit)
    return results


async def _fetch_artist_top_tracks(
    spotify: Any, artist: str, limit: int = 8
) -> list[dict[str, str]]:
    """Fetch an artist's top tracks via Spotify."""
    results = await spotify.get_artist_top_tracks(artist, limit=limit)
    return results


async def _fetch_era_tracks(
    spotify: Any, genre: str, era: str, limit: int = 10
) -> list[dict[str, str]]:
    """Fetch era-specific tracks via Spotify search."""
    decade_map = {
        "70s": "1970", "1970s": "1970",
        "80s": "1980", "1980s": "1980",
        "90s": "1990", "1990s": "1990",
        "00s": "2000", "2000s": "2000",
        "10s": "2010", "2010s": "2010",
    }
    decade_year = decade_map.get(era.lower(), "1990")
    results = await spotify.search_tracks_by_era(genre, decade_year, limit=limit)
    return results


def _detect_throwback_eras(profile: dict) -> list[tuple[str, str]]:
    """Detect era signals from the listener profile.

    Looks for era keywords in skip_patterns, recent_mood_trajectory,
    and genre affinity. Returns (era, genre) pairs.
    """
    eras = []

    # Check recent mood trajectory for era hints
    trajectory = profile.get("recent_mood_trajectory", "").lower()
    skip_patterns = profile.get("skip_patterns", "").lower()
    genre_affinity = profile.get("genre_affinity", {})

    era_keywords = ["90s", "80s", "70s", "00s", "2000s", "1990s", "1980s"]
    top_genre = next(iter(sorted(genre_affinity, key=lambda g: genre_affinity[g], reverse=True)), "")

    for era in era_keywords:
        if era in trajectory or era in skip_patterns:
            eras.append((era, top_genre or "pop"))

    # Default: 90s throwback if no era detected but user has classic genres
    classic_genres = {"jazz", "soul", "funk", "r&b", "classic rock", "blues"}
    if not eras and any(g in classic_genres for g in genre_affinity):
        matching = [g for g in genre_affinity if g in classic_genres]
        if matching:
            eras.append(("90s", matching[0]))

    return eras
