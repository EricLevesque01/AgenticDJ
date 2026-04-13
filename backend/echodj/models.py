"""
EchoDJ Data Models
==================
Core data structures shared across all LangGraph nodes.
These are the exact contracts from the product specification.

References:
    - Spec §3.2 (Supporting Data Models)
    - Spec §5.8 (Listener Profile Schema)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


# ── Spotify Track ────────────────────────────────────────────────────────────
# Spec §3.2 — the canonical representation of a Spotify track across all nodes.

@dataclass(frozen=True, slots=True)
class SpotifyTrack:
    """A Spotify track with essential metadata.

    Used by: Observer (write), Historian (read), Discoverer (read),
    Curator (read/write), Scriptwriter (read), Memory Manager (read).
    """

    spotify_uri: str          # "spotify:track:4iV5W9uYEdYUVa79Axb7Rh"
    track_name: str
    artist_name: str
    album_name: str
    duration_ms: int
    album_art_url: str | None = None
    genres: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.spotify_uri.startswith("spotify:track:"):
            raise ValueError(
                f"Invalid Spotify URI: {self.spotify_uri!r}. "
                "Must start with 'spotify:track:'"
            )
        if self.duration_ms <= 0:
            raise ValueError(
                f"Invalid duration: {self.duration_ms}ms. Must be positive."
            )


# ── Trivia Link ──────────────────────────────────────────────────────────────
# Spec §3.2 — output of the Historian agent (GraphRAG).

@dataclass(frozen=True, slots=True)
class TriviaLink:
    """A knowledge-graph link between two artists, discovered via SPARQL.

    Used by: Historian (write), Curator (read), Scriptwriter (read),
    Memory Manager (read).
    """

    link_type: str            # "shared_producer", "same_studio", "genre_movement", "influence"
    entity_a: str             # Artist/track name (previous)
    entity_b: str             # Artist/track name (next)
    connecting_entity: str    # The shared element (producer name, studio, etc.)
    description: str          # Human-readable: "Both produced by Brian Eno"
    confidence: float         # 0.0 – 1.0
    wikidata_qids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )


# ── Candidate Track ──────────────────────────────────────────────────────────
# Spec §3.2 — tracks proposed by the Discoverer or Historian for Curator scoring.
# Graph RAG enrichment per Diamantini et al. (2026): candidates carry KG context.

@dataclass(frozen=True, slots=True)
class CandidateTrack:
    """A candidate track for the Curator to score and potentially queue.

    Used by: Discoverer (write), Curator (read).

    Graph RAG enrichment (Diamantini et al.):
        - kg_relationships: Known KG links to current/previous artists
        - kg_explanation: LLM-generated reason for this candidate's relevance
    """

    spotify_uri: str
    track_name: str
    artist_name: str
    source: str               # "lastfm", "listenbrainz", "historian", "spotify_top"
    relevance_score: float    # 0.0 – 1.0
    trivia_link: TriviaLink | None = None  # Only for Historian candidates
    kg_relationships: tuple[dict, ...] = ()   # KG context from MusicKnowledgeGraph
    kg_explanation: str = ""                   # LLM-generated relevance explanation

    def __post_init__(self) -> None:
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(
                f"Relevance score must be between 0.0 and 1.0, "
                f"got {self.relevance_score}"
            )


# ── User Intent ──────────────────────────────────────────────────────────────
# Spec §3.2 — parsed intent from PTT voice commands.

class UserIntent(Enum):
    """Classified user intent from push-to-talk voice input.

    Used by: Observer (write), Curator (read).
    See Spec §5.4.1 for the intent classifier prompt.
    """

    CHANGE_VIBE = "change_vibe"       # "Play something more upbeat"
    SKIP = "skip"                     # "Skip this" / "Next"
    MORE_INFO = "more_info"           # "Tell me more about this artist"
    SPECIFIC_REQUEST = "specific"     # "Play some Miles Davis"
    POSITIVE_FEEDBACK = "positive"    # "I love this" / "This is great"
    NEGATIVE_FEEDBACK = "negative"    # "Not feeling this"


# ── Listener Profile ─────────────────────────────────────────────────────────
# Spec §5.8 — persisted in LangGraph SqliteStore, updated by Memory Manager.

class ListenerProfile(TypedDict, total=False):
    """Long-term user preference profile stored in LangGraph SqliteStore.

    Updated by: Memory Manager (write).
    Read by: Curator (read), Memory Manager (read).
    """

    user_id: str
    updated_at: str                    # ISO 8601 timestamp
    genre_affinity: dict[str, float]   # genre -> 0.0–1.0 affinity score
    artist_favorites: list[str]
    artist_dislikes: list[str]
    vibe_preference: str               # "chill-to-moderate", "energetic", etc.
    discovery_openness: float          # 0.0 (only familiar) – 1.0 (very adventurous)
    avg_session_length_tracks: int
    total_sessions: int
    recent_mood_trajectory: str        # "shifting from jazz toward funk/soul"
    skip_patterns: str                 # "skips tracks over 7 minutes, dislikes country"
    skip_patterns_uris: list[str]      # Specific Spotify URIs skipped (negative signal)
    trivia_discussed: list[str]        # Cross-session trivia dedup


# ── Session Summary ──────────────────────────────────────────────────────────
# Append-only log stored per session in SqliteStore.

class SessionSummary(TypedDict, total=False):
    """Summary of a single listening session, stored for long-term reference."""

    session_id: str
    started_at: str
    ended_at: str
    tracks_played: int
    dominant_genres: list[str]
    vibe_trajectory: str
    user_feedback_summary: str
