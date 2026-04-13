"""
Shared test fixtures for EchoDJ backend tests.
"""

from __future__ import annotations

import pytest

from echodj.models import (
    CandidateTrack,
    ListenerProfile,
    SpotifyTrack,
    TriviaLink,
)


# ── SpotifyTrack Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_track() -> SpotifyTrack:
    """A typical SpotifyTrack for testing."""
    return SpotifyTrack(
        spotify_uri="spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
        track_name="Just Like Heaven",
        artist_name="The Cure",
        album_name="Kiss Me, Kiss Me, Kiss Me",
        duration_ms=211_000,
        album_art_url="https://i.scdn.co/image/test",
        genres=["new wave", "post-punk", "alternative rock"],
    )


@pytest.fixture
def sample_track_b() -> SpotifyTrack:
    """A second SpotifyTrack for transition tests."""
    return SpotifyTrack(
        spotify_uri="spotify:track:6dGnYIeXmHdcikdzNNDMm2",
        track_name="Wandering Star",
        artist_name="Portishead",
        album_name="Dummy",
        duration_ms=293_000,
        album_art_url="https://i.scdn.co/image/test2",
        genres=["trip hop", "electronic"],
    )


@pytest.fixture
def sample_track_minimal() -> SpotifyTrack:
    """A SpotifyTrack with minimal data (no album art, no genres)."""
    return SpotifyTrack(
        spotify_uri="spotify:track:0000000000000000000000",
        track_name="Unknown Song",
        artist_name="Unknown Artist",
        album_name="Unknown Album",
        duration_ms=180_000,
    )


# ── TriviaLink Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_trivia_link() -> TriviaLink:
    """A high-confidence shared-producer trivia link."""
    return TriviaLink(
        link_type="shared_producer",
        entity_a="Radiohead",
        entity_b="Portishead",
        connecting_entity="Bristol Studios",
        description="Both Radiohead and Portishead recorded at studios in Bristol in the mid-90s",
        confidence=0.85,
        wikidata_qids=["Q188451", "Q207272"],
    )


@pytest.fixture
def sample_trivia_link_weak() -> TriviaLink:
    """A low-confidence genre-based trivia link (fallback case)."""
    return TriviaLink(
        link_type="genre_movement",
        entity_a="The Cure",
        entity_b="Siouxsie and the Banshees",
        connecting_entity="post-punk",
        description="Both are icons of the post-punk movement",
        confidence=0.4,
    )


# ── CandidateTrack Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def sample_candidates() -> list[CandidateTrack]:
    """A list of candidate tracks from mixed sources."""
    return [
        CandidateTrack(
            spotify_uri="spotify:track:aaa",
            track_name="Glory Box",
            artist_name="Portishead",
            source="lastfm",
            relevance_score=0.9,
        ),
        CandidateTrack(
            spotify_uri="spotify:track:bbb",
            track_name="Teardrop",
            artist_name="Massive Attack",
            source="listenbrainz",
            relevance_score=0.85,
        ),
        CandidateTrack(
            spotify_uri="spotify:track:ccc",
            track_name="Roads",
            artist_name="Portishead",
            source="historian",
            relevance_score=0.75,
            trivia_link=TriviaLink(
                link_type="shared_producer",
                entity_a="The Cure",
                entity_b="Portishead",
                connecting_entity="Flood",
                description="Both produced by Flood",
                confidence=0.7,
            ),
        ),
        CandidateTrack(
            spotify_uri="spotify:track:ddd",
            track_name="Half Light",
            artist_name="Arcade Fire",
            source="spotify_top",
            relevance_score=0.6,
        ),
    ]


# ── ListenerProfile Fixtures ────────────────────────────────────────────────


@pytest.fixture
def sample_listener_profile() -> ListenerProfile:
    """A populated listener profile for testing."""
    return {
        "user_id": "test_user",
        "updated_at": "2026-04-12T21:00:00Z",
        "genre_affinity": {
            "trip hop": 0.85,
            "post-punk": 0.72,
            "indie rock": 0.65,
            "country": 0.15,
        },
        "artist_favorites": ["Portishead", "Radiohead", "Massive Attack"],
        "artist_dislikes": [],
        "vibe_preference": "chill-to-moderate",
        "discovery_openness": 0.7,
        "avg_session_length_tracks": 15,
        "total_sessions": 5,
        "recent_mood_trajectory": "shifting from trip-hop toward post-punk",
        "skip_patterns": "skips tracks over 7 minutes",
        "trivia_discussed": ["bristol_studios", "shared_producer_flood"],
    }


@pytest.fixture
def empty_listener_profile() -> ListenerProfile:
    """An empty listener profile (cold start)."""
    return {
        "user_id": "new_user",
        "updated_at": "2026-04-12T21:00:00Z",
        "genre_affinity": {},
        "artist_favorites": [],
        "artist_dislikes": [],
        "vibe_preference": "moderate",
        "discovery_openness": 0.5,
        "avg_session_length_tracks": 0,
        "total_sessions": 0,
        "recent_mood_trajectory": "",
        "skip_patterns": "",
        "trivia_discussed": [],
    }
