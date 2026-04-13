"""
Tests for EchoDJ Data Models
=============================
Validates all data models from Spec §3.2:
    - SpotifyTrack
    - TriviaLink
    - CandidateTrack
    - UserIntent
    - ListenerProfile
"""

from __future__ import annotations

import pytest

from echodj.models import (
    CandidateTrack,
    ListenerProfile,
    SpotifyTrack,
    TriviaLink,
    UserIntent,
)


# ── SpotifyTrack ─────────────────────────────────────────────────────────────


class TestSpotifyTrack:
    """Tests for the SpotifyTrack dataclass."""

    def test_valid_construction(self, sample_track: SpotifyTrack) -> None:
        """A properly formed track should construct without errors."""
        assert sample_track.track_name == "Just Like Heaven"
        assert sample_track.artist_name == "The Cure"
        assert sample_track.spotify_uri.startswith("spotify:track:")

    def test_minimal_construction(
        self, sample_track_minimal: SpotifyTrack
    ) -> None:
        """Track with no album art or genres should construct fine."""
        assert sample_track_minimal.album_art_url is None
        assert sample_track_minimal.genres == []

    def test_invalid_uri_raises(self) -> None:
        """URI not starting with 'spotify:track:' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid Spotify URI"):
            SpotifyTrack(
                spotify_uri="spotify:album:invalid",
                track_name="Test",
                artist_name="Test",
                album_name="Test",
                duration_ms=1000,
            )

    def test_zero_duration_raises(self) -> None:
        """Zero or negative duration should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid duration"):
            SpotifyTrack(
                spotify_uri="spotify:track:valid",
                track_name="Test",
                artist_name="Test",
                album_name="Test",
                duration_ms=0,
            )

    def test_negative_duration_raises(self) -> None:
        """Negative duration should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid duration"):
            SpotifyTrack(
                spotify_uri="spotify:track:valid",
                track_name="Test",
                artist_name="Test",
                album_name="Test",
                duration_ms=-100,
            )

    def test_frozen_immutability(self, sample_track: SpotifyTrack) -> None:
        """SpotifyTrack is frozen — attributes cannot be reassigned."""
        with pytest.raises(AttributeError):
            sample_track.track_name = "Modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two SpotifyTracks with the same data should be equal."""
        a = SpotifyTrack(
            spotify_uri="spotify:track:abc",
            track_name="Test",
            artist_name="Artist",
            album_name="Album",
            duration_ms=1000,
        )
        b = SpotifyTrack(
            spotify_uri="spotify:track:abc",
            track_name="Test",
            artist_name="Artist",
            album_name="Album",
            duration_ms=1000,
        )
        assert a == b

    def test_default_genres(self) -> None:
        """Genres should default to an empty list."""
        track = SpotifyTrack(
            spotify_uri="spotify:track:test",
            track_name="Test",
            artist_name="Artist",
            album_name="Album",
            duration_ms=1000,
        )
        assert track.genres == []


# ── TriviaLink ───────────────────────────────────────────────────────────────


class TestTriviaLink:
    """Tests for the TriviaLink dataclass."""

    def test_valid_construction(self, sample_trivia_link: TriviaLink) -> None:
        """A properly formed trivia link should construct without errors."""
        assert sample_trivia_link.link_type == "shared_producer"
        assert sample_trivia_link.confidence == 0.85
        assert len(sample_trivia_link.wikidata_qids) == 2

    def test_confidence_out_of_range_high(self) -> None:
        """Confidence > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            TriviaLink(
                link_type="test",
                entity_a="A",
                entity_b="B",
                connecting_entity="C",
                description="Test",
                confidence=1.5,
            )

    def test_confidence_out_of_range_low(self) -> None:
        """Confidence < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be"):
            TriviaLink(
                link_type="test",
                entity_a="A",
                entity_b="B",
                connecting_entity="C",
                description="Test",
                confidence=-0.1,
            )

    def test_boundary_confidence_zero(self) -> None:
        """Confidence of exactly 0.0 should be valid."""
        link = TriviaLink(
            link_type="test",
            entity_a="A",
            entity_b="B",
            connecting_entity="C",
            description="Test",
            confidence=0.0,
        )
        assert link.confidence == 0.0

    def test_boundary_confidence_one(self) -> None:
        """Confidence of exactly 1.0 should be valid."""
        link = TriviaLink(
            link_type="test",
            entity_a="A",
            entity_b="B",
            connecting_entity="C",
            description="Test",
            confidence=1.0,
        )
        assert link.confidence == 1.0

    def test_default_empty_qids(self) -> None:
        """wikidata_qids should default to empty list."""
        link = TriviaLink(
            link_type="test",
            entity_a="A",
            entity_b="B",
            connecting_entity="C",
            description="Test",
            confidence=0.5,
        )
        assert link.wikidata_qids == []


# ── CandidateTrack ───────────────────────────────────────────────────────────


class TestCandidateTrack:
    """Tests for the CandidateTrack dataclass."""

    def test_valid_construction(self) -> None:
        """A properly formed candidate should construct without errors."""
        candidate = CandidateTrack(
            spotify_uri="spotify:track:test",
            track_name="Test",
            artist_name="Artist",
            source="lastfm",
            relevance_score=0.85,
        )
        assert candidate.source == "lastfm"
        assert candidate.trivia_link is None

    def test_with_trivia_link(
        self, sample_trivia_link: TriviaLink
    ) -> None:
        """A historian candidate should carry a trivia link."""
        candidate = CandidateTrack(
            spotify_uri="spotify:track:test",
            track_name="Test",
            artist_name="Artist",
            source="historian",
            relevance_score=0.75,
            trivia_link=sample_trivia_link,
        )
        assert candidate.trivia_link is not None
        assert candidate.trivia_link.link_type == "shared_producer"

    def test_invalid_relevance_score(self) -> None:
        """Relevance score out of range should raise ValueError."""
        with pytest.raises(ValueError, match="Relevance score must be"):
            CandidateTrack(
                spotify_uri="spotify:track:test",
                track_name="Test",
                artist_name="Artist",
                source="lastfm",
                relevance_score=1.5,
            )

    def test_valid_sources(self) -> None:
        """All spec-defined sources should be valid."""
        for source in ("lastfm", "listenbrainz", "historian", "spotify_top"):
            candidate = CandidateTrack(
                spotify_uri="spotify:track:test",
                track_name="Test",
                artist_name="Artist",
                source=source,
                relevance_score=0.5,
            )
            assert candidate.source == source


# ── UserIntent ───────────────────────────────────────────────────────────────


class TestUserIntent:
    """Tests for the UserIntent enum."""

    def test_all_intents_defined(self) -> None:
        """All 6 intents from Spec §3.2 should be defined."""
        expected = {
            "change_vibe",
            "skip",
            "more_info",
            "specific",
            "positive",
            "negative",
        }
        actual = {intent.value for intent in UserIntent}
        assert actual == expected

    def test_intent_count(self) -> None:
        """Exactly 6 intents should exist (Spec §3.2)."""
        assert len(UserIntent) == 6

    def test_intent_from_value(self) -> None:
        """Intents should be constructable from string values."""
        assert UserIntent("skip") == UserIntent.SKIP
        assert UserIntent("change_vibe") == UserIntent.CHANGE_VIBE


# ── ListenerProfile ──────────────────────────────────────────────────────────


class TestListenerProfile:
    """Tests for the ListenerProfile TypedDict."""

    def test_populated_profile(
        self, sample_listener_profile: ListenerProfile
    ) -> None:
        """A fully populated profile should have all expected fields."""
        assert sample_listener_profile["user_id"] == "test_user"
        assert "trip hop" in sample_listener_profile["genre_affinity"]
        assert len(sample_listener_profile["artist_favorites"]) == 3
        assert sample_listener_profile["discovery_openness"] == 0.7

    def test_empty_profile(
        self, empty_listener_profile: ListenerProfile
    ) -> None:
        """A cold-start profile should have sensible defaults."""
        assert empty_listener_profile["total_sessions"] == 0
        assert empty_listener_profile["discovery_openness"] == 0.5
        assert empty_listener_profile["genre_affinity"] == {}

    def test_profile_is_dict(
        self, sample_listener_profile: ListenerProfile
    ) -> None:
        """ListenerProfile should be a regular dict at runtime."""
        assert isinstance(sample_listener_profile, dict)
