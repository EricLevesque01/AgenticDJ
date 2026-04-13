"""
Tests for SegmentCache and SegmentBuilder
===========================================
Validates the pre-curated segment playlists and builder logic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
import sqlite3

import pytest

from echodj.services.segment_cache import SegmentCache, SegmentBuilder


@pytest.fixture
def empty_cache(tmp_path):
    """Provide a new SegmentCache with a temporary SQLite db."""
    db_path = tmp_path / "test_segments.db"
    cache = SegmentCache(db_path=db_path)
    yield cache
    cache.close()


class TestSegmentCache:
    """Tests for the SegmentCache database layer."""

    def test_store_and_check_segment(self, empty_cache: SegmentCache) -> None:
        """Should store a segment and check its validity."""
        empty_cache.store_segment(
            user_id="user1",
            segment_type="genre_deep_dive",
            label="test_dive",
            tracks=[{"uri": "spotify:track:1", "track_name": "T1", "artist_name": "A1"}],
        )
        assert empty_cache.has_valid_segment("user1", "test_dive")

    def test_get_next_segment_track(self, empty_cache: SegmentCache) -> None:
        """Should return the next unused track not in recent_uris."""
        empty_cache.store_segment(
            user_id="user1",
            segment_type="throwback",
            label="test_throwback",
            tracks=[
                {"uri": "spotify:track:1", "track_name": "T1", "artist_name": "A1"},
                {"uri": "spotify:track:2", "track_name": "T2", "artist_name": "A2"},
            ],
            context={"era": "90s", "genre": "pop"},
        )
        
        # Track 1 is in recent_uris, so it should skip to Track 2
        track = empty_cache.get_next_segment_track(
            user_id="user1",
            active_segment_label="test_throwback",
            recent_uris={"spotify:track:1"},
        )
        assert track is not None
        assert track["uri"] == "spotify:track:2"
        assert track["track_name"] == "T2"
        assert track["segment_label"] == "test_throwback"
        assert "90s" in track["script"]

    def test_get_next_segment_track_exhausted(self, empty_cache: SegmentCache) -> None:
        """Should return None if all tracks are in recent_uris."""
        empty_cache.store_segment(
            user_id="user1",
            segment_type="artist_spotlight",
            label="test_spotlight",
            tracks=[
                {"uri": "spotify:track:1", "track_name": "T1", "artist_name": "A1"},
            ],
        )
        track = empty_cache.get_next_segment_track(
            user_id="user1",
            active_segment_label="test_spotlight",
            recent_uris={"spotify:track:1"},
        )
        assert track is None

    def test_expired_segment_is_invalid(self, empty_cache: SegmentCache) -> None:
        """Segments past their expiration should not be returned."""
        # Store with negative TTL
        empty_cache.store_segment(
            user_id="user1",
            segment_type="genre_deep_dive",
            label="test_expired",
            tracks=[{"uri": "spotify:track:1", "track_name": "T1", "artist_name": "A1"}],
            ttl_hours=-1,
        )
        assert not empty_cache.has_valid_segment("user1", "test_expired")
        
class MockSpotifyClient:
    async def search_tracks_by_genre(self, genre, limit):
        return [{"uri": "spotify:track:genre1", "track_name": "G1", "artist_name": "GA"}]
        
    async def get_artist_top_tracks(self, artist, limit):
        return [{"uri": "spotify:track:artist1", "track_name": "A1", "artist_name": artist}]

    async def search_tracks_by_era(self, genre, era, limit):
        return [{"uri": "spotify:track:era1", "track_name": "E1", "artist_name": "EA"}]

@pytest.mark.asyncio
class TestSegmentBuilder:
    """Tests for the SegmentBuilder background task."""

    async def test_build_from_profile(self, empty_cache: SegmentCache) -> None:
        builder = SegmentBuilder(empty_cache)
        profile = {
            "genre_affinity": {"rock": 1.0},
            "artist_favorites": ["The Beatles"],
            "skip_patterns_uris": ["spotify:track:skipme"],
            "recent_mood_trajectory": "feeling 90s vibes",
        }
        
        count = await builder.build_from_profile("user1", profile, MockSpotifyClient())
        
        # Should build 1 genre dive, 1 artist spotlight, and 1 throwback (detect 90s)
        assert count == 3
        
        segments = empty_cache.get_available_segments("user1")
        assert len(segments) == 3
        labels = [s["label"] for s in segments]
        assert any(l.startswith("genre_dive") for l in labels)
        assert any(l.startswith("artist_spotlight") for l in labels)
        assert any(l.startswith("throwback") for l in labels)
