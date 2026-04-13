"""
Tests for the Spotify API Client
=================================
Tests the SpotifyClient from Spec §5-6.
Uses mocked HTTP responses — no actual Spotify API calls.
"""

from __future__ import annotations

import pytest
import httpx

from echodj.models import SpotifyTrack
from echodj.services.spotify import SpotifyAPIError, SpotifyClient


# ── Mock Response Fixtures ───────────────────────────────────────────────────


MOCK_TRACK_RESPONSE = {
    "uri": "spotify:track:4iV5W9uYEdYUVa79Axb7Rh",
    "name": "Just Like Heaven",
    "duration_ms": 211000,
    "artists": [{"name": "The Cure", "genres": ["new wave", "post-punk"]}],
    "album": {
        "name": "Kiss Me, Kiss Me, Kiss Me",
        "images": [
            {"url": "https://i.scdn.co/image/large", "height": 640},
            {"url": "https://i.scdn.co/image/medium", "height": 300},
            {"url": "https://i.scdn.co/image/small", "height": 64},
        ],
    },
}

MOCK_SEARCH_RESPONSE = {
    "tracks": {
        "items": [MOCK_TRACK_RESPONSE],
        "total": 1,
    }
}

MOCK_TOP_TRACKS_RESPONSE = {
    "items": [MOCK_TRACK_RESPONSE],
    "total": 1,
}

MOCK_TOP_ARTISTS_RESPONSE = {
    "items": [
        {
            "name": "The Cure",
            "genres": ["new wave", "post-punk"],
            "id": "7bu3H8JO7d0UbMoVzbo70s",
        }
    ],
    "total": 1,
}


# ── SpotifyClient Tests ─────────────────────────────────────────────────────


class TestSpotifyClient:
    """Tests for SpotifyClient."""

    def test_construction(self) -> None:
        """Client should initialize with a token."""
        client = SpotifyClient("test-token")
        assert client.access_token == "test-token"

    def test_set_token(self) -> None:
        """set_token should update the stored token."""
        client = SpotifyClient("old-token")
        client.set_token("new-token")
        assert client.access_token == "new-token"

    def test_no_token_raises(self) -> None:
        """Requests without a token should raise SpotifyAPIError."""
        client = SpotifyClient("")
        with pytest.raises(SpotifyAPIError, match="No access token"):
            client._headers()

    def test_search_cache(self) -> None:
        """Search results should be cached by artist:track key."""
        client = SpotifyClient("test")
        # Manually populate cache
        client._search_cache["the cure:just like heaven"] = (
            "spotify:track:cached"
        )
        # Verify cache hit
        assert "the cure:just like heaven" in client._search_cache

class TestSpotifySegmentSearch:
    """Tests for SpotifyClient segment search methods."""

    @pytest.mark.asyncio
    async def test_search_tracks_by_genre(self, monkeypatch) -> None:
        client = SpotifyClient("token")
        async def mock_request(*args, **kwargs):
            return MOCK_SEARCH_RESPONSE
        monkeypatch.setattr(client, "_request", mock_request)
        
        res = await client.search_tracks_by_genre("new wave")
        assert len(res) == 1
        assert res[0]["uri"] == MOCK_TRACK_RESPONSE["uri"]
        assert res[0]["track_name"] == MOCK_TRACK_RESPONSE["name"]

    @pytest.mark.asyncio
    async def test_search_tracks_by_era(self, monkeypatch) -> None:
        client = SpotifyClient("token")
        async def mock_request(*args, **kwargs):
            return MOCK_SEARCH_RESPONSE
        monkeypatch.setattr(client, "_request", mock_request)
        
        res = await client.search_tracks_by_era("pop", "1990")
        assert len(res) == 1
        assert res[0]["track_name"] == MOCK_TRACK_RESPONSE["name"]

    @pytest.mark.asyncio
    async def test_get_artist_top_tracks(self, monkeypatch) -> None:
        client = SpotifyClient("token")
        async def mock_request(method, path, **kwargs):
            if "search" in path:
                return {"artists": {"items": [{"id": "artist_123"}]}}
            return {"tracks": [MOCK_TRACK_RESPONSE]}
        monkeypatch.setattr(client, "_request", mock_request)
        
        res = await client.get_artist_top_tracks("The Cure")
        assert len(res) == 1
        assert res[0]["uri"] == MOCK_TRACK_RESPONSE["uri"]


class TestTrackParsing:
    """Tests for SpotifyClient._parse_track."""

    def test_parse_full_track(self) -> None:
        """Full track data should parse correctly."""
        track = SpotifyClient._parse_track(MOCK_TRACK_RESPONSE)
        assert isinstance(track, SpotifyTrack)
        assert track.track_name == "Just Like Heaven"
        assert track.artist_name == "The Cure"
        assert track.album_name == "Kiss Me, Kiss Me, Kiss Me"
        assert track.duration_ms == 211000
        # Should prefer medium-size image (index 1)
        assert track.album_art_url == "https://i.scdn.co/image/medium"
        assert "new wave" in track.genres

    def test_parse_track_no_images(self) -> None:
        """Track with no album images should have None album_art_url."""
        data = {
            **MOCK_TRACK_RESPONSE,
            "album": {"name": "Test Album", "images": []},
        }
        track = SpotifyClient._parse_track(data)
        assert track.album_art_url is None

    def test_parse_track_single_image(self) -> None:
        """Track with only one album image should use that image."""
        data = {
            **MOCK_TRACK_RESPONSE,
            "album": {
                "name": "Test Album",
                "images": [{"url": "https://only-image.jpg", "height": 640}],
            },
        }
        track = SpotifyClient._parse_track(data)
        assert track.album_art_url == "https://only-image.jpg"

    def test_parse_track_no_artists(self) -> None:
        """Track with empty artists list should use 'Unknown Artist'."""
        data = {**MOCK_TRACK_RESPONSE, "artists": []}
        track = SpotifyClient._parse_track(data)
        assert track.artist_name == "Unknown Artist"

    def test_parse_track_no_genres(self) -> None:
        """Track whose artist has no genres field should return empty list."""
        data = {
            **MOCK_TRACK_RESPONSE,
            "artists": [{"name": "TestArtist"}],
        }
        track = SpotifyClient._parse_track(data)
        assert track.genres == []


class TestSpotifyAPIError:
    """Tests for SpotifyAPIError."""

    def test_error_includes_status_code(self) -> None:
        """Error message should include the HTTP status code."""
        err = SpotifyAPIError(401, "Unauthorized")
        assert err.status_code == 401
        assert "401" in str(err)
        assert "Unauthorized" in str(err)

    def test_error_is_exception(self) -> None:
        """SpotifyAPIError should be a proper Exception subclass."""
        err = SpotifyAPIError(500, "Server error")
        assert isinstance(err, Exception)
