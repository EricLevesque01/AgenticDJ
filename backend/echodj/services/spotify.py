"""
Spotify Web API Client
======================
Handles all interactions with the Spotify Web API.
The access token is received from the frontend via WebSocket after OAuth PKCE.

References:
    - Spec §5.1 (Observer — track metadata)
    - Spec §5.3 (Discoverer — user top tracks/artists, search)
    - Spec §5.4 (Curator — queue track)
    - Spec §6 (API Strategy — 3s timeout, standard rate limits)
    - Spec §11 (Authentication)

IMPORTANT — Deprecated endpoints that MUST NOT be used (Spec §1.2):
    - /v1/recommendations
    - /v1/audio-features
    - /v1/audio-analysis
    - /v1/artists/{id}/related-artists
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from echodj.models import SpotifyTrack

logger = logging.getLogger(__name__)

# Spotify API base URL
_BASE_URL = "https://api.spotify.com/v1"

# Timeout for all Spotify API requests (Spec §6: 3s)
_TIMEOUT = httpx.Timeout(10.0, connect=3.0)


class SpotifyAPIError(Exception):
    """Raised when a Spotify API call fails after retries."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"Spotify API error {status_code}: {message}")


class SpotifyClient:
    """Async client for the Spotify Web API.

    The access token is set externally (received from frontend via WebSocket)
    and can be refreshed mid-session via ``set_token()``.
    """

    def __init__(self, access_token: str = "") -> None:
        self._access_token = access_token
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=_TIMEOUT,
        )
        # Session-level cache: track search results (Spec §5.3)
        self._search_cache: dict[str, str] = {}  # "artist:track" -> spotify_uri

    @property
    def access_token(self) -> str:
        return self._access_token

    def set_token(self, token: str) -> None:
        """Update the access token (e.g., after a refresh).

        Spec §11.3: Token refresh is triggered at the 50-minute mark.
        The frontend sends the new token via WebSocket.
        """
        self._access_token = token
        logger.info("Spotify access token updated")

    def _headers(self) -> dict[str, str]:
        if not self._access_token:
            raise SpotifyAPIError(401, "No access token set")
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an authenticated request to the Spotify API.

        Raises SpotifyAPIError on non-2xx responses.
        """
        try:
            response = await self._client.request(
                method,
                path,
                headers=self._headers(),
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise SpotifyAPIError(408, f"Request timed out: {path}") from exc

        if response.status_code == 204:
            return {}

        if response.status_code == 401:
            raise SpotifyAPIError(401, "Access token expired or invalid")

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise SpotifyAPIError(
                429, f"Rate limited. Retry after {retry_after}s"
            )

        if not response.is_success:
            raise SpotifyAPIError(
                response.status_code,
                response.text[:200],
            )

        return response.json()

    # ── User Profile ─────────────────────────────────────────────────────

    async def get_current_user(self) -> dict[str, Any]:
        """Get the current user's profile.

        Used for: session initialization, user_id extraction.
        """
        return await self._request("GET", "/me")

    # ── Top Tracks / Artists ─────────────────────────────────────────────
    # Spec §5.3 (Discoverer — seed/fallback), §5.8 (Memory Manager — cold start)

    async def get_top_tracks(
        self,
        time_range: str = "medium_term",
        limit: int = 20,
    ) -> list[SpotifyTrack]:
        """Get the user's top tracks.

        Args:
            time_range: "short_term" (4 weeks), "medium_term" (6 months),
                        "long_term" (all time).
            limit: Max number of tracks (1-50).

        Returns:
            List of SpotifyTrack objects.
        """
        data = await self._request(
            "GET",
            "/me/top/tracks",
            params={"time_range": time_range, "limit": limit},
        )
        return [self._parse_track(item) for item in data.get("items", [])]

    async def get_top_artists(
        self,
        time_range: str = "medium_term",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get the user's top artists.

        Args:
            time_range: "short_term", "medium_term", "long_term".
            limit: Max number of artists (1-50).

        Returns:
            List of artist data dicts with name, genres, etc.
        """
        data = await self._request(
            "GET",
            "/me/top/artists",
            params={"time_range": time_range, "limit": limit},
        )
        return data.get("items", [])

    # ── Search ───────────────────────────────────────────────────────────
    # Spec §5.3: Resolve Last.fm/ListenBrainz results to Spotify URIs.

    async def search_track(
        self,
        track_name: str,
        artist_name: str,
    ) -> str | None:
        """Search for a track on Spotify and return its URI.

        Uses a session-level cache to avoid redundant searches.

        Args:
            track_name: Track title.
            artist_name: Artist name.

        Returns:
            Spotify URI (e.g., "spotify:track:...") or None if not found.
        """
        cache_key = f"{artist_name.lower()}:{track_name.lower()}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        query = f"track:{track_name} artist:{artist_name}"
        try:
            data = await self._request(
                "GET",
                "/search",
                params={"q": query, "type": "track", "limit": 1},
            )
        except SpotifyAPIError:
            logger.warning("Spotify search failed for %s", cache_key)
            return None

        items = data.get("tracks", {}).get("items", [])
        if not items:
            return None

        uri = items[0]["uri"]
        self._search_cache[cache_key] = uri
        return uri

    # ── Queue ────────────────────────────────────────────────────────────
    # Spec §5.4: Curator queues the selected track.

    async def queue_track(self, spotify_uri: str) -> None:
        """Add a track to the user's playback queue.

        Spec §5.4: POST /v1/me/player/queue?uri={...}
        Error handling: retry once on failure, then raise.
        """
        try:
            await self._request(
                "POST",
                "/me/player/queue",
                params={"uri": spotify_uri},
            )
            logger.info("Queued track: %s", spotify_uri)
        except SpotifyAPIError as exc:
            # Spec §5.4: retry once
            logger.warning("Queue failed, retrying: %s", exc)
            await self._request(
                "POST",
                "/me/player/queue",
                params={"uri": spotify_uri},
            )

    # ── Playback Control ─────────────────────────────────────────────────

    async def skip_to_next(self) -> None:
        """Skip to the next track in the queue."""
        await self._request("POST", "/me/player/next")

    # ── Track Parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_track(data: dict[str, Any]) -> SpotifyTrack:
        """Parse a Spotify API track object into a SpotifyTrack model.

        Handles missing fields gracefully.
        """
        album = data.get("album", {})
        artists = data.get("artists", [])

        # Get album art — prefer 300px size (index 1), fall back to first
        images = album.get("images", [])
        album_art_url = None
        if len(images) > 1:
            album_art_url = images[1].get("url")
        elif images:
            album_art_url = images[0].get("url")

        # Extract genres from artist data if available
        genres: list[str] = []
        if artists and "genres" in artists[0]:
            genres = artists[0]["genres"]

        return SpotifyTrack(
            spotify_uri=data.get("uri", "spotify:track:unknown"),
            track_name=data.get("name", "Unknown Track"),
            artist_name=artists[0]["name"] if artists else "Unknown Artist",
            album_name=album.get("name", "Unknown Album"),
            duration_ms=data.get("duration_ms", 0) or 1,  # Avoid 0
            album_art_url=album_art_url,
            genres=genres,
        )

    # ── Segment Cache Support ─────────────────────────────────────────────
    # Used by SegmentBuilder to pre-curate throwback/genre/artist playlists.

    async def search_tracks_by_genre(
        self, genre: str, limit: int = 12
    ) -> list[dict[str, str]]:
        """Search for tracks in a given genre.

        Returns list of {uri, track_name, artist_name} dicts.
        """
        try:
            data = await self._request(
                "GET",
                "/search",
                params={"q": f'genre:"{genre}"', "type": "track", "limit": limit},
            )
            return [
                {
                    "uri": t["uri"],
                    "track_name": t["name"],
                    "artist_name": t["artists"][0]["name"] if t.get("artists") else "",
                }
                for t in data.get("tracks", {}).get("items", [])
            ]
        except SpotifyAPIError:
            logger.warning("Spotify: genre search failed for %r", genre)
            return []

    async def get_artist_top_tracks(
        self, artist_name: str, limit: int = 8
    ) -> list[dict[str, str]]:
        """Get an artist's top tracks on Spotify.

        First resolves artist name → ID, then fetches top tracks.
        Returns list of {uri, track_name, artist_name} dicts.
        """
        try:
            # Resolve artist name to ID
            search_data = await self._request(
                "GET",
                "/search",
                params={"q": f'artist:"{artist_name}"', "type": "artist", "limit": 1},
            )
            artists = search_data.get("artists", {}).get("items", [])
            if not artists:
                return []

            artist_id = artists[0]["id"]
            tracks_data = await self._request(
                "GET",
                f"/artists/{artist_id}/top-tracks",
                params={"market": "US"},
            )
            return [
                {
                    "uri": t["uri"],
                    "track_name": t["name"],
                    "artist_name": artist_name,
                }
                for t in tracks_data.get("tracks", [])[:limit]
            ]
        except SpotifyAPIError:
            logger.warning("Spotify: artist top tracks failed for %r", artist_name)
            return []

    async def search_tracks_by_era(
        self, genre: str, decade_year: str, limit: int = 10
    ) -> list[dict[str, str]]:
        """Search for tracks from a specific decade and genre.

        Returns list of {uri, track_name, artist_name} dicts.
        """
        end_year = str(int(decade_year) + 9)
        try:
            data = await self._request(
                "GET",
                "/search",
                params={
                    "q": f'genre:"{genre}" year:{decade_year}-{end_year}',
                    "type": "track",
                    "limit": limit,
                },
            )
            return [
                {
                    "uri": t["uri"],
                    "track_name": t["name"],
                    "artist_name": t["artists"][0]["name"] if t.get("artists") else "",
                }
                for t in data.get("tracks", {}).get("items", [])
            ]
        except SpotifyAPIError:
            logger.warning("Spotify: era search failed for genre=%r year=%s", genre, decade_year)
            return []

    # ── Cleanup ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
