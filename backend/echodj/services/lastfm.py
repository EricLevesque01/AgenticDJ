"""
Last.fm API Client
===================
Fetches similar artists and tracks for the Discoverer agent.
Uses collaborative filtering to find taste-matching candidates.

References:
    - Spec §5.3 (Discoverer — Data Source 1: Last.fm)
    - Spec §6 (Rate limit: ~5 req/s, 3s timeout)

Endpoints used:
    - artist.getSimilar → similar artists to current track's artist
    - track.getSimilar → similar tracks to current track
    - artist.getTopTracks → top track for each similar artist
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
# Spec §6: 3s timeout for Last.fm
_TIMEOUT = httpx.Timeout(3.0, connect=3.0)


class LastFMError(Exception):
    """Raised when Last.fm API is unavailable."""


class LastFMClient:
    """Async client for the Last.fm Web Services API."""

    def __init__(self) -> None:
        if not settings.lastfm_api_key:
            raise ValueError(
                "LASTFM_API_KEY is not set. "
                "Add it to your .env file."
            )
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=_TIMEOUT,
        )
        self._api_key = settings.lastfm_api_key

    async def _get(self, method: str, **params: str) -> dict[str, Any]:
        """Make a Last.fm API call and return the parsed JSON."""
        try:
            response = await self._client.get(
                "",
                params={
                    "method": method,
                    "api_key": self._api_key,
                    "format": "json",
                    **params,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Last.fm wraps errors in a JSON error field
            if "error" in data:
                raise LastFMError(
                    f"Last.fm error {data['error']}: {data.get('message', '')}"
                )

            return data

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise LastFMError(f"Last.fm unavailable: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise LastFMError(
                f"Last.fm HTTP {exc.response.status_code}"
            ) from exc

    async def get_similar_artists(
        self, artist_name: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fetch artists similar to the given artist.

        Spec §5.3: GET ?method=artist.getSimilar&artist=...&limit=10

        Returns:
            List of dicts with keys: name, match (similarity 0-1).
        """
        try:
            data = await self._get(
                "artist.getSimilar",
                artist=artist_name,
                limit=str(limit),
            )
            artists = data.get("similarartists", {}).get("artist", [])
            return [
                {
                    "name": a.get("name", ""),
                    "match": float(a.get("match", 0)),
                }
                for a in artists
                if a.get("name")
            ]
        except LastFMError as exc:
            logger.warning("get_similar_artists failed for %r: %s", artist_name, exc)
            return []

    async def get_similar_tracks(
        self,
        artist_name: str,
        track_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch tracks similar to the given track.

        Spec §5.3: GET ?method=track.getSimilar&artist=...&track=...

        Returns:
            List of dicts with keys: name, artist, match.
        """
        try:
            data = await self._get(
                "track.getSimilar",
                artist=artist_name,
                track=track_name,
                limit=str(limit),
            )
            tracks = data.get("similartracks", {}).get("track", [])
            return [
                {
                    "name": t.get("name", ""),
                    "artist": t.get("artist", {}).get("name", ""),
                    "match": float(t.get("match", 0)),
                }
                for t in tracks
                if t.get("name") and t.get("artist", {}).get("name")
            ]
        except LastFMError as exc:
            logger.warning(
                "get_similar_tracks failed for %r/%r: %s",
                artist_name, track_name, exc,
            )
            return []

    async def get_top_tracks(
        self, artist_name: str, limit: int = 1
    ) -> list[dict[str, Any]]:
        """Get the top tracks for an artist.

        Spec §5.3: Used to pick one track per similar artist.

        Returns:
            List of dicts with keys: name, artist.
        """
        try:
            data = await self._get(
                "artist.getTopTracks",
                artist=artist_name,
                limit=str(limit),
            )
            tracks = data.get("toptracks", {}).get("track", [])
            return [
                {
                    "name": t.get("name", ""),
                    "artist": artist_name,
                }
                for t in tracks
                if t.get("name")
            ]
        except LastFMError as exc:
            logger.warning("get_top_tracks failed for %r: %s", artist_name, exc)
            return []

    async def close(self) -> None:
        await self._client.aclose()
