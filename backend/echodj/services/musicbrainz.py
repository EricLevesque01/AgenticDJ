"""
MusicBrainz REST API Client
============================
Resolves artist names to MusicBrainz IDs (MBIDs), which the Historian
uses to look up Wikidata entities via the P434 property.

References:
    - Spec §5.2 (Historian — Artist Name → MBID resolution)
    - Spec §6 (Rate limit: 1 req/s, 2s timeout)

Rate limiting: 1 request per second (hard requirement from MusicBrainz).
The client uses a session-level cache to avoid redundant lookups.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://musicbrainz.org/ws/2"
# Spec §5.2: 2s timeout for MBID resolution
_TIMEOUT = httpx.Timeout(2.0, connect=2.0)
# Spec §5.2: MusicBrainz requires a descriptive User-Agent
_USER_AGENT = "EchoDJ/1.0 ({email})"
# Spec §6: 1 request per second
_MIN_REQUEST_INTERVAL_S = 1.0


class MusicBrainzClient:
    """Async client for MusicBrainz artist MBID resolution.

    Maintains a session-level cache so each artist is only resolved once
    per session (Spec §5.2: "Cache resolved MBIDs for the session").
    """

    def __init__(self) -> None:
        user_agent = _USER_AGENT.format(
            email=settings.musicbrainz_contact_email
        )
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=_TIMEOUT,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )
        # Session-level MBID cache: artist_name_lower -> mbid
        self._cache: dict[str, str | None] = {}
        self._last_request_time: float = 0.0

    async def resolve_mbid(self, artist_name: str) -> str | None:
        """Resolve an artist name to a MusicBrainz ID.

        Uses session-level cache to avoid repeat lookups.
        Respects the 1 req/s rate limit.

        Args:
            artist_name: The artist name to resolve.

        Returns:
            MBID string (e.g., "a74b1b7f-71a5-4011-9441-d0b5e4122711")
            or None if not found or request fails.
        """
        cache_key = artist_name.lower().strip()

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Enforce 1 req/s rate limit
        await self._rate_limit()

        try:
            response = await self._client.get(
                "/artist/",
                params={
                    "query": f"artist:{artist_name}",
                    "fmt": "json",
                    "limit": 1,
                },
            )
            response.raise_for_status()
            data = response.json()

            artists = data.get("artists", [])
            mbid = artists[0]["id"] if artists else None

            self._cache[cache_key] = mbid
            if mbid:
                logger.debug("Resolved MBID: %r → %s", artist_name, mbid)
            else:
                logger.debug("No MBID found for: %r", artist_name)

            return mbid

        except httpx.TimeoutException:
            logger.debug("MusicBrainz timeout for: %r", artist_name)
            self._cache[cache_key] = None
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("MusicBrainz rate limited (429)")
            else:
                logger.debug(
                    "MusicBrainz error %s for: %r",
                    exc.response.status_code,
                    artist_name,
                )
            self._cache[cache_key] = None
            return None
        except Exception:
            logger.debug(
                "MusicBrainz resolve failed for: %r", artist_name, exc_info=True
            )
            self._cache[cache_key] = None
            return None

    async def _rate_limit(self) -> None:
        """Enforce the 1 req/s rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL_S:
            await asyncio.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)
        self._last_request_time = time.monotonic()

    def cache_size(self) -> int:
        """Return the number of cached MBID resolutions."""
        return len(self._cache)

    async def close(self) -> None:
        await self._client.aclose()
