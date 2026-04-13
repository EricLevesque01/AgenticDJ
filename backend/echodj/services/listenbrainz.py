"""
ListenBrainz API Client
========================
Fetches collaborative filtering recommendations from ListenBrainz.
Optional data source — gracefully skipped if user has no account.

References:
    - Spec §5.3 (Discoverer — Data Source 2: ListenBrainz)
    - Spec §6 (3s timeout, skip gracefully if unavailable)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from echodj.config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.listenbrainz.org"
_TIMEOUT = httpx.Timeout(3.0, connect=3.0)


class ListenBrainzClient:
    """Async client for the ListenBrainz collaborative filtering API.

    Spec §5.3: If the user doesn't have a ListenBrainz account,
    this source is skipped gracefully.
    """

    def __init__(self) -> None:
        self._token = settings.listenbrainz_user_token
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=_TIMEOUT,
        )

    def is_configured(self) -> bool:
        """Return True if a ListenBrainz token is configured."""
        return bool(self._token)

    async def get_recommendations(
        self, username: str | None = None, count: int = 25
    ) -> list[dict[str, Any]]:
        """Fetch collaborative filtering recommendations.

        Spec §5.3: GET /1/cf/recommendation/user/{username}/recording

        Args:
            username: ListenBrainz username. Falls back to extracting
                      from the token metadata if not provided.
            count: Maximum number of recommendations.

        Returns:
            List of dicts with keys: recording_mbid, artist_mbid_list,
            release_mbid, track_metadata (name, artist_name).
            Empty list if not configured or request fails.
        """
        if not self._token:
            logger.debug("ListenBrainz token not configured, skipping")
            return []

        # Get username from token if not provided
        if not username:
            username = await self._get_username()
            if not username:
                return []

        try:
            response = await self._client.get(
                f"/1/cf/recommendation/user/{username}/recording",
                headers={"Authorization": f"Token {self._token}"},
                params={"count": count},
            )

            if response.status_code == 404:
                logger.info("No ListenBrainz CF recommendations for %s", username)
                return []

            response.raise_for_status()
            data = response.json()

            payload = data.get("payload", {})
            recordings = payload.get("mbids", []) or []

            return [
                {
                    "recording_mbid": r.get("recording_mbid"),
                    "artist_mbid": r.get("artist_mbid_list", [None])[0],
                    "track_name": r.get("track_metadata", {}).get("track_name", ""),
                    "artist_name": r.get("track_metadata", {}).get("artist_name", ""),
                }
                for r in recordings
                if r.get("recording_mbid")
            ]

        except httpx.TimeoutException:
            logger.warning("ListenBrainz recommendations timed out")
            return []
        except Exception:
            logger.warning("ListenBrainz recommendations failed", exc_info=True)
            return []

    async def _get_username(self) -> str | None:
        """Extract the ListenBrainz username from the token."""
        try:
            response = await self._client.get(
                "/1/validate-token",
                headers={"Authorization": f"Token {self._token}"},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("user_name")
        except Exception:
            logger.warning("ListenBrainz token validation failed", exc_info=True)
            return None

    async def close(self) -> None:
        await self._client.aclose()
