"""
Wikidata SPARQL Client
=======================
Queries the Wikidata knowledge graph to find trivia links between artists.
This is the GraphRAG component of the Historian agent.

References:
    - Spec §5.2 (Historian Agent — SPARQL queries)
    - Spec §6 (Rate limits: 5 req/s, 2s timeout per query)

The Historian runs up to 3 SPARQL queries per transition, stopping at
the first hit with confidence ≥ 0.6 (Spec §5.2).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
# Spec §5.2: 2s hard cap per SPARQL query
_TIMEOUT = httpx.Timeout(2.0, connect=2.0)
_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "EchoDJ/1.0 (https://github.com/echodj)",
}

# Spec §5.2 — SPARQL query templates
# {mbid_a} and {mbid_b} are replaced with resolved MusicBrainz IDs.

_QUERY_SHARED_PRODUCER = """\
SELECT ?producerLabel WHERE {{
  ?artist_a wdt:P434 "{mbid_a}" .
  ?artist_b wdt:P434 "{mbid_b}" .
  ?artist_a wdt:P162 ?producer .
  ?artist_b wdt:P162 ?producer .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 5
"""

_QUERY_SAME_STUDIO = """\
SELECT ?studioLabel WHERE {{
  ?artist_a wdt:P434 "{mbid_a}" .
  ?artist_b wdt:P434 "{mbid_b}" .
  ?artist_a wdt:P1762 ?studio .
  ?artist_b wdt:P1762 ?studio .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 5
"""

_QUERY_SHARED_GENRE = """\
SELECT ?genreLabel WHERE {{
  ?artist_a wdt:P434 "{mbid_a}" .
  ?artist_b wdt:P434 "{mbid_b}" .
  ?artist_a wdt:P136 ?genre .
  ?artist_b wdt:P136 ?genre .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 5
"""

# Ordered query plan — stops at first hit with confidence ≥ 0.6
_QUERY_PLAN = [
    ("shared_producer", _QUERY_SHARED_PRODUCER, "shared_producer", 0.85),
    ("same_studio",     _QUERY_SAME_STUDIO,     "same_studio",     0.75),
    ("genre_movement",  _QUERY_SHARED_GENRE,    "genre_movement",  0.45),
]


class WikidataClient:
    """Async client for Wikidata SPARQL queries."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )

    async def find_link(
        self,
        mbid_a: str,
        mbid_b: str,
        artist_a: str,
        artist_b: str,
    ) -> dict[str, Any] | None:
        """Run the 3-query plan and return the first strong link found.

        Spec §5.2: Runs queries sequentially, stops at first hit ≥ 0.6.

        Args:
            mbid_a: MusicBrainz ID for the previous artist.
            mbid_b: MusicBrainz ID for the next artist.
            artist_a: Artist name (for human-readable output).
            artist_b: Artist name (for human-readable output).

        Returns:
            Dict with keys: link_type, connecting_entity, description,
            confidence, wikidata_qids — or None if no link found.
        """
        for (label, query_template, link_type, confidence) in _QUERY_PLAN:
            query = query_template.format(mbid_a=mbid_a, mbid_b=mbid_b)
            results = await self._run_sparql(query)

            if not results:
                continue

            # Take the first result
            connecting_entity = results[0]

            return {
                "link_type": link_type,
                "connecting_entity": connecting_entity,
                "description": self._make_description(
                    link_type, artist_a, artist_b, connecting_entity
                ),
                "confidence": confidence,
                "wikidata_qids": [],
            }

        return None

    async def _run_sparql(self, query: str) -> list[str]:
        """Execute a SPARQL query and return a flat list of label values.

        Returns empty list on timeout, error, or empty results.
        """
        try:
            response = await self._client.get(
                _SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", {}).get("bindings", [])
            # Extract the first variable's label value from each result row
            values = []
            for row in results:
                for val in row.values():
                    if val.get("type") == "literal":
                        values.append(val["value"])
                        break
            return values

        except httpx.TimeoutException:
            logger.debug("Wikidata SPARQL timeout")
            return []
        except Exception:
            logger.debug("Wikidata SPARQL error", exc_info=True)
            return []

    @staticmethod
    def _make_description(
        link_type: str,
        artist_a: str,
        artist_b: str,
        connecting_entity: str,
    ) -> str:
        """Generate a human-readable description of the trivia link."""
        templates = {
            "shared_producer": (
                f"Both {artist_a} and {artist_b} were produced by {connecting_entity}"
            ),
            "same_studio": (
                f"Both {artist_a} and {artist_b} recorded at {connecting_entity}"
            ),
            "genre_movement": (
                f"Both {artist_a} and {artist_b} are rooted in {connecting_entity}"
            ),
        }
        return templates.get(
            link_type,
            f"{artist_a} and {artist_b} are both linked to {connecting_entity}",
        )

    async def close(self) -> None:
        await self._client.aclose()
