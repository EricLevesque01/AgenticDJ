"""
Historian Agent — Graph RAG Node
=================================
Queries the Music Knowledge Graph (local-first) and Wikidata SPARQL
to find trivia links between consecutive artists.

Graph RAG architecture per Diamantini et al. (2026):
    - Local KG lookup first (accumulated knowledge)
    - SPARQL fallback for uncached relationships
    - Results stored back to KG for future sessions
    - Returns ALL known relationships as trivia_context (not just best)

References:
    - Paper §3 (Two-layer KG design)
    - Paper §4.1 (KG-enriched context extraction)
    - Spec §5.2 (Historian Agent)
    - Fallback cascade: Local KG → SPARQL → genre-based → None
    - Latency target: < 2s total
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Awaitable

from echodj.models import CandidateTrack, TriviaLink
from echodj.services.lastfm import LastFMClient
from echodj.services.musicbrainz import MusicBrainzClient
from echodj.services.music_knowledge_graph import MusicKnowledgeGraph
from echodj.services.spotify import SpotifyClient
from echodj.services.wikidata import WikidataClient
from echodj.state import DJState

logger = logging.getLogger(__name__)

# Spec §5.2: Confidence threshold for a "hit"
_CONFIDENCE_THRESHOLD = 0.6
# Spec §5.2: Genre-based fallback confidence
_GENRE_FALLBACK_CONFIDENCE = 0.4


async def historian_node(
    state: DJState,
    musicbrainz: MusicBrainzClient,
    wikidata: WikidataClient,
    music_kg: MusicKnowledgeGraph | None = None,
    lastfm: LastFMClient | None = None,
    spotify: SpotifyClient | None = None,
    on_status: Callable[[str], Awaitable[None]] | None = None,
) -> dict:
    """LangGraph node: find trivia links between the last two artists.

    Graph RAG flow (per Diamantini et al.):
        1. Check local Music KG first (cache hit = skip external APIs)
        2. If not cached, run SPARQL queries
        3. Store any new results back to the KG
        4. Return ALL known relationships as trivia_context

    Input reads (Spec §3.3):
        - current_track: The track that just played (or is playing)
        - previous_tracks: History for picking artist_a
        - discussed_trivia: Prevents repeating already-used trivia

    Output writes (Spec §3.3):
        - trivia_link: Best TriviaLink object or None
        - trivia_confidence: Float 0.0–1.0
        - trivia_context: ALL known TriviaLinks between the artists (Graph RAG)
        - taste_candidates: CandidateTrack list from trivia-linked artist (GAP 6)
    """
    current = state.get("current_track")
    previous = state.get("previous_tracks", [])
    discussed = state.get("discussed_trivia", [])

    empty_result = {
        "trivia_link": None,
        "trivia_confidence": 0.0,
        "trivia_context": [],
        "taste_candidates": [],
    }

    # Need at least one previous track to compare
    if not current or not previous:
        logger.debug("Historian: insufficient track history, skipping")
        return empty_result

    artist_a_name = previous[-1].artist_name  # Most recent previous
    artist_b_name = current.artist_name

    # Same artist — skip trivia
    if artist_a_name.lower() == artist_b_name.lower():
        logger.debug("Historian: same artist on both tracks, skipping")
        return empty_result

    logger.info(
        "Historian: looking for link between %r and %r",
        artist_a_name, artist_b_name,
    )
    if on_status:
        await on_status(f"Searching for {artist_a_name} ↔ {artist_b_name}...")

    # ── Step 1: Check local Music KG first ────────────────────────────────
    # Paper §4.1: Local KG lookup before external APIs
    kg_relationships = []
    if music_kg:
        kg_relationships = music_kg.get_relationships(artist_a_name, artist_b_name)
        if kg_relationships:
            logger.info(
                "Historian: found %d cached KG relationships",
                len(kg_relationships),
            )

    # ── Step 2: Resolve MBIDs (use KG cache when available) ──────────────
    if music_kg:
        mbid_a = music_kg.get_cached_mbid(artist_a_name)
        mbid_b = music_kg.get_cached_mbid(artist_b_name)
    else:
        mbid_a, mbid_b = None, None

    if not mbid_a or not mbid_b:
        mbid_a_new, mbid_b_new = await _resolve_both_mbids(
            musicbrainz, artist_a_name, artist_b_name
        )
        mbid_a = mbid_a or mbid_a_new
        mbid_b = mbid_b or mbid_b_new

        # Store resolved MBIDs back to KG
        if music_kg:
            if mbid_a:
                music_kg.upsert_artist(artist_a_name, mbid=mbid_a)
            if mbid_b:
                music_kg.upsert_artist(artist_b_name, mbid=mbid_b)

    # ── Step 3: Run SPARQL queries (if no strong KG hit) ─────────────────
    # Only query SPARQL if we don't already have a strong link locally
    best_kg_confidence = max((r["confidence"] for r in kg_relationships), default=0.0)
    trivia_data = None

    if best_kg_confidence < _CONFIDENCE_THRESHOLD and mbid_a and mbid_b:
        if on_status:
            await on_status(f"Querying Wikidata SPARQL...")
        trivia_data = await wikidata.find_link(
            mbid_a, mbid_b, artist_a_name, artist_b_name
        )

        # Store SPARQL results back to the KG (accumulate knowledge)
        if trivia_data and music_kg:
            music_kg.add_relationship(
                artist_a=artist_a_name,
                artist_b=artist_b_name,
                rel_type=trivia_data["link_type"],
                connecting_entity=trivia_data["connecting_entity"],
                confidence=trivia_data["confidence"],
                description=trivia_data["description"],
                source="sparql",
                wikidata_qids=trivia_data.get("wikidata_qids", []),
            )

    # ── Step 4: Build trivia_context (all known links) ───────────────────
    # Paper §4.4: Enrich with ALL KG context, not just the best link
    if music_kg:
        # Refresh after potential SPARQL write
        kg_relationships = music_kg.get_relationships(artist_a_name, artist_b_name)

    trivia_context = [
        TriviaLink(
            link_type=r["rel_type"],
            entity_a=r.get("artist_a", artist_a_name),
            entity_b=r.get("artist_b", artist_b_name),
            connecting_entity=r["connecting_entity"],
            description=r["description"],
            confidence=r["confidence"],
            wikidata_qids=r.get("wikidata_qids", []),
        )
        for r in kg_relationships
    ]

    # ── Step 5: Select best link + apply repetition guard ────────────────
    best_link = None
    best_confidence = 0.0

    # Check SPARQL result first (it's the freshest)
    if trivia_data and trivia_data["confidence"] >= _CONFIDENCE_THRESHOLD:
        if trivia_data["description"] not in discussed:
            best_link = TriviaLink(
                link_type=trivia_data["link_type"],
                entity_a=artist_a_name,
                entity_b=artist_b_name,
                connecting_entity=trivia_data["connecting_entity"],
                description=trivia_data["description"],
                confidence=trivia_data["confidence"],
                wikidata_qids=trivia_data.get("wikidata_qids", []),
            )
            best_confidence = trivia_data["confidence"]

    # Fall back to cached KG links if SPARQL didn't produce a fresh hit
    if not best_link:
        for r in kg_relationships:
            if r["confidence"] >= _CONFIDENCE_THRESHOLD and r["description"] not in discussed:
                best_link = TriviaLink(
                    link_type=r["rel_type"],
                    entity_a=r.get("artist_a", artist_a_name),
                    entity_b=r.get("artist_b", artist_b_name),
                    connecting_entity=r["connecting_entity"],
                    description=r["description"],
                    confidence=r["confidence"],
                    wikidata_qids=r.get("wikidata_qids", []),
                )
                best_confidence = r["confidence"]
                break

    # ── Step 6: Build historian candidates + genre fallback ──────────────
    if best_link:
        logger.info(
            "Historian: found link type=%s confidence=%.2f (SPARQL=%s, KG_cached=%d)",
            best_link.link_type, best_link.confidence,
            trivia_data is not None, len(kg_relationships),
        )
        historian_candidates = await _build_historian_candidates(
            best_link, artist_b_name, lastfm, spotify
        )
        return {
            "trivia_link": best_link,
            "trivia_confidence": best_confidence,
            "trivia_context": trivia_context,
            "taste_candidates": historian_candidates,
        }

    # Genre fallback (Spec §5.2)
    genre_link = _try_genre_fallback(
        artist_a_name, artist_b_name,
        previous[-1].genres, current.genres,
        discussed,
    )
    if genre_link:
        logger.info("Historian: using genre fallback link")
        # Store genre link to KG too
        if music_kg:
            music_kg.add_relationship(
                artist_a=artist_a_name,
                artist_b=artist_b_name,
                rel_type="genre_movement",
                connecting_entity=genre_link.connecting_entity,
                confidence=genre_link.confidence,
                description=genre_link.description,
                source="spotify_genres",
            )
        return {
            "trivia_link": genre_link,
            "trivia_confidence": genre_link.confidence,
            "trivia_context": trivia_context + [genre_link],
            "taste_candidates": [],
        }

    logger.info("Historian: no link found")
    return {
        "trivia_link": None,
        "trivia_confidence": 0.0,
        "trivia_context": trivia_context,
        "taste_candidates": [],
    }


async def _resolve_both_mbids(
    client: MusicBrainzClient, name_a: str, name_b: str
) -> tuple[str | None, str | None]:
    """Resolve both artist names to MBIDs concurrently."""
    import asyncio
    mbid_a, mbid_b = await asyncio.gather(
        client.resolve_mbid(name_a),
        client.resolve_mbid(name_b),
    )
    return mbid_a, mbid_b


def _try_genre_fallback(
    artist_a: str,
    artist_b: str,
    genres_a: list[str],
    genres_b: list[str],
    discussed: list[str],
) -> TriviaLink | None:
    """Generate a genre-based trivia link if both artists share genres.

    Spec §5.2: "check if artists share genres from Spotify metadata →
    generate a genre-based link (confidence 0.4)"
    """
    if not genres_a or not genres_b:
        return None

    shared = sorted(set(genres_a) & set(genres_b))
    if not shared:
        return None

    genre = shared[0]
    description = (
        f"Both {artist_a} and {artist_b} are rooted in {genre}"
    )

    if description in discussed:
        return None

    return TriviaLink(
        link_type="genre_movement",
        entity_a=artist_a,
        entity_b=artist_b,
        connecting_entity=genre,
        description=description,
        confidence=_GENRE_FALLBACK_CONFIDENCE,
    )


async def _build_historian_candidates(
    link: TriviaLink,
    linked_artist: str,
    lastfm: LastFMClient | None,
    spotify: SpotifyClient | None,
) -> list[CandidateTrack]:
    """Produce CandidateTrack objects from the trivia-linked artist.

    Spec §5.4: "Historian candidates (from trivia_link — if the Historian
    found a linked artist, that artist's top tracks are candidates with
    source = 'historian')".
    """
    if not lastfm or not spotify:
        return []

    try:
        top_tracks = await lastfm.get_top_tracks(linked_artist, limit=3)
        candidates: list[CandidateTrack] = []

        for track in top_tracks:
            uri = await spotify.search_track(track["name"], track["artist"])
            if uri:
                candidates.append(CandidateTrack(
                    spotify_uri=uri,
                    track_name=track["name"],
                    artist_name=track["artist"],
                    source="historian",
                    relevance_score=round(link.confidence, 3),
                    trivia_link=link,
                ))

        logger.info(
            "Historian: produced %d historian candidates for %r",
            len(candidates), linked_artist,
        )
        return candidates

    except Exception:
        logger.warning(
            "Historian: failed to build candidates for %r", linked_artist,
            exc_info=True,
        )
        return []
