"""
Discoverer Agent — Taste Profile Node
======================================
Finds tracks the user will enjoy using collaborative filtering from
Last.fm and ListenBrainz, falling back to Spotify top tracks.

References:
    - Spec §5.3 (Discoverer Agent)
    - Latency target: < 2s total (parallel API calls)
    - Output: ranked list of CandidateTrack objects (max 20)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Awaitable

from echodj.models import CandidateTrack
from echodj.services.lastfm import LastFMClient
from echodj.services.listenbrainz import ListenBrainzClient
from echodj.services.spotify import SpotifyClient
from echodj.state import DJState

logger = logging.getLogger(__name__)

# Spec §5.3: max 20 candidates
_MAX_CANDIDATES = 20


async def discoverer_node(
    state: DJState,
    lastfm: LastFMClient,
    listenbrainz: ListenBrainzClient,
    spotify: SpotifyClient,
    on_status: Callable[[str], Awaitable[None]] | None = None,
) -> dict:
    """LangGraph node: find taste-matching candidate tracks.

    Runs Last.fm and ListenBrainz queries in parallel, then falls back to
    Spotify top tracks if both external sources fail.

    Input reads (Spec §3.3):
        - current_track: Seed for similar artist/track queries
        - previous_tracks: Used to avoid repeating recent artists

    Output writes (Spec §3.3):
        - taste_candidates: Ranked list of CandidateTrack (max 20)
    """
    current = state.get("current_track")
    previous = state.get("previous_tracks", [])

    if not current:
        logger.debug("Discoverer: no current track, returning empty")
        return {"taste_candidates": []}

    if on_status:
        await on_status(f"Finding similar tracks to {current.artist_name}...")

    recent_artists = {t.artist_name.lower() for t in previous[-5:]}

    # ── Run all sources in parallel ───────────────────────────────────────
    lastfm_task = asyncio.create_task(
        _fetch_lastfm(lastfm, spotify, current.artist_name, current.track_name)
    )
    lb_task = asyncio.create_task(
        _fetch_listenbrainz(listenbrainz, spotify)
    )

    lastfm_results, lb_results = await asyncio.gather(
        lastfm_task, lb_task, return_exceptions=True
    )

    # Collect valid results
    all_candidates: list[CandidateTrack] = []

    if isinstance(lastfm_results, list):
        all_candidates.extend(lastfm_results)
    else:
        logger.warning("Discoverer: Last.fm task failed: %s", lastfm_results)

    if isinstance(lb_results, list):
        all_candidates.extend(lb_results)
    else:
        logger.warning("Discoverer: ListenBrainz task failed: %s", lb_results)

    # Fallback: Spotify top tracks (Spec §5.3)
    if not all_candidates:
        logger.info("Discoverer: all external sources failed, using Spotify top tracks")
        all_candidates = await _fetch_spotify_top(spotify)

    # De-duplicate by artist and filter recent artists
    candidates = _deduplicate_and_rank(all_candidates, recent_artists)

    logger.info("Discoverer: %d candidates found", len(candidates))
    return {"taste_candidates": candidates}


async def _fetch_lastfm(
    lastfm: LastFMClient,
    spotify: SpotifyClient,
    artist_name: str,
    track_name: str,
) -> list[CandidateTrack]:
    """Fetch and merge artist.getSimilar and track.getSimilar results."""
    similar_artists, similar_tracks = await asyncio.gather(
        lastfm.get_similar_artists(artist_name, limit=10),
        lastfm.get_similar_tracks(artist_name, track_name, limit=10),
    )

    candidates: list[CandidateTrack] = []

    # From similar artists: pick each artist's top track
    for i, artist in enumerate(similar_artists):
        top_tracks = await lastfm.get_top_tracks(artist["name"], limit=1)
        if not top_tracks:
            continue
        track = top_tracks[0]
        uri = await spotify.search_track(track["name"], track["artist"])
        if uri:
            score = 1.0 - (i / max(len(similar_artists), 1))
            candidates.append(CandidateTrack(
                spotify_uri=uri,
                track_name=track["name"],
                artist_name=track["artist"],
                source="lastfm",
                relevance_score=round(min(score, 1.0), 3),
            ))

    # From similar tracks (direct track similarity)
    for i, track in enumerate(similar_tracks):
        uri = await spotify.search_track(track["name"], track["artist"])
        if uri:
            # Interleave with lower base score
            score = 0.8 - (i / max(len(similar_tracks), 1) * 0.3)
            candidates.append(CandidateTrack(
                spotify_uri=uri,
                track_name=track["name"],
                artist_name=track["artist"],
                source="lastfm",
                relevance_score=round(min(max(score, 0.0), 1.0), 3),
            ))

    return candidates


async def _fetch_listenbrainz(
    listenbrainz: ListenBrainzClient,
    spotify: SpotifyClient,
) -> list[CandidateTrack]:
    """Fetch ListenBrainz CF recommendations and resolve to Spotify URIs."""
    if not listenbrainz.is_configured():
        return []

    recs = await listenbrainz.get_recommendations(count=20)
    candidates: list[CandidateTrack] = []

    for i, rec in enumerate(recs):
        track_name = rec.get("track_name", "")
        artist_name = rec.get("artist_name", "")
        if not track_name or not artist_name:
            continue

        uri = await spotify.search_track(track_name, artist_name)
        if uri:
            score = 1.0 - (i / max(len(recs), 1))
            candidates.append(CandidateTrack(
                spotify_uri=uri,
                track_name=track_name,
                artist_name=artist_name,
                source="listenbrainz",
                relevance_score=round(min(score, 1.0), 3),
            ))

    return candidates


async def _fetch_spotify_top(spotify: SpotifyClient) -> list[CandidateTrack]:
    """Fallback: user's Spotify top tracks (Spec §5.3)."""
    try:
        tracks = await spotify.get_top_tracks(limit=20)
        import random
        random.shuffle(tracks)  # Shuffle for variety per spec
        return [
            CandidateTrack(
                spotify_uri=t.spotify_uri,
                track_name=t.track_name,
                artist_name=t.artist_name,
                source="spotify_top",
                relevance_score=round(0.5 + (i / max(len(tracks), 1)) * -0.3, 3),
            )
            for i, t in enumerate(tracks)
        ]
    except Exception:
        logger.exception("Discoverer: Spotify top tracks fallback failed")
        return []


def _deduplicate_and_rank(
    candidates: list[CandidateTrack],
    recent_artists: set[str],
) -> list[CandidateTrack]:
    """De-duplicate by artist, filter recent, sort by relevance, cap at 20.

    Spec §5.3: "De-duplicate by artist."
    Spec §5.4: No repeat artist in last 5 tracks (enforced in Curator,
               but we pre-filter here for cleaner candidate lists).
    """
    seen_artists: set[str] = set()
    unique: list[CandidateTrack] = []

    for c in sorted(candidates, key=lambda x: x.relevance_score, reverse=True):
        artist_key = c.artist_name.lower()
        if artist_key in seen_artists:
            continue
        seen_artists.add(artist_key)
        # Filter recently played artists (Spec §5.4 pre-filter)
        if artist_key not in recent_artists:
            unique.append(c)

    return unique[:_MAX_CANDIDATES]
