"""
Curator Agent — KG-Enriched Orchestrator Node
===============================================
Merges candidates from Historian and Discoverer, enriches them with
Knowledge Graph context, uses LLM-driven ranking (when KG context is
available), and queues the winning track on Spotify.

Graph RAG architecture per Diamantini et al. (2026):
    - Candidates enriched with KG relationships before ranking
    - LLM-driven ranking with full KG context (18.2% improvement over static)
    - Structured explainable reasoning in output
    - Fallback to static scoring when LLM is unavailable

References:
    - Paper §4.4 (Preference-Oriented Solution Ranking and Explanation)
    - Paper Table 3 (KG context improves ranking accuracy by 18.2%)
    - Spec §5.4 (Curator Agent)
    - Spec §5.4.1 (Intent Classifier Prompt)
    - Latency target: < 1.5s including Spotify queue API call
"""

from __future__ import annotations

import json
import logging
from typing import Any

from echodj.llm.provider import LLMProvider
from echodj.models import CandidateTrack, ListenerProfile, SpotifyTrack, TriviaLink, UserIntent
from echodj.services.music_knowledge_graph import MusicKnowledgeGraph
from echodj.services.segment_cache import SegmentCache
from echodj.services.spotify import SpotifyClient, SpotifyAPIError
from echodj.state import DJState

logger = logging.getLogger(__name__)

# Spec §5.4: Default scoring weights (fallback when LLM ranking unavailable)
_W_TRIVIA = 0.4
_W_TASTE = 0.4
_W_PROFILE = 0.2

# Spec §5.4: No repeat artist in last N tracks
_RECENT_ARTIST_WINDOW = 5

# Spec §5.4.1: Intent classifier prompt
_INTENT_CLASSIFIER_SYSTEM = """\
You are an intent classifier for a music DJ application. Given a user's spoken
command, classify it into exactly one of these categories:

- CHANGE_VIBE: User wants different energy/mood (e.g., "more upbeat", "something chill")
- SKIP: User wants to skip the current track (e.g., "next", "skip this")
- MORE_INFO: User wants more information about current track/artist
- SPECIFIC_REQUEST: User requests a specific artist/track/genre (e.g., "play some jazz")
- POSITIVE_FEEDBACK: User expresses they like the current selection
- NEGATIVE_FEEDBACK: User expresses dislike

Respond with ONLY the category name, nothing else."""

_INTENT_CLASSIFIER_USER = 'User said: "{utterance}"'

# Graph RAG: KG-enriched ranking prompt (Diamantini et al. §4.4)
_KG_RANKING_SYSTEM = """\
You are a music curator for EchoDJ. Your job is to select the best next track
from a list of candidates, considering the listener's taste profile, the current
session vibe, and known relationships between artists from our music knowledge graph.

RULES:
1. Prefer candidates with strong knowledge graph connections (shared producers,
   same studios, genre movements) to the current artist — these create interesting
   DJ transitions with trivia.
2. Respect the session vibe — if the listener wants "chill", don't pick high-energy tracks.
3. Avoid artists that appeared in the last 5 tracks.
4. Consider the listener's genre affinities and favorite artists.
5. Balance familiarity and discovery based on the listener's discovery_openness score.

Output ONLY valid JSON in this exact format:
{
  "selection_index": 0,
  "reasoning": "One sentence explaining why this track was selected, referencing specific KG connections or taste signals."
}"""

_KG_RANKING_USER = """\
## Listener Profile
{profile_summary}

## Session Context
- Current vibe: {session_vibe}
- Previous tracks: {previous_summary}

## Knowledge Graph Context
{kg_context}

## Candidates (pick the best one by index)
{candidates_summary}"""


async def curator_node(
    state: DJState,
    spotify: SpotifyClient,
    llm: LLMProvider,
    listener_profile: ListenerProfile | None = None,
    music_kg: MusicKnowledgeGraph | None = None,
    segment_cache: SegmentCache | None = None,
) -> dict:
    """LangGraph node: select and queue the next track.

    Priority order (latency-optimized):
        1. Pre-curated segment cache (near-zero latency)
        2. LLM ranking with KG context (Graph RAG)
        3. Static weighted scoring (fallback)

    Input reads (Spec §3.3):
        - taste_candidates, trivia_link, trivia_confidence, trivia_context
        - previous_tracks, session_vibe
        - user_utterance, user_intent
        - skip_detected, skipped_tracks (skip tracking signals)
        - listener_profile (from LangGraph Store)

    Output writes (Spec §3.3):
        - next_track, curator_reasoning, queue_action
        - tracks_since_last_memory_update (incremented)
        - active_segment_label (if using pre-curated segment)
    """
    user_intent = state.get("user_intent")
    user_utterance = state.get("user_utterance")

    # ── PTT Interrupt Mode ────────────────────────────────────────────────
    if user_intent is not None:
        return await _handle_ptt_interrupt(state, spotify, llm, user_intent)

    # Classify intent from utterance if we have one but no parsed intent
    if user_utterance:
        user_intent = await _classify_intent(llm, user_utterance)
        if user_intent is not None:
            return await _handle_ptt_interrupt(state, spotify, llm, user_intent)

    # ── Normal Transition Mode ────────────────────────────────────────────
    candidates = state.get("taste_candidates", [])
    trivia_link = state.get("trivia_link")
    trivia_confidence = state.get("trivia_confidence", 0.0)
    trivia_context = state.get("trivia_context", [])
    previous_tracks = state.get("previous_tracks", [])
    session_vibe = state.get("session_vibe", "moderate")
    tracks_since = state.get("tracks_since_last_memory_update", 0)
    user_id = state.get("user_id", "default")
    skipped_tracks = set(state.get("skipped_tracks", []))
    active_segment = state.get("active_segment_label")

    if not candidates:
        logger.warning("Curator: no candidates available")
        return {
            "next_track": None,
            "curator_reasoning": "No candidates available from Discoverer or Historian",
            "queue_action": "continue",
            "tracks_since_last_memory_update": tracks_since,
        }

    # Build recent URI set (skip signal + recent artist filter)
    recent_uris: set[str] = {t.spotify_uri for t in previous_tracks[-10:]}
    recent_artists = {t.artist_name.lower() for t in previous_tracks[-_RECENT_ARTIST_WINDOW:]}

    # ── Priority 1: Pre-curated Segment Cache ────────────────────────────
    # Near-zero latency path — use pre-built playlist if one is active.
    # Skip signal: if user just skipped, exit the active segment.
    skip_detected = state.get("skip_detected", False)
    if skip_detected and active_segment:
        logger.info("Curator: skip detected — exiting segment %r", active_segment)
        active_segment = None

    if segment_cache and active_segment:
        seg_track = segment_cache.get_next_segment_track(
            user_id=user_id,
            active_segment_label=active_segment,
            recent_uris=recent_uris | skipped_tracks,
        )
        if seg_track and seg_track["artist_name"].lower() not in recent_artists:
            next_track = SpotifyTrack(
                spotify_uri=seg_track["uri"],
                track_name=seg_track["track_name"],
                artist_name=seg_track["artist_name"],
                album_name="",
                duration_ms=1,
            )
            try:
                await spotify.queue_track(seg_track["uri"])
                queue_action = "play_next"
            except SpotifyAPIError:
                queue_action = "continue"

            return {
                "next_track": next_track,
                "curator_reasoning": f"[Segment: {active_segment}] {seg_track['script']}",
                "queue_action": queue_action,
                "active_segment_label": active_segment,
                "tracks_since_last_memory_update": tracks_since + 1,
            }

    # Filter out recent artists
    filtered = [c for c in candidates if c.artist_name.lower() not in recent_artists]
    if not filtered:
        logger.warning("Curator: all candidates filtered by recency, relaxing constraint")
        filtered = candidates

    # ── Graph RAG: Try LLM ranking with KG context ───────────────────────
    # Paper §4.4: LLM ranking with KG-enriched profiles
    current_track = state.get("current_track")
    current_artist = current_track.artist_name if current_track else ""

    llm_result = await _try_llm_ranking(
        llm=llm,
        candidates=filtered,
        listener_profile=listener_profile,
        session_vibe=session_vibe,
        previous_tracks=previous_tracks,
        trivia_context=trivia_context,
        current_artist=current_artist,
        music_kg=music_kg,
    )

    if llm_result:
        winner_candidate, reasoning = llm_result
    else:
        # Fallback: static scoring (Spec §5.4 original algorithm)
        scored = _score_candidates(
            filtered, trivia_confidence, session_vibe, listener_profile
        )
        if not scored:
            return {
                "next_track": None,
                "curator_reasoning": "All candidates exhausted",
                "queue_action": "continue",
                "tracks_since_last_memory_update": tracks_since,
            }
        winner_candidate, _, reasoning = scored[0]

    # Convert CandidateTrack → SpotifyTrack
    next_track = SpotifyTrack(
        spotify_uri=winner_candidate.spotify_uri,
        track_name=winner_candidate.track_name,
        artist_name=winner_candidate.artist_name,
        album_name="",
        duration_ms=1,
    )

    # Queue on Spotify (Spec §5.4)
    try:
        await spotify.queue_track(winner_candidate.spotify_uri)
        queue_action = "play_next"
    except SpotifyAPIError as exc:
        logger.error("Curator: failed to queue track: %s", exc)
        queue_action = "continue"

    return {
        "next_track": next_track,
        "curator_reasoning": reasoning,
        "queue_action": queue_action,
        "tracks_since_last_memory_update": tracks_since + 1,
    }


async def _try_llm_ranking(
    llm: LLMProvider,
    candidates: list[CandidateTrack],
    listener_profile: ListenerProfile | None,
    session_vibe: str,
    previous_tracks: list,
    trivia_context: list[TriviaLink],
    current_artist: str,
    music_kg: MusicKnowledgeGraph | None,
) -> tuple[CandidateTrack, str] | None:
    """Attempt LLM-driven ranking with KG context.

    Paper §4.4: The LLM ranks candidates with full KG context, producing
    structured reasoning. Returns None if LLM fails (triggers fallback).
    """
    import asyncio

    # Build KG context string
    kg_context_parts = []

    # Add trivia_context from Historian
    if trivia_context:
        kg_context_parts.append("Known artist relationships:")
        for link in trivia_context[:5]:  # Limit to avoid prompt bloat
            kg_context_parts.append(f"  - {link.description} (confidence: {link.confidence:.2f})")

    # Add per-candidate KG enrichment
    if music_kg and current_artist:
        for i, c in enumerate(candidates[:10]):
            rels = music_kg.get_relationships(current_artist, c.artist_name)
            if rels:
                for r in rels[:2]:
                    kg_context_parts.append(
                        f"  - Candidate {i} ({c.artist_name}): {r['description']}"
                    )

    kg_context = "\n".join(kg_context_parts) if kg_context_parts else "No KG relationships found for these candidates."

    # Build profile summary
    if listener_profile:
        genres = listener_profile.get("genre_affinity", {})
        top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:5]
        profile_summary = (
            f"Favorite genres: {', '.join(f'{g} ({s:.1f})' for g, s in top_genres)}\n"
            f"Favorite artists: {', '.join(listener_profile.get('artist_favorites', [])[:5])}\n"
            f"Discovery openness: {listener_profile.get('discovery_openness', 0.5):.1f}\n"
            f"Vibe preference: {listener_profile.get('vibe_preference', 'moderate')}"
        )
    else:
        profile_summary = "No listener profile available (cold start)."

    # Build previous tracks summary
    prev_summary = ", ".join(
        f'"{t.track_name}" by {t.artist_name}'
        for t in previous_tracks[-5:]
    ) or "None"

    # Build candidates summary
    cand_lines = []
    for i, c in enumerate(candidates[:10]):
        trivia_note = f" [trivia: {c.trivia_link.description}]" if c.trivia_link else ""
        cand_lines.append(
            f"  [{i}] \"{c.track_name}\" by {c.artist_name} "
            f"(source: {c.source}, relevance: {c.relevance_score:.2f}){trivia_note}"
        )
    candidates_summary = "\n".join(cand_lines)

    user_prompt = _KG_RANKING_USER.format(
        profile_summary=profile_summary,
        session_vibe=session_vibe,
        previous_summary=prev_summary,
        kg_context=kg_context,
        candidates_summary=candidates_summary,
    )

    try:
        response = await asyncio.wait_for(
            llm.generate(
                system_prompt=_KG_RANKING_SYSTEM,
                user_prompt=user_prompt,
            ),
            timeout=3.0,
        )

        if not response:
            return None

        # Parse JSON response
        # Strip markdown code fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]

        parsed = json.loads(clean)
        idx = int(parsed.get("selection_index", 0))
        reasoning = parsed.get("reasoning", "LLM selected this candidate.")

        if 0 <= idx < len(candidates):
            logger.info("Curator: LLM selected candidate %d: %s", idx, reasoning)
            return candidates[idx], f"[Graph RAG] {reasoning}"

    except asyncio.TimeoutError:
        logger.warning("Curator: LLM ranking timed out, falling back to static scoring")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Curator: LLM ranking parse error: %s", exc)
    except Exception:
        logger.warning("Curator: LLM ranking failed", exc_info=True)

    return None


def _score_candidates(
    candidates: list[CandidateTrack],
    trivia_confidence: float,
    session_vibe: str,
    profile: ListenerProfile | None,
) -> list[tuple[CandidateTrack, float, str]]:
    """Static score-based ranking (fallback when LLM unavailable).

    score = (w_trivia × trivia_bonus) + (w_taste × relevance_score)
              + (w_profile × profile_match)
    """
    genre_affinity = profile.get("genre_affinity", {}) if profile else {}
    results: list[tuple[CandidateTrack, float, str]] = []

    for c in candidates:
        # Trivia bonus: only for Historian candidates
        trivia_bonus = trivia_confidence if c.source == "historian" else 0.0

        # Profile match: genre overlap (Spec §5.4)
        profile_match = 0.0

        # Session vibe boost (Spec §5.4: "BOOST if genre matches session_vibe")
        vibe_boost = 0.0
        if session_vibe and session_vibe.lower() in str(genre_affinity):
            vibe_boost = 0.05

        score = (
            _W_TRIVIA * trivia_bonus
            + _W_TASTE * c.relevance_score
            + _W_PROFILE * profile_match
            + vibe_boost
        )

        reasoning = (
            f"source={c.source} "
            f"relevance={c.relevance_score:.2f} "
            f"trivia_bonus={trivia_bonus:.2f} "
            f"vibe_boost={vibe_boost:.2f} "
            f"total={score:.3f}"
        )
        results.append((c, score, reasoning))

    return sorted(results, key=lambda x: x[1], reverse=True)


async def _handle_ptt_interrupt(
    state: DJState,
    spotify: SpotifyClient,
    llm: LLMProvider,
    intent: UserIntent,
) -> dict:
    """Handle PTT interrupt routing per Spec §5.4."""
    candidates = state.get("taste_candidates", [])
    tracks_since = state.get("tracks_since_last_memory_update", 0)
    user_utterance = state.get("user_utterance", "")

    logger.info("Curator: handling PTT intent=%s", intent.value)

    if intent == UserIntent.SKIP:
        # Select top Discoverer candidate immediately
        if candidates:
            top = candidates[0]
            next_track = SpotifyTrack(
                spotify_uri=top.spotify_uri,
                track_name=top.track_name,
                artist_name=top.artist_name,
                album_name="",
                duration_ms=1,
            )
            try:
                await spotify.queue_track(top.spotify_uri)
                queue_action = "interrupt"
            except SpotifyAPIError:
                queue_action = "continue"
            return {
                "next_track": next_track,
                "curator_reasoning": "User requested skip → top Discoverer candidate",
                "queue_action": queue_action,
                "tracks_since_last_memory_update": tracks_since + 1,
            }

    elif intent == UserIntent.CHANGE_VIBE:
        # Extract vibe from utterance — simple keyword mapping
        new_vibe = _extract_vibe(user_utterance or "")
        return {
            "next_track": None,
            "curator_reasoning": f"Vibe changed to: {new_vibe}",
            "queue_action": "continue",
            "session_vibe": new_vibe,
            "tracks_since_last_memory_update": tracks_since,
        }

    elif intent in (UserIntent.POSITIVE_FEEDBACK, UserIntent.NEGATIVE_FEEDBACK):
        # Log feedback — no track change
        return {
            "next_track": None,
            "curator_reasoning": f"User feedback: {intent.value}",
            "queue_action": "continue",
            "tracks_since_last_memory_update": tracks_since,
        }

    elif intent == UserIntent.SPECIFIC_REQUEST:
        # Try to search Spotify for the requested entity
        if user_utterance:
            uri = await _search_specific_request(spotify, llm, user_utterance)
            if uri:
                next_track = SpotifyTrack(
                    spotify_uri=uri,
                    track_name="Requested Track",
                    artist_name="",
                    album_name="",
                    duration_ms=1,
                )
                try:
                    await spotify.queue_track(uri)
                    queue_action = "interrupt"
                except SpotifyAPIError:
                    queue_action = "continue"
                return {
                    "next_track": next_track,
                    "curator_reasoning": f"Specific request: {user_utterance}",
                    "queue_action": queue_action,
                    "tracks_since_last_memory_update": tracks_since + 1,
                }

    # Default: continue without change
    return {
        "next_track": None,
        "curator_reasoning": f"PTT intent={intent.value}, no track change",
        "queue_action": "continue",
        "tracks_since_last_memory_update": tracks_since,
    }


async def _classify_intent(
    llm: LLMProvider, utterance: str
) -> UserIntent | None:
    """Classify user utterance into a UserIntent using the LLM.

    Spec §5.4.1: Intent Classifier Prompt.
    """
    response = await llm.generate(
        system_prompt=_INTENT_CLASSIFIER_SYSTEM,
        user_prompt=_INTENT_CLASSIFIER_USER.format(utterance=utterance),
    )
    response = response.strip().upper()

    intent_map = {
        "CHANGE_VIBE": UserIntent.CHANGE_VIBE,
        "SKIP": UserIntent.SKIP,
        "MORE_INFO": UserIntent.MORE_INFO,
        "SPECIFIC_REQUEST": UserIntent.SPECIFIC_REQUEST,
        "POSITIVE_FEEDBACK": UserIntent.POSITIVE_FEEDBACK,
        "NEGATIVE_FEEDBACK": UserIntent.NEGATIVE_FEEDBACK,
    }
    return intent_map.get(response)


def _extract_vibe(utterance: str) -> str:
    """Simple keyword-based vibe extraction from user utterance."""
    u = utterance.lower()
    if any(w in u for w in ("upbeat", "energetic", "fast", "danceable", "hype")):
        return "energetic"
    if any(w in u for w in ("chill", "calm", "mellow", "relax", "slow", "quiet")):
        return "chill"
    if any(w in u for w in ("happy", "positive", "fun", "joyful")):
        return "upbeat"
    if any(w in u for w in ("sad", "melancholy", "moody", "dark")):
        return "melancholic"
    return "moderate"


async def _search_specific_request(
    spotify: SpotifyClient,
    llm: LLMProvider,
    utterance: str,
) -> str | None:
    """Extract artist/track from utterance and search Spotify."""
    # Use LLM to extract artist and track name from free-form request
    extraction = await llm.generate(
        system_prompt=(
            "Extract the artist name and track/genre from the user's music request. "
            "Reply ONLY as: ARTIST: <name> | TRACK: <name or genre>. "
            "If only an artist is mentioned, leave TRACK empty."
        ),
        user_prompt=f'User said: "{utterance}"',
    )
    # Parse the extraction
    artist, track = "", ""
    for part in extraction.split("|"):
        part = part.strip()
        if part.startswith("ARTIST:"):
            artist = part[7:].strip()
        elif part.startswith("TRACK:"):
            track = part[6:].strip()

    if artist:
        return await spotify.search_track(track or "", artist)
    return None
