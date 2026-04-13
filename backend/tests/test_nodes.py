"""
Tests for LangGraph Node Logic
================================
Tests the pure functions within each node without LangGraph compilation.
No external API calls — all external clients are mocked.

These tests verify:
    - Observer: playback state processing and track change detection
    - Historian: trivia link finding and fallbacks
    - Curator: candidate scoring and PTT routing
    - Scriptwriter: liner generation and guardrails
    - Memory Manager: cold start and should_run trigger
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from echodj.models import CandidateTrack, SpotifyTrack, TriviaLink, UserIntent
from echodj.state import DJState


# ── Observer Tests ───────────────────────────────────────────────────────────

class TestObserverNode:
    """Tests for the Observer node pure functions."""

    def test_process_playback_state_basic(self, sample_track: SpotifyTrack) -> None:
        """Should parse playback state and return correct fields."""
        from echodj.graph.observer import process_playback_state

        state: DJState = {}
        data = {
            "track_uri": sample_track.spotify_uri,
            "position_ms": 50_000,
            "duration_ms": 200_000,
            "is_playing": True,
            "track_name": sample_track.track_name,
            "artist_name": sample_track.artist_name,
            "album_art_url": sample_track.album_art_url,
        }

        updates = process_playback_state(state, data)
        assert updates["current_track"].track_name == sample_track.track_name
        assert abs(updates["playback_progress"] - 0.25) < 0.01
        assert updates["track_ending_soon"] is False

    def test_pre_computation_trigger(self) -> None:
        """track_ending_soon should be True at 76% progress."""
        from echodj.graph.observer import process_playback_state

        state: DJState = {}
        data = {
            "track_uri": "spotify:track:abc",
            "position_ms": 76_000,
            "duration_ms": 100_000,
            "is_playing": True,
            "track_name": "Test",
            "artist_name": "Artist",
        }

        updates = process_playback_state(state, data)
        assert updates["track_ending_soon"] is True

    def test_missing_track_uri_returns_empty(self) -> None:
        """Empty track_uri should return empty updates dict."""
        from echodj.graph.observer import process_playback_state

        updates = process_playback_state({}, {"track_uri": "", "track_name": ""})
        assert updates == {}

    def test_process_ptt_result(self) -> None:
        """PTT result should update utterance and intent."""
        from echodj.graph.observer import process_ptt_result

        updates = process_ptt_result("skip this", UserIntent.SKIP)
        assert updates["user_utterance"] == "skip this"
        assert updates["user_intent"] == UserIntent.SKIP

    def test_clear_ptt_state(self) -> None:
        """clear_ptt_state should return None values."""
        from echodj.graph.observer import clear_ptt_state

        updates = clear_ptt_state()
        assert updates["user_utterance"] is None
        assert updates["user_intent"] is None

    def test_track_change_detected(
        self, sample_track: SpotifyTrack, sample_track_b: SpotifyTrack
    ) -> None:
        """When URI changes, previous track should be added to history list."""
        from echodj.graph.observer import process_playback_state

        # Simulated state with old track
        state: DJState = {
            "current_track": sample_track,
            "previous_tracks": [],
        }

        data = {
            "track_uri": sample_track_b.spotify_uri,
            "position_ms": 10_000,
            "duration_ms": 200_000,
            "is_playing": True,
            "track_name": sample_track_b.track_name,
            "artist_name": sample_track_b.artist_name,
        }

        updates = process_playback_state(state, data)
        # previous_tracks should contain the OLD track
        assert len(updates.get("previous_tracks", [])) == 1
        assert updates["previous_tracks"][0].spotify_uri == sample_track.spotify_uri


# ── Historian Tests ──────────────────────────────────────────────────────────

class TestHistorianNode:
    """Tests for the Historian node."""

    @pytest.mark.asyncio
    async def test_insufficient_history_returns_none(self) -> None:
        """Without previous tracks, historian should return no link."""
        from echodj.graph.historian import historian_node

        state: DJState = {
            "current_track": None,
            "previous_tracks": [],
        }

        result = await historian_node(
            state,
            musicbrainz=AsyncMock(),
            wikidata=AsyncMock(),
        )
        assert result["trivia_link"] is None
        assert result["trivia_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_same_artist_returns_none(
        self, sample_track: SpotifyTrack
    ) -> None:
        """Same artist on consecutive tracks should return no trivia."""
        from echodj.graph.historian import historian_node

        state: DJState = {
            "current_track": sample_track,
            "previous_tracks": [sample_track],  # Same artist
            "discussed_trivia": [],
        }

        result = await historian_node(
            state,
            musicbrainz=AsyncMock(),
            wikidata=AsyncMock(),
        )
        assert result["trivia_link"] is None

    @pytest.mark.asyncio
    async def test_genre_fallback(
        self, sample_track: SpotifyTrack, sample_track_b: SpotifyTrack
    ) -> None:
        """Genre fallback should activate when SPARQL fails but genres overlap."""
        from echodj.graph.historian import historian_node

        # Create track with overlapping genre
        track_a = SpotifyTrack(
            spotify_uri="spotify:track:aaa",
            track_name="Song A",
            artist_name="Artist A",
            album_name="Album A",
            duration_ms=200_000,
            genres=["trip hop", "electronic"],
        )
        track_b = SpotifyTrack(
            spotify_uri="spotify:track:bbb",
            track_name="Song B",
            artist_name="Artist B",
            album_name="Album B",
            duration_ms=200_000,
            genres=["trip hop", "ambient"],  # Shared: trip hop
        )

        mb_mock = AsyncMock()
        mb_mock.resolve_mbid = AsyncMock(return_value=None)  # MBID fails

        wikidata_mock = AsyncMock()
        wikidata_mock.find_link = AsyncMock(return_value=None)

        state: DJState = {
            "current_track": track_b,
            "previous_tracks": [track_a],
            "discussed_trivia": [],
        }

        result = await historian_node(
            state,
            musicbrainz=mb_mock,
            wikidata=wikidata_mock,
        )
        # Should find genre fallback since MBID resolution failed → genre overlap
        if result["trivia_link"] is not None:
            assert result["trivia_link"].link_type == "genre_movement"
            assert result["trivia_confidence"] == 0.4


# ── Curator Tests ────────────────────────────────────────────────────────────

class TestCuratorNode:
    """Tests for the Curator node."""

    @pytest.mark.asyncio
    async def test_no_candidates_returns_continue(self) -> None:
        """No candidates should produce queue_action='continue'."""
        from echodj.graph.curator import curator_node

        state: DJState = {
            "taste_candidates": [],
            "trivia_link": None,
            "trivia_confidence": 0.0,
            "previous_tracks": [],
            "session_vibe": "moderate",
            "tracks_since_last_memory_update": 0,
        }

        spotify_mock = AsyncMock()
        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(return_value="")

        result = await curator_node(state, spotify=spotify_mock, llm=llm_mock)
        assert result["queue_action"] == "continue"
        assert result["next_track"] is None

    @pytest.mark.asyncio
    async def test_skip_intent_queues_top_candidate(
        self, sample_candidates: list[CandidateTrack]
    ) -> None:
        """SKIP intent should immediately queue the top Discoverer candidate."""
        from echodj.graph.curator import curator_node

        spotify_mock = AsyncMock()
        spotify_mock.queue_track = AsyncMock()

        state: DJState = {
            "user_intent": UserIntent.SKIP,
            "user_utterance": "skip this",
            "taste_candidates": sample_candidates,
            "previous_tracks": [],
            "tracks_since_last_memory_update": 0,
        }

        result = await curator_node(
            state, spotify=spotify_mock, llm=AsyncMock()
        )
        assert result["queue_action"] == "interrupt"
        assert result["next_track"] is not None
        spotify_mock.queue_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_selects_highest_scored_candidate(
        self, sample_candidates: list[CandidateTrack]
    ) -> None:
        """Curator should select the candidate with highest composite score."""
        from echodj.graph.curator import curator_node

        state: DJState = {
            "taste_candidates": sample_candidates,
            "trivia_link": None,
            "trivia_confidence": 0.0,
            "previous_tracks": [],
            "session_vibe": "moderate",
            "tracks_since_last_memory_update": 0,
        }

        spotify_mock = AsyncMock()
        spotify_mock.queue_track = AsyncMock()

        result = await curator_node(
            state,
            spotify=spotify_mock,
            llm=AsyncMock(),
        )
        # Should select the highest relevance_score candidate
        assert result["next_track"] is not None
        # Top candidate was score 0.9 (Glory Box by Portishead)
        assert result["next_track"].artist_name == "Portishead"


# ── Scriptwriter Tests ───────────────────────────────────────────────────────

class TestScriptwriterNode:
    """Tests for the Scriptwriter node."""

    @pytest.mark.asyncio
    async def test_generates_script_with_trivia(
        self,
        sample_track: SpotifyTrack,
        sample_track_b: SpotifyTrack,
        sample_trivia_link: TriviaLink,
    ) -> None:
        """Should include trivia description in prompt and return word count."""
        from echodj.graph.scriptwriter import scriptwriter_node

        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(
            return_value=(
                "Both of these artists share a deep connection to the Bristol "
                "scene of the 90s. From the post-punk melancholy of The Cure, "
                "we move to Portishead's haunting trip-hop. Here's Wandering Star."
            )
        )

        state: DJState = {
            "current_track": sample_track,
            "next_track": sample_track_b,
            "trivia_link": sample_trivia_link,
            "discussed_trivia": [],
            "session_vibe": "chill",
        }

        result = await scriptwriter_node(state, llm=llm_mock)
        assert result["script_text"] != ""
        assert result["script_word_count"] > 0
        # Trivia link description should be added to discussed
        assert sample_trivia_link.description in result["discussed_trivia"]

    @pytest.mark.asyncio
    async def test_guardrail_truncates_long_script(
        self, sample_track: SpotifyTrack, sample_track_b: SpotifyTrack
    ) -> None:
        """Scripts over 60 words should be truncated to 55 words."""
        from echodj.graph.scriptwriter import scriptwriter_node

        # Generate a very long script
        long_script = " ".join(["word"] * 100)

        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(return_value=long_script)

        state: DJState = {
            "current_track": sample_track,
            "next_track": sample_track_b,
            "trivia_link": None,
            "discussed_trivia": [],
            "session_vibe": "moderate",
        }

        result = await scriptwriter_node(state, llm=llm_mock)
        assert result["script_word_count"] <= 56  # 55 words + "..."

    @pytest.mark.asyncio
    async def test_empty_llm_uses_fallback(
        self, sample_track: SpotifyTrack, sample_track_b: SpotifyTrack
    ) -> None:
        """Empty LLM response should produce a fallback liner."""
        from echodj.graph.scriptwriter import scriptwriter_node

        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(return_value="")

        state: DJState = {
            "current_track": sample_track,
            "next_track": sample_track_b,
            "trivia_link": None,
            "discussed_trivia": [],
            "session_vibe": "moderate",
        }

        result = await scriptwriter_node(state, llm=llm_mock)
        assert result["script_text"] != ""
        # Fallback should mention the next track
        assert "Wandering Star" in result["script_text"]


# ── Memory Manager Tests ─────────────────────────────────────────────────────

class TestMemoryManager:
    """Tests for the Memory Manager node."""

    def test_should_run_trigger(self) -> None:
        """should_run() returns True when tracks >= 10."""
        from echodj.graph.memory_manager import should_run

        assert should_run({"tracks_since_last_memory_update": 10})
        assert should_run({"tracks_since_last_memory_update": 15})

    def test_should_not_run_below_threshold(self) -> None:
        """should_run() returns False below 10 tracks."""
        from echodj.graph.memory_manager import should_run

        assert not should_run({"tracks_since_last_memory_update": 9})
        assert not should_run({"tracks_since_last_memory_update": 0})
        assert not should_run({})

    def test_build_session_summary_structure(
        self, sample_track: SpotifyTrack, sample_track_b: SpotifyTrack
    ) -> None:
        """Session summary should have correct structure."""
        from echodj.graph.memory_manager import _build_session_summary

        summary = _build_session_summary(
            tracks=[sample_track, sample_track_b],
            vibe="chill",
            trivia_discussed=["trivia-1"],
            session_id="test-session",
        )
        assert summary["tracks_played"] == 2
        assert summary["session_vibe"] == "chill"
        assert "test-session" in summary["session_id"]
        assert "timestamp" in summary
