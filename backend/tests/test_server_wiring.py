"""
Tests for the Server Session Lifecycle
=======================================
Tests the server wiring: graph invocation, PTT buffering,
state merge logic, and Memory Manager flush on disconnect.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── State merge tests ────────────────────────────────────────────────────────

class TestMergeState:
    """Tests for the _merge_state utility in server.py."""

    def test_simple_field_update(self) -> None:
        """Simple fields should be overwritten."""
        from echodj.server import _merge_state

        state = {"current_track": None, "playback_progress": 0.0}
        _merge_state(state, {"playback_progress": 0.75})
        assert state["playback_progress"] == 0.75

    def test_append_field_extends_list(self) -> None:
        """Annotated[list, add] fields should append, not replace."""
        from echodj.server import _merge_state

        state = {"previous_tracks": ["track_a"]}
        _merge_state(state, {"previous_tracks": ["track_b"]})
        assert state["previous_tracks"] == ["track_a", "track_b"]

    def test_messages_appended(self) -> None:
        """messages field should also use the append reducer."""
        from echodj.server import _merge_state

        state = {"messages": [{"role": "system", "content": "hi"}]}
        _merge_state(state, {"messages": [{"role": "user", "content": "hello"}]})
        assert len(state["messages"]) == 2

    def test_non_list_append_field_replaced(self) -> None:
        """If an append field has non-list value, replace it."""
        from echodj.server import _merge_state

        state = {"previous_tracks": "not_a_list"}
        _merge_state(state, {"previous_tracks": ["new"]})
        assert state["previous_tracks"] == ["new"]

    def test_new_fields_added(self) -> None:
        """Fields not in state yet should be added."""
        from echodj.server import _merge_state

        state = {}
        _merge_state(state, {"new_field": "value"})
        assert state["new_field"] == "value"

    def test_mixed_update(self) -> None:
        """Multiple fields should update correctly in a single call."""
        from echodj.server import _merge_state

        state = {
            "playback_progress": 0.0,
            "previous_tracks": [],
            "session_vibe": "moderate",
        }
        _merge_state(state, {
            "playback_progress": 0.5,
            "previous_tracks": ["new_track"],
            "session_vibe": "chill",
        })
        assert state["playback_progress"] == 0.5
        assert state["previous_tracks"] == ["new_track"]
        assert state["session_vibe"] == "chill"


# ── Memory Manager cross-session discussed_trivia tests ──────────────────────

class TestCrossSessionTrivia:
    """Tests for cross-session trivia loading and flushing (GAP 8)."""

    def test_load_discussed_trivia_empty_store(self) -> None:
        """Empty store should return empty list."""
        from echodj.graph.memory_manager import load_discussed_trivia

        store = MagicMock()
        store.get.return_value = None
        result = load_discussed_trivia(store, "user1")
        assert result == []

    def test_load_discussed_trivia_with_data(self) -> None:
        """Store with data should return the list."""
        from echodj.graph.memory_manager import load_discussed_trivia

        store = MagicMock()
        mock_item = MagicMock()
        mock_item.value = ["trivia_a", "trivia_b"]
        store.get.return_value = mock_item
        result = load_discussed_trivia(store, "user1")
        assert result == ["trivia_a", "trivia_b"]

    def test_load_discussed_trivia_exception_returns_empty(self) -> None:
        """Exception in store read should return empty list."""
        from echodj.graph.memory_manager import load_discussed_trivia

        store = MagicMock()
        store.get.side_effect = RuntimeError("corrupt")
        result = load_discussed_trivia(store, "user1")
        assert result == []

    def test_write_discussed_trivia(self) -> None:
        """Trivia should be written to correct namespace."""
        from echodj.graph.memory_manager import _write_discussed_trivia

        store = MagicMock()
        _write_discussed_trivia(store, "user1", ["a", "b"])
        store.put.assert_called_once_with(
            ("users", "user1", "trivia_discussed"), "all", ["a", "b"]
        )


# ── Historian candidate pipeline tests (GAP 6) ──────────────────────────────

class TestHistorianCandidates:
    """Tests for the Historian's CandidateTrack pipeline."""

    @pytest.mark.asyncio
    async def test_historian_returns_taste_candidates_key(self) -> None:
        """Historian should always return a taste_candidates key."""
        from echodj.graph.historian import historian_node

        state = {"current_track": None, "previous_tracks": []}
        result = await historian_node(
            state, musicbrainz=AsyncMock(), wikidata=AsyncMock()
        )
        assert "taste_candidates" in result
        assert result["taste_candidates"] == []

    @pytest.mark.asyncio
    async def test_build_historian_candidates_no_clients(self) -> None:
        """Without lastfm/spotify clients, should return empty list."""
        from echodj.graph.historian import _build_historian_candidates
        from echodj.models import TriviaLink

        link = TriviaLink(
            link_type="shared_producer",
            entity_a="A",
            entity_b="B",
            connecting_entity="C",
            description="test",
            confidence=0.8,
        )
        result = await _build_historian_candidates(link, "Artist B", None, None)
        assert result == []

    @pytest.mark.asyncio
    async def test_build_historian_candidates_with_clients(self) -> None:
        """With clients, should produce CandidateTrack objects."""
        from echodj.graph.historian import _build_historian_candidates
        from echodj.models import CandidateTrack, TriviaLink

        link = TriviaLink(
            link_type="shared_producer",
            entity_a="RadioheadTest",
            entity_b="PortisheadTest",
            connecting_entity="Bristol Studios",
            description="Shared studios",
            confidence=0.85,
        )

        lastfm = AsyncMock()
        lastfm.get_top_tracks = AsyncMock(return_value=[
            {"name": "Glory Box", "artist": "PortisheadTest"},
            {"name": "Sour Times", "artist": "PortisheadTest"},
        ])

        spotify = AsyncMock()
        spotify.search_track = AsyncMock(side_effect=[
            "spotify:track:glory",
            "spotify:track:sour",
        ])

        candidates = await _build_historian_candidates(
            link, "PortisheadTest", lastfm, spotify
        )
        assert len(candidates) == 2
        assert all(isinstance(c, CandidateTrack) for c in candidates)
        assert all(c.source == "historian" for c in candidates)
        assert all(c.trivia_link is not None for c in candidates)
        assert candidates[0].spotify_uri == "spotify:track:glory"

    @pytest.mark.asyncio
    async def test_build_historian_candidates_search_fails(self) -> None:
        """Spotify search failure for a track should drop that candidate."""
        from echodj.graph.historian import _build_historian_candidates
        from echodj.models import TriviaLink

        link = TriviaLink(
            link_type="shared_producer",
            entity_a="A",
            entity_b="B",
            connecting_entity="C",
            description="test",
            confidence=0.7,
        )

        lastfm = AsyncMock()
        lastfm.get_top_tracks = AsyncMock(return_value=[
            {"name": "Track1", "artist": "B"},
        ])

        spotify = AsyncMock()
        spotify.search_track = AsyncMock(return_value=None)  # Search fails

        candidates = await _build_historian_candidates(link, "B", lastfm, spotify)
        assert len(candidates) == 0  # Dropped because URI is None


# ── Memory Manager TypedDict construction test (GAP 9) ───────────────────────

class TestMemoryManagerProfileConstruction:
    """Verify that the profile construction doesn't crash with TypedDict."""

    @pytest.mark.asyncio
    async def test_create_initial_profile_is_dict(self) -> None:
        """Cold start profile should be a plain dict, not a TypedDict instance call."""
        from echodj.graph.memory_manager import _create_initial_profile

        spotify = AsyncMock()
        spotify.get_top_artists = AsyncMock(return_value=[
            {"name": "Radiohead", "genres": ["alt rock", "electronic"]},
            {"name": "Portishead", "genres": ["trip hop"]},
        ])

        profile = await _create_initial_profile(spotify, "test_user")
        assert isinstance(profile, dict)
        assert profile["user_id"] == "test_user"
        assert "alt rock" in profile["genre_affinity"]
        assert profile["discovery_openness"] == 0.5

    @pytest.mark.asyncio
    async def test_update_profile_with_llm_valid_json(self) -> None:
        """LLM returning valid JSON should produce a filtered dict."""
        from echodj.graph.memory_manager import _update_profile_with_llm
        import json

        existing = {
            "user_id": "test",
            "total_sessions": 3,
            "genre_affinity": {"jazz": 0.8},
        }

        updated_json = json.dumps({
            "user_id": "test",
            "total_sessions": 4,
            "genre_affinity": {"jazz": 0.85, "funk": 0.3},
            "unknown_field": "should_be_filtered",
        })

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=updated_json)

        result = await _update_profile_with_llm(llm, existing, {})
        assert isinstance(result, dict)
        assert result["total_sessions"] == 4
        assert "unknown_field" not in result

    @pytest.mark.asyncio
    async def test_update_profile_with_llm_invalid_json(self) -> None:
        """Invalid JSON from LLM should return existing profile."""
        from echodj.graph.memory_manager import _update_profile_with_llm

        existing = {"user_id": "test", "total_sessions": 3}

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="not valid json at all")

        result = await _update_profile_with_llm(llm, existing, {})
        assert result == existing
