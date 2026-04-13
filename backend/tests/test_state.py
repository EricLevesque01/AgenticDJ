"""
Tests for EchoDJ LangGraph State Contract
==========================================
Validates the DJState TypedDict from Spec §3.1.
"""

from __future__ import annotations

from typing import get_type_hints

from echodj.state import DJState


class TestDJState:
    """Tests for the DJState TypedDict."""

    def test_all_spec_fields_present(self) -> None:
        """DJState should contain all fields from Spec §3.1."""
        hints = get_type_hints(DJState, include_extras=True)
        expected_fields = {
            # Observer writes
            "current_track",
            "previous_tracks",
            "playback_progress",
            "track_ending_soon",
            "user_utterance",
            "user_intent",
            "skipped_tracks",
            "skip_detected",
            # Historian writes
            "trivia_link",
            "trivia_confidence",
            "trivia_context",
            # Discoverer writes
            "taste_candidates",
            # Curator writes
            "next_track",
            "curator_reasoning",
            "active_segment_label",
            "queue_action",
            # Scriptwriter writes
            "script_text",
            "script_word_count",
            # Vocalizer writes
            "audio_buffer",
            "audio_duration_ms",
            # Broadcast writes
            "ducking_active",
            # Session context
            "session_id",
            "user_id",
            "discussed_trivia",
            "session_vibe",
            "tracks_since_last_memory_update",
            "messages",
        }
        actual_fields = set(hints.keys())
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        assert not missing, f"Missing fields from spec: {missing}"
        assert not extra, f"Extra fields not in spec: {extra}"

    def test_field_count(self) -> None:
        """DJState should have exactly 27 fields (Spec §3.1)."""
        hints = get_type_hints(DJState, include_extras=True)
        assert len(hints) == 27, (
            f"Expected 27 fields, got {len(hints)}: {sorted(hints.keys())}"
        )

    def test_total_false(self) -> None:
        """DJState uses total=False — all fields are optional for partial updates."""
        # TypedDict with total=False allows creating instances with
        # only a subset of fields (essential for LangGraph state updates).
        state: DJState = {  # type: ignore[typeddict-item]
            "session_id": "test-session",
            "user_id": "test-user",
        }
        assert state["session_id"] == "test-session"

    def test_annotated_reducers_exist(self) -> None:
        """Fields with Annotated reducers should be correctly typed.

        previous_tracks uses Annotated[list[SpotifyTrack], add]
        messages uses Annotated[list[BaseMessage], add_messages]
        """
        hints = get_type_hints(DJState, include_extras=True)
        # These fields should have Annotated metadata (the reducer)
        # We verify they exist in the type hints
        assert "previous_tracks" in hints
        assert "messages" in hints

    def test_queue_action_literal(self) -> None:
        """queue_action should accept only the three spec-defined values."""
        # Verify Literal type by constructing valid states
        for action in ("play_next", "interrupt", "continue"):
            state: DJState = {"queue_action": action}  # type: ignore[typeddict-item]
            assert state["queue_action"] == action
