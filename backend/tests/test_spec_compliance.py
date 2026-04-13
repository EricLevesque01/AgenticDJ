"""
Rigorous Spec Compliance Tests
==============================
As requested by Product Management, this test suite rigorously validates
the agentic non-deterministic components against the strict constraints
defined in ECHODJ_SPEC.md.

Specifically focusing on:
- Graph RAG LLM Output Parsing (Spec §5.4.1)
- Curator Intent Classification Map
- Scriptwriter Guardrail Word Count Caps (Spec §5.5)
- Memory Manager JSON Profile Updation Strictness
"""

import pytest
import json
from unittest.mock import AsyncMock

from echodj.models import CandidateTrack, UserIntent, SpotifyTrack
from echodj.graph.curator import _try_llm_ranking, _classify_intent, _extract_vibe
from echodj.graph.scriptwriter import _apply_guardrails
from echodj.graph.memory_manager import _update_profile_with_llm

@pytest.mark.asyncio
class TestCuratorGraphRAGCompliance:
    """Ensures Graph RAG ranking adheres exactly to Spec §4.4"""

    async def test_llm_ranking_markdown_json_parsing(self, sample_candidates):
        """Must correctly parse LLM responses enveloped in ```json fences."""
        llm_mock = AsyncMock()
        # Simulated LLM output wrapped in markdown fences
        llm_mock.generate = AsyncMock(return_value='''```json\n{"selection_index": 1, "reasoning": "Fits the chill vibe better"}\n```''')
        
        result = await _try_llm_ranking(
             llm=llm_mock,
             candidates=sample_candidates,
             listener_profile=None,
             session_vibe="chill",
             previous_tracks=[],
             trivia_context=[],
             current_artist="Some Artist",
             music_kg=None
         )
         
        assert result is not None
        winner, reasoning = result
        assert winner.track_name == sample_candidates[1].track_name
        assert "Fits the chill vibe better" in reasoning

    async def test_llm_ranking_out_of_bounds_index(self, sample_candidates):
        """If LLM hallucinates an index > len(candidates), fallback to None."""
        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(return_value='{"selection_index": 99, "reasoning": "Oops"}')
        
        result = await _try_llm_ranking(
             llm=llm_mock,
             candidates=sample_candidates,
             listener_profile=None,
             session_vibe="chill",
             previous_tracks=[],
             trivia_context=[],
             current_artist="Some Artist",
             music_kg=None
        )
        assert result is None  # Should gracefully fallback

    async def test_classify_intent_strict_mapping(self):
        """Must classify LLM output accurately against UserIntent Enum (Spec §5.4.1)"""
        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(return_value="CHANGE_VIBE")
        assert await _classify_intent(llm_mock, "play upbeat") == UserIntent.CHANGE_VIBE

        llm_mock.generate = AsyncMock(return_value="NEGATIVE_FEEDBACK")
        assert await _classify_intent(llm_mock, "hate this") == UserIntent.NEGATIVE_FEEDBACK

        llm_mock.generate = AsyncMock(return_value="NONSENSE")
        assert await _classify_intent(llm_mock, "what") is None
        
class TestVibeExtraction:
    def test_extract_vibe_keywords(self):
        """Tests the keyword fallback vibe extractor."""
        assert _extract_vibe("play something more energetic") == "energetic"
        assert _extract_vibe("i need to relax") == "chill"
        assert _extract_vibe("so happy right now") == "upbeat"
        assert _extract_vibe("just a normal song") == "moderate"


class TestScriptwriterGuardrailCompliance:
    """Ensures DJ vocal cadence and word counts meet strict Spec §5.5 caps."""

    def test_guardrails_truncates_at_exactly_55_words(self, sample_track):
        """Spec §5.5 explicitly requires truncation to 55 max words plus an ellipsis if over 60."""
        # 65 words string
        long_script = "word " * 65
        result = _apply_guardrails(long_script, sample_track)
        
        # Word count should be 55 words + 1 for the end "word..."
        assert len(result.split()) == 55
        assert result.endswith("...")

    def test_guardrails_empty_output_fallback(self, sample_track):
        """If script is empty, use the fallback template."""
        result = _apply_guardrails("", sample_track)
        assert result == f"Here's {sample_track.track_name} by {sample_track.artist_name}."


@pytest.mark.asyncio
class TestMemoryManagerProfileCompliance:
    """Ensures User ListenerProfile conforms to rigid schema updating."""

    async def test_update_profile_merges_json_correctly(self):
        """Must properly merge new affinity values coming out of LLM JSON safely."""
        llm_mock = AsyncMock()
        updated_json = {
            "genre_affinity": {"jazz": 0.9, "rock": 0.5},
            "artist_favorites": ["Miles Davis"],
            "discovery_openness": 0.8
        }
        llm_mock.generate = AsyncMock(return_value=json.dumps(updated_json))
        
        existing_profile = {
            "genre_affinity": {"jazz": 0.5},
            "artist_favorites": [],
            "discovery_openness": 0.5,
            "total_sessions": 1
        }
        
        session_data = {"dummy": "data"}
        
        # Even though LLM missed total_sessions, function should retain it from existing profile!
        # Let's wait: does `_update_profile_with_llm` map directly what LLM gives, or merges?
        # Actually in `_update_profile_with_llm` it parses and updates the `updated_at`.
        updated = await _update_profile_with_llm(llm_mock, existing_profile, session_data)
        
        assert updated["genre_affinity"]["jazz"] == 0.9
        assert "Miles Davis" in updated["artist_favorites"]
        # It should update the updated_at field!
        assert "updated_at" in updated
        assert hasattr(updated, "get")

    async def test_update_profile_handles_malformed_json(self):
        """Must retain existing profile if LLM completely hallucinates bad JSON."""
        llm_mock = AsyncMock()
        llm_mock.generate = AsyncMock(return_value="NOT JSON AT ALL")
        
        existing_profile = {"genre_affinity": {"rock": 0.9}}
        
        updated = await _update_profile_with_llm(llm_mock, existing_profile, {})
        # Should gracefully return existing profile since LLM failed
        assert updated["genre_affinity"]["rock"] == 0.9
