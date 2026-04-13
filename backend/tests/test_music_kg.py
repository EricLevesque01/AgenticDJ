"""
Tests for MusicKnowledgeGraph — persistent two-layer KG.

Validates:
    - Artist CRUD (upsert, get, cached MBID)
    - Relationship storage and retrieval (both directions)
    - Neighborhood queries
    - Candidate enrichment
    - Source profile layer (play events)
    - Statistics
"""

import pytest
import tempfile
from pathlib import Path

from echodj.services.music_knowledge_graph import MusicKnowledgeGraph


@pytest.fixture
def kg(tmp_path):
    """Create a fresh MusicKnowledgeGraph with a temp database."""
    db = tmp_path / "test_music_kg.db"
    g = MusicKnowledgeGraph(db_path=db)
    yield g
    g.close()


class TestArtistOperations:
    """Background layer: artist entity CRUD."""

    def test_upsert_and_get_artist(self, kg):
        artist_id = kg.upsert_artist("Miles Davis", mbid="mbid_001", genres=["jazz", "fusion"])
        assert artist_id > 0

        artist = kg.get_artist("Miles Davis")
        assert artist is not None
        assert artist["name"] == "Miles Davis"
        assert artist["mbid"] == "mbid_001"
        assert "jazz" in artist["genres"]

    def test_upsert_is_case_insensitive(self, kg):
        id1 = kg.upsert_artist("Miles Davis")
        id2 = kg.upsert_artist("miles davis")
        assert id1 == id2

    def test_upsert_merges_genres(self, kg):
        kg.upsert_artist("Miles Davis", genres=["jazz"])
        kg.upsert_artist("Miles Davis", genres=["fusion", "jazz"])

        artist = kg.get_artist("Miles Davis")
        assert sorted(artist["genres"]) == ["fusion", "jazz"]

    def test_upsert_does_not_overwrite_existing_mbid(self, kg):
        kg.upsert_artist("Miles Davis", mbid="original")
        kg.upsert_artist("Miles Davis", mbid="should_not_replace")

        artist = kg.get_artist("Miles Davis")
        assert artist["mbid"] == "original"

    def test_get_nonexistent_artist_returns_none(self, kg):
        assert kg.get_artist("Nonexistent") is None

    def test_get_cached_mbid(self, kg):
        kg.upsert_artist("Miles Davis", mbid="mbid_001")
        assert kg.get_cached_mbid("Miles Davis") == "mbid_001"
        assert kg.get_cached_mbid("miles davis") == "mbid_001"  # case insensitive
        assert kg.get_cached_mbid("Unknown") is None


class TestRelationshipOperations:
    """Background layer: artist relationship storage."""

    def test_add_and_get_relationship(self, kg):
        kg.add_relationship(
            artist_a="Radiohead",
            artist_b="David Bowie",
            rel_type="shared_producer",
            connecting_entity="Brian Eno",
            confidence=0.85,
            description="Both Radiohead and David Bowie were produced by Brian Eno",
        )

        rels = kg.get_relationships("Radiohead", "David Bowie")
        assert len(rels) == 1
        assert rels[0]["rel_type"] == "shared_producer"
        assert rels[0]["connecting_entity"] == "Brian Eno"
        assert rels[0]["confidence"] == 0.85

    def test_relationship_is_bidirectional(self, kg):
        kg.add_relationship(
            artist_a="Radiohead",
            artist_b="David Bowie",
            rel_type="shared_producer",
            connecting_entity="Brian Eno",
            confidence=0.85,
            description="Both produced by Brian Eno",
        )

        # Query in reverse direction should also find it
        rels = kg.get_relationships("David Bowie", "Radiohead")
        assert len(rels) == 1
        assert rels[0]["connecting_entity"] == "Brian Eno"

    def test_upsert_updates_higher_confidence(self, kg):
        kg.add_relationship(
            artist_a="A", artist_b="B",
            rel_type="genre_movement", connecting_entity="rock",
            confidence=0.4, description="Both rooted in rock",
        )
        kg.add_relationship(
            artist_a="A", artist_b="B",
            rel_type="genre_movement", connecting_entity="rock",
            confidence=0.9, description="Both rooted in rock",
        )

        rels = kg.get_relationships("A", "B")
        assert len(rels) == 1
        assert rels[0]["confidence"] == 0.9

    def test_multiple_relationship_types(self, kg):
        kg.add_relationship(
            artist_a="A", artist_b="B",
            rel_type="shared_producer", connecting_entity="Producer X",
            confidence=0.85, description="Shared producer",
        )
        kg.add_relationship(
            artist_a="A", artist_b="B",
            rel_type="genre_movement", connecting_entity="rock",
            confidence=0.45, description="Genre overlap",
        )

        rels = kg.get_relationships("A", "B")
        assert len(rels) == 2
        # Ordered by confidence DESC
        assert rels[0]["confidence"] > rels[1]["confidence"]

    def test_get_relationships_empty(self, kg):
        rels = kg.get_relationships("Unknown1", "Unknown2")
        assert rels == []


class TestNeighborhood:
    """Background layer: artist neighborhood queries."""

    def test_get_artist_neighborhood(self, kg):
        kg.add_relationship("A", "B", "genre_movement", "rock", 0.5, "genre link")
        kg.add_relationship("A", "C", "shared_producer", "X", 0.8, "producer link")
        kg.add_relationship("D", "E", "same_studio", "Y", 0.7, "studio link")

        neighborhood = kg.get_artist_neighborhood("A")
        assert len(neighborhood) == 2
        artists_mentioned = {n["artist_b"] for n in neighborhood} | {n["artist_a"] for n in neighborhood}
        assert "B" in artists_mentioned or "C" in artists_mentioned

    def test_neighborhood_empty_for_unknown(self, kg):
        assert kg.get_artist_neighborhood("Unknown") == []


class TestCandidateEnrichment:
    """Graph RAG: per-candidate KG enrichment for Curator ranking."""

    def test_enriched_candidate_context(self, kg):
        kg.add_relationship("Current", "Candidate1", "shared_producer", "P", 0.8, "Shared producer P")
        kg.add_relationship("Current", "Candidate2", "genre_movement", "jazz", 0.5, "Both jazz")

        candidates = [
            {"artist_name": "Candidate1"},
            {"artist_name": "Candidate2"},
            {"artist_name": "Unknown"},
        ]

        enriched = kg.get_enriched_candidate_context(candidates, "Current")
        assert len(enriched) == 3

        # Candidate1 has a direct relationship
        assert len(enriched[0]["direct_relationships"]) == 1
        assert enriched[0]["known_connections"] == ["Shared producer P"]

        # Candidate2 has a relationship too
        assert len(enriched[1]["direct_relationships"]) == 1

        # Unknown has no relationships
        assert len(enriched[2]["direct_relationships"]) == 0


class TestSourceProfileLayer:
    """Source profile layer: session play data."""

    def test_record_and_count_plays(self, kg):
        kg.record_play("Miles Davis", "spotify:track:123", "So What", session_id="s1")
        kg.record_play("Miles Davis", "spotify:track:456", "Blue In Green", session_id="s1")
        kg.record_play("Miles Davis", "spotify:track:123", "So What", session_id="s2")

        assert kg.get_play_count("Miles Davis") == 3

    def test_play_count_zero_for_unknown(self, kg):
        assert kg.get_play_count("Unknown Artist") == 0


class TestStatistics:
    """KG statistics for monitoring."""

    def test_stats_empty(self, kg):
        stats = kg.stats()
        assert stats["artists"] == 0
        assert stats["relationships"] == 0
        assert stats["play_events"] == 0

    def test_stats_after_operations(self, kg):
        kg.upsert_artist("A")
        kg.upsert_artist("B")
        kg.add_relationship("A", "B", "genre_movement", "rock", 0.5, "test")
        kg.record_play("A", "spotify:track:1", "Track 1")

        stats = kg.stats()
        assert stats["artists"] == 2
        assert stats["relationships"] == 1
        assert stats["play_events"] == 1
