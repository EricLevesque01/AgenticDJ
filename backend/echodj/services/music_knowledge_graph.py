"""
Music Knowledge Graph — Persistent Local KG
=============================================
Two-layer knowledge graph design inspired by Diamantini et al. (2026)
"A Graph RAG Approach to Enhance Explainability in Dataset Discovery".

Background Layer (stable):
    - Artist entities (name, genres, MBIDs, Wikidata QIDs)
    - Relationships (shared_producer, same_studio, genre_movement, influence)

Source Profile Layer (session-dynamic):
    - Play counts, user feedback, discussed trivia

The graph accumulates across sessions — every SPARQL result, every
successful trivia link, every MusicBrainz resolution is persisted.
Over time, lookups become local-first before hitting external APIs.

References:
    - Paper §3 (Data Model — BKG + SKG layers)
    - Paper §4.1 (Request enrichment via KG context)
    - Paper §4.4 (Ranking with KG-enriched profiles)
    - Spec §5.2 (Historian Agent)
    - Spec §10 (Memory Architecture)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default DB path — relative to project root
_DEFAULT_DB = Path(__file__).resolve().parent.parent.parent.parent / "data" / "echodj_music_kg.db"


class MusicKnowledgeGraph:
    """Persistent SQLite-backed music knowledge graph.

    Two-layer design per Diamantini et al.:
    - Background layer: artist entities + relationships (stable knowledge)
    - Source profile layer: session-dynamic play data and user signals

    All write operations are auto-committed. The graph is designed to be
    long-lived (survives server restarts) and accumulates knowledge over time.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path or _DEFAULT_DB)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("MusicKnowledgeGraph initialized: %s", self._db_path)

    def _create_tables(self) -> None:
        """Create the KG schema if it doesn't exist."""
        self._conn.executescript("""
            -- Background Layer: Artist entities
            CREATE TABLE IF NOT EXISTS artists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                name_lower TEXT NOT NULL UNIQUE,
                mbid TEXT,
                wikidata_qid TEXT,
                genres TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_artists_name_lower ON artists(name_lower);
            CREATE INDEX IF NOT EXISTS idx_artists_mbid ON artists(mbid);

            -- Background Layer: Relationships between artists
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist_a_id INTEGER NOT NULL,
                artist_b_id INTEGER NOT NULL,
                rel_type TEXT NOT NULL,
                connecting_entity TEXT NOT NULL,
                confidence REAL NOT NULL,
                description TEXT NOT NULL,
                source TEXT DEFAULT 'sparql',
                wikidata_qids TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (artist_a_id) REFERENCES artists(id),
                FOREIGN KEY (artist_b_id) REFERENCES artists(id),
                UNIQUE(artist_a_id, artist_b_id, rel_type, connecting_entity)
            );

            CREATE INDEX IF NOT EXISTS idx_rel_artists ON relationships(artist_a_id, artist_b_id);

            -- Source Profile Layer: Session play data
            CREATE TABLE IF NOT EXISTS play_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist_id INTEGER NOT NULL,
                track_uri TEXT NOT NULL,
                track_name TEXT NOT NULL,
                session_id TEXT,
                played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_feedback TEXT,
                FOREIGN KEY (artist_id) REFERENCES artists(id)
            );
        """)
        self._conn.commit()

    # ── Background Layer: Artist Operations ──────────────────────────────

    def upsert_artist(
        self,
        name: str,
        mbid: str | None = None,
        wikidata_qid: str | None = None,
        genres: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert or update an artist entity. Returns the artist ID.

        If the artist already exists (by name_lower), updates any newly
        provided fields without overwriting existing data.
        """
        name_lower = name.lower().strip()
        existing = self._conn.execute(
            "SELECT id, mbid, wikidata_qid, genres, metadata FROM artists WHERE name_lower = ?",
            (name_lower,),
        ).fetchone()

        if existing:
            artist_id = existing["id"]
            updates = {}
            if mbid and not existing["mbid"]:
                updates["mbid"] = mbid
            if wikidata_qid and not existing["wikidata_qid"]:
                updates["wikidata_qid"] = wikidata_qid
            if genres:
                existing_genres = json.loads(existing["genres"] or "[]")
                merged = sorted(set(existing_genres) | set(genres))
                updates["genres"] = json.dumps(merged)
            if metadata:
                existing_meta = json.loads(existing["metadata"] or "{}")
                existing_meta.update(metadata)
                updates["metadata"] = json.dumps(existing_meta)

            if updates:
                updates["updated_at"] = "CURRENT_TIMESTAMP"
                set_clause = ", ".join(
                    f"{k} = ?" if k != "updated_at" else f"{k} = CURRENT_TIMESTAMP"
                    for k in updates
                )
                values = [v for k, v in updates.items() if k != "updated_at"]
                self._conn.execute(
                    f"UPDATE artists SET {set_clause} WHERE id = ?",
                    (*values, artist_id),
                )
                self._conn.commit()
            return artist_id

        self._conn.execute(
            "INSERT INTO artists (name, name_lower, mbid, wikidata_qid, genres, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (name, name_lower, mbid, wikidata_qid,
             json.dumps(genres or []), json.dumps(metadata or {})),
        )
        self._conn.commit()
        return self._conn.execute(
            "SELECT id FROM artists WHERE name_lower = ?", (name_lower,)
        ).fetchone()["id"]

    def get_artist(self, name: str) -> dict[str, Any] | None:
        """Look up an artist by name. Returns dict or None."""
        row = self._conn.execute(
            "SELECT * FROM artists WHERE name_lower = ?", (name.lower().strip(),)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "mbid": row["mbid"],
            "wikidata_qid": row["wikidata_qid"],
            "genres": json.loads(row["genres"] or "[]"),
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    def get_cached_mbid(self, name: str) -> str | None:
        """Quick MBID lookup from the local KG (avoids MusicBrainz API)."""
        row = self._conn.execute(
            "SELECT mbid FROM artists WHERE name_lower = ? AND mbid IS NOT NULL",
            (name.lower().strip(),),
        ).fetchone()
        return row["mbid"] if row else None

    # ── Background Layer: Relationship Operations ────────────────────────

    def add_relationship(
        self,
        artist_a: str,
        artist_b: str,
        rel_type: str,
        connecting_entity: str,
        confidence: float,
        description: str,
        source: str = "sparql",
        wikidata_qids: list[str] | None = None,
    ) -> None:
        """Store a relationship between two artists.

        Upserts — if the exact relationship already exists, updates confidence
        if the new value is higher.
        """
        a_id = self.upsert_artist(artist_a)
        b_id = self.upsert_artist(artist_b)

        # Try both orderings (A→B and B→A)
        existing = self._conn.execute(
            """SELECT id, confidence FROM relationships
               WHERE ((artist_a_id = ? AND artist_b_id = ?) OR (artist_a_id = ? AND artist_b_id = ?))
               AND rel_type = ? AND connecting_entity = ?""",
            (a_id, b_id, b_id, a_id, rel_type, connecting_entity),
        ).fetchone()

        if existing:
            if confidence > existing["confidence"]:
                self._conn.execute(
                    "UPDATE relationships SET confidence = ? WHERE id = ?",
                    (confidence, existing["id"]),
                )
                self._conn.commit()
            return

        self._conn.execute(
            """INSERT INTO relationships
               (artist_a_id, artist_b_id, rel_type, connecting_entity, confidence, description, source, wikidata_qids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (a_id, b_id, rel_type, connecting_entity, confidence, description,
             source, json.dumps(wikidata_qids or [])),
        )
        self._conn.commit()

    def get_relationships(
        self, artist_a: str, artist_b: str
    ) -> list[dict[str, Any]]:
        """Get ALL known relationships between two artists (both directions).

        This is the local-first KG lookup — replaces SPARQL when data exists.
        """
        a = self.get_artist(artist_a)
        b = self.get_artist(artist_b)
        if not a or not b:
            return []

        rows = self._conn.execute(
            """SELECT r.*, a1.name as artist_a_name, a2.name as artist_b_name
               FROM relationships r
               JOIN artists a1 ON r.artist_a_id = a1.id
               JOIN artists a2 ON r.artist_b_id = a2.id
               WHERE (r.artist_a_id = ? AND r.artist_b_id = ?)
                  OR (r.artist_a_id = ? AND r.artist_b_id = ?)
               ORDER BY r.confidence DESC""",
            (a["id"], b["id"], b["id"], a["id"]),
        ).fetchall()

        return [
            {
                "rel_type": row["rel_type"],
                "connecting_entity": row["connecting_entity"],
                "confidence": row["confidence"],
                "description": row["description"],
                "source": row["source"],
                "wikidata_qids": json.loads(row["wikidata_qids"] or "[]"),
                "artist_a": row["artist_a_name"],
                "artist_b": row["artist_b_name"],
            }
            for row in rows
        ]

    def get_artist_neighborhood(
        self, artist: str, depth: int = 1
    ) -> list[dict[str, Any]]:
        """Get all relationships involving an artist (1-hop neighborhood).

        Paper §4.1: KG context enrichment — provides the full local
        neighborhood around an entity for prompt enrichment.
        """
        a = self.get_artist(artist)
        if not a:
            return []

        rows = self._conn.execute(
            """SELECT r.*, a1.name as artist_a_name, a2.name as artist_b_name
               FROM relationships r
               JOIN artists a1 ON r.artist_a_id = a1.id
               JOIN artists a2 ON r.artist_b_id = a2.id
               WHERE r.artist_a_id = ? OR r.artist_b_id = ?
               ORDER BY r.confidence DESC
               LIMIT 20""",
            (a["id"], a["id"]),
        ).fetchall()

        return [
            {
                "rel_type": row["rel_type"],
                "connecting_entity": row["connecting_entity"],
                "confidence": row["confidence"],
                "description": row["description"],
                "artist_a": row["artist_a_name"],
                "artist_b": row["artist_b_name"],
            }
            for row in rows
        ]

    def get_enriched_candidate_context(
        self, candidates: list[dict[str, str]], current_artist: str
    ) -> list[dict[str, Any]]:
        """For each candidate, return known KG relationships to the current artist.

        Paper §4.4: Enriching solutions with KG context before LLM ranking.
        This is the key Graph RAG enrichment step.

        Args:
            candidates: List of dicts with at least 'artist_name' key
            current_artist: The currently playing artist

        Returns:
            List of enrichment dicts, one per candidate, with KG relationships.
        """
        enriched = []
        for cand in candidates:
            rels = self.get_relationships(current_artist, cand["artist_name"])
            neighborhood = self.get_artist_neighborhood(cand["artist_name"])

            enriched.append({
                "artist_name": cand["artist_name"],
                "direct_relationships": rels,
                "neighborhood_size": len(neighborhood),
                "known_connections": [
                    r["description"] for r in rels
                ],
            })
        return enriched

    # ── Source Profile Layer ──────────────────────────────────────────────

    def record_play(
        self,
        artist_name: str,
        track_uri: str,
        track_name: str,
        session_id: str | None = None,
        feedback: str | None = None,
    ) -> None:
        """Record a track play event in the source profile layer."""
        artist_id = self.upsert_artist(artist_name)
        self._conn.execute(
            "INSERT INTO play_events (artist_id, track_uri, track_name, session_id, user_feedback) VALUES (?, ?, ?, ?, ?)",
            (artist_id, track_uri, track_name, session_id, feedback),
        )
        self._conn.commit()

    def get_play_count(self, artist_name: str) -> int:
        """Get total play count for an artist across all sessions."""
        a = self.get_artist(artist_name)
        if not a:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM play_events WHERE artist_id = ?",
            (a["id"],),
        ).fetchone()
        return row["cnt"] if row else 0

    # ── Statistics ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return basic KG statistics."""
        artists = self._conn.execute("SELECT COUNT(*) as cnt FROM artists").fetchone()["cnt"]
        rels = self._conn.execute("SELECT COUNT(*) as cnt FROM relationships").fetchone()["cnt"]
        plays = self._conn.execute("SELECT COUNT(*) as cnt FROM play_events").fetchone()["cnt"]
        return {"artists": artists, "relationships": rels, "play_events": plays}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
