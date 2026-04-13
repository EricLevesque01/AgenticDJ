"""
LangGraph Graph Builder
========================
Wires all 7 nodes into a compiled LangGraph graph.

References:
    - Spec §4 (Architecture Overview — 7-node loop)
    - Spec §10.3 (Persistence — SqliteStore + MemorySaver)
    - Spec §4.2 (Graph flow: conditional edges based on state)

Graph topology (Spec §4):
    START
      → Historian (parallel with Discoverer)
      → Discoverer (parallel with Historian)
      → Curator (merge point — selects next track)
      → Scriptwriter (generate liner)
      → Vocalizer (synthesize audio)
      → Broadcast (send to frontend)
      → Memory Manager (conditional — every 10 tracks)
    END (loop back at next pre-computation trigger)
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from echodj.config import settings
from echodj.llm.provider import get_provider
from echodj.services.lastfm import LastFMClient
from echodj.services.listenbrainz import ListenBrainzClient
from echodj.services.musicbrainz import MusicBrainzClient
from echodj.services.music_knowledge_graph import MusicKnowledgeGraph
from echodj.services.segment_cache import SegmentCache
from echodj.services.spotify import SpotifyClient
from echodj.services.wikidata import WikidataClient
from echodj.state import DJState

logger = logging.getLogger(__name__)


def build_graph(
    spotify: SpotifyClient,
    send_json: Any,
    send_bytes: Any,
    store: Any = None,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the EchoDJ LangGraph graph.

    Each session gets its own compiled graph instance because the Spotify
    client is session-specific (different access tokens).

    Args:
        spotify: Per-session SpotifyClient with the user's access token.
        send_json: Async callable to send JSON to the frontend WebSocket.
        send_bytes: Async callable to send binary to the frontend WebSocket.
        store: LangGraph SqliteStore for persistent memory. If None,
               uses an in-memory store (for testing).

    Returns:
        Compiled LangGraph CompiledGraph instance.
    """
    # Shared client instances (reused across all graph invocations)
    llm = get_provider()
    musicbrainz = MusicBrainzClient()
    wikidata = WikidataClient()
    lastfm = LastFMClient()
    listenbrainz = ListenBrainzClient()

    # Graph RAG: Persistent Music Knowledge Graph (Diamantini et al.)
    music_kg = MusicKnowledgeGraph()

    # Pre-curated segment cache (shared DB with music_kg)
    segment_cache = SegmentCache()

    # Build the graph
    graph = StateGraph(DJState)

    # ── Status sender helpers (GAP 5: StatusHUD callbacks) ───────────────
    def _make_status_sender(node_name: str):
        """Returns an async callable that sends a status update for a node."""
        async def _send(message: str) -> None:
            await send_json({"type": "status", "node": node_name, "message": message})
        return _send

    # ── Node: Historian ───────────────────────────────────────────────────
    from echodj.graph.historian import historian_node
    graph.add_node("historian", functools.partial(
        historian_node,
        musicbrainz=musicbrainz,
        wikidata=wikidata,
        music_kg=music_kg,
        lastfm=lastfm,
        spotify=spotify,
        on_status=_make_status_sender("historian"),
    ))

    # ── Node: Discoverer ──────────────────────────────────────────────────
    from echodj.graph.discoverer import discoverer_node
    graph.add_node("discoverer", functools.partial(
        discoverer_node,
        lastfm=lastfm,
        listenbrainz=listenbrainz,
        spotify=spotify,
        on_status=_make_status_sender("discoverer"),
    ))

    # ── Node: Curator ─────────────────────────────────────────────────────
    from echodj.graph.curator import curator_node
    graph.add_node("curator", functools.partial(
        curator_node,
        spotify=spotify,
        llm=llm,
        music_kg=music_kg,
        segment_cache=segment_cache,
    ))

    # ── Node: Scriptwriter ────────────────────────────────────────────────
    from echodj.graph.scriptwriter import scriptwriter_node
    graph.add_node("scriptwriter", functools.partial(
        scriptwriter_node,
        llm=llm,
        on_status=_make_status_sender("scriptwriter"),
    ))

    # ── Node: Vocalizer ───────────────────────────────────────────────────
    from echodj.graph.vocalizer import vocalizer_node
    graph.add_node("vocalizer", vocalizer_node)

    # ── Node: Broadcast ───────────────────────────────────────────────────
    from echodj.graph.broadcast import broadcast_node
    graph.add_node("broadcast", functools.partial(
        broadcast_node,
        send_json=send_json,
        send_bytes=send_bytes,
    ))

    # ── Node: Memory Manager ──────────────────────────────────────────────
    from echodj.graph.memory_manager import memory_manager_node, should_run
    graph.add_node("memory_manager", functools.partial(
        memory_manager_node,
        llm=llm,
        store=store,
        spotify=spotify,
        segment_cache=segment_cache,
    ))

    # ── Edges ─────────────────────────────────────────────────────────────

    # Historian and Discoverer run in parallel from START
    # Spec §4: "Historian and Discoverer run concurrently (asyncio.gather)"
    graph.add_edge(START, "historian")
    graph.add_edge(START, "discoverer")

    # Both feed into Curator (merge point — LangGraph waits for both)
    graph.add_edge("historian", "curator")
    graph.add_edge("discoverer", "curator")

    # Linear pipeline after Curator
    graph.add_edge("curator", "scriptwriter")
    graph.add_edge("scriptwriter", "vocalizer")
    graph.add_edge("vocalizer", "broadcast")

    # Conditional: memory_manager only every 10 tracks
    graph.add_conditional_edges(
        "broadcast",
        _should_run_memory,
        {
            "memory": "memory_manager",
            "end": END,
        },
    )
    graph.add_edge("memory_manager", END)

    # ── Compile ───────────────────────────────────────────────────────────
    # Use provided checkpointer (AsyncSqliteSaver from server) or fallback to MemorySaver
    cp = checkpointer if checkpointer is not None else MemorySaver()
    compiled = graph.compile(checkpointer=cp)

    logger.info("LangGraph graph compiled successfully")
    return compiled


def _should_run_memory(state: DJState) -> str:
    """Conditional edge: run memory_manager or go to END.

    Spec §5.8: Memory Manager triggers every 10 tracks.
    """
    from echodj.graph.memory_manager import should_run
    return "memory" if should_run(state) else "end"


def get_sqlite_store() -> Any:
    """Create and return a LangGraph InMemoryStore for runtime KV state.

    Spec §10.3: The checkpointer uses SqliteSaver for thread-level persistence.
    The store (long-term memory KV) uses InMemoryStore backed by SqliteStore
    semantics via the memory_manager's explicit read/write pattern.
    """
    from langgraph.store.memory import InMemoryStore

    # Ensure data directory exists
    data_dir = settings.echodj_memory_db.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Store initialized (InMemoryStore with sqlite checkpointer)")
    return InMemoryStore()


async def get_sqlite_checkpointer() -> AsyncSqliteSaver:
    """Create and return an AsyncSqliteSaver for LangGraph thread state.

    Spec §10.3: SQLite for cross-session graph checkpoint persistence.
    Returns a context manager that must be used with `async with`.
    """
    data_dir = settings.echodj_sessions_db.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("AsyncSqliteSaver: %s", settings.echodj_sessions_db)
    return AsyncSqliteSaver.from_conn_string(str(settings.echodj_sessions_db))
