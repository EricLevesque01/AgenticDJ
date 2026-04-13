"""
EchoDJ FastAPI Server
=====================
WebSocket-first API server for the EchoDJ frontend.
Wires the LangGraph agent loop to WebSocket sessions.

References:
    - Spec §7 (Frontend ↔ Backend Protocol)
    - Spec §7.2 (Connection Lifecycle)
    - Spec §7.3 (Message Types)
    - Spec §5.1 (Observer — pre-computation trigger at 75%)
    - Spec §5.8 (Memory Manager — flush on disconnect)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from echodj.graph.observer import (
    clear_ptt_state,
    process_playback_state,
    process_ptt_result,
)
from echodj.models import UserIntent
from echodj.services.spotify import SpotifyClient

logger = logging.getLogger(__name__)


# ── Per-Session State ────────────────────────────────────────────────────────

class SessionState:
    """Mutable per-session state holder.

    Tracks the DJState between graph invocations, PTT audio buffering,
    and the compiled LangGraph graph for this session.
    """

    def __init__(
        self,
        session_id: str,
        spotify: SpotifyClient,
        websocket: WebSocket,
    ) -> None:
        self.session_id = session_id
        self.spotify = spotify
        self.websocket = websocket
        self.compiled_graph: Any = None
        self.store: Any = None

        # DJState accumulator — seeded on first playback_state
        self.dj_state: dict[str, Any] = {
            "session_id": session_id,
            "user_id": "default",
            "current_track": None,
            "previous_tracks": [],
            "playback_progress": 0.0,
            "track_ending_soon": False,
            "user_utterance": None,
            "user_intent": None,
            "trivia_link": None,
            "trivia_confidence": 0.0,
            "taste_candidates": [],
            "next_track": None,
            "curator_reasoning": "",
            "queue_action": "continue",
            "script_text": "",
            "script_word_count": 0,
            "audio_buffer": None,
            "audio_duration_ms": 0,
            "ducking_active": False,
            "discussed_trivia": [],
            "session_vibe": "moderate",
            "tracks_since_last_memory_update": 0,
            "messages": [],
        }

        # PTT audio buffer (Spec §5.1)
        self.ptt_buffer: bytearray = bytearray()
        self.ptt_active: bool = False

        # Pre-computation guard: don't re-trigger while pipeline is running
        self._pipeline_running: bool = False
        self._pre_compute_triggered: bool = False


# ── Active connections ───────────────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections and their associated sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    async def connect(
        self,
        websocket: WebSocket,
        access_token: str,
    ) -> SessionState:
        """Accept a WebSocket connection, build the graph, and create a session.

        Spec §7.2: Server validates token, responds with session_id.
        """
        await websocket.accept()
        session_id = str(uuid.uuid4())

        spotify = SpotifyClient(access_token)
        session = SessionState(session_id, spotify, websocket)

        # Build the LangGraph graph for this session
        try:
            from echodj.graph.builder import (
                build_graph,
                get_sqlite_store,
                get_sqlite_checkpointer,
            )

            session.store = get_sqlite_store()

            # AsyncSqliteSaver.from_conn_string returns a context manager;
            # enter it and keep the saver for the session lifetime.
            try:
                checkpointer_cm = get_sqlite_checkpointer()
                checkpointer = await checkpointer_cm.__aenter__()
                session._checkpointer_cm = checkpointer_cm  # prevent GC
            except Exception as cp_err:
                logger.warning("SQLite checkpointer failed (%s), using MemorySaver", cp_err)
                from langgraph.checkpoint.memory import MemorySaver
                checkpointer = MemorySaver()

            session.compiled_graph = build_graph(
                spotify=spotify,
                send_json=lambda data: self._send_json_ws(websocket, data),
                send_bytes=lambda data: self._send_bytes_ws(websocket, data),
                store=session.store,
                checkpointer=checkpointer,
            )

            # Load cross-session discussed_trivia (GAP 8)
            from echodj.graph.memory_manager import load_discussed_trivia
            session.dj_state["discussed_trivia"] = load_discussed_trivia(
                session.store, session.dj_state["user_id"]
            )

            logger.info(
                "Graph compiled for session=%s (sqlite checkpointer), loaded %d discussed trivia",
                session_id,
                len(session.dj_state["discussed_trivia"]),
            )
        except Exception:
            logger.exception("Failed to build graph for session=%s", session_id)
            # Session will still work for playback, just without agentic features

        self._sessions[session_id] = session

        # Spec §7.2: Send connected message
        await self.send_json(session_id, {
            "type": "connected",
            "session_id": session_id,
        })

        logger.info("Client connected: session=%s", session_id)
        return session

    async def disconnect(self, session_id: str) -> None:
        """Clean up a disconnected session.

        Spec §5.8: Run Memory Manager on disconnect.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        # GAP 7: Flush memory on disconnect
        await self._flush_memory_on_disconnect(session)

        # Clean up Spotify client
        await session.spotify.close()
        del self._sessions[session_id]
        logger.info("Client disconnected: session=%s", session_id)

    async def _flush_memory_on_disconnect(self, session: SessionState) -> None:
        """Run Memory Manager one final time on session end (Spec §5.8)."""
        if not session.store or not session.compiled_graph:
            return

        tracks_played = session.dj_state.get("tracks_since_last_memory_update", 0)
        if tracks_played == 0:
            return  # Nothing to flush

        try:
            from echodj.graph.memory_manager import memory_manager_node
            from echodj.llm.provider import get_provider

            llm = get_provider()
            await memory_manager_node(
                session.dj_state,
                llm=llm,
                store=session.store,
                spotify=session.spotify,
            )
            logger.info(
                "Memory flushed on disconnect: session=%s (%d tracks)",
                session.session_id, tracks_played,
            )
        except Exception:
            logger.warning(
                "Memory flush failed on disconnect: session=%s",
                session.session_id,
                exc_info=True,
            )

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    async def send_json(self, session_id: str, data: dict[str, Any]) -> None:
        """Send a JSON message to a specific session."""
        session = self._sessions.get(session_id)
        if session:
            await self._send_json_ws(session.websocket, data)

    async def send_bytes(self, session_id: str, data: bytes) -> None:
        """Send binary data (DJ audio) to a specific session."""
        session = self._sessions.get(session_id)
        if session:
            await self._send_bytes_ws(session.websocket, data)

    async def send_status(
        self, session_id: str, node: str, message: str
    ) -> None:
        """Send a status update for the HUD (Spec §7.3)."""
        await self.send_json(session_id, {
            "type": "status", "node": node, "message": message,
        })

    async def send_error(
        self, session_id: str, message: str, recoverable: bool = True
    ) -> None:
        """Send an error message to the frontend (Spec §7.3)."""
        await self.send_json(session_id, {
            "type": "error", "message": message, "recoverable": recoverable,
        })

    @staticmethod
    async def _send_json_ws(ws: WebSocket, data: dict[str, Any]) -> None:
        try:
            await ws.send_json(data)
        except Exception:
            logger.warning("send_json failed")

    @staticmethod
    async def _send_bytes_ws(ws: WebSocket, data: bytes) -> None:
        try:
            await ws.send_bytes(data)
        except Exception:
            logger.warning("send_bytes failed")


# ── Global connection manager ────────────────────────────────────────────────
manager = ConnectionManager()


# ── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Application startup and shutdown."""
    logger.info("EchoDJ server starting")
    yield
    logger.info("EchoDJ server shutting down")


# ── FastAPI Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="EchoDJ",
    description="Agentic AI radio station API",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend (Spec: local-only deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "service": "echodj"}


# ── WebSocket Endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Primary WebSocket endpoint for frontend communication.

    Spec §7.1: ws://localhost:8000/ws
    Spec §7.2: Connection lifecycle — auth via query param.
    """
    token = websocket.query_params.get("token", "")
    if not token:
        await websocket.close(code=4001, reason="Missing access token")
        return

    session = await manager.connect(websocket, token)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                await _handle_text_message(session, message["text"])
            elif "bytes" in message:
                await _handle_binary_message(session, message["bytes"])

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session.session_id)
    except Exception:
        logger.exception("WebSocket error: session=%s", session.session_id)
    finally:
        await manager.disconnect(session.session_id)


# ── Message handlers ─────────────────────────────────────────────────────────

async def _handle_text_message(session: SessionState, raw: str) -> None:
    """Route incoming JSON messages from the frontend.

    Spec §7.3 — Client → Server message types.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from session=%s", session.session_id)
        return

    msg_type = data.get("type")

    if msg_type == "ping":
        await manager.send_json(session.session_id, {"type": "pong"})

    elif msg_type == "playback_state":
        await _handle_playback_state(session, data.get("data", {}))

    elif msg_type == "ptt_start":
        _handle_ptt_start(session)

    elif msg_type == "ptt_end":
        await _handle_ptt_end(session)

    elif msg_type == "skip":
        await _handle_skip(session)

    elif msg_type == "feedback":
        sentiment = data.get("sentiment", "")
        await _handle_feedback(session, sentiment)

    elif msg_type == "token_refresh":
        new_token = data.get("access_token", "")
        if new_token:
            session.spotify.set_token(new_token)
            logger.info("Token refreshed: session=%s", session.session_id)

    else:
        logger.warning(
            "Unknown message type: %s from session=%s",
            msg_type, session.session_id,
        )


async def _handle_playback_state(
    session: SessionState, playback_data: dict[str, Any]
) -> None:
    """Process playback_state updates and trigger pre-computation.

    Spec §5.1:
        - Update DJState via Observer
        - When track_ending_soon transitions False→True, trigger the full pipeline
        - Pre-computation guard prevents re-triggering while pipeline runs
    """
    state_updates = process_playback_state(session.dj_state, playback_data)

    if not state_updates:
        return

    # Merge observer updates into session state
    _merge_state(session.dj_state, state_updates)

    # Detect pre-computation trigger: track_ending_soon became True
    track_ending_soon = session.dj_state.get("track_ending_soon", False)
    current_track = session.dj_state.get("current_track")

    if (
        track_ending_soon
        and not session._pre_compute_triggered
        and not session._pipeline_running
        and current_track is not None
    ):
        session._pre_compute_triggered = True
        logger.info(
            "Pre-computation triggered for session=%s track=%s",
            session.session_id,
            current_track.track_name,
        )
        # Run pipeline in background so we don't block message loop
        asyncio.create_task(
            _run_agent_pipeline(session, mode="transition")
        )

    # Reset pre-compute flag when a new track starts (track changed)
    if "previous_tracks" in state_updates:
        session._pre_compute_triggered = False


def _handle_ptt_start(session: SessionState) -> None:
    """Begin buffering PTT audio frames (Spec §5.1, §9)."""
    session.ptt_buffer = bytearray()
    session.ptt_active = True
    logger.info("PTT start: session=%s", session.session_id)


async def _handle_ptt_end(session: SessionState) -> None:
    """Process PTT buffer through Whisper and trigger interrupt pipeline.

    Spec §5.1: buffer → Faster-Whisper → intent classifier → Curator re-route.
    """
    session.ptt_active = False
    audio_bytes = bytes(session.ptt_buffer)
    session.ptt_buffer = bytearray()

    if len(audio_bytes) < 8000:  # < 500ms of 16kHz/16bit/mono
        logger.info("PTT too short: session=%s (%d bytes)", session.session_id, len(audio_bytes))
        return

    logger.info("PTT end: session=%s (%d bytes)", session.session_id, len(audio_bytes))

    # Transcribe (Spec §5.1)
    await manager.send_status(session.session_id, "observer", "Listening...")
    try:
        from echodj.stt.whisper import transcriber
        user_utterance = transcriber.transcribe(audio_bytes)
    except Exception:
        logger.exception("Whisper transcription failed: session=%s", session.session_id)
        user_utterance = None

    if not user_utterance:
        await manager.send_json(session.session_id, {
            "type": "error",
            "message": "Didn't catch that",
            "recoverable": True,
        })
        return

    logger.info("PTT transcribed: %r (session=%s)", user_utterance, session.session_id)

    # Classify intent (Spec §5.4.1 — done inside Curator)
    ptt_updates = process_ptt_result(user_utterance, None)
    _merge_state(session.dj_state, ptt_updates)

    # Run interrupt pipeline
    asyncio.create_task(
        _run_agent_pipeline(session, mode="interrupt")
    )


async def _handle_skip(session: SessionState) -> None:
    """Handle explicit skip button press (Spec §7.3)."""
    ptt_updates = process_ptt_result("skip this", UserIntent.SKIP)
    _merge_state(session.dj_state, ptt_updates)

    asyncio.create_task(
        _run_agent_pipeline(session, mode="interrupt")
    )


async def _handle_feedback(session: SessionState, sentiment: str) -> None:
    """Handle user feedback (Spec §7.3)."""
    intent = (
        UserIntent.POSITIVE_FEEDBACK if sentiment == "positive"
        else UserIntent.NEGATIVE_FEEDBACK
    )
    ptt_updates = process_ptt_result(sentiment, intent)
    _merge_state(session.dj_state, ptt_updates)

    # Feedback doesn't require full pipeline — just log for Memory Manager
    logger.info(
        "Feedback logged: session=%s sentiment=%s",
        session.session_id, sentiment,
    )


async def _handle_binary_message(session: SessionState, data: bytes) -> None:
    """Handle binary WebSocket messages (PTT audio frames).

    Spec §9.2: PCM 16kHz 16-bit mono, ~1-second chunks.
    """
    if session.ptt_active:
        session.ptt_buffer.extend(data)
        logger.debug(
            "PTT audio chunk: session=%s +%d bytes (total=%d)",
            session.session_id, len(data), len(session.ptt_buffer),
        )


# ── Agent Pipeline ───────────────────────────────────────────────────────────

async def _run_agent_pipeline(
    session: SessionState, mode: str = "transition"
) -> None:
    """Invoke the full LangGraph agent pipeline.

    Spec §2.3:
        - "transition": Historian + Discoverer → Curator → Scriptwriter → Vocalizer → Broadcast
        - "interrupt": Curator → Scriptwriter → Vocalizer → Broadcast

    The compiled graph handles node sequencing and parallelism automatically
    via add_edge() topology defined in builder.py.
    """
    if session._pipeline_running:
        logger.warning(
            "Pipeline already running for session=%s, skipping",
            session.session_id,
        )
        return

    if not session.compiled_graph:
        logger.warning(
            "No compiled graph for session=%s, skipping pipeline",
            session.session_id,
        )
        return

    session._pipeline_running = True

    try:
        logger.info(
            "Agent pipeline starting: session=%s mode=%s",
            session.session_id, mode,
        )

        # Send status to frontend
        await manager.send_status(
            session.session_id,
            "observer",
            "Preparing next track..." if mode == "transition" else "Processing command...",
        )

        # Invoke the graph with current state
        config = {
            "configurable": {
                "thread_id": session.session_id,
            }
        }

        result = await session.compiled_graph.ainvoke(
            session.dj_state,
            config=config,
        )

        # Merge graph output back into session state
        if isinstance(result, dict):
            _merge_state(session.dj_state, result)

        # Send trivia to frontend if available (Spec §7.3)
        trivia = session.dj_state.get("trivia_link")
        if trivia and hasattr(trivia, "description"):
            await manager.send_json(session.session_id, {
                "type": "trivia",
                "data": {
                    "link_type": trivia.link_type,
                    "description": trivia.description,
                },
            })

        # Clear PTT state for next interaction
        clear_updates = clear_ptt_state()
        _merge_state(session.dj_state, clear_updates)

        logger.info(
            "Agent pipeline complete: session=%s next_track=%s",
            session.session_id,
            session.dj_state.get("next_track"),
        )

    except Exception:
        logger.exception(
            "Agent pipeline failed: session=%s mode=%s",
            session.session_id, mode,
        )
        await manager.send_error(
            session.session_id,
            "DJ pipeline error — music will continue",
            recoverable=True,
        )

    finally:
        session._pipeline_running = False


# ── State merge utility ──────────────────────────────────────────────────────

def _merge_state(state: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge partial state updates into the accumulated session state.

    Handles the Annotated[list, add] reducer pattern by appending to list
    fields rather than replacing them.
    """
    # Fields that use the `add` reducer (Spec §3.1)
    _APPEND_FIELDS = {"previous_tracks", "messages"}

    for key, value in updates.items():
        if key in _APPEND_FIELDS and isinstance(value, list):
            existing = state.get(key, [])
            if isinstance(existing, list):
                state[key] = existing + value
            else:
                state[key] = value
        else:
            state[key] = value
