"""
EchoDJ LangGraph State Contract
================================
The single source of truth for data flowing between all 7 LangGraph nodes.
Every field has a defined owner (writer) and consumers (readers).

References:
    - Spec §3.1 (Core State Schema)
    - Spec §3.3 (Node Read/Write Permissions)
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from echodj.models import CandidateTrack, SpotifyTrack, TriviaLink, UserIntent


class DJState(TypedDict, total=False):
    """Core state schema for the EchoDJ LangGraph graph.

    Each field documents which node writes it (W) and which nodes read it (R).
    See Spec §3.3 for the full permissions matrix.

    Fields use LangGraph annotated reducers where appropriate:
    - ``Annotated[list[...], add]`` for append-only lists
    - ``Annotated[list[BaseMessage], add_messages]`` for message history
    """

    # ── Observer Writes ──────────────────────────────────────────────────
    # W: Observer | R: Historian, Discoverer, Curator, Scriptwriter, Memory Mgr
    current_track: SpotifyTrack | None

    # W: Observer | R: Historian, Discoverer, Curator, Scriptwriter, Memory Mgr
    # Rolling history (max 20 tracks). Uses `add` reducer for append semantics.
    previous_tracks: Annotated[list[SpotifyTrack], add]

    # W: Observer | R: (internal Observer logic)
    playback_progress: float      # 0.0 – 1.0

    # W: Observer | R: (triggers pre-computation)
    track_ending_soon: bool       # True when progress > 0.75

    # W: Observer | R: Curator, Scriptwriter
    user_utterance: str | None    # Faster-Whisper transcription of PTT audio

    # W: Observer | R: Curator
    user_intent: UserIntent | None  # Parsed from user_utterance via LLM

    # ── Historian Writes ─────────────────────────────────────────────────
    # W: Historian | R: Curator, Scriptwriter, Memory Mgr
    trivia_link: TriviaLink | None

    # W: Historian | R: Curator
    trivia_confidence: float      # 0.0 – 1.0

    # W: Historian | R: Curator, Scriptwriter
    # Graph RAG (Diamantini et al.): ALL known relationships, not just best one.
    trivia_context: list[TriviaLink]

    # ── Discoverer Writes ────────────────────────────────────────────────
    # W: Discoverer | R: Curator
    taste_candidates: list[CandidateTrack]  # Ranked by taste match, max 20

    # ── Curator Writes ───────────────────────────────────────────────────
    # W: Curator | R: Scriptwriter, Memory Mgr
    next_track: SpotifyTrack | None

    # W: Curator | R: (logging/debugging)
    curator_reasoning: str

    # W: Curator | R: Broadcast
    queue_action: Literal["play_next", "interrupt", "continue"]

    # ── Scriptwriter Writes ──────────────────────────────────────────────
    # W: Scriptwriter | R: Vocalizer
    script_text: str

    # W: Scriptwriter | R: (validation)
    script_word_count: int

    # ── Vocalizer Writes ─────────────────────────────────────────────────
    # W: Vocalizer | R: Broadcast
    audio_buffer: bytes | None

    # W: Vocalizer | R: Broadcast
    audio_duration_ms: int

    # ── Broadcast Writes ─────────────────────────────────────────────────
    # W: Broadcast | R: (frontend via WebSocket)
    ducking_active: bool

    # ── Session Context ──────────────────────────────────────────────────
    session_id: str
    user_id: str

    # W: Scriptwriter (R/W) | R: Memory Mgr
    # Prevents repetition of trivia links within and across sessions.
    discussed_trivia: list[str]

    # W: Memory Mgr | R: Curator, Scriptwriter
    session_vibe: str             # "chill", "energetic", etc.

    # W: Curator (W) | R: Memory Mgr (R/W)
    # Triggers Memory Manager when >= 10.
    tracks_since_last_memory_update: int

    # ── Skip Tracking ────────────────────────────────────────────────────
    # W: Observer | R: Curator, Memory Mgr
    # Track URIs that were skipped (progress < 30% at change) — negative signal.
    skipped_tracks: Annotated[list[str], add]

    # W: Observer | R: Curator (triggers segment override logic)
    skip_detected: bool

    # ── Pre-curated Segment ──────────────────────────────────────────────
    # W: Curator | R: Scriptwriter, Curator (for continuity)
    # Active segment label if the Curator is running a themed playlist.
    active_segment_label: str | None

    # LangGraph message history for debugging and checkpointing.
    messages: Annotated[list[BaseMessage], add_messages]
