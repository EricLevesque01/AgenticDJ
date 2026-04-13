"""
Scriptwriter Agent — DJ Liner Generation Node
==============================================
Generates a personality-driven spoken liner bridging the previous
track to the next track, using the trivia link and curator reasoning.

Graph RAG explainability (Diamantini et al.):
    The Scriptwriter receives the Curator's structured reasoning
    (which includes KG-derived connections) and weaves it into
    natural DJ patter — making explainability feel like insider knowledge.

References:
    - Paper §4.4 (Explainable ranking)
    - Spec §5.5 (Scriptwriter Agent)
    - Spec §5.5 (System Prompt — verbatim from spec)
    - Latency target: < 1s
    - Word count: 40–55 words (15–20s spoken)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Awaitable

from echodj.llm.provider import LLMProvider
from echodj.state import DJState

logger = logging.getLogger(__name__)

# Spec §5.5: Word count constraints
_MIN_WORDS = 40
_MAX_WORDS = 60
_TARGET_MAX_WORDS = 55

# Spec §5.5: Scriptwriter persona — verbatim from spec
_SYSTEM_PROMPT = """\
You are the voice of EchoDJ — a late-night radio DJ with deep music knowledge
and a warm, conversational style. Think: a knowledgeable Zane Lowe meets the
intimacy of a college radio host at 2am.

RULES:
1. Never cite raw data. No "120 BPM", no "released in 1994", no Wikipedia-style facts.
2. Speak in second person ("you're going to love this", "this next one").
3. Bridge between the PREVIOUS track and the NEXT track using the trivia link.
4. Keep liners between 40–55 words (15–20 seconds when read aloud).
5. Never repeat trivia from previous breaks (check discussed_trivia).
6. If no trivia is available, introduce the next song with genuine enthusiasm.
7. If responding to a user command, acknowledge it naturally first.
8. Match the session vibe in your tone (chill = mellow, energetic = excited).
9. If curator reasoning mentions a knowledge graph connection, weave it in
   naturally — make it sound like insider knowledge, not a database lookup."""

# Spec §5.5 + Graph RAG: User prompt template with curator reasoning
_USER_PROMPT_TEMPLATE = """\
Previous track: "{prev_track}" by {prev_artist}
Next track: "{next_track}" by {next_artist}
Trivia link: {trivia}
Selection reasoning: {curator_reasoning}
Session vibe: {vibe}
User command: {command}
Already discussed: {discussed}

Write your DJ liner."""


async def scriptwriter_node(
    state: DJState,
    llm: LLMProvider,
    on_status: Callable[[str], Awaitable[None]] | None = None,
) -> dict:
    """LangGraph node: generate a spoken DJ liner.

    Input reads (Spec §3.3 + Graph RAG):
        - trivia_link, current_track, next_track
        - curator_reasoning (Graph RAG: explains WHY this track)
        - discussed_trivia, user_utterance, session_vibe

    Output writes (Spec §3.3):
        - script_text, script_word_count
        - discussed_trivia (appended with new trivia description)
    """
    current = state.get("current_track")
    next_track = state.get("next_track")
    trivia_link = state.get("trivia_link")
    curator_reasoning = state.get("curator_reasoning", "")
    discussed = list(state.get("discussed_trivia", []))
    user_utterance = state.get("user_utterance")
    session_vibe = state.get("session_vibe", "moderate")

    # Build user prompt
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        prev_track=current.track_name if current else "the last song",
        prev_artist=current.artist_name if current else "the artist",
        next_track=next_track.track_name if next_track else "the next song",
        next_artist=next_track.artist_name if next_track else "the next artist",
        trivia=trivia_link.description if trivia_link else "None available",
        curator_reasoning=curator_reasoning if curator_reasoning else "Standard taste-based selection",
        vibe=session_vibe,
        command=user_utterance if user_utterance else "None",
        discussed=", ".join(discussed) if discussed else "None",
    )

    # Generate liner
    if on_status:
        next_name = next_track.track_name if next_track else "next track"
        await on_status(f"Writing liner for {next_name}...")
    script = await llm.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # Apply guardrails (Spec §5.5)
    script = _apply_guardrails(script, next_track)

    word_count = len(script.split())
    logger.info("Scriptwriter: %d words generated", word_count)

    # Append trivia to discussed list to prevent repetition (Spec §5.5)
    new_discussed = discussed.copy()
    if trivia_link and trivia_link.description not in new_discussed:
        new_discussed.append(trivia_link.description)

    return {
        "script_text": script,
        "script_word_count": word_count,
        "discussed_trivia": new_discussed,
    }


def _apply_guardrails(script: str, next_track) -> str:  # type: ignore[no-untyped-def]
    """Apply word count and content guardrails.

    Spec §5.5:
        - If word_count > 60: truncate to first 55 words, append "..."
        - If empty/nonsensical: use fallback template
    """
    script = script.strip()

    # Fallback for empty or very short output
    if not script or len(script) < 10:
        if next_track:
            return (
                f"Here's {next_track.track_name} by {next_track.artist_name}."
            )
        return "Here's the next one."

    # Truncate long output — do NOT re-prompt (latency budget)
    words = script.split()
    if len(words) > _MAX_WORDS:
        logger.info(
            "Scriptwriter: truncating from %d to %d words",
            len(words), _TARGET_MAX_WORDS,
        )
        script = " ".join(words[:_TARGET_MAX_WORDS]) + "..."

    return script
