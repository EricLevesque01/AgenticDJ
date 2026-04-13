# 📋 EchoDJ Specification Review Notes

**Date:** 2026-04-12  
**Spec Version:** v2.0 ([ECHODJ_SPEC.md](file:///c:/Users/ericl/OneDrive/Desktop/DJv3/ECHODJ_SPEC.md))

---

## Review Summary

The EchoDJ v2.0 specification is **implementation-ready**. It was produced after a thorough review
of the original v1.0 draft, which identified 28 issues (10 blockers, 11 major, 7 minor). All blockers
have been resolved in v2.0. This document captures the key findings, architectural decisions, and
risk areas for the engineering team.

---

## Spec Maturity Assessment

| Dimension | Status | Notes |
|:----------|:-------|:------|
| Data contracts | ✅ Complete | `DJState(TypedDict)` with 20+ typed fields, annotated reducers |
| Interface definitions | ✅ Complete | `LLMProvider` protocol, per-node read/write table |
| Database schema | ✅ Complete | LangGraph `SqliteStore` namespace schema, `ListenerProfile` JSON |
| Latency targets | ✅ Quantified | Per-node budgets for both pipelines (§14) |
| Error handling | ✅ Complete | 14 failure scenarios with detection → behavior → UI (§12) |
| Prompt templates | ✅ Complete | Scriptwriter system prompt, Memory Manager prompt, Intent Classifier |
| WebSocket protocol | ✅ Complete | 12 message types, connection lifecycle (§7) |
| API strategy | ✅ Complete | 6 external APIs with auth, rate limits, timeouts, fallbacks (§6) |
| Testing plan | ✅ Complete | Unit + integration + manual verification (§16) |

---

## Key Architectural Decisions

### 1. No Deprecated Spotify APIs
The following Spotify endpoints are **restricted for new apps** and MUST NOT be used:
- `/v1/recommendations` — replaced by Last.fm/ListenBrainz collaborative filtering
- `/v1/audio-features` — no alternative needed; not used in scoring
- `/v1/audio-analysis` — not needed
- `/v1/artists/{id}/related-artists` — replaced by Last.fm `artist.getSimilar`

### 2. 3-Agent Discovery System
Since Spotify's recommendation engine is unavailable, track discovery uses three data sources:
- **Historian** (GraphRAG): Wikidata SPARQL for trivia links between artists
- **Discoverer** (Collaborative Filtering): Last.fm + ListenBrainz for taste-based candidates
- **Curator** (Merge + Select): Combines both sources, applies session rules, selects final track

### 3. Memory Architecture (No Vector DB)
Long-term memory uses **LangGraph's built-in `SqliteStore`** — not ChromaDB or any external vector DB.
This simplifies the architecture significantly:
- Working memory: `DJState` (current graph execution)
- Session memory: `SqliteSaver` (checkpointing, enables session resume)
- Long-term memory: `SqliteStore` (cross-session `ListenerProfile`)

### 4. Audio Routing
Two independent audio channels in the browser:
- **Spotify**: Web Playback SDK → `player.setVolume()` for ducking
- **DJ Voice**: `<audio>` element → Web Audio API `AnalyserNode` for visualizer

The Spotify SDK does NOT expose its audio through Web Audio API. The visualizer only works on DJ voice.

### 5. LLM Provider Abstraction
All LLM calls go through a `LLMProvider` protocol with a single `generate()` method.
Provider is swappable via `ECHODJ_LLM_PROVIDER` env var (gemini, ollama, openai).

### 6. Pre-computation Strategy
The agent pipeline starts at **75% track progress**, not at track end. This gives ~5.5s of
pre-computation time so the DJ audio is buffered and ready to play immediately when the track ends.
User-perceived gap at track end: < 1s.

---

## Risk Areas

### High Risk
| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Wikidata SPARQL reliability | Trivia links unavailable | 3-query fallback cascade + genre-based fallback (§5.2) |
| GPU VRAM contention | Faster-Whisper + local LLM don't fit | Default to Gemini API; Whisper alone fits easily |
| Spotify token expiry mid-session | Playback stops | Proactive refresh at 50-min mark + auto-reconnect |

### Medium Risk
| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| MusicBrainz rate limiting (1 req/s) | Slow MBID resolution | Session-level MBID cache |
| LLM output quality | DJ liners sound robotic | Few-shot examples in prompt + truncation guardrail |
| PTT audio quality in browser | Transcription fails | 500ms minimum duration, max 10s, ignore noise |

### Low Risk
| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| edge-tts service outage | No DJ voice | Skip DJ break, play next track silently |
| Last.fm API down | Fewer candidates | Fallback to ListenBrainz + Spotify top tracks |

---

## Dependency Matrix

```
Frontend (Next.js 15) ←→ Backend (FastAPI + LangGraph)
                             │
                             ├── Spotify Web API (OAuth, queue, search, top tracks)
                             ├── Wikidata SPARQL (artist trivia)
                             ├── MusicBrainz REST (artist ID resolution)
                             ├── Last.fm API (similar artists/tracks)
                             ├── ListenBrainz API (CF recommendations)
                             ├── edge-tts (text-to-speech)
                             ├── Faster-Whisper (speech-to-text, local GPU)
                             └── LLM Provider (Gemini/Ollama/OpenAI)
```

---

## Changes From v1.0 → v2.0

| Area | v1.0 (Original) | v2.0 (Rewritten) |
|:-----|:-----------------|:------------------|
| State contract | ❌ None | ✅ `DJState(TypedDict)` with 20+ fields |
| Node count | 6 (ambiguous roles) | 7 + Memory Manager (clear responsibilities) |
| Memory | ChromaDB embeddings | LangGraph `SqliteStore` (zero extra deps) |
| Next.js version | "16" (doesn't exist) | 15.x (latest stable) |
| LLM | "Gemini or Llama" | Pluggable `LLMProvider` protocol |
| Spotify auth | Not mentioned | Full OAuth PKCE + token refresh lifecycle |
| WebSocket protocol | Not defined | 12 message types with TypeScript schemas |
| PTT interaction | "Mic button" | Hold-to-talk with PCM format, visual states, debounce |
| Error handling | None | 14 failure scenarios with recovery |
| Latency | "Real-time" | Per-node budgets: 5.5s pre-computed, 4.8s PTT |
| SPARQL queries | None | 3 working query templates with property IDs |
| Prompts | None | Full system prompts for Scriptwriter, Memory Manager, Intent Classifier |
