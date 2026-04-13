/**
 * Shared TypeScript types for the EchoDJ frontend.
 * Mirrors the backend data models from Spec §3.2.
 * Also defines the WebSocket message protocol from Spec §7.3.
 */

// ── Spotify Track ──────────────────────────────────────────────────────────

export interface SpotifyTrack {
  spotify_uri: string;
  track_name: string;
  artist_name: string;
  album_name: string;
  album_art_url: string | null;
  duration_ms: number;
  genres: string[];
}

// ── Trivia Link ────────────────────────────────────────────────────────────

export interface TriviaLink {
  link_type: string;
  description: string;
}

// ── Playback State ─────────────────────────────────────────────────────────

export interface PlaybackState {
  track_uri: string;
  position_ms: number;
  duration_ms: number;
  is_playing: boolean;
  track_name: string;
  artist_name: string;
  album_art_url: string;
}

// ── WebSocket Messages: Client → Server ────────────────────────────────────
// Spec §7.3

export type ClientMessage =
  | { type: 'playback_state'; data: PlaybackState }
  | { type: 'ptt_start' }
  | { type: 'ptt_end' }
  | { type: 'skip' }
  | { type: 'feedback'; sentiment: 'positive' | 'negative' }
  | { type: 'ping' }
  | { type: 'token_refresh'; access_token: string };

// ── Curator Reasoning (Graph RAG explainability) ───────────────────────────
// Paper §4.4: Structured explanation of why a track was selected

export interface CuratorReasoning {
  reasoning: string;
  ranking?: Array<{
    track: string;
    artist: string;
    score: number;
    reason: string;
  }>;
}

// ── WebSocket Messages: Server → Client ────────────────────────────────────
// Spec §7.3

export type ServerMessage =
  | { type: 'connected'; session_id: string }
  | { type: 'pong' }
  | { type: 'duck_start'; fade_ms: number }
  | { type: 'duck_end'; fade_ms: number }
  | { type: 'skip_to_next' }
  | { type: 'now_playing'; data: SpotifyTrack }
  | { type: 'status'; node: string; message: string }
  | { type: 'trivia'; data: TriviaLink }
  | { type: 'curator_reasoning'; data: CuratorReasoning }
  | { type: 'error'; message: string; recoverable: boolean };

// ── Auth ───────────────────────────────────────────────────────────────────

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_at: number; // Unix timestamp in ms
}

// ── PTT States (Spec §9.3) ────────────────────────────────────────────────

export type PTTState = 'idle' | 'recording' | 'processing' | 'responding';

// ── Connection State ───────────────────────────────────────────────────────

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
