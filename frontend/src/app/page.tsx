/**
 * Main DJ Page — The primary EchoDJ interface
 *
 * References:
 *   - Spec §13.1 (Layout)
 *   - Spec §13.2 (Component Specs)
 *   - Spec §7.3 (All message types)
 *   - Spec §8.2 (Ducking)
 */

'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { useSpotifyPlayer } from '@/hooks/useSpotifyPlayer';
import { useWebSocket } from '@/hooks/useWebSocket';
import NowPlaying from '@/components/NowPlaying';
import StatusHUD, { type StatusMessage } from '@/components/StatusHUD';
import TriviaCard, { type TriviaData } from '@/components/TriviaCard';
import ReasoningCard from '@/components/ReasoningCard';
import Visualizer from '@/components/Visualizer';
import PTTButton from '@/components/PTTButton';
import type { PTTState } from '@/hooks/usePTT';
import type { CuratorReasoning, ServerMessage } from '@/lib/types';

export default function DJPage() {
  const router = useRouter();
  const { tokens, isAuthenticated, isLoading } = useAuth();
  const player = useSpotifyPlayer(tokens?.access_token ?? null);
  const ws = useWebSocket(tokens?.access_token ?? null);

  // Phase 5 component state
  const [agentStatus, setAgentStatus] = useState<StatusMessage | null>(null);
  const [trivia, setTrivia] = useState<TriviaData | null>(null);
  const [curatorReasoning, setCuratorReasoning] = useState<CuratorReasoning | null>(null);
  const [isDJActive, setIsDJActive] = useState(false);

  // Redirect to login if not authenticated (only after auth has finished loading)
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, isLoading, router]);

  // Send playback state to backend at 1 Hz (Spec §7.3)
  useEffect(() => {
    if (!player.currentTrack || !player.isPlaying) return;

    const interval = setInterval(() => {
      ws.sendMessage({
        type: 'playback_state',
        data: {
          track_uri: player.currentTrack!.spotify_uri,
          position_ms: player.positionMs,
          duration_ms: player.durationMs,
          is_playing: player.isPlaying,
          track_name: player.currentTrack!.track_name,
          artist_name: player.currentTrack!.artist_name,
          album_art_url: player.currentTrack!.album_art_url ?? '',
        },
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [player.currentTrack, player.isPlaying, player.positionMs, player.durationMs, ws]);

  // Clear trivia and reasoning on track change (Spec §13.2: "persists until next track change")
  useEffect(() => {
    setTrivia(null);
    setCuratorReasoning(null);
  }, [player.currentTrack?.spotify_uri]);

  // Handle all server → client messages (Spec §7.3)
  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case 'duck_start':
        // Spec §8.2: Fade Spotify volume down
        setIsDJActive(true);
        duckVolume(player.setVolume, msg.fade_ms ?? 300);
        break;

      case 'duck_end':
        // Spec §8.2: Fade Spotify volume back up
        setIsDJActive(false);
        restoreVolume(player.setVolume, msg.fade_ms ?? 500);
        break;

      case 'skip_to_next':
        // Spec §5.7: Advance to queued track
        player.skipToNext?.();
        break;

      case 'status':
        // Spec §7.3 + §13.2: Update Status HUD
        if (msg.node && msg.message) {
          setAgentStatus({ node: msg.node, message: msg.message });
        }
        break;

      case 'trivia':
        // Spec §7.3 + §13.2: Show TriviaCard
        if (msg.data) {
          setTrivia({ link_type: msg.data.link_type, description: msg.data.description });
        }
        break;

      case 'curator_reasoning':
        // Graph RAG explainability (Diamantini et al.)
        if (msg.data) {
          setCuratorReasoning(msg.data);
        }
        break;

      case 'now_playing':
        // Server confirms what's playing
        break;

      case 'error':
        console.error('[EchoDJ Server Error]', msg.message, 'recoverable:', msg.recoverable);
        break;

      default:
        break;
    }
  }, [player.setVolume, player.skipToNext]);

  useEffect(() => {
    ws.onMessage(handleMessage);
  }, [ws, handleMessage]);

  // Play DJ audio received as binary WebSocket frame (Spec §8.3)
  useEffect(() => {
    ws.onBinary((data: ArrayBuffer) => {
      const blob = new Blob([data], { type: 'audio/mpeg' });
      const audioUrl = URL.createObjectURL(blob);
      const djAudio = document.getElementById('dj-audio') as HTMLAudioElement;
      if (djAudio) {
        djAudio.src = audioUrl;
        djAudio.play().catch((err) => console.warn('[DJ Audio] Play failed:', err));
        djAudio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          setIsDJActive(false);
        };
      }
    });
  }, [ws]);

  // Map PTT state to agent status
  const handlePTTStateChange = useCallback((state: PTTState) => {
    if (state === 'processing') {
      setAgentStatus({ node: 'observer', message: 'Listening...' });
    } else if (state === 'responding') {
      setAgentStatus({ node: 'curator', message: 'Processing your request...' });
    }
  }, []);

  if (isLoading || !isAuthenticated) return null;

  return (
    <main className="dj-page">
      {/* Status HUD — Spec §13.2 */}
      <StatusHUD
        status={agentStatus}
        connectionState={ws.connectionState}
      />

      {/* Now Playing — Spec §13.2 */}
      <NowPlaying
        track={player.currentTrack}
        progress={player.progress}
        isPlaying={player.isPlaying}
      />

      {/* Trivia Link Card — Spec §13.2 */}
      <TriviaCard trivia={trivia} />

      {/* Curator Reasoning — Graph RAG Explainability */}
      <ReasoningCard reasoning={curatorReasoning} />

      {/* Audio Visualizer — Spec §13.2 */}
      <Visualizer isDJActive={isDJActive} audioElementId="dj-audio" />

      {/* PTT Button — Spec §9, §13.2 */}
      <PTTButton
        ws={ws.socket}
        onStateChange={handlePTTStateChange}
        disabled={!player.currentTrack}
      />

      {/* Player error banner */}
      {player.error && (
        <div className="error-banner" id="player-error">
          {player.error}
        </div>
      )}

      {/* Hidden DJ audio element — Spec §8.1 */}
      {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
      <audio id="dj-audio" preload="none" />
    </main>
  );
}

// ── Ducking Helpers (Spec §8.2) ─────────────────────────────────────────────

/**
 * Fade Spotify volume down for DJ audio.
 * Spec §8.2: Linear fade from 1.0 → 0.2 over fadeMs.
 */
function duckVolume(
  setVolume: (level: number) => Promise<void>,
  fadeMs: number,
) {
  const startVolume = 1.0;
  const targetVolume = 0.2;
  const startTime = performance.now();

  function step(currentTime: number) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / fadeMs, 1.0);
    const volume = startVolume + (targetVolume - startVolume) * progress;
    setVolume(volume);
    if (progress < 1.0) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

/**
 * Fade Spotify volume back up after DJ audio.
 * Spec §8.2: Linear fade from 0.2 → 1.0 over fadeMs.
 */
function restoreVolume(
  setVolume: (level: number) => Promise<void>,
  fadeMs: number,
) {
  const startVolume = 0.2;
  const targetVolume = 1.0;
  const startTime = performance.now();

  function step(currentTime: number) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / fadeMs, 1.0);
    const volume = startVolume + (targetVolume - startVolume) * progress;
    setVolume(volume);
    if (progress < 1.0) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}
