/**
 * useSpotifyPlayer — Spotify Web Playback SDK hook
 *
 * Manages the Spotify Web Playback SDK lifecycle: loads the SDK,
 * creates a player instance, tracks playback state, and provides
 * volume control for audio ducking.
 *
 * References:
 *   - Spec §5.1 (Observer — playback state monitoring)
 *   - Spec §8.1 (Audio Routing — SDK volume control)
 *   - Spec §1.2 (Requires Spotify Premium)
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { SpotifyTrack } from '@/lib/types';

// Note: Window.Spotify and Window.onSpotifyWebPlaybackSDKReady are declared
// by @types/spotify-web-playback-sdk

export interface UseSpotifyPlayerReturn {
  isReady: boolean;
  isActive: boolean;
  currentTrack: SpotifyTrack | null;
  isPlaying: boolean;
  positionMs: number;
  durationMs: number;
  progress: number; // 0.0 – 1.0
  deviceId: string | null;
  setVolume: (level: number) => Promise<void>;
  togglePlay: () => Promise<void>;
  skipToNext: () => Promise<void>;
  error: string | null;
}

export function useSpotifyPlayer(accessToken: string | null): UseSpotifyPlayerReturn {
  const [isReady, setIsReady] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [currentTrack, setCurrentTrack] = useState<SpotifyTrack | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [positionMs, setPositionMs] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const playerRef = useRef<Spotify.Player | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Calculate progress (0.0 – 1.0)
  const progress = durationMs > 0 ? positionMs / durationMs : 0;

  // Load the Spotify Web Playback SDK script
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (window.Spotify) return; // Already loaded

    const script = document.createElement('script');
    script.src = 'https://sdk.scdn.co/spotify-player.js';
    script.async = true;
    document.body.appendChild(script);

    return () => {
      // Don't remove the script on unmount — SDK should persist
    };
  }, []);

  // Initialize the player when SDK is ready and we have a token
  useEffect(() => {
    if (!accessToken) return;

    const initPlayer = () => {
      if (!window.Spotify) return;

      const player = new window.Spotify.Player({
        name: 'EchoDJ',
        getOAuthToken: (cb) => cb(accessToken),
        volume: 1.0,
      });

      // Ready — device is available for playback
      player.addListener('ready', ({ device_id }) => {
        console.log('[Spotify] Player ready, device:', device_id);
        setDeviceId(device_id);
        setIsReady(true);
        setError(null);

        // Transfer playback to this device
        fetch('https://api.spotify.com/v1/me/player', {
          method: 'PUT',
          headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            device_ids: [device_id],
            play: false,
          }),
        }).catch((err) => console.warn('[Spotify] Transfer failed:', err));
      });

      // Not ready — device went offline
      player.addListener('not_ready', ({ device_id }) => {
        console.warn('[Spotify] Player not ready:', device_id);
        setIsReady(false);
      });

      // Playback state changed (Spec §5.1)
      player.addListener('player_state_changed', (state) => {
        if (!state) {
          setIsActive(false);
          return;
        }

        setIsActive(true);
        setIsPlaying(!state.paused);
        setPositionMs(state.position);
        setDurationMs(state.duration);

        const track = state.track_window.current_track;
        if (track) {
          setCurrentTrack({
            spotify_uri: track.uri,
            track_name: track.name,
            artist_name: track.artists.map((a) => a.name).join(', '),
            album_name: track.album.name,
            album_art_url: track.album.images?.[0]?.url ?? null,
            duration_ms: state.duration,
            genres: [], // SDK doesn't provide genres
          });
        }
      });

      // Error handling
      player.addListener('initialization_error', ({ message }) => {
        console.error('[Spotify] Init error:', message);
        setError(`Initialization failed: ${message}`);
      });

      player.addListener('authentication_error', ({ message }) => {
        console.error('[Spotify] Auth error:', message);
        setError(`Authentication failed: ${message}`);
      });

      player.addListener('account_error', ({ message }) => {
        console.error('[Spotify] Account error:', message);
        setError(`Account error (Premium required): ${message}`);
      });

      player.addListener('playback_error', ({ message }) => {
        console.error('[Spotify] Playback error:', message);
        setError(`Playback error: ${message}`);
      });

      player.connect();
      playerRef.current = player;
    };

    // SDK might already be loaded
    if (window.Spotify) {
      initPlayer();
    } else {
      window.onSpotifyWebPlaybackSDKReady = initPlayer;
    }

    return () => {
      if (playerRef.current) {
        playerRef.current.disconnect();
        playerRef.current = null;
      }
    };
  }, [accessToken]);

  // Update position at ~1 Hz when playing (Spec §5.1: progress tracking)
  useEffect(() => {
    if (isPlaying && playerRef.current) {
      progressIntervalRef.current = setInterval(async () => {
        const state = await playerRef.current?.getCurrentState();
        if (state) {
          setPositionMs(state.position);
        }
      }, 1000);
    } else if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }

    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, [isPlaying]);

  // Volume control (Spec §8.2: ducking via player.setVolume)
  const setVolume = useCallback(async (level: number) => {
    if (playerRef.current) {
      await playerRef.current.setVolume(Math.max(0, Math.min(1, level)));
    }
  }, []);

  const togglePlay = useCallback(async () => {
    if (playerRef.current) {
      await playerRef.current.togglePlay();
    }
  }, []);

  const skipToNext = useCallback(async () => {
    if (playerRef.current) {
      await playerRef.current.nextTrack();
    }
  }, []);

  return {
    isReady,
    isActive,
    currentTrack,
    isPlaying,
    positionMs,
    durationMs,
    progress,
    deviceId,
    setVolume,
    togglePlay,
    skipToNext,
    error,
  };
}
