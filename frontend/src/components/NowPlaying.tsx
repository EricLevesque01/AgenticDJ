/**
 * NowPlaying — Album art, track info, and progress bar
 *
 * References:
 *   - Spec §13.2 (Now Playing component spec)
 *   - Album art: 256×256px from Spotify API
 *   - Inter font family
 *   - Progress bar: thin line, updated at 1 Hz
 */

'use client';

import type { SpotifyTrack } from '@/lib/types';

interface NowPlayingProps {
  track: SpotifyTrack | null;
  progress: number; // 0.0 – 1.0
  isPlaying: boolean;
}

export default function NowPlaying({ track, progress, isPlaying }: NowPlayingProps) {
  if (!track) {
    return (
      <div className="now-playing now-playing--empty">
        <div className="now-playing__art-placeholder">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            width="64"
            height="64"
          >
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
          <p>Waiting for playback...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="now-playing" id="now-playing">
      {/* Album Art */}
      <div className="now-playing__art-container">
        {track.album_art_url ? (
          <img
            src={track.album_art_url}
            alt={`${track.album_name} album art`}
            className="now-playing__art"
            width={256}
            height={256}
          />
        ) : (
          <div className="now-playing__art-placeholder">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              width="64"
              height="64"
            >
              <path d="M9 18V5l12-2v13" />
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="16" r="3" />
            </svg>
          </div>
        )}

        {/* Spinning indicator when playing */}
        {isPlaying && <div className="now-playing__pulse" />}
      </div>

      {/* Track Info */}
      <div className="now-playing__info">
        <h2 className="now-playing__track" id="track-name">{track.track_name}</h2>
        <p className="now-playing__artist" id="artist-name">{track.artist_name}</p>
        <p className="now-playing__album">{track.album_name}</p>
      </div>

      {/* Progress Bar */}
      <div className="now-playing__progress-container" id="progress-bar">
        <div
          className="now-playing__progress-bar"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </div>
  );
}
