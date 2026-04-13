/**
 * Login page — Spotify OAuth entry point
 *
 * References:
 *   - Spec §11.1 (OAuth 2.0 PKCE flow)
 *   - Spec §1.2 (Spotify Premium required)
 */

'use client';

import { useState } from 'react';
import { initiateSpotifyAuth, getClientId } from '@/lib/spotify-auth';

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Only clear the PKCE verifier, NOT the auth tokens
  // (the main page redirect might bring us here before tokens are hydrated)
  useState(() => {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('spotify_code_verifier');
    }
  });

  const handleLogin = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const clientId = getClientId();
      await initiateSpotifyAuth(clientId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start login');
      setIsLoading(false);
    }
  };

  return (
    <main className="login-page">
      <div className="login-card">
        {/* Branding */}
        <div className="login-card__header">
          <h1 className="login-card__title">
            <span className="login-card__icon">📻</span>
            EchoDJ
          </h1>
          <p className="login-card__subtitle">
            Your AI radio station — powered by your music taste.
          </p>
        </div>

        {/* Login Button */}
        <button
          id="spotify-login-btn"
          className="login-card__button"
          onClick={handleLogin}
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <span className="login-card__spinner" />
              Connecting...
            </>
          ) : (
            <>
              <svg
                className="login-card__spotify-icon"
                viewBox="0 0 24 24"
                fill="currentColor"
                width="24"
                height="24"
              >
                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
              </svg>
              Connect with Spotify
            </>
          )}
        </button>

        {/* Error Display */}
        {error && (
          <p className="login-card__error" id="login-error">
            {error}
          </p>
        )}

        {/* Premium Notice */}
        <p className="login-card__notice">
          Requires Spotify Premium for playback.
        </p>
      </div>
    </main>
  );
}
