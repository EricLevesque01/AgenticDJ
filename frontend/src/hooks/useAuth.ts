/**
 * useAuth — Authentication state management hook
 *
 * Manages Spotify OAuth tokens, handles proactive refresh,
 * and provides auth state to the application.
 *
 * References:
 *   - Spec §11.3 (Token Management — refresh at 50-minute mark)
 *   - Spec §11.4 (Error Recovery)
 */

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { refreshAccessToken, getClientId } from '@/lib/spotify-auth';
import type { AuthTokens } from '@/lib/types';

const STORAGE_KEY = 'echodj_auth';
// Spec §11.3: Refresh at 50-minute mark (10-minute buffer on 60-min expiry)
const REFRESH_BUFFER_MS = 10 * 60 * 1000; // 10 minutes

export interface UseAuthReturn {
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  setTokens: (accessToken: string, refreshToken: string, expiresIn: number) => void;
  clearTokens: () => void;
}

export function useAuth(): UseAuthReturn {
  const [tokens, setTokensState] = useState<AuthTokens | null>(null);
  const refreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load tokens from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed: AuthTokens = JSON.parse(stored);
        // Check if tokens are still valid
        if (parsed.expires_at > Date.now()) {
          setTokensState(parsed);
        } else {
          localStorage.removeItem(STORAGE_KEY);
        }
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Set up proactive token refresh
  useEffect(() => {
    if (!tokens) return;

    const timeUntilRefresh = tokens.expires_at - Date.now() - REFRESH_BUFFER_MS;

    if (timeUntilRefresh <= 0) {
      // Token needs immediate refresh
      handleRefresh(tokens.refresh_token);
      return;
    }

    // Schedule refresh
    refreshTimerRef.current = setTimeout(() => {
      handleRefresh(tokens.refresh_token);
    }, timeUntilRefresh);

    return () => {
      if (refreshTimerRef.current) {
        clearTimeout(refreshTimerRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tokens?.expires_at]);

  const handleRefresh = useCallback(async (refreshToken: string) => {
    try {
      // Spec §11.3: Use server-side route so SPOTIFY_CLIENT_SECRET stays on server
      const res = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (!res.ok) {
        throw new Error(`Refresh failed: ${res.status}`);
      }

      const data = await res.json();
      setTokens(
        data.access_token,
        data.refresh_token || refreshToken,
        data.expires_in,
      );
    } catch (error) {
      console.error('Server-side token refresh failed:', error);
      // Fallback to PKCE client-side refresh
      try {
        const clientId = getClientId();
        const data = await refreshAccessToken(refreshToken, clientId);
        setTokens(
          data.access_token,
          data.refresh_token || refreshToken,
          data.expires_in,
        );
      } catch (fallbackError) {
        console.error('PKCE fallback refresh also failed:', fallbackError);
        // Spec §11.4: If all refresh fails, redirect to login
        clearTokens();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setTokens = useCallback(
    (accessToken: string, refreshToken: string, expiresIn: number) => {
      const authTokens: AuthTokens = {
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_at: Date.now() + expiresIn * 1000,
      };
      setTokensState(authTokens);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(authTokens));
    },
    [],
  );

  const clearTokens = useCallback(() => {
    setTokensState(null);
    localStorage.removeItem(STORAGE_KEY);
    if (refreshTimerRef.current) {
      clearTimeout(refreshTimerRef.current);
    }
  }, []);

  return {
    tokens,
    isAuthenticated: tokens !== null && tokens.expires_at > Date.now(),
    setTokens,
    clearTokens,
  };
}
