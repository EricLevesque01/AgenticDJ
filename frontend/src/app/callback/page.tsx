/**
 * OAuth Callback — Exchanges authorization code for tokens
 *
 * References:
 *   - Spec §11.1 (OAuth PKCE callback)
 */

'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { exchangeCodeForTokens, getClientId } from '@/lib/spotify-auth';
import { useAuth } from '@/hooks/useAuth';
import { Suspense } from 'react';

function CallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { setTokens } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const code = searchParams.get('code');
    const authError = searchParams.get('error');

    if (authError) {
      setError(`Spotify authorization failed: ${authError}`);
      return;
    }

    if (!code) {
      setError('No authorization code received.');
      return;
    }

    const exchange = async () => {
      try {
        const clientId = getClientId();
        const data = await exchangeCodeForTokens(code, clientId);
        setTokens(data.access_token, data.refresh_token, data.expires_in);
        router.push('/');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Token exchange failed');
      }
    };

    exchange();
  }, [searchParams, setTokens, router]);

  if (error) {
    return (
      <main className="login-page">
        <div className="login-card">
          <h1 className="login-card__title">Authentication Error</h1>
          <p className="login-card__error">{error}</p>
          <button
            className="login-card__button"
            onClick={() => router.push('/login')}
          >
            Try Again
          </button>
        </div>
      </main>
    );
  }

  return (
    <main className="login-page">
      <div className="login-card">
        <span className="login-card__spinner" />
        <p>Connecting to Spotify...</p>
      </div>
    </main>
  );
}

export default function CallbackPage() {
  return (
    <Suspense fallback={
      <main className="login-page">
        <div className="login-card">
          <span className="login-card__spinner" />
          <p>Loading...</p>
        </div>
      </main>
    }>
      <CallbackContent />
    </Suspense>
  );
}
