/**
 * Spotify OAuth 2.0 Authorization Code Flow with PKCE
 *
 * References:
 *   - Spec §11 (Spotify Authentication & Token Lifecycle)
 *   - Spec §11.2 (Required Scopes)
 *
 * PKCE is used because the client-side Next.js app cannot safely store
 * a client secret (Spec §11.1).
 */

// ── Configuration ──────────────────────────────────────────────────────────

const SPOTIFY_AUTH_URL = 'https://accounts.spotify.com/authorize';
const SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token';

// Must match the redirect URI registered in the Spotify Developer Dashboard
const REDIRECT_URI =
  typeof window !== 'undefined'
    ? `${window.location.origin}/callback`
    : 'http://localhost:3000/callback';

// Spec §11.2: Required scopes
const SCOPES = [
  'streaming',                      // Web Playback SDK
  'user-modify-playback-state',     // Queue tracks, skip, volume
  'user-read-playback-state',       // Read current track state
  'user-read-currently-playing',    // Observer node
  'user-top-read',                  // Discoverer (top tracks/artists), cold start
  'user-read-recently-played',      // Fallback history source
].join(' ');

// ── PKCE Helpers ───────────────────────────────────────────────────────────

/**
 * Generate a random code verifier for PKCE.
 * Must be 43-128 characters of [A-Za-z0-9-._~].
 */
function generateCodeVerifier(length = 64): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
  const values = crypto.getRandomValues(new Uint8Array(length));
  return Array.from(values, (v) => chars[v % chars.length]).join('');
}

/**
 * Generate the code challenge from the verifier using SHA-256.
 */
async function generateCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return btoa(String.fromCharCode(...new Uint8Array(digest)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Initiate the Spotify OAuth PKCE authorization flow.
 * Redirects the user to Spotify's authorization page.
 *
 * @param clientId - The Spotify application client ID
 */
export async function initiateSpotifyAuth(clientId: string): Promise<void> {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = await generateCodeChallenge(codeVerifier);

  // Store verifier for the callback to use
  sessionStorage.setItem('spotify_code_verifier', codeVerifier);

  const params = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: REDIRECT_URI,
    scope: SCOPES,
    code_challenge_method: 'S256',
    code_challenge: codeChallenge,
    show_dialog: 'false',
  });

  window.location.href = `${SPOTIFY_AUTH_URL}?${params.toString()}`;
}

/**
 * Exchange the authorization code for access and refresh tokens.
 * Called from the /callback page after Spotify redirects back.
 *
 * @param code - The authorization code from the callback URL
 * @param clientId - The Spotify application client ID
 * @returns The token response including access_token, refresh_token, and expires_in
 */
export async function exchangeCodeForTokens(
  code: string,
  clientId: string,
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
}> {
  const codeVerifier = sessionStorage.getItem('spotify_code_verifier');
  if (!codeVerifier) {
    throw new Error('No code verifier found. Please restart the login flow.');
  }

  const response = await fetch(SPOTIFY_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: clientId,
      grant_type: 'authorization_code',
      code,
      redirect_uri: REDIRECT_URI,
      code_verifier: codeVerifier,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Token exchange failed: ${error.error_description || error.error}`);
  }

  // Clean up
  sessionStorage.removeItem('spotify_code_verifier');

  return response.json();
}

/**
 * Refresh the access token using the refresh token.
 * Spec §11.3: Triggered proactively at the 50-minute mark.
 *
 * @param refreshToken - The refresh token
 * @param clientId - The Spotify application client ID
 * @returns New token data
 */
export async function refreshAccessToken(
  refreshToken: string,
  clientId: string,
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
}> {
  const response = await fetch(SPOTIFY_TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: clientId,
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Token refresh failed: ${error.error_description || error.error}`);
  }

  return response.json();
}

/**
 * Get the Spotify client ID from environment variables.
 * This is safe to expose client-side (it's not a secret).
 */
export function getClientId(): string {
  const clientId = process.env.NEXT_PUBLIC_SPOTIFY_CLIENT_ID;
  if (!clientId) {
    throw new Error(
      'NEXT_PUBLIC_SPOTIFY_CLIENT_ID is not set. ' +
      'Add it to your .env.local file.'
    );
  }
  return clientId;
}
