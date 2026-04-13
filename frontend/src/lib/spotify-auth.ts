/**
 * Spotify OAuth 2.0 Authorization Code Flow with PKCE
 * 
 * Based on Spotify's official example, adapted for Next.js.
 * PKCE challenge generation is done server-side via /api/auth/pkce
 * to work around crypto.subtle being unavailable on http://127.0.0.1.
 *
 * Spotify 2025 redirect_uri rules:
 *   - HTTPS required for non-loopback
 *   - http://127.0.0.1:PORT allowed (loopback exception)
 *   - "localhost" is NOT allowed
 *
 * References:
 *   - Spec §11 (Spotify Authentication & Token Lifecycle)
 *   - Spec §11.2 (Required Scopes)
 *   - https://github.com/spotify/web-api-examples/tree/master/authorization/authorization_code_pkce
 */

// ── Configuration ──────────────────────────────────────────────────────────

const SPOTIFY_AUTH_URL = 'https://accounts.spotify.com/authorize';
const SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token';

// Spotify requires loopback IP (not "localhost") for HTTP redirect URIs
const REDIRECT_URI = 'http://127.0.0.1:3000/callback';

// Spec §11.2: Required scopes
// user-read-email + user-read-private are required by Web Playback SDK
const SCOPES = [
  'streaming',                      // Web Playback SDK
  'user-read-email',                // Required by Web Playback SDK
  'user-read-private',              // Required by Web Playback SDK
  'user-modify-playback-state',     // Queue tracks, skip, volume
  'user-read-playback-state',       // Read current track state
  'user-read-currently-playing',    // Observer node
  'user-top-read',                  // Discoverer (top tracks/artists), cold start
  'user-read-recently-played',      // Fallback history source
].join(' ');

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Initiate the Spotify OAuth PKCE authorization flow.
 * 
 * Fetches PKCE code_verifier and code_challenge from the server-side
 * API route (/api/auth/pkce) to avoid needing crypto.subtle in the browser.
 * Then redirects the user to Spotify's authorization page.
 */
export async function initiateSpotifyAuth(clientId: string): Promise<void> {
  // Generate PKCE pair server-side (bypasses crypto.subtle restriction)
  const pkceResponse = await fetch('/api/auth/pkce');
  if (!pkceResponse.ok) {
    throw new Error('Failed to generate PKCE challenge');
  }
  const { code_verifier, code_challenge } = await pkceResponse.json();

  // Store verifier for the callback to use
  localStorage.setItem('spotify_code_verifier', code_verifier);

  const authUrl = new URL(SPOTIFY_AUTH_URL);
  authUrl.search = new URLSearchParams({
    client_id: clientId,
    response_type: 'code',
    redirect_uri: REDIRECT_URI,
    scope: SCOPES,
    code_challenge_method: 'S256',
    code_challenge: code_challenge,
    show_dialog: 'true',
  }).toString();

  window.location.href = authUrl.toString();
}

/**
 * Exchange the authorization code for access and refresh tokens.
 * Called from the /callback page after Spotify redirects back.
 */
export async function exchangeCodeForTokens(
  code: string,
  clientId: string,
): Promise<{
  access_token: string;
  refresh_token: string;
  expires_in: number;
}> {
  const codeVerifier = localStorage.getItem('spotify_code_verifier');
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
  localStorage.removeItem('spotify_code_verifier');

  return response.json();
}

/**
 * Refresh the access token using the refresh token.
 * Spec §11.3: Triggered proactively at the 50-minute mark.
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
