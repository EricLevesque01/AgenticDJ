/**
 * Spotify Token Refresh — Server-Side Route
 * ==========================================
 * Handles token refresh using the client secret on the server side,
 * so the SPOTIFY_CLIENT_SECRET never touches the browser.
 *
 * Spec §11.3: "A Next.js API route (/api/auth/refresh) handles
 *              silent token refresh server-side"
 */

import { NextRequest, NextResponse } from 'next/server';

const SPOTIFY_TOKEN_URL = 'https://accounts.spotify.com/api/token';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const refreshToken = body.refresh_token;

    if (!refreshToken) {
      return NextResponse.json(
        { error: 'Missing refresh_token' },
        { status: 400 }
      );
    }

    const clientId = process.env.NEXT_PUBLIC_SPOTIFY_CLIENT_ID;
    const clientSecret = process.env.SPOTIFY_CLIENT_SECRET;

    if (!clientId || !clientSecret) {
      return NextResponse.json(
        { error: 'Server configuration error: missing Spotify credentials' },
        { status: 500 }
      );
    }

    // Exchange refresh token for new access token (Spec §11.3)
    const params = new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
    });

    const authHeader = Buffer.from(`${clientId}:${clientSecret}`).toString('base64');

    const response = await fetch(SPOTIFY_TOKEN_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${authHeader}`,
      },
      body: params.toString(),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Spotify token refresh failed:', response.status, errorText);
      return NextResponse.json(
        { error: 'Token refresh failed', details: errorText },
        { status: response.status }
      );
    }

    const data = await response.json();

    // Return the new tokens to the client
    return NextResponse.json({
      access_token: data.access_token,
      expires_in: data.expires_in,
      // Spotify may return a new refresh token
      refresh_token: data.refresh_token || refreshToken,
    });

  } catch (error) {
    console.error('Token refresh error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
