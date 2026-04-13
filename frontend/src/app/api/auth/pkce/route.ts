/**
 * /api/auth/pkce — Server-side PKCE code challenge generation
 * 
 * Generates the code_verifier and code_challenge on the server using Node.js
 * crypto, bypassing the browser's crypto.subtle secure-context restriction.
 * 
 * This is necessary because:
 *   - Spotify requires http://127.0.0.1 as redirect_uri (no localhost)
 *   - Browsers only expose crypto.subtle on HTTPS or http://localhost
 *   - http://127.0.0.1 is NOT considered a secure context by browsers
 *   - Node.js crypto works regardless of browser context
 */

import { NextResponse } from 'next/server';
import crypto from 'crypto';

export async function GET() {
  // Generate code verifier (43-128 chars of [A-Za-z0-9])
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  const randomBytes = crypto.randomBytes(64);
  const codeVerifier = Array.from(randomBytes, (b) => possible[b % possible.length]).join('');

  // Generate code challenge (SHA-256 hash, base64url-encoded)
  const hash = crypto.createHash('sha256').update(codeVerifier).digest();
  const codeChallenge = hash
    .toString('base64')
    .replace(/=/g, '')
    .replace(/\+/g, '-')
    .replace(/\//g, '_');

  return NextResponse.json({
    code_verifier: codeVerifier,
    code_challenge: codeChallenge,
  });
}
