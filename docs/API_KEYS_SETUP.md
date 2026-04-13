# 🔑 EchoDJ — API Keys Setup Guide

## Required API Keys

### 1. Spotify (REQUIRED)

**What you need:** `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Log in with your **Spotify Premium** account
3. Click "Create App"
   - App name: `EchoDJ`
   - App description: `Agentic AI radio station`
   - Redirect URI: `http://localhost:3000/callback`
   - APIs used: Check **"Web Playback SDK"** and **"Web API"**
4. Copy the **Client ID** and **Client Secret**
5. Paste them into both:
   - `DJv3/.env` → `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`
   - `DJv3/frontend/.env.local` → `NEXT_PUBLIC_SPOTIFY_CLIENT_ID` (same client ID)

> ⚠️ **Spotify Premium is required.** The Web Playback SDK does not work with free accounts.

**Required Scopes** (configured automatically in code):
- `streaming` — Web Playback SDK
- `user-modify-playback-state` — Queue tracks, skip, volume
- `user-read-playback-state` — Read current track state
- `user-read-currently-playing` — Observer node
- `user-top-read` — Discoverer (top tracks/artists), cold start
- `user-read-recently-played` — Fallback history source

---

### 2. Last.fm (REQUIRED)

**What you need:** `LASTFM_API_KEY`

1. Go to [Last.fm API Account Creation](https://www.last.fm/api/account/create)
2. Fill in the form:
   - Application name: `EchoDJ`
   - Application description: `AI radio station`
   - Callback URL: (leave blank)
3. Copy the **API Key**

---

### 3. LLM Provider (REQUIRED — pick one)

#### Option A: Google Gemini (Recommended)
**What you need:** `GEMINI_API_KEY`

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key

Set in `.env`:
```
ECHODJ_LLM_PROVIDER=gemini
ECHODJ_LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your-key-here
```

#### Option B: Ollama (Local, no API key needed)
1. Install [Ollama](https://ollama.com/)
2. Pull a model: `ollama pull gemma3:4b`
3. Start Ollama: `ollama serve`

Set in `.env`:
```
ECHODJ_LLM_PROVIDER=ollama
ECHODJ_LLM_MODEL=gemma3:4b
OLLAMA_BASE_URL=http://localhost:11434
```

> **Ollama advantage:** Fully local — no API key needed, no internet required for LLM calls. Good for privacy-conscious users or offline demos.

---

### 4. ListenBrainz (OPTIONAL — enhances recommendations)

**What you need:** `LISTENBRAINZ_USER_TOKEN`

1. Create an account at [ListenBrainz](https://listenbrainz.org/)
2. Link your Spotify account for scrobbling (Settings → Music Services)
3. Go to [Settings](https://listenbrainz.org/settings/) and copy your **User Token**

> **Note:** ListenBrainz uses its own token system — it cannot be combined with Spotify OAuth. If you don't have a ListenBrainz account, the Discoverer will gracefully skip this data source and use Last.fm + Spotify top tracks instead.

---

### 5. MusicBrainz (No key needed)

MusicBrainz is a free, open API. No registration required. However, their API requires a **contact email** in the User-Agent header (to contact you if your app misbehaves).

Set in `.env`:
```
MUSICBRAINZ_CONTACT_EMAIL=your-real-email@example.com
```

---

## Complete `.env` Setup

### Root: `DJv3/.env`

```bash
cp .env.example .env
```

Then fill in:
- `SPOTIFY_CLIENT_ID` — from Spotify Dashboard
- `SPOTIFY_CLIENT_SECRET` — from Spotify Dashboard
- `LASTFM_API_KEY` — from Last.fm
- `GEMINI_API_KEY` — from Google AI Studio (or use Ollama)
- `MUSICBRAINZ_CONTACT_EMAIL` — your real email

### Frontend: `DJv3/frontend/.env.local`

```bash
cp frontend/.env.local.example frontend/.env.local
```

Then fill in:
- `NEXT_PUBLIC_SPOTIFY_CLIENT_ID` — same as `SPOTIFY_CLIENT_ID` above

---

## Verification

After setting up your `.env`, verify each key:

### Spotify
```bash
# Should return your user profile
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.spotify.com/v1/me
```

### Last.fm
```bash
# Should return similar artists
curl "https://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist=Radiohead&api_key=YOUR_KEY&format=json&limit=3"
```

### Gemini
```bash
# Should return a response
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
```
