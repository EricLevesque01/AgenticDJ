# 🛠️ EchoDJ — Development Guide

## Environment Setup

### Backend (Python)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -e ".[dev]"
```

### Frontend (Next.js)

```bash
cd frontend
npm install
```

### Environment Variables

1. Copy the template: `cp .env.example .env`
2. Fill in your API keys (see [API_KEYS_SETUP.md](./API_KEYS_SETUP.md))
3. For the frontend: `cp frontend/.env.local.example frontend/.env.local`
4. Set `NEXT_PUBLIC_SPOTIFY_CLIENT_ID` in `frontend/.env.local`

---

## Running

### Backend

```bash
cd backend
.venv\Scripts\activate
uvicorn echodj.server:app --reload --port 8000
```

The backend serves:
- `GET /health` — Health check
- `WebSocket /ws?token={access_token}` — Primary real-time connection

### Frontend

```bash
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

---

## Testing

### Backend Tests

```bash
cd backend
.venv\Scripts\activate
pytest tests/ -v              # All tests
pytest tests/ -v --tb=short   # Compact output
pytest tests/ -v -k "model"   # Filter by keyword
pytest tests/ --cov=echodj    # With coverage
```

### Linting

```bash
cd backend
ruff check echodj/ tests/     # Check for issues
ruff check echodj/ tests/ --fix  # Auto-fix
```

### Frontend Type Checking

```bash
cd frontend
npm run build                 # Includes TypeScript type check
```

---

## Project Structure

```
DJv3/
├── backend/                  # Python FastAPI + LangGraph
│   ├── echodj/               # Main package
│   │   ├── config.py         # Pydantic settings
│   │   ├── models.py         # Data models (SpotifyTrack, TriviaLink, etc.)
│   │   ├── state.py          # DJState TypedDict (LangGraph state contract)
│   │   ├── server.py         # FastAPI app + WebSocket
│   │   ├── graph/            # LangGraph nodes (7 agents + memory manager)
│   │   ├── services/         # External API clients
│   │   ├── llm/              # LLM provider abstraction
│   │   └── stt/              # Speech-to-text (Faster-Whisper)
│   └── tests/                # pytest test suite
│
├── frontend/                 # Next.js 15 (App Router)
│   └── src/
│       ├── app/              # Pages (login, callback, main DJ)
│       ├── components/       # React components
│       ├── hooks/            # Custom hooks (auth, player, WebSocket)
│       └── lib/              # Utilities and types
│
├── docs/                     # Documentation
└── ECHODJ_SPEC.md            # Product specification
```

---

## WebSocket Testing

You can test the WebSocket connection using `websocat` or a browser console:

```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8000/ws?token=test-token');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onopen = () => ws.send(JSON.stringify({ type: 'ping' }));
```

---

## Development Workflow

1. **Write code** in the relevant module
2. **Write tests** alongside the code
3. **Run `pytest` and `ruff`** before committing
4. **Run `npm run build`** to verify frontend types
5. **Update PHASE_NOTES.md** with decisions and learnings
