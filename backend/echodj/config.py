"""
EchoDJ Configuration
====================
All settings are read from environment variables / .env file.
Copy .env.example → .env and fill in your values.

References:
    - Spec §4 (Technical Stack)
    - Spec §6.1 (Required API Keys)
    - Spec §11 (Spotify Authentication)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve the .env file relative to the backend directory
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Spotify (Required)
    # Spec §6.1, §11
    # ------------------------------------------------------------------
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_redirect_uri: str = "http://localhost:3000/callback"

    # ------------------------------------------------------------------
    # Last.fm (Required)
    # Spec §6.1
    # ------------------------------------------------------------------
    lastfm_api_key: str = ""

    # ------------------------------------------------------------------
    # ListenBrainz (Optional — enhances recommendations)
    # Spec §6.1
    # ------------------------------------------------------------------
    listenbrainz_user_token: str = ""

    # ------------------------------------------------------------------
    # LLM Provider — plug-and-play via protocol
    # Spec §4.1
    # ------------------------------------------------------------------
    echodj_llm_provider: Literal["gemini", "ollama"] = "gemini"
    echodj_llm_model: str = "gemini-2.0-flash"

    # Gemini
    gemini_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"



    # ------------------------------------------------------------------
    # TTS — Provider and voice selection
    # Spec §5.6
    # ------------------------------------------------------------------
    echodj_tts_provider: Literal["edge", "gemini"] = "gemini"
    
    # edge-tts voice config
    echodj_tts_voice: str = "en-US-GuyNeural"
    
    # Gemini voice config (Puck, Charon, Kore, Fenrir, Aoede)
    echodj_gemini_voice: str = "Puck"

    # ------------------------------------------------------------------
    # STT — Faster-Whisper configuration
    # Spec §5.1
    # ------------------------------------------------------------------
    echodj_whisper_model: str = "large-v3"
    echodj_whisper_device: str = "cuda"
    echodj_whisper_compute_type: str = "int8"
    echodj_whisper_beam_size: int = 2

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    echodj_host: str = "0.0.0.0"
    echodj_port: int = 8000

    # ------------------------------------------------------------------
    # Database paths (LangGraph persistence)
    # Spec §10.3
    # ------------------------------------------------------------------
    echodj_sessions_db: Path = _PROJECT_ROOT / "data" / "echodj_sessions.db"
    echodj_memory_db: Path = _PROJECT_ROOT / "data" / "echodj_memory.db"

    # ------------------------------------------------------------------
    # MusicBrainz
    # Spec §5.2 — requires User-Agent header
    # ------------------------------------------------------------------
    musicbrainz_contact_email: str = "echodj@example.com"


settings = Settings()
