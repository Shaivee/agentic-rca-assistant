"""
Settings, read from environment variables (and a local .env file, which is git-ignored).

    LLM_PROVIDER      groq | openai          (default: whichever API key is present)
    LLM_MODEL         override the default model for the provider
    GROQ_API_KEY      free key from console.groq.com
    OPENAI_API_KEY    optional
    EMBEDDINGS        fastembed | openai | none   (default: fastembed if installed)
"""
from __future__ import annotations

import os
from pathlib import Path

try:                                    # python-dotenv is optional
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# provider: (base_url, api_key_env_var, default_model)
# Groq exposes an OpenAI-compatible API, so ONE client library works for both.
# Model line-ups change: Groq retired its free Llama chat models in Aug 2026. Override with LLM_MODEL.
PROVIDERS = {
    "groq":   ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "openai": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
}


def llm_provider() -> str | None:
    explicit = os.getenv("LLM_PROVIDER")
    if explicit:
        return explicit.lower()
    for name, (_, key_var, _) in PROVIDERS.items():
        if os.getenv(key_var):
            return name
    return None


def embeddings_provider() -> str:
    explicit = os.getenv("EMBEDDINGS")
    if explicit:
        return explicit.lower()
    try:
        import fastembed  # noqa: F401
        return "fastembed"
    except ImportError:
        return "openai" if os.getenv("OPENAI_API_KEY") else "none"
