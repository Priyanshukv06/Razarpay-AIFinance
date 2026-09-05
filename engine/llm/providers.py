"""Provider endpoints and key handling -- the single source of truth.

Every provider here speaks the OpenAI chat-completions API, so one client class
covers all of them and only the base URL and key pool differ. Adding a provider
is one line in each dict below plus one line in .env; no other code changes.
"""

import os


BASE_URLS = {
    "groq":       "https://api.groq.com/openai/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral":    "https://api.mistral.ai/v1",
    "nvidia":     "https://integrate.api.nvidia.com/v1",
}

KEY_ENV_VARS = {
    "groq":       "GROQ_API_KEYS",
    "gemini":     "GEMINI_API_KEYS",
    "openrouter": "OPENROUTER_API_KEYS",
    "mistral":    "MISTRAL_API_KEYS",
    "nvidia":     "NVIDIA_API_KEYS",
}

# Endpoints that reject OpenAI's presence_penalty / frequency_penalty.
NO_PENALTY_PROVIDERS = frozenset({"gemini", "mistral"})

# Groq's WAF blocks urllib's default User-Agent. Only needed for raw
# urllib calls; the openai SDK sends its own UA.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)


def read_keys(provider: str) -> list[str]:
    """Keys for one provider, from its comma-separated env var."""
    env_var = KEY_ENV_VARS.get(provider)
    if not env_var:
        return []
    return [k.strip() for k in os.getenv(env_var, "").split(",") if k.strip()]


def all_keys() -> dict[str, list[str]]:
    """All configured keys, keyed by provider name."""
    return {provider: read_keys(provider) for provider in KEY_ENV_VARS}
