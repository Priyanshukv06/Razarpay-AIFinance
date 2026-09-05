"""Configuration constants for the LLM router and reconciliation engine.

Kept in one place so tuning is a single-file change.
"""

import os
from pathlib import Path

# Project root -- used for locating .env, health files, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# LLM call parameters
# ---------------------------------------------------------------------------
LLM_MAX_TOKENS = 1024        # Enough for a JSON classification response
LLM_TEMPERATURE = 0.2        # Lower than reference (0.3) for more deterministic JSON
LLM_TOP_P = 0.9
LLM_PRESENCE_PENALTY = 0.0   # No penalty -- we want consistent JSON, not creative text
LLM_FREQUENCY_PENALTY = 0.0
LLM_TIMEOUT_SECONDS = 60     # Same as reference -- generous for free-tier latency

# ---------------------------------------------------------------------------
# Router behaviour
# ---------------------------------------------------------------------------
ROUTER_MAX_ATTEMPTS = 5       # Max total API calls before giving up on one record
ROUTER_ROTATE_KEYS = True     # On key-specific failure, try next key on same model
ROUTER_ROUND_ROBIN_KEYS = True  # After success, advance key cursor for load balancing

# Cooldown durations (seconds)
ROUTER_RATE_LIMIT_COOLDOWN_S = 45    # 429 -- per-minute limits clear in <60s
ROUTER_SERVER_ERROR_COOLDOWN_S = 20  # 5xx -- brief pause before retrying
ROUTER_AUTH_FAILURE_COOLDOWN_S = 3600  # 401/403 -- key is bad, long cooldown

# Health state persistence -- use temp dir if project root is read-only (Streamlit Cloud)
import tempfile as _tempfile
_health_dir = PROJECT_ROOT if os.access(PROJECT_ROOT, os.W_OK) else Path(_tempfile.gettempdir())
ROUTER_HEALTH_FILE = _health_dir / "provider_health.json"
