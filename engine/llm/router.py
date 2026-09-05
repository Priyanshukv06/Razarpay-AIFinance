"""Multi-provider LLM router with automatic failover.

Designed for batch reconciliation:
  - No streaming (batch processing, not real-time)
  - No cancel tokens (no user-interrupt scenario)
  - No vision chain (text-only JSON classification)

Core architecture preserved:
  - Priority-ordered chain derived from model priorities
  - Failure classification (rate_limited / auth_failed / bad_request / server_error)
  - Provider health tracking with disk persistence
  - Key rotation (round-robin for load balancing + failover on 429/401)
  - strip_reasoning() to remove <think> blocks from reasoning models
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from engine import config
from engine.llm.models import MODELS, ModelInfo
from engine.llm.providers import (BASE_URLS, KEY_ENV_VARS, NO_PENALTY_PROVIDERS,
                                  all_keys)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
RATE_LIMITED = "rate_limited"
AUTH_FAILED = "auth_failed"
BAD_REQUEST = "bad_request"
SERVER_ERROR = "server_error"
UNKNOWN = "unknown"

# Failures about the KEY rather than the provider. A per-key quota or a
# single revoked key says nothing about the provider's other keys.
KEY_SPECIFIC_FAILURES = frozenset({RATE_LIMITED, AUTH_FAILED})


def classify_error(exc: Exception) -> str:
    """Map a provider exception to a failure kind.

    Status codes are checked before message text, since wording varies
    between providers while codes do not.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None)

    if status == 429:
        return RATE_LIMITED
    if status in (401, 403):
        return AUTH_FAILED
    if status == 400:
        return BAD_REQUEST
    if status and 500 <= int(status) < 600:
        return SERVER_ERROR

    text = str(exc).lower()
    if ("429" in text or "rate limit" in text or "ratelimit" in text
            or "quota" in text or "too many requests" in text
            or "try again in" in text or "retry after" in text):
        return RATE_LIMITED
    if ("401" in text or "403" in text or "unauthorized" in text
            or "forbidden" in text or "invalid api key" in text
            or "authentication" in text):
        return AUTH_FAILED
    if "timeout" in text or "timed out" in text or "connection" in text:
        return SERVER_ERROR
    if "context length" in text or "too long" in text or "maximum context" in text:
        return BAD_REQUEST
    return UNKNOWN


def parse_retry_after(exc: Exception) -> float | None:
    """Seconds to wait, if the provider said so."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

    match = re.search(r"try again in ([\d.]+)\s*(ms|s|m)\b", str(exc), re.I)
    if match:
        value, unit = float(match.group(1)), match.group(2).lower()
        return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
    return None


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models leak.

    Some models (Qwen3.x, magistral-*) include internal reasoning in
    their output. This is noise for our JSON parsing.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Provider health tracking
# ---------------------------------------------------------------------------
@dataclass
class _Cooldown:
    until: float
    reason: str


class ProviderHealth:
    """Remembers which providers are temporarily unusable.

    Persisted to disk so restarting mid-session does not lose the knowledge
    that a provider is throttled.
    """

    def __init__(self, path=None):
        self.path = path or config.ROUTER_HEALTH_FILE
        self._cooldowns: dict[str, _Cooldown] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                now = time.time()
                self._cooldowns = {
                    provider: _Cooldown(entry["until"], entry.get("reason", ""))
                    for provider, entry in raw.items()
                    if entry.get("until", 0) > now
                }
        except Exception:
            self._cooldowns = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {p: {"until": c.until, "reason": c.reason}
                 for p, c in self._cooldowns.items()},
                indent=2), encoding="utf-8")
        except Exception:
            pass

    def is_available(self, provider: str) -> bool:
        cooldown = self._cooldowns.get(provider)
        if not cooldown:
            return True
        if cooldown.until <= time.time():
            del self._cooldowns[provider]
            self._save()
            return True
        return False

    def seconds_remaining(self, provider: str) -> float:
        cooldown = self._cooldowns.get(provider)
        return max(0.0, cooldown.until - time.time()) if cooldown else 0.0

    def penalise(self, provider: str, kind: str, retry_after: float | None = None) -> float:
        """Put a provider on cooldown. Returns cooldown seconds."""
        if kind == RATE_LIMITED:
            seconds = retry_after if retry_after else config.ROUTER_RATE_LIMIT_COOLDOWN_S
        elif kind == AUTH_FAILED:
            seconds = config.ROUTER_AUTH_FAILURE_COOLDOWN_S
        elif kind == SERVER_ERROR:
            seconds = config.ROUTER_SERVER_ERROR_COOLDOWN_S
        else:
            return 0.0

        seconds = float(seconds)
        self._cooldowns[provider] = _Cooldown(time.time() + seconds, kind)
        self._save()
        return seconds

    def clear(self, provider: str | None = None) -> None:
        if provider:
            self._cooldowns.pop(provider, None)
        else:
            self._cooldowns.clear()
        self._save()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass
class _Attempt:
    model: ModelInfo
    kind: str
    detail: str
    elapsed: float
    key_index: int = 0
    key_count: int = 1


@dataclass
class RouteResult:
    """What the router returns on success."""
    text: str
    model: ModelInfo
    attempts: int
    elapsed: float
    fell_back: bool
    notes: list[str]


class RoutedLLMBackend:
    """Tries models in priority order until one answers.

    Simplified for batch reconciliation:
    - Only blocking route() calls (no streaming)
    - No cancel tokens
    - Text-only models
    """

    def __init__(self, health: ProviderHealth | None = None):
        from dotenv import load_dotenv
        load_dotenv(config.PROJECT_ROOT / ".env")

        self._keys = all_keys()
        self._index = {provider: 0 for provider in KEY_ENV_VARS}
        self.health = health or ProviderHealth()
        self.last_attempts: list[_Attempt] = []

    def chain(self) -> list[ModelInfo]:
        """Models sorted by priority, filtered to providers with keys."""
        available = {p for p, keys in self._keys.items() if keys}
        return sorted(
            [m for m in MODELS if m.provider in available],
            key=lambda m: m.priority,
        )

    def route(self, messages: list[dict]) -> RouteResult | None:
        """Try models in priority order until one answers.

        Returns RouteResult on success, None if every attempt failed.
        None means the caller should degrade gracefully (mark as
        'flagged for manual review'), not crash.
        """
        order = self.chain()
        if not order:
            log.error("No models available -- no API keys configured")
            return None

        self.last_attempts = []
        started = time.time()
        notes: list[str] = []
        attempted = 0
        models_tried = 0

        for candidate in order:
            if attempted >= config.ROUTER_MAX_ATTEMPTS:
                break

            if not self.health.is_available(candidate.provider):
                remaining = self.health.seconds_remaining(candidate.provider)
                notes.append(f"skipped {candidate.provider} ({remaining:.0f}s cooldown)")
                continue

            key_count = self._key_pass_count(candidate.provider)
            for key_pass in range(key_count):
                if attempted >= config.ROUTER_MAX_ATTEMPTS:
                    break
                if key_pass == 0:
                    models_tried += 1

                attempted += 1
                key_index = self._current_key_index(candidate.provider)
                attempt_started = time.time()
                try:
                    text = self._call(candidate, messages)
                    self._advance_after_success(candidate.provider)
                    return RouteResult(
                        text=text,
                        model=candidate,
                        attempts=attempted,
                        elapsed=time.time() - started,
                        fell_back=models_tried > 1,
                        notes=notes,
                    )
                except Exception as exc:
                    kind = classify_error(exc)
                    rotated = self._rotate_if_key_specific(
                        candidate.provider, kind, key_pass, key_count)
                    if not rotated:
                        self.health.penalise(
                            candidate.provider, kind, parse_retry_after(exc))
                    self.last_attempts.append(_Attempt(
                        candidate, kind, str(exc)[:200],
                        time.time() - attempt_started, key_index, key_count))
                    note = f"{candidate.name}: {kind}"
                    if key_count > 1:
                        note += f" [key {key_index + 1}/{key_count}]"
                    if rotated:
                        note += " -> retrying on next key"
                    notes.append(note)
                    if not rotated:
                        break

        return None

    def failure_summary(self) -> str:
        """Human-readable summary of what went wrong."""
        if not self.last_attempts:
            return "No attempts made (no models available?)"
        lines = ["All providers failed:"]
        for a in self.last_attempts:
            lines.append(f"  {a.model.name} ({a.model.provider}): "
                         f"{a.kind} -- {a.detail[:100]}")
        return "\n".join(lines)

    # -- internals ---------------------------------------------------------
    def _client(self, provider: str) -> OpenAI:
        keys = self._keys.get(provider) or []
        if not keys:
            raise RuntimeError(f"No API key configured for {provider}")
        return OpenAI(
            api_key=keys[self._index[provider] % len(keys)],
            base_url=BASE_URLS[provider],
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,  # Disable SDK retries -- our router handles failover
        )

    def _call(self, model: ModelInfo, messages: list[dict]) -> str:
        """Blocking completion call. Returns answer text or raises."""
        client = self._client(model.provider)
        kwargs = dict(
            model=model.id,
            messages=messages,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            top_p=config.LLM_TOP_P,
        )
        if model.provider not in NO_PENALTY_PROVIDERS:
            kwargs["presence_penalty"] = config.LLM_PRESENCE_PENALTY
            kwargs["frequency_penalty"] = config.LLM_FREQUENCY_PENALTY

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        answer = strip_reasoning(content)
        if not answer.strip():
            raise RuntimeError("model returned no answer (empty, or reasoning only)")
        return answer

    # -- key pool ----------------------------------------------------------
    def _advance_after_success(self, provider: str) -> None:
        """Move to next key so the following request uses a different one."""
        if config.ROUTER_ROUND_ROBIN_KEYS:
            self._rotate_key(provider)

    def _key_pass_count(self, provider: str) -> int:
        """How many key attempts one model is allowed."""
        if not config.ROUTER_ROTATE_KEYS:
            return 1
        return max(1, len(self._keys.get(provider) or []))

    def _current_key_index(self, provider: str) -> int:
        keys = self._keys.get(provider) or []
        return self._index.get(provider, 0) % len(keys) if keys else 0

    def _rotate_key(self, provider: str) -> None:
        keys = self._keys.get(provider) or []
        if len(keys) > 1:
            self._index[provider] = (self._index[provider] + 1) % len(keys)

    def _rotate_if_key_specific(self, provider: str, kind: str,
                                key_pass: int, key_count: int) -> bool:
        """Advance to next key instead of cooling the provider down.

        Returns True if rotation happened (caller should retry same model).
        Returns False if caller should penalise provider and move down chain.
        """
        if kind not in KEY_SPECIFIC_FAILURES or key_pass >= key_count - 1:
            return False
        self._rotate_key(provider)
        return True
