"""Model registry -- curated for text-only structured JSON output.

Each model carries a priority number. The router builds its chain by sorting
on priority, so adding a model is one dict + one number, with no separate
ordered list to drift out of sync.

Selection criteria:
  - Text-only (no vision needed -- reconciliation is pure data)
  - Good at structured JSON output (the prompt demands strict JSON)
  - Fast (batch processing, but we still want quick turnaround)
  - Free-tier friendly (low per-minute limits are fine, we do ~35 calls/run)

Priority ordering rationale (lower = tried first):
  1. Groq models first -- fastest inference (LPU hardware), 30 RPM
  2. Gemini Flash-Lite -- fast, generous daily quota (500 req/day)
  3. Mistral Small -- fast, good JSON output
  4. NVIDIA NIM -- credit-based, good reasoning, mid-chain
  5. OpenRouter free -- slowest but free fallback
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One model in the registry."""
    name: str
    id: str
    provider: str
    strength: int = 3     # 1-5, higher = more capable
    priority: int = 100   # lower = tried first in the chain

    @classmethod
    def from_dict(cls, d: dict) -> "ModelInfo":
        return cls(
            name=d["name"],
            id=d["id"],
            provider=d["provider"],
            strength=d.get("strength", 3),
            priority=d.get("priority", 100),
        )


_MODEL_DICTS = [
    # ==================================================================
    # Groq -- fastest inference, 30 RPM free tier
    # ==================================================================
    {"name": "Groq GPT-OSS 120B (fast, strong reasoning)",
     "id": "openai/gpt-oss-120b", "provider": "groq", "strength": 4, "priority": 10},

    {"name": "Groq GPT-OSS 20B (fast, lighter fallback)",
     "id": "openai/gpt-oss-20b", "provider": "groq", "strength": 2, "priority": 20},

    # ==================================================================
    # Gemini -- generous daily quota (500 req/day on Flash)
    # ==================================================================
    {"name": "Gemini 3.5 Flash-Lite (fast, generous quota)",
     "id": "gemini-3.5-flash-lite", "provider": "gemini", "strength": 3, "priority": 30},

    {"name": "Gemini 3.5 Flash (strong all-round)",
     "id": "gemini-3.5-flash", "provider": "gemini", "strength": 4, "priority": 40},

    # ==================================================================
    # Mistral -- fast, good JSON output
    # ==================================================================
    {"name": "Mistral Small (fastest measured, good JSON)",
     "id": "mistral-small-latest", "provider": "mistral", "strength": 3, "priority": 50},

    # ==================================================================
    # NVIDIA NIM -- credit-based, strong reasoning models
    # ==================================================================
    {"name": "NVIDIA Nemotron 3 Super 120B (strong reasoning)",
     "id": "nvidia/nemotron-3-super-120b-a12b", "provider": "nvidia", "strength": 5, "priority": 60},

    {"name": "NVIDIA DeepSeek V4 Flash (good JSON output)",
     "id": "deepseek-ai/deepseek-v4-flash-0731", "provider": "nvidia", "strength": 4, "priority": 70},

    # ==================================================================
    # OpenRouter free -- last resort fallback (slowest, most variable)
    # ==================================================================
    {"name": "OR Nemotron 3.5 Lightning (free, 1M ctx)",
     "id": "nvidia/nemotron-3.5-lightning:free", "provider": "openrouter", "strength": 3, "priority": 200},

    {"name": "OR GLM 5.2 (free)",
     "id": "z-ai/glm-5.2:free", "provider": "openrouter", "strength": 4, "priority": 210},

    {"name": "OR Nemotron 3 Super 120B (free, last resort)",
     "id": "nvidia/nemotron-3-super-120b-a12b:free", "provider": "openrouter", "strength": 5, "priority": 220},
]

MODELS: list[ModelInfo] = [ModelInfo.from_dict(d) for d in _MODEL_DICTS]
