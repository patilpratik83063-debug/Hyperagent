"""Central settings. All values come from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # pragma: no cover - dotenv is optional at import time
    pass


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Classification LLM — DeepSeek by default (provider: deepseek | openai).
    llm_provider: str = field(default_factory=lambda: _str("LLM_PROVIDER", "deepseek").lower())
    deepseek_api_key: str = field(default_factory=lambda: _str("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: _str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: _str("DEEPSEEK_MODEL", "deepseek-chat"))
    llm_timeout_seconds: float = field(default_factory=lambda: _float("LLM_TIMEOUT_SECONDS", 60.0))
    llm_max_retries: int = field(default_factory=lambda: _int("LLM_MAX_RETRIES", 2))
    # Optional legacy OpenAI path (only used when LLM_PROVIDER=openai).
    openai_api_key: str = field(default_factory=lambda: _str("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _str("OPENAI_MODEL", "gpt-4o"))

    # Database
    supabase_url: str = field(default_factory=lambda: _str("SUPABASE_URL"))
    supabase_service_role_key: str = field(default_factory=lambda: _str("SUPABASE_SERVICE_ROLE_KEY"))

    # Discovery — Google SERP API (Serper.dev-style wrapper over Google search)
    discovery_provider: str = field(default_factory=lambda: _str("DISCOVERY_PROVIDER", "auto").lower())
    mock_mode: bool = field(default_factory=lambda: _str("MOCK_MODE", "0") in {"1", "true", "yes"})

    serper_api_key: str = field(default_factory=lambda: _str("SERPER_API_KEY"))
    serper_base_url: str = field(default_factory=lambda: _str("SERPER_BASE_URL", "https://google.serper.dev"))
    # Restriction appended to every query, e.g. "linkedin.com/posts".
    serper_site_restriction: str = field(default_factory=lambda: _str("SERPER_SITE_RESTRICTION", "linkedin.com/posts"))
    # Results per query. NOTE: Serper's FREE tier caps `num` at 10 and rejects
    # larger values with HTTP 400 — keep at 10 unless you are on a paid plan.
    serper_results_per_query: int = field(default_factory=lambda: _int("SERPER_RESULTS_PER_QUERY", 10))
    serper_gl: str = field(default_factory=lambda: _str("SERPER_GL", ""))   # optional Google country ('us', 'in', ...)
    serper_hl: str = field(default_factory=lambda: _str("SERPER_HL", "en"))  # optional Google language
    serper_timeout_seconds: float = field(default_factory=lambda: _float("SERPER_TIMEOUT_SECONDS", 30.0))

    # Usage budget
    max_searches_per_day: int = field(default_factory=lambda: _int("MAX_SEARCHES_PER_DAY", 20))

    # Scoring gates
    min_overall_score: float = field(default_factory=lambda: _float("MIN_OVERALL_SCORE", 60.0))
    min_service_match: float = field(default_factory=lambda: _float("MIN_SERVICE_MATCH", 50.0))
    min_intent_strength: str = field(default_factory=lambda: _str("MIN_INTENT_STRENGTH", "recommendation"))

    # Engine loop — the exact-count loop keeps running until N qualified leads
    # are found or discovery is genuinely exhausted; 30 iterations of fresh
    # query diversity is the budget before giving up (never padding).
    engine_max_iterations: int = field(default_factory=lambda: _int("ENGINE_MAX_ITERATIONS", 30))
    engine_deadline_seconds: int = field(default_factory=lambda: _int("ENGINE_DEADLINE_SECONDS", 1800))
    engine_early_stop_empty_rounds: int = field(default_factory=lambda: _int("ENGINE_EARLY_STOP_EMPTY_ROUNDS", 3))
    classifier_concurrency: int = field(default_factory=lambda: _int("CLASSIFIER_CONCURRENCY", 8))

    @property
    def storage_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def llm_configured(self) -> bool:
        if self.llm_provider == "deepseek":
            return bool(self.deepseek_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        return False

    @property
    def llm_key_env(self) -> str:
        """Name of the env var that must hold the key for the active provider."""
        return "DEEPSEEK_API_KEY" if self.llm_provider == "deepseek" else "OPENAI_API_KEY"

    @property
    def serp_configured(self) -> bool:
        return bool(self.serper_api_key)


settings = Settings()
