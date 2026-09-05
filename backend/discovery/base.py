"""DiscoveryClient interface — the single seam every provider plugs into.

§0: discovery = Google search over LinkedIn posts via a SERP API (Serper.dev
style), i.e. "<query> site:linkedin.com/posts after:YYYY-MM-DD" — not
LinkedIn-native scraping. Coverage is partial and lags 1-3 days (a tight
query/window legitimately returns zero hits); every candidate still goes
through GPT-4o classification upstream. The rest of the codebase only talks
to search_posts() on this interface, so swapping the SERP vendor later never
touches the engine.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit


class DiscoveryError(RuntimeError):
    """Transient provider failure (timeouts, 5xx, 429, failed run)."""


class DiscoveryConfigError(DiscoveryError):
    """Provider is missing credentials/config or cannot do the requested job."""


@dataclass(slots=True)
class RawPost:
    """One candidate LinkedIn post as returned by a discovery provider."""

    post_url: str
    text: str
    author_name: str | None = None
    author_profile_url: str | None = None
    posted_at: datetime | None = None  # timezone-aware UTC when known
    query_used: str = ""
    provider: str = "unknown"


def canonical_post_url(url: str) -> str:
    """Canonical identity used for de-duplication (host+path, lowercased).

    LinkedIn share URLs carry tracking query params that would otherwise
    make the same post look unique.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return (url or "").strip().lower()
    host = (parts.netloc or "").lower()
    path = parts.path or ""
    return f"{host}{path}".rstrip("/").lower()


@dataclass(slots=True)
class SearchBatchResult:
    """Results + diagnostics from one search round."""

    posts: list[RawPost] = field(default_factory=list)
    queries_used: list[str] = field(default_factory=list)
    provider_errors: list[str] = field(default_factory=list)
    raw_rows_seen: int = 0
    undated_dropped: int = 0
    missing_url_dropped: int = 0


class DiscoveryClient(ABC):
    """Keyword post-search for candidates posted at/after `since`."""

    name: str = "base"

    @abstractmethod
    def search_posts(
        self,
        queries: list[str],
        since: datetime,
        *,
        results_per_query: int = 25,
    ) -> SearchBatchResult:
        """Run the small query set together, return deduplicated raw posts.

        Implementations dedupe internally by canonical_post_url. A provider
        that returns nothing it can attribute to the task (as opposed to a
        genuinely empty result) must raise DiscoveryError — never silently
        return [] and let callers believe "no leads exist".
        """

    @property
    def config_errors(self) -> list[str]:
        """Human-readable list of missing config; empty means ready to run."""
        return []


# ---------------------------------------------------------------------------
# Shared row-mapping helpers (provider rows arrive with unpredictable field
# names, so mapping is always "pick the first present candidate key").
# ---------------------------------------------------------------------------

def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def field_value(item: dict[str, Any], candidates: list[str]) -> Any:
    """Pick the first present candidate field from a provider row.

    Handles flat scalars and nested payloads ({"author": {...}}). When the
    candidates are URL-ish, nested lookups prefer url keys; otherwise they
    prefer name keys — so a row with only {"author": {"name": ...}} does not
    masquerade as an author-profile URL.
    """
    want_url = any("url" in c.lower() for c in candidates)
    nested_keys = ("url", "profileUrl", "profile_url", "personUrl", "handle") if want_url else (
        "name", "fullName", "displayName", "username", "handle",
    )
    for key in candidates:
        if key in item and item[key] not in (None, ""):
            value = item[key]
            if isinstance(value, dict):
                for nk in nested_keys:
                    if nk in value and value[nk] not in (None, ""):
                        return value[nk]
                return value
            return value
    author = item.get("author")
    if isinstance(author, dict):
        for nk in nested_keys:
            if nk in author and author[nk] not in (None, ""):
                return author[nk]
    return None


_ISO_START = re.compile(r"^\d{4}-\d{2}-\d{2}")
_RELATIVE = re.compile(
    r"^(?:just\s+now|yesterday)"
    r"|(\d+)\s*(minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|w|months?)\s*(?:ago)?$",
    re.IGNORECASE,
)


def parse_posted_at(value: Any, now: datetime | None = None) -> datetime | None:
    """Parse absolute ISO dates and the relative date strings scrapers emit
    ("2 days ago", "Yesterday", "just now", "3h") into an aware UTC datetime."""
    if value in (None, ""):
        return None
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        try:
            ms = int(value)
            if ms > 10_000_000_000:  # milliseconds vs seconds
                ms //= 1000
            return datetime.fromtimestamp(ms, tz=UTC)
        except (ValueError, OSError, OverflowError):
            return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "just now":
        return now
    if lowered == "yesterday":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    rel = _RELATIVE.match(lowered)
    if rel and rel.group(1):
        amount = int(rel.group(1))
        unit = rel.group(2).lower().rstrip("s")
        scale = {
            "minute": 60, "min": 60,
            "hour": 3600, "hr": 3600, "h": 3600,
            "day": 86400, "d": 86400,
            "week": 604800, "wk": 604800, "w": 604800,
            "month": 2_592_000,
        }.get(unit)
        if scale:
            return now - timedelta(seconds=amount * scale)
    text_tz = text.replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
    ):
        try:
            dt = datetime.strptime(text_tz, fmt)
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
        except ValueError:
            continue
    if _ISO_START.match(text):
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            return None
    return None
