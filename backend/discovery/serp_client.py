"""Google SERP discovery client (Serper.dev-style) — the discovery engine.

Discovery = Google search over LinkedIn posts, not LinkedIn-native scraping:

    <query> site:linkedin.com/posts after:YYYY-MM-DD

Design notes (§0 — these are provider trade-offs, not bugs):
* Coverage is partial/inconsistent: Google does not fully or promptly crawl
  linkedin.com/posts, so a precise query + tight window will OFTEN return
  zero organic results. Returning an empty batch is normal provider
  behavior — it is never translated into a fake failure upstream.
* The 24h window structurally underperforms (Google's crawl lag on LinkedIn
  is typically 1–3 days); the UI says so under that option.
* The date filter rides in the query string itself (after:YYYY-MM-DD),
  recomputed fresh from `since` on every call.
* Every result still goes through GPT-4o classification upstream — SERP
  relevance is a starting filter, not a verdict.
"""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from .base import (
    DiscoveryClient,
    DiscoveryConfigError,
    DiscoveryError,
    RawPost,
    SearchBatchResult,
    canonical_post_url,
    parse_posted_at,
)

log = logging.getLogger(__name__)

_TITLE_AUTHOR = re.compile(r"^(?P<author>.+?)\s+on\s+LinkedIn\s*:\s*(?P<body>.+)$", re.IGNORECASE | re.DOTALL)


class SerperDiscoveryClient(DiscoveryClient):
    """Google SERP (Serper.dev or any Serper-compatible endpoint) -> RawPosts."""

    name = "serp"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://google.serper.dev",
        site_restriction: str = "linkedin.com/posts",
        results_per_query: int = 10,
        gl: str = "",
        hl: str = "en",
        timeout_seconds: float = 30.0,
    ) -> None:
        """`results_per_query` maps to Serper's `num`. Serper's FREE tier caps
        `num` at 10 and rejects larger values with HTTP 400 — the default is
        10 for that reason; raise it only on a paid Serper plan."""
        if not api_key:
            raise DiscoveryConfigError("SERPER_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.site_restriction = (site_restriction or "").strip()
        self.default_results_per_query = results_per_query
        self.gl = (gl or "").strip().lower()
        self.hl = (hl or "").strip()
        self.timeout_seconds = timeout_seconds
        self._headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        self.calls = 0

    @property
    def config_errors(self) -> list[str]:
        return [] if self.api_key else ["SERPER_API_KEY is not set — discovery cannot run (or set MOCK_MODE=1 for an offline demo)"]

    # ------------------------------------------------------------------ query
    def _full_query(self, query: str, since: datetime) -> str:
        parts = [query]
        if self.site_restriction:
            parts.append(f"site:{self.site_restriction}")
        parts.append(f"after:{since.date().isoformat()}")  # fresh per request (§3)
        return " ".join(parts)

    # ------------------------------------------------------------------ http
    def _search(self, full_query: str, num: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"q": full_query, "num": max(1, min(num, 100))}
        if self.gl:
            payload["gl"] = self.gl
        if self.hl:
            payload["hl"] = self.hl
        try:
            resp = httpx.post(
                f"{self.base_url}/search",
                headers=self._headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise DiscoveryError(f"Serper request failed: {exc}") from exc
        if resp.status_code in (401, 403):
            raise DiscoveryError(
                f"Serper rejected the API key (HTTP {resp.status_code}) — check SERPER_API_KEY"
            )
        if resp.status_code == 400:
            # Free Serper accounts reject num > 10 with a misleading
            # "Query pattern not allowed" message — say what actually happened.
            hint = ""
            if num > 10:
                hint = " Serper's free tier caps `num` at 10 — set SERPER_RESULTS_PER_QUERY=10 (or use a paid plan)."
            raise DiscoveryError(f"Serper request failed (HTTP 400): {(resp.text or '')[:200]}{hint}")
        if resp.status_code != 200:
            detail = (resp.text or "")[:300]
            raise DiscoveryError(f"Serper request failed (HTTP {resp.status_code}): {detail}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise DiscoveryError("Serper returned non-JSON output") from exc
        if isinstance(body, dict) and body.get("error"):
            raise DiscoveryError(f"Serper error: {body['error']}")
        organic = body.get("organic") if isinstance(body, dict) else None
        return organic if isinstance(organic, list) else []

    # ------------------------------------------------------------------ map
    def _map_organic(self, item: dict[str, Any], query: str) -> RawPost | None:
        link = str(item.get("link") or "").strip()
        if not link:
            return None
        # Google obeys site:, but keep a belt-and-braces guard so results from
        # news/knowledge boxes never slip in as "posts".
        if self.site_restriction and "linkedin.com/posts/" not in link.lower() and self.site_restriction.lower() not in link.lower():
            return None
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        text = snippet or title
        if not text:
            return None

        author: str | None = None
        body = text
        m = _TITLE_AUTHOR.match(title)
        if m:
            author = m.group("author").strip() or None
            body = (m.group("body") or snippet or text).strip() or text
        else:
            body = snippet or title

        return RawPost(
            post_url=link,
            text=body,
            author_name=author,
            author_profile_url=None,  # SERP organic results don't carry the profile link
            posted_at=parse_posted_at(item.get("date")),
            query_used=query,
            provider=self.name,
        )

    # -------------------------------------------------------------- discovery
    def search_posts(
        self,
        queries: list[str],
        since: datetime,
        *,
        results_per_query: int | None = None,
    ) -> SearchBatchResult:
        queries = [q for q in (queries or []) if q.strip()]
        if not queries:
            return SearchBatchResult()
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        per_q = results_per_query or self.default_results_per_query

        posts: list[RawPost] = []
        errors: list[str] = []
        for query in queries:
            self.calls += 1
            full = self._full_query(query, since)
            try:
                organic = self._search(full, per_q)
                for item in organic:
                    post = self._map_organic(item, query)
                    if post:
                        posts.append(post)
                log.info("Serper %r -> %d organic results", full[:160], len(organic))
            except DiscoveryConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - one query must not kill the round
                errors.append(f"query {query!r} failed: {exc}")
                log.warning("Serper query %r failed: %s", query, exc)

        # Dedupe by canonical post URL (same post found under several phrasings).
        seen: set[str] = set()
        unique: list[RawPost] = []
        for p in posts:
            key = canonical_post_url(p.post_url)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(p)
        # Zero hits is EXPECTED for tight queries/windows (§0): report it as
        # empty data, never as a broken discovery task.
        return SearchBatchResult(
            posts=unique,
            queries_used=queries,
            provider_errors=errors,
            raw_rows_seen=len(unique),
        )
