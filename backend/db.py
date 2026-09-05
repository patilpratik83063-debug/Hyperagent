"""Persistence layer.

Two interchangeable stores behind one protocol:
  * SupabaseStore  — production (Postgres via supabase-py, service-role key)
  * MemoryStore    — offline runs / demo mode / tests
The engine and the API only ever talk to the protocol below.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable

log = logging.getLogger(__name__)

SEARCH_STATUSES = ("queued", "running", "completed", "failed", "no_results")


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class Store(ABC):
    """Store protocol used by the engine + API routes."""

    # searches ------------------------------------------------------------
    @abstractmethod
    def create_search(self, *, service: str, country: str, lead_type: str, time_window: str,
                      leads_needed: int) -> dict[str, Any]: ...

    @abstractmethod
    def get_search(self, search_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def update_search(self, search_id: str, **fields: Any) -> None: ...

    @abstractmethod
    def list_searches(self, limit: int = 50) -> list[dict[str, Any]]: ...

    # leads ------------------------------------------------------------------
    @abstractmethod
    def insert_leads_many(self, rows: Iterable[dict[str, Any]]) -> int: ...

    @abstractmethod
    def find_existing_post_urls(self, urls: Iterable[str]) -> set[str]:
        """Post URLs already saved across ANY search (leads.post_url is
        globally unique). The engine uses this to skip already-owned posts
        and keep looping until it delivers exactly N NEW leads."""

    @abstractmethod
    def list_leads(self, *, search_id: str | None = None, time_window: str | None = None,
                   status: str | None = None, limit: int = 500) -> list[dict[str, Any]]: ...

    @abstractmethod
    def patch_lead(self, lead_id: str, *, status: str | None = None, notes: str | None = None) -> dict[str, Any] | None: ...

    # usage ----------------------------------------------------------------
    @abstractmethod
    def searches_used_today(self) -> int: ...


# ---------------------------------------------------------------------------
# In-memory (offline / tests)
# ---------------------------------------------------------------------------

class MemoryStore(Store):
    """Thread-safe in-memory implementation of the store protocol."""

    def __init__(self) -> None:
        self._lock = __import__("threading").RLock()
        self._searches: dict[str, dict[str, Any]] = {}
        self._leads: dict[str, dict[str, Any]] = {}
        self._lead_by_url: dict[str, str] = {}

    # ids -------------------------------------------------------------------
    @staticmethod
    def _new_id() -> str:
        import uuid

        return str(uuid.uuid4())

    # searches ---------------------------------------------------------------
    def create_search(self, *, service, country, lead_type, time_window, leads_needed) -> dict[str, Any]:
        with self._lock:
            row = {
                "id": self._new_id(),
                "service": service,
                "country": country or "",
                "lead_type": lead_type,
                "time_window": time_window,
                "leads_needed": leads_needed,
                "status": "queued",
                "found_count": 0,
                "accepted_count": 0,
                "scanned_count": 0,
                "error": None,
                "finished_at": None,
                "created_at": _utcnow(),
            }
            self._searches[row["id"]] = row
            return dict(row)

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._searches.get(search_id)
            return dict(row) if row else None

    def update_search(self, search_id: str, **fields: Any) -> None:
        with self._lock:
            row = self._searches.get(search_id)
            if row is None:
                raise KeyError(f"search {search_id} not found")
            for key, value in fields.items():
                if key in row:
                    row[key] = value

    def list_searches(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(self._searches.values(), key=lambda r: r["created_at"], reverse=True)
            return [dict(r) for r in rows[:limit]]

    # leads ---------------------------------------------------------------------
    def insert_leads_many(self, rows: Iterable[dict[str, Any]]) -> int:
        inserted = 0
        with self._lock:
            for row in rows:
                url = (row.get("post_url") or "").strip()
                if not url or url in self._lead_by_url:
                    continue
                lead_id = self._new_id()
                lead = {
                    "id": lead_id,
                    "search_id": row.get("search_id"),
                    "lead_type": row.get("lead_type"),
                    "time_window": row.get("time_window"),
                    "post_url": url,
                    "author_name": row.get("author_name"),
                    "author_profile_url": row.get("author_profile_url"),
                    "post_text": row.get("post_text"),
                    "post_date": row.get("post_date"),
                    "overall_quality_score": row.get("overall_quality_score"),
                    "service_match_score": row.get("service_match_score"),
                    "intent_strength": row.get("intent_strength"),
                    "status": "new",
                    "notes": None,
                    "created_at": _utcnow(),
                }
                self._leads[lead_id] = lead
                self._lead_by_url[url] = lead_id
                inserted += 1
        return inserted

    def find_existing_post_urls(self, urls: Iterable[str]) -> set[str]:
        wanted = {u.strip() for u in urls if u and u.strip()}
        if not wanted:
            return set()
        with self._lock:
            return {u for u in wanted if u in self._lead_by_url}

    def list_leads(self, *, search_id=None, time_window=None, status=None, limit: int = 500) -> list[dict[str, Any]]:
        from models import TimeWindow

        with self._lock:
            after: datetime | None = None
            if time_window:
                try:
                    after = TimeWindow(time_window).cutoff()
                except ValueError:
                    after = None
            out = []
            for lead in self._leads.values():
                if search_id and lead.get("search_id") != search_id:
                    continue
                if status and lead.get("status") != status:
                    continue
                if after is not None:
                    pdate = lead.get("post_date")
                    if pdate is None or pdate < after.date():
                        continue
                out.append(dict(lead))
            out.sort(key=lambda r: r.get("post_date") or datetime.min.date(), reverse=True)
            return out[:limit]

    def patch_lead(self, lead_id: str, *, status=None, notes=None) -> dict[str, Any] | None:
        with self._lock:
            lead = self._leads.get(lead_id)
            if lead is None:
                return None
            if status is not None:
                lead["status"] = status
            if notes is not None:
                lead["notes"] = notes
            return dict(lead)

    def searches_used_today(self) -> int:
        with self._lock:
            start = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            return sum(1 for s in self._searches.values() if (s.get("created_at") or _utcnow()) >= start)


# ---------------------------------------------------------------------------
# Supabase (production)
# ---------------------------------------------------------------------------

class SupabaseStore(Store):
    def __init__(self, url: str, service_role_key: str) -> None:
        if not url or not service_role_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        from supabase import create_client

        self._client = create_client(url, service_role_key)

    # -- JSON-safe outbound values -----------------------------------------
    # supabase-py serializes payloads with httpx, which rejects native
    # date/datetime objects. Every outbound payload passes through here.
    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(UTC).isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: SupabaseStore._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SupabaseStore._jsonable(v) for v in value]
        return value

    @staticmethod
    def _err(exc: Exception, what: str) -> None:
        log.error("Supabase %s failed: %s", what, exc)

    # searches ------------------------------------------------------------
    def create_search(self, *, service, country, lead_type, time_window, leads_needed) -> dict[str, Any]:
        resp = self._client.table("searches").insert({
            "service": service, "country": country or "", "lead_type": lead_type,
            "time_window": time_window, "leads_needed": leads_needed,
        }).execute()
        return resp.data[0]

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        resp = self._client.table("searches").select("*").eq("id", search_id).maybe_single().execute()
        return resp.data

    def update_search(self, search_id: str, **fields: Any) -> None:
        payload = {k: v for k, v in fields.items() if v is not None or k == "error"}
        if not payload:
            return
        self._client.table("searches").update(self._jsonable(payload)).eq("id", search_id).execute()

    def list_searches(self, limit: int = 50) -> list[dict[str, Any]]:
        resp = (
            self._client.table("searches")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data

    # leads -----------------------------------------------------------------
    def insert_leads_many(self, rows: Iterable[dict[str, Any]]) -> int:
        payload = [self._jsonable(dict(r)) for r in rows]
        if not payload:
            return 0
        try:
            resp = (
                self._client.table("leads")
                .upsert(payload, on_conflict="post_url", ignore_duplicates=True)
                .execute()
            )
            return len(resp.data or [])
        except Exception as exc:  # noqa: BLE001
            # Older supabase-py may not support ignore_duplicates; fall back to
            # per-row inserts and skip conflicts (post_url unique).
            self._err(exc, "bulk insert (falling back to per-row)")
            inserted = 0
            for row in payload:
                try:
                    self._client.table("leads").insert(row).execute()
                    inserted += 1
                except Exception as per_row:  # noqa: BLE001
                    log.warning("Supabase per-row insert skipped (duplicate or error): %s", per_row)
            return inserted

    def find_existing_post_urls(self, urls: Iterable[str]) -> set[str]:
        wanted = sorted({u.strip() for u in urls if u and u.strip()})
        if not wanted:
            return set()
        existing: set[str] = set()
        # postgrest `in` accepts a bounded list; chunk defensively.
        for i in range(0, len(wanted), 200):
            chunk = wanted[i : i + 200]
            try:
                resp = self._client.table("leads").select("post_url").in_("post_url", chunk).execute()
                existing.update(row["post_url"] for row in (resp.data or []))
            except Exception as exc:  # noqa: BLE001 - treat as "none known"; insert dedupes anyway
                log.warning("Supabase find_existing_post_urls failed (assuming new): %s", exc)
        return existing

    def list_leads(self, *, search_id=None, time_window=None, status=None, limit: int = 500) -> list[dict[str, Any]]:
        from models import TimeWindow

        q = self._client.table("leads").select("*")
        if search_id:
            q = q.eq("search_id", search_id)
        if status:
            q = q.eq("status", status)
        if time_window:
            try:
                after = TimeWindow(time_window).cutoff().date().isoformat()
                q = q.gte("post_date", after)
            except ValueError:
                pass
        resp = q.order("post_date", desc=True).limit(limit).execute()
        return resp.data

    def patch_lead(self, lead_id: str, *, status=None, notes=None) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if notes is not None:
            payload["notes"] = notes
        if not payload:
            return self._client.table("leads").select("*").eq("id", lead_id).maybe_single().execute().data
        resp = self._client.table("leads").update(self._jsonable(payload)).eq("id", lead_id).execute()
        return resp.data[0] if resp.data else None

    def searches_used_today(self) -> int:
        start = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        resp = self._client.table("searches").select("id").gte("created_at", start).execute()
        return len(resp.data or [])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_store(settings) -> Store:
    """Pick the store from settings (mock_mode or supabase availability)."""
    if settings.mock_mode or not settings.storage_configured:
        if settings.mock_mode:
            log.info("MOCK_MODE=1 -> using in-memory store")
        else:
            log.warning("SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY missing -> using in-memory store (data is NOT persisted)")
        return MemoryStore()
    log.info("Using Supabase store at %s", settings.supabase_url)
    return SupabaseStore(settings.supabase_url, settings.supabase_service_role_key)
