"""Hyperclients — FastAPI app: routes, background search worker, static UI.

Runs the search engine on a daemon thread per request (Google SERP discovery
and GPT-4o classification can take a while), returns the search_id
immediately, and exposes progress + lead CRUD for the single-file frontend.
"""
from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from classifier import GptClassifier
from config import Settings, settings as _settings
from db import Store, build_store
from discovery.base import DiscoveryClient, DiscoveryConfigError
from discovery.serp_client import SerperDiscoveryClient
from engine import run_search
from geography import normalize_country
from models import LeadPatch, SearchCreateResponse, SearchRequest, TimeWindow, UsageOut

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings: Settings = _settings
store: Store = build_store(settings)

# ---------------------------------------------------------------------------
# Component wiring (all swappable; mock mode runs fully offline)
# ---------------------------------------------------------------------------

class _UnconfiguredDiscovery(DiscoveryClient):
    name = "unconfigured"

    def __init__(self, errors: list[str]) -> None:
        self._errors = errors

    @property
    def config_errors(self) -> list[str]:
        return self._errors

    def search_posts(self, queries, since, *, results_per_query=25):  # type: ignore[no-untyped-def]
        raise DiscoveryConfigError("; ".join(self._errors))


def build_discovery(s: Settings) -> DiscoveryClient:
    mode = (s.discovery_provider or "auto").lower()
    if mode == "mock" or (mode == "auto" and s.mock_mode):
        from testing.mock_providers import MockDiscoveryClient

        log.info("Discovery provider: mock (offline corpus)")
        return MockDiscoveryClient()
    if mode in ("auto", "serp"):
        if s.serp_configured:
            log.info("Discovery provider: Google SERP (Serper) restricted to %s", s.serper_site_restriction or "all sites")
            return SerperDiscoveryClient(
                s.serper_api_key,
                base_url=s.serper_base_url,
                site_restriction=s.serper_site_restriction,
                results_per_query=s.serper_results_per_query,
                gl=s.serper_gl,
                hl=s.serper_hl,
                timeout_seconds=s.serper_timeout_seconds,
            )
        return _UnconfiguredDiscovery([
            "SERPER_API_KEY is not set — discovery cannot run (or set MOCK_MODE=1 for an offline demo)"
        ])
    raise RuntimeError(f"Unknown DISCOVERY_PROVIDER={mode!r} (auto | serp | mock)")


def build_classifier(s: Settings):
    if s.mock_mode:
        from testing.mock_providers import MockClassifier

        log.info("Classifier: mock (offline, deterministic)")
        return MockClassifier()
    if s.llm_provider == "deepseek":
        log.info("Classifier: DeepSeek %s @ %s (json_object, fail-closed)", s.deepseek_model, s.deepseek_base_url)
        return GptClassifier(
            s.deepseek_api_key,
            model=s.deepseek_model,
            base_url=s.deepseek_base_url,
            provider="deepseek",
            json_mode="json_object",
            timeout_seconds=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )
    if s.llm_provider == "openai":
        log.info("Classifier: OpenAI %s (json_schema, fail-closed)", s.openai_model)
        return GptClassifier(
            s.openai_api_key,
            model=s.openai_model,
            provider="openai",
            json_mode="json_schema",
            timeout_seconds=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )
    raise RuntimeError(f"Unknown LLM_PROVIDER={s.llm_provider!r} (deepseek | openai)")


discovery = build_discovery(settings)
classifier = build_classifier(settings)

# ---------------------------------------------------------------------------
# In-process progress registry (polled by the UI while a search runs)
# ---------------------------------------------------------------------------

_PROGRESS: dict[str, dict[str, Any]] = {}
_PROGRESS_LOCK = threading.Lock()


def _progress_push(search_id: str, stage: str, found: int, accepted: int, scanned: int, message: str = "") -> None:
    with _PROGRESS_LOCK:
        if len(_PROGRESS) > 500:
            _PROGRESS.pop(next(iter(_PROGRESS)), None)
        _PROGRESS[search_id] = {
            "search_id": search_id, "stage": stage, "found": found,
            "accepted": accepted, "scanned": scanned, "message": message,
        }


def _run_search_worker(search_id: str) -> None:
    def cb(stage: str, found: int, accepted: int, scanned: int, message: str = "") -> None:
        _progress_push(search_id, stage, found, accepted, scanned, message)

    try:
        summary = run_search(
            search_id,
            store=store,
            discovery=discovery,
            classifier=classifier,
            settings=settings,
            progress=cb,
        )
        _progress_push(search_id, summary.status, summary.found, summary.accepted, summary.scanned,
                       summary.detail or summary.status)
    except Exception as exc:  # noqa: BLE001 - never leave a search stuck in "running"
        log.exception("Search worker crashed for %s", search_id)
        try:
            store.update_search(search_id, status="failed", error=f"internal error: {exc}",
                                finished_at=datetime.now(UTC))
        except Exception:  # noqa: BLE001
            log.exception("Could not mark search %s failed", search_id)
        _progress_push(search_id, "failed", 0, 0, 0, f"internal error: {exc}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Hyperclients", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchCreateResponse)
def start_search(req: SearchRequest) -> dict[str, str]:
    used = store.searches_used_today()
    if used >= settings.max_searches_per_day:
        raise HTTPException(
            status_code=429,
            detail=f"Daily search limit reached ({used}/{settings.max_searches_per_day}). Try again tomorrow.",
        )
    missing = list(discovery.config_errors) + list(classifier.config_errors)
    if missing:
        raise HTTPException(status_code=503, detail={"message": "Engine is not fully configured", "missing": missing})

    # §0/§6: normalize the free-text country to a canonical code; an unknown
    # value degrades gracefully to the raw text (never a hard reject).
    country_info = normalize_country(req.country)
    country_stored = country_info.code if country_info.recognized else req.country

    row = store.create_search(
        service=req.service,
        country=country_stored,
        lead_type=req.lead_type.value,
        time_window=req.time_window.value,
        leads_needed=req.leads_needed,
    )
    search_id = row["id"]
    _progress_push(search_id, "queued", 0, 0, 0, "queued")
    thread = threading.Thread(target=_run_search_worker, args=(search_id,), daemon=True, name=f"search-{search_id[:8]}")
    thread.start()
    log.info("Search %s queued: service=%r country=%r type=%s window=%s need=%d",
             search_id, req.service, req.country, req.lead_type.value, req.time_window.value, req.leads_needed)
    return {"search_id": search_id, "status": "queued"}


@app.get("/api/search/{search_id}/status")
def search_status(search_id: str) -> dict[str, Any]:
    row = store.get_search(search_id)
    if row is None:
        raise HTTPException(status_code=404, detail="search not found")
    with _PROGRESS_LOCK:
        live = _PROGRESS.get(search_id)
    status = row.get("status") or "queued"
    payload: dict[str, Any] = {
        "search_id": search_id,
        "status": status,
        "service": row.get("service"),
        "country": row.get("country"),
        "lead_type": row.get("lead_type"),
        "time_window": row.get("time_window"),
        "leads_needed": row.get("leads_needed"),
        "found": int(row.get("found_count") or 0),
        "accepted": int(row.get("accepted_count") or 0),
        "scanned": int(row.get("scanned_count") or 0),
        "created_at": row.get("created_at"),
        "finished_at": row.get("finished_at"),
        "error": row.get("error"),
        "message": (live or {}).get("message") or row.get("error"),
    }
    if live and status in ("queued", "running"):
        payload.update({"found": int(live.get("found") or payload["found"]),
                        "accepted": int(live.get("accepted") or payload["accepted"]),
                        "scanned": int(live.get("scanned") or payload["scanned"])})
    return payload


@app.get("/api/searches")
def list_searches(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    rows = store.list_searches(limit=limit)
    return {"searches": rows, "count": len(rows)}


@app.get("/api/leads")
def list_leads(
    search_id: str | None = Query(default=None),
    time_window: str | None = Query(default=None, description="24h | 7d | 14d | 28d — re-filter saved leads by posted_at"),
    status: str | None = Query(default=None),
    limit: int = Query(500, ge=1, le=1000),
) -> dict[str, Any]:
    if time_window is not None and time_window != "all":
        try:
            TimeWindow(time_window)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"invalid time_window {time_window!r}") from exc
    rows = store.list_leads(search_id=search_id, time_window=None if time_window == "all" else time_window,
                            status=status, limit=limit)
    return {"leads": rows, "count": len(rows)}


@app.patch("/api/leads/{lead_id}")
def patch_lead(lead_id: str, body: LeadPatch) -> dict[str, Any]:
    updated = store.patch_lead(lead_id, status=body.status, notes=body.notes)
    if updated is None:
        raise HTTPException(status_code=404, detail="lead not found")
    return updated


@app.get("/api/usage", response_model=UsageOut)
def usage() -> dict[str, Any]:
    used = store.searches_used_today()
    now = datetime.now(UTC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "used": used,
        "limit": settings.max_searches_per_day,
        "remaining": max(settings.max_searches_per_day - used, 0),
        "resets_in_seconds": int((midnight - now).total_seconds()),
    }


# Serve the single-file frontend (last, so /api/* wins).
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    def _no_frontend() -> dict[str, str]:
        return {"message": "frontend/ not found next to backend/"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
