"""LIVE strict-type proof: real GPT-4o + real Serper + real Supabase.

For each requested lead type we wipe the tables, run a real search, and assert
EVERY delivered lead has exactly the requested lead_type (no other buyer type
and no trap leaks in). Ends by leaving one hiring_buyer search's results in
the DB for the UI. Run:  python testing/typed_proof.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import os

import httpx  # noqa: E402

from config import settings  # noqa: E402
from classifier import GptClassifier  # noqa: E402
from db import build_store  # noqa: E402
from discovery.serp_client import SerperDiscoveryClient  # noqa: E402
from engine import run_search  # noqa: E402
from models import LeadType  # noqa: E402

PROJECT_REF = "gtklleletrpajrfypofi"
# Management token comes from the environment — NEVER commit it.
MGMT_TOKEN = os.environ.get("SUPABASE_MGMT_TOKEN", "")


def wipe_tables() -> None:
    r = httpx.post(
        f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query",
        headers={"Authorization": f"Bearer {MGMT_TOKEN}", "Content-Type": "application/json"},
        json={"query": "delete from leads; delete from searches;"}, timeout=60,
    )
    assert r.status_code < 300, r.text


def main() -> int:
    store = build_store(settings)
    if settings.llm_provider == "deepseek":
        classifier = GptClassifier(settings.deepseek_api_key, model=settings.deepseek_model,
                                   base_url=settings.deepseek_base_url, provider="deepseek",
                                   json_mode="json_object", timeout_seconds=60.0, max_retries=2)
    else:
        classifier = GptClassifier(settings.openai_api_key, model=settings.openai_model,
                                   provider="openai", json_mode="json_schema",
                                   timeout_seconds=60.0, max_retries=2)
    discovery = SerperDiscoveryClient(settings.serper_api_key, results_per_query=10)

    failures = 0
    for lead_type in LeadType:
        wipe_tables()
        sid = store.create_search(
            service="a video editor", country="United States", lead_type=lead_type.value,
            time_window="28d", leads_needed=5,
        )["id"]
        t0 = time.time()
        summary = run_search(sid, store=store, discovery=discovery,
                             classifier=classifier, settings=settings)
        leads = store.list_leads(search_id=sid)
        secs = time.time() - t0
        kinds = sorted({l["lead_type"] for l in leads})
        ok = all(l["lead_type"] == lead_type.value for l in leads)
        failures += 0 if ok else 1
        print(f"[{lead_type.value}] status={summary.status} accepted={summary.accepted} "
              f"scanned={summary.scanned} stored={len(leads)} kinds={kinds} "
              f"strict={'OK' if ok else 'FAIL'} ({secs:.0f}s)")
        if summary.detail:
            print(f"   detail: {summary.detail[:300]}")
        for l in leads[:3]:
            print(f"   - {l['lead_type']} | {l['post_url']}")
    # Leave one hiring_buyer search's results visible in the UI.
    wipe_tables()
    lead_type = LeadType.HIRING_BUYER
    sid = store.create_search(service="a video editor", country="United States",
                              lead_type=lead_type.value, time_window="28d", leads_needed=5)["id"]
    summary = run_search(sid, store=store, discovery=discovery, classifier=classifier, settings=settings)
    print(f"\n[final keeper: {lead_type.value}] status={summary.status} "
          f"accepted={summary.accepted} search_id={sid}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
